"""Offline reference for asynchronous phone events; no Android app code."""

import csv
import json
from pathlib import Path
import re

import numpy as np

from .normalization import canonical_units
from .spec import MAX_GAP_NS, PERIOD_NS, STRIDE, WINDOW_SIZE

EVENT_COLUMNS = ["timestamp_ns", "sensor", "x", "y", "z", "accuracy"]


def load_metadata(path: Path) -> dict:
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Recording metadata must be a JSON object")
    required_text = ("session_id", "participant_id", "device_model", "os_version",
                     "placement", "orientation")
    for name in required_text:
        value = metadata.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or empty recording metadata: {name}")
    for name in ("session_id", "participant_id"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", metadata[name]):
            raise ValueError(f"Invalid anonymous identifier: {name}")
    expected = {
        "schema_version": "sensor-record-v1",
        "placement": "waist",
        "accelerometer_unit": "m/s^2",
        "gyroscope_unit": "rad/s",
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"Recording metadata {name} must be {value}")
    if metadata.get("purpose") not in ("development", "final_test"):
        raise ValueError("Recording purpose must be development or final_test")
    activity = metadata.get("activity_id")
    if type(activity) is not int or activity not in range(1, 7):
        raise ValueError("Recording activity_id must be the original UCI ID 1..6")
    rate = metadata.get("requested_sampling_hz")
    if (type(rate) not in (int, float) or not np.isfinite(rate) or rate <= 0):
        raise ValueError("requested_sampling_hz must be positive and finite")
    for name in ("stable_start_ns", "stable_end_ns"):
        value = metadata.get(name)
        if type(value) is not int or not 0 <= value < np.iinfo(np.int64).max:
            raise ValueError(f"{name} must be a nonnegative int64 timestamp")
    if metadata["stable_start_ns"] >= metadata["stable_end_ns"]:
        raise ValueError("Stable interval must have positive duration")
    return metadata


def read_events(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    times = {sensor: [] for sensor in ("accelerometer", "gyroscope")}
    values = {sensor: [] for sensor in times}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EVENT_COLUMNS:
            raise ValueError(f"Event CSV header must be {','.join(EVENT_COLUMNS)}")
        for line, row in enumerate(reader, start=2):
            try:
                sensor = row["sensor"]
                if sensor not in times or None in row or None in row.values():
                    raise ValueError("Unknown sensor or malformed event columns")
                timestamp = int(row["timestamp_ns"])
                if not 0 <= timestamp < np.iinfo(np.int64).max:
                    raise ValueError("Timestamp is outside int64 range")
                if times[sensor] and timestamp <= times[sensor][-1]:
                    raise ValueError("Per-sensor timestamps must strictly increase")
                xyz = [float(row[axis]) for axis in ("x", "y", "z")]
                if not np.isfinite(xyz).all():
                    raise ValueError("Sensor readings must be finite")
                if int(row["accuracy"]) not in range(-1, 4):
                    raise ValueError("Invalid sensor accuracy status")
                times[sensor].append(timestamp)
                values[sensor].append(xyz)
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid event at CSV line {line}: {error}") from error
    result = {}
    for sensor in times:
        if len(times[sensor]) < 2:
            raise ValueError(f"At least two events are required for {sensor}")
        result[sensor] = (np.asarray(times[sensor], dtype=np.int64),
                          np.asarray(values[sensor], dtype=np.float64))
    return result


def timing_summary(events: dict) -> dict:
    result = {}
    for sensor, (timestamps, _) in events.items():
        intervals = np.diff(timestamps).astype(np.float64)
        result[sensor] = {
            "events": len(timestamps),
            "observed_mean_hz": float((len(timestamps) - 1) * 1e9 /
                                      (int(timestamps[-1]) - int(timestamps[0]))),
            "median_interval_ms": float(np.median(intervals) / 1e6),
            "maximum_interval_ms": float(intervals.max() / 1e6),
            "gaps_over_100ms": int((intervals > MAX_GAP_NS).sum()),
        }
    return result


def window_recording(events: dict, metadata: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Interpolate within the stable overlap, splitting at excessive gaps."""
    stable_events = {}
    for sensor, (t, xyz) in events.items():
        begin = np.searchsorted(t, metadata["stable_start_ns"], side="left")
        stop = np.searchsorted(t, metadata["stable_end_ns"], side="right")
        if stop - begin < 2:
            raise ValueError(f"At least two stable events are required for {sensor}")
        stable_events[sensor] = (t[begin:stop], xyz[begin:stop])
    start = max(int(t[0]) for t, _ in stable_events.values())
    end = min(int(t[-1]) for t, _ in stable_events.values())
    if end <= start:
        raise ValueError("Sensors have no common stable interval")
    count = (end - start) // PERIOD_NS + 1
    if count > 1_000_000:
        raise ValueError("Recording exceeds the offline reference's one-million-point limit")
    grid = np.arange(count, dtype=np.int64) * PERIOD_NS + start
    valid = np.ones(count, dtype=bool)
    channels = []
    for sensor in ("accelerometer", "gyroscope"):
        t, xyz = stable_events[sensor]
        right = np.searchsorted(t, grid, side="left")
        exact = t[np.minimum(right, len(t) - 1)] == grid
        left = np.maximum(right - 1, 0)
        next_index = np.minimum(right, len(t) - 1)
        valid &= exact | ((right > 0) & (right < len(t)) &
                          ((t[next_index] - t[left]) <= MAX_GAP_NS))
        # Subtract the integer origin before converting nanoseconds to floats.
        source_time = (t - start).astype(np.float64)
        target_time = (grid - start).astype(np.float64)
        channels.extend(np.interp(target_time, source_time, xyz[:, axis])
                        for axis in range(3))
    aligned = canonical_units(np.stack(channels, axis=-1))
    boundaries = np.diff(np.r_[False, valid, False].astype(np.int8))
    windows, starts = [], []
    for begin, stop in zip(np.flatnonzero(boundaries == 1),
                           np.flatnonzero(boundaries == -1)):
        for offset in range(int(begin), int(stop) - WINDOW_SIZE + 1, STRIDE):
            windows.append(aligned[offset:offset + WINDOW_SIZE])
            starts.append(grid[offset])
    x = np.asarray(windows, dtype=np.float64).reshape(-1, WINDOW_SIZE, 6)
    summary = {
        "grid_start_ns": start,
        "grid_end_ns": int(grid[-1]),
        "grid_points": count,
        "invalid_grid_points": int((~valid).sum()),
        "complete_windows": len(x),
        "timing": timing_summary(events),
    }
    return x, np.asarray(starts, dtype=np.int64), summary
