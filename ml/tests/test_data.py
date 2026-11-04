"""Data boundaries: subject isolation, numerical conventions, and raw timing."""

import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from harss_data.__main__ import main
from harss_data.dataset import (
    Windows, batches, load_prepared, load_split, partition_subjects, validate_windows,
)
from harss_data.normalization import canonical_units, fit_training, standardize
from harss_data.prepare import prepare_recording
from harss_data.recording import (
    EVENT_COLUMNS, load_metadata, read_events, window_recording,
)
from harss_data.source import download, open_dataset
from harss_data.spec import (
    CHANNELS, OFFICIAL_TEST_SUBJECTS, OFFICIAL_TRAIN_SUBJECTS, PERIOD_NS, STANDARD_GRAVITY,
)

REPO = Path(__file__).resolve().parents[2]


def synthetic_windows(subjects):
    """Every subject has every class; source rows are distinct."""
    ids = np.repeat(subjects, 6)
    labels = np.tile(np.arange(6), len(subjects))
    x = np.broadcast_to((ids + labels / 10)[:, None, None], (len(ids), 128, 6)).copy()
    return Windows(x, labels.astype(np.int64), ids.astype(np.int64),
                   np.arange(len(ids), dtype=np.int64))


def metadata():
    return {
        "schema_version": "sensor-record-v1", "session_id": "synthetic-01",
        "participant_id": "p01", "purpose": "development", "activity_id": 1,
        "device_model": "synthetic", "os_version": "synthetic", "placement": "waist",
        "orientation": "portrait, top toward head, screen outward",
        "accelerometer_unit": "m/s^2", "gyroscope_unit": "rad/s",
        "requested_sampling_hz": 50, "stable_start_ns": 0,
        "stable_end_ns": 10_000_000_000,
    }


def event_arrays(count=320, gyro_offset=0, origin=0):
    t = origin + np.arange(count, dtype=np.int64) * PERIOD_NS
    acceleration = np.tile([0, 0, STANDARD_GRAVITY], (count, 1))
    gyro = np.zeros((count, 3))
    return {"accelerometer": (t, acceleration), "gyroscope": (t + gyro_offset, gyro)}


class DataTests(unittest.TestCase):
    def setUp(self):
        self.split = load_split(REPO / "ml/configs/subjects-v1.json")

    def test_held_out_values_cannot_change_fitted_statistics(self):
        source = synthetic_windows(OFFICIAL_TRAIN_SUBJECTS)
        test = synthetic_windows(OFFICIAL_TEST_SUBJECTS)
        first = partition_subjects(source, test, self.split)
        before = fit_training(first["train"], self.split["train"])
        source.x[np.isin(source.subject_id, self.split["validation"])] += 1e6
        test.x += 2e6
        second = partition_subjects(source, test, self.split)
        after = fit_training(second["train"], self.split["train"])
        self.assertEqual(before, after)
        self.assertGreater(float(standardize(second["validation"].x, after).mean()), 1000)

    def test_fit_rejects_wrong_subjects(self):
        with self.assertRaisesRegex(ValueError, "declared training"):
            fit_training(synthetic_windows(OFFICIAL_TRAIN_SUBJECTS), self.split["train"])

    def test_split_rejects_overlap_and_changed_official_test(self):
        for field, subject in (("train", 5), ("test", 1)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                config = copy.deepcopy(self.split)
                config[field].append(subject)
                path = Path(temporary) / "split.json"
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    load_split(path)

    def test_split_rejects_missing_or_noninteger_subjects(self):
        for value in ([], [1.5], [True], [1, 1]):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                config = copy.deepcopy(self.split)
                config["train"] = value
                path = Path(temporary) / "split.json"
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    load_split(path)

    def test_all_windows_retained_once_with_complete_subjects(self):
        source = synthetic_windows(OFFICIAL_TRAIN_SUBJECTS)
        result = partition_subjects(source, synthetic_windows(OFFICIAL_TEST_SUBJECTS), self.split)
        rows = np.concatenate([result[n].source_row for n in ("train", "validation")])
        np.testing.assert_array_equal(np.sort(rows), source.source_row)
        self.assertFalse(set(result["train"].subject_id) & set(result["validation"].subject_id))

    def test_fit_uses_population_statistics_and_handles_constant_channels(self):
        data = synthetic_windows([1])
        data.x[:] = 4
        data.x[:, :, 0] = np.arange(128)
        stats = fit_training(data, [1])
        self.assertAlmostEqual(stats["mean"][0], 63.5)
        self.assertAlmostEqual(stats["std"][0], np.std(np.arange(128), ddof=0))
        self.assertEqual(stats["std"][1], 0)
        self.assertEqual(stats["scale"][1], 1)
        self.assertEqual(stats["fit_window_count"], 6)
        result = standardize(data.x, stats)
        np.testing.assert_array_equal(result[:, :, 1:], 0)
        self.assertEqual(result.dtype, np.float32)

    def test_synthetic_cross_platform_reference(self):
        fixture = json.loads((REPO / "testdata/preprocessing-v1.json").read_text())
        raw = np.tile(fixture["android_input_pattern"], (fixture["repeat_pattern"], 1))
        expected = np.tile(fixture["expected_standardized_pattern"],
                           (fixture["repeat_pattern"], 1))
        actual = standardize(canonical_units(raw), fixture["normalization"])
        self.assertEqual(actual.shape, (128, 6))
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_rejects_changed_channel_order_and_nonfinite_inputs(self):
        data = synthetic_windows([1])
        stats = fit_training(data, [1])
        stats["channels"] = list(reversed(CHANNELS))
        with self.assertRaisesRegex(ValueError, "contract"):
            standardize(data.x, stats)
        data.x[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validate_windows(data)

    def test_invalid_layout_and_labels_are_rejected(self):
        for kind in ("layout", "labels", "rows"):
            with self.subTest(kind=kind):
                data = synthetic_windows([1])
                if kind == "layout":
                    data.x = data.x.transpose(0, 2, 1)
                elif kind == "labels":
                    data.y[0] = 6
                else:
                    data.source_row[1] = data.source_row[0]
                with self.assertRaises(ValueError):
                    validate_windows(data)

    def test_batch_reader_retains_last_batch(self):
        data = synthetic_windows([1])
        self.assertEqual([len(b.x) for b in batches(data, 4)], [4, 2])
        with self.assertRaises(ValueError):
            list(batches(data, 0))

    def test_prepared_roundtrip(self):
        data = synthetic_windows([1])
        data.x = data.x.astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "windows.npz"
            np.savez_compressed(path, **vars(data))
            restored = load_prepared(path)
            np.testing.assert_array_equal(restored.x, data.x)
            np.testing.assert_array_equal(restored.source_row, data.source_row)


class SourceTests(unittest.TestCase):
    def test_rejects_unpinned_archive_before_parsing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.zip"
            path.write_bytes(b"not the official archive")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                open_dataset(path)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                download(path)

    def test_interrupted_or_wrong_download_does_not_create_accepted_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.zip"
            with patch("harss_data.source.urllib.request.urlopen",
                       return_value=io.BytesIO(b"incomplete source")):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    download(path)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])


class RecordingTests(unittest.TestCase):
    def test_asynchronous_streams_are_aligned_without_extrapolation(self):
        events = event_arrays(320, gyro_offset=5_000_000)
        x, starts, summary = window_recording(events, metadata())
        self.assertEqual(len(x), 3)
        self.assertEqual(int(starts[0]), 5_000_000)
        self.assertEqual(summary["invalid_grid_points"], 0)
        np.testing.assert_array_equal(x[:, :, 2], 1)
        np.testing.assert_array_equal(x[:, :, 3:], 0)

    def test_long_gap_splits_windows(self):
        events = event_arrays(400)
        t, xyz = events["gyroscope"]
        keep = (np.arange(400) < 140) | (np.arange(400) >= 220)
        events["gyroscope"] = (t[keep], xyz[keep])
        x, starts, summary = window_recording(events, metadata())
        self.assertGreater(summary["invalid_grid_points"], 0)
        self.assertEqual(len(x), 2)
        for start in starts:
            end = start + 127 * PERIOD_NS
            self.assertTrue(end <= 139 * PERIOD_NS or start >= 220 * PERIOD_NS)

    def test_short_recording_emits_no_partial_window(self):
        x, starts, _ = window_recording(event_arrays(127), metadata())
        self.assertEqual(x.shape, (0, 128, 6))
        self.assertEqual(len(starts), 0)

    def test_stable_interval_is_enforced(self):
        info = metadata()
        info["stable_start_ns"] = 100 * PERIOD_NS
        info["stable_end_ns"] = 300 * PERIOD_NS
        x, starts, _ = window_recording(event_arrays(400), info)
        self.assertEqual(len(x), 2)
        self.assertTrue((starts >= info["stable_start_ns"]).all())
        self.assertTrue((starts + 127 * PERIOD_NS <= info["stable_end_ns"]).all())

    def test_interpolation_does_not_use_values_outside_stable_interval(self):
        events = event_arrays(192)
        for _, xyz in events.values():
            xyz[0] = 1e6
        info = metadata()
        info["stable_start_ns"] = 5_000_000
        x, starts, _ = window_recording(events, info)
        self.assertEqual(int(starts[0]), PERIOD_NS)
        np.testing.assert_array_equal(x[:, :, 0], 0)
        np.testing.assert_array_equal(x[:, :, 2], 1)

    def test_large_monotonic_timestamps_keep_relative_precision(self):
        origin = 8_000_000_000_000_003
        info = metadata()
        info["stable_start_ns"] += origin
        info["stable_end_ns"] += origin
        x, starts, _ = window_recording(event_arrays(192, origin=origin), info)
        np.testing.assert_array_equal(starts, [origin, origin + 64 * PERIOD_NS])
        np.testing.assert_array_equal(x[:, :, 2], 1)

    def write_csv(self, path, rows):
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(EVENT_COLUMNS)
            writer.writerows(rows)

    def test_events_reject_bad_timing_missing_sensors_and_nonfinite_values(self):
        good = [[0, "accelerometer", 0, 0, 9.80665, 3],
                [PERIOD_NS, "accelerometer", 0, 0, 9.80665, 3],
                [0, "gyroscope", 0, 0, 0, 3],
                [PERIOD_NS, "gyroscope", 0, 0, 0, 3]]
        cases = [good + [good[1]], good[:2], good + [[-1, "gyroscope", 0, 0, 0, 3]],
                 good + [[2 * PERIOD_NS, "gyroscope", "nan", 0, 0, 3]],
                 good + [[2 * PERIOD_NS, "unknown", 0, 0, 0, 3]]]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.csv"
            for rows in cases:
                with self.subTest(rows=rows):
                    self.write_csv(path, rows)
                    with self.assertRaises(ValueError):
                        read_events(path)

    def test_metadata_rejects_wrong_units_labels_and_intervals(self):
        cases = [("accelerometer_unit", "g"), ("activity_id", 0),
                 ("activity_id", True), ("stable_end_ns", 0),
                 ("requested_sampling_hz", -50), ("purpose", "train")]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "session.json"
            for field, value in cases:
                with self.subTest(field=field, value=value):
                    info = metadata()
                    info[field] = value
                    path.write_text(json.dumps(info))
                    with self.assertRaises(ValueError):
                        load_metadata(path)

    def test_offline_recording_command_preserves_session_and_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            info = metadata()
            (root / "session.json").write_text(json.dumps(info))
            fixture = json.loads((REPO / "testdata/preprocessing-v1.json").read_text())
            stats_path = root / "normalization.json"
            stats_path.write_text(json.dumps(fixture["normalization"]))
            before = stats_path.read_bytes()
            rows = []
            for sensor, (timestamps, xyz) in event_arrays(192).items():
                rows.extend([int(t), sensor, *values, 3] for t, values in zip(timestamps, xyz))
            self.write_csv(root / "events.csv", rows)
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["prepare-recording", "--metadata", str(root / "session.json"),
                             "--events", str(root / "events.csv"),
                             "--normalization", str(stats_path), "--output", str(root / "out")])
            self.assertEqual(code, 0)
            self.assertEqual(stats_path.read_bytes(), before)
            with np.load(root / "out/recording.npz", allow_pickle=False) as result:
                self.assertEqual(result["x"].shape, (2, 128, 6))
                np.testing.assert_array_equal(result["x"], 0)
                self.assertEqual(set(result["session_id"]), {"synthetic-01"})
                np.testing.assert_array_equal(result["y"], [0, 0])
            summary = json.loads((root / "out/recording-summary.json").read_text())
            self.assertEqual(summary["complete_windows"], 2)
            self.assertAlmostEqual(summary["timing"]["accelerometer"]["observed_mean_hz"], 50)

    def test_final_test_requires_explicit_enablement_before_reading_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            info = metadata()
            info["purpose"] = "final_test"
            (root / "session.json").write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError, "--allow-final-test"):
                prepare_recording(root / "session.json", root / "missing.csv",
                                  root / "missing.json", root / "out")


if __name__ == "__main__":
    unittest.main()
