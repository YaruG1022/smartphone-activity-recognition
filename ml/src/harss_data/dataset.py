"""Read six-channel UCI windows and partition complete subjects."""

from dataclasses import dataclass
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .spec import (
    CHANNELS, CLASSES, OFFICIAL_TEST_SUBJECTS, OFFICIAL_TRAIN_SUBJECTS, WINDOW_SIZE,
)


@dataclass
class Windows:
    x: np.ndarray
    y: np.ndarray
    subject_id: np.ndarray
    source_row: np.ndarray

    def take(self, selection) -> "Windows":
        return Windows(*(a[selection] for a in
                         (self.x, self.y, self.subject_id, self.source_row)))


def validate_windows(data: Windows) -> None:
    count = len(data.x)
    if data.x.shape != (count, WINDOW_SIZE, len(CHANNELS)) or count == 0:
        raise ValueError("Expected nonempty [N, 128, 6] signal windows")
    if not np.isfinite(data.x).all():
        raise ValueError("Signal windows contain nonfinite values")
    for name in ("y", "subject_id", "source_row"):
        values = getattr(data, name)
        if values.shape != (count,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must be an integer vector matching the windows")
    if not np.isin(data.y, np.arange(len(CLASSES))).all():
        raise ValueError("Prepared labels must be in 0..5")
    if not np.isin(data.subject_id, np.arange(1, 31)).all():
        raise ValueError("UCI subject IDs must be in 1..30")
    if (data.source_row < 0).any() or len(np.unique(data.source_row)) != count:
        raise ValueError("Source row indices must be nonnegative and unique")


def read_official(archive: ZipFile, partition: str) -> Windows:
    if partition not in ("train", "test"):
        raise ValueError("Official partition must be train or test")
    prefix = f"UCI HAR Dataset/{partition}/"
    signals = []
    for channel in CHANNELS:
        name = f"{prefix}Inertial Signals/{channel}_{partition}.txt"
        values = np.loadtxt(io.BytesIO(archive.read(name)), dtype=np.float64, ndmin=2)
        signals.append(values)
    x = np.stack(signals, axis=-1)
    # Read as floats first so malformed fractional IDs cannot be truncated.
    def integers(filename: str) -> np.ndarray:
        values = np.loadtxt(io.BytesIO(archive.read(prefix + filename)), ndmin=1)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Noninteger metadata in {filename}")
        return values.astype(np.int64)

    y = integers(f"y_{partition}.txt") - 1
    subjects = integers(f"subject_{partition}.txt")
    data = Windows(x, y, subjects, np.arange(len(x), dtype=np.int64))
    validate_windows(data)
    expected_count = 7352 if partition == "train" else 2947
    expected_subjects = (OFFICIAL_TRAIN_SUBJECTS if partition == "train"
                         else OFFICIAL_TEST_SUBJECTS)
    if len(x) != expected_count or set(subjects) != set(expected_subjects):
        raise ValueError(f"Unexpected official {partition} rows or subject IDs")
    if set(y) != set(range(6)):
        raise ValueError(f"Official {partition} does not contain all six classes")
    return data


def load_split(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_split(config)
    return config


def validate_split(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("Subject split must be a JSON object")
    if config.get("version") != "subjects-v1" or config.get("dataset") != "uci-har-v1":
        raise ValueError("Unsupported subject split version or dataset")
    groups = []
    for name in ("train", "validation", "test"):
        values = config.get(name)
        if (not isinstance(values, list) or not values
                or any(type(v) is not int for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"Invalid subject list: {name}")
        groups.append(set(values))
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Subject partitions overlap")
    if groups[0] | groups[1] != set(OFFICIAL_TRAIN_SUBJECTS):
        raise ValueError("Train/validation must partition official training subjects")
    if groups[2] != set(OFFICIAL_TEST_SUBJECTS):
        raise ValueError("Official test subjects must remain unchanged")


def partition_subjects(train: Windows, test: Windows, split: dict) -> dict[str, Windows]:
    validate_split(split)
    result = {
        "train": train.take(np.isin(train.subject_id, split["train"])),
        "validation": train.take(np.isin(train.subject_id, split["validation"])),
        "test": test,
    }
    for name, data in result.items():
        validate_windows(data)
        if set(data.subject_id) != set(split[name]) or set(data.y) != set(range(6)):
            raise ValueError(f"Missing subjects or classes in {name}")
    if len(result["train"].x) + len(result["validation"].x) != len(train.x):
        raise ValueError("Some official training windows were lost")
    return result


def load_prepared(path: Path) -> Windows:
    with np.load(path, allow_pickle=False) as arrays:
        data = Windows(*(arrays[k] for k in ("x", "y", "subject_id", "source_row")))
    validate_windows(data)
    if data.x.dtype != np.float32:
        raise ValueError("Prepared signal dtype must be float32")
    return data


def batches(data: Windows, batch_size: int = 32):
    """Yield all arrays in source order, retaining the final partial batch."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    for start in range(0, len(data.x), batch_size):
        yield data.take(slice(start, start + batch_size))
