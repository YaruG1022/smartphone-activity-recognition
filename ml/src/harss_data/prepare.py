"""Create traceable data artifacts without training or selecting a model."""

import json
from pathlib import Path
import platform

import numpy as np

from . import __version__
from .dataset import (
    batches, load_prepared, load_split, partition_subjects, read_official, validate_split,
)
from .normalization import fit_training, standardize, validate_statistics
from .recording import load_metadata, read_events, window_recording
from .source import open_dataset, sha256
from .spec import (
    CHANNELS, CLASSES, CONTRACT_VERSION, DATASET_SHA256, SAMPLE_HZ, STRIDE, UNITS,
    WINDOW_SIZE, WRAPPER_SHA256,
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def new_output_directory(path: Path) -> None:
    """Do not mix a new run with old artifacts or overwrite an existing run."""
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def prepare_public(archive_path: Path, split_path: Path, output: Path) -> dict:
    split = load_split(split_path)
    with open_dataset(archive_path) as archive:
        expected_labels = "\n".join(f"{i + 1} {name}" for i, name in enumerate(CLASSES))
        actual_labels = archive.read("UCI HAR Dataset/activity_labels.txt").decode().strip()
        if actual_labels != expected_labels:
            raise ValueError("Source activity label mapping does not match v1")
        official_train = read_official(archive, "train")
        official_test = read_official(archive, "test")
    partitions = partition_subjects(official_train, official_test, split)
    stats = fit_training(partitions["train"], split["train"])
    new_output_directory(output)
    write_json(output / "normalization.json", stats)
    partition_info = {}
    for name, data in partitions.items():
        x = standardize(data.x, stats)
        path = output / f"{name}.npz"
        np.savez_compressed(path, x=x, y=data.y, subject_id=data.subject_id,
                            source_row=data.source_row)
        partition_info[name] = {
            "file": path.name,
            "official_partition": "test" if name == "test" else "train",
            "windows": len(x),
            "shape": list(x.shape),
            "subjects": sorted(map(int, set(data.subject_id))),
            "class_counts": {label: int((data.y == i).sum())
                             for i, label in enumerate(CLASSES)},
        }
    training = partitions["train"]
    indices = [int(np.flatnonzero(training.y == i)[0]) for i in range(6)]
    reference = training.take(indices)
    write_json(output / "replay.json", {
        "preprocessing_version": CONTRACT_VERSION,
        "purpose": "training-source preprocessing replay; not classifier predictions",
        "channels": list(CHANNELS),
        "source_partition": "train",
        "source_rows": reference.source_row.tolist(),
        "subject_ids": reference.subject_id.tolist(),
        "labels": reference.y.tolist(),
        "canonical_input": reference.x.tolist(),
        "standardized_output": standardize(reference.x, stats).tolist(),
        "normalization_file": "normalization.json",
    })
    files = ["train.npz", "validation.npz", "test.npz", "normalization.json", "replay.json"]
    manifest = {
        "schema_version": "data-manifest-v1",
        "dataset": "uci-har-v1",
        "preprocessing_version": CONTRACT_VERSION,
        "software": {"harss_data": __version__, "python": platform.python_version(),
                     "numpy": np.__version__},
        "source_archive_sha256": sha256(archive_path),
        "split_config_sha256": sha256(split_path),
        "split": split,
        "channels": list(CHANNELS),
        "canonical_units": list(UNITS),
        "classes": list(CLASSES),
        "sample_hz": SAMPLE_HZ,
        "window_size": WINDOW_SIZE,
        "source_stride": STRIDE,
        "partitions": partition_info,
        "artifacts": {name: sha256(output / name) for name in files},
    }
    write_json(output / "manifest.json", manifest)
    return verify_prepared(output)


def verify_prepared(output: Path) -> dict:
    """Reopen every artifact and check lineage, counts, layout, and numeric replay."""
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "data-manifest-v1"
            or manifest.get("preprocessing_version") != CONTRACT_VERSION
            or manifest.get("channels") != list(CHANNELS)
            or manifest.get("classes") != list(CLASSES)):
        raise ValueError("Unsupported prepared-data manifest")
    expected_files = {"train.npz", "validation.npz", "test.npz",
                      "normalization.json", "replay.json"}
    if set(manifest["artifacts"]) != expected_files:
        raise ValueError("Prepared manifest has an unexpected artifact list")
    for name, digest in manifest["artifacts"].items():
        if sha256(output / name) != digest:
            raise ValueError(f"Artifact hash mismatch: {name}")
    stats = json.loads((output / "normalization.json").read_text(encoding="utf-8"))
    validate_statistics(stats)
    split = manifest["split"]
    validate_split(split)
    if manifest["source_archive_sha256"] not in (DATASET_SHA256, WRAPPER_SHA256):
        raise ValueError("Manifest does not identify a pinned source archive")
    if set(stats["fit_subjects"]) != set(split["train"]):
        raise ValueError("Normalization fit subjects do not match training split")
    prepared = {}
    for name in ("train", "validation", "test"):
        data = load_prepared(output / f"{name}.npz")
        info = manifest["partitions"][name]
        if (set(data.subject_id) != set(split[name]) or len(data.x) != info["windows"]
                or list(data.x.shape) != info["shape"]
                or info["subjects"] != sorted(split[name])):
            raise ValueError(f"Prepared {name} does not match its manifest")
        if info["class_counts"] != {label: int((data.y == i).sum())
                                     for i, label in enumerate(CLASSES)}:
            raise ValueError(f"Class counts changed for {name}")
        prepared[name] = data
    if set(prepared["train"].source_row) & set(prepared["validation"].source_row):
        raise ValueError("Train and validation share source windows")
    source_rows = np.concatenate([prepared[n].source_row for n in ("train", "validation")])
    if (not np.array_equal(np.sort(source_rows), np.arange(7352))
            or not np.array_equal(prepared["test"].source_row, np.arange(2947))):
        raise ValueError("Prepared partitions do not retain the official source rows exactly once")
    training = prepared["train"]
    if (stats["fit_window_count"] != len(training.x)
            or stats["fit_time_point_count"] != len(training.x) * WINDOW_SIZE):
        raise ValueError("Normalization fit count does not match training data")
    # Constant channels have zero variance and use scale 1 by contract.
    expected_std = np.where(np.asarray(stats["std"]) < 1e-8, 0.0, 1.0)
    np.testing.assert_allclose(training.x.mean(axis=(0, 1), dtype=np.float64), 0, atol=1e-6)
    np.testing.assert_allclose(training.x.std(axis=(0, 1), dtype=np.float64),
                               expected_std, atol=1e-6)
    fixture = json.loads((output / "replay.json").read_text(encoding="utf-8"))
    transformed = standardize(np.asarray(fixture["canonical_input"]), stats)
    np.testing.assert_allclose(transformed, fixture["standardized_output"],
                               atol=1e-7, rtol=1e-7)
    for i, row in enumerate(fixture["source_rows"]):
        index = np.flatnonzero(training.source_row == row)
        if (len(index) != 1 or training.y[index[0]] != fixture["labels"][i]
                or training.subject_id[index[0]] != fixture["subject_ids"][i]):
            raise ValueError("Replay fixture does not identify its training-source window")
        np.testing.assert_array_equal(transformed[i], training.x[index[0]])
    batch = next(batches(training))
    return {
        "verified": True,
        "windows": {name: len(data.x) for name, data in prepared.items()},
        "first_batch_shape": list(batch.x.shape),
        "dtype": str(batch.x.dtype),
        "normalization_fit_subjects": stats["fit_subjects"],
        "source_archive_sha256": manifest["source_archive_sha256"],
    }


def prepare_recording(metadata_path: Path, events_path: Path, normalization_path: Path,
                      output: Path, allow_final_test: bool = False) -> dict:
    metadata = load_metadata(metadata_path)
    if metadata["purpose"] == "final_test" and not allow_final_test:
        raise ValueError("Final-test processing requires --allow-final-test after method freezing")
    stats = json.loads(normalization_path.read_text(encoding="utf-8"))
    validate_statistics(stats)
    events = read_events(events_path)
    raw, starts, summary = window_recording(events, metadata)
    if not len(raw):
        raise ValueError("No complete stable windows remain after timing-gap checks")
    x = standardize(raw, stats)
    new_output_directory(output)
    path = output / "recording.npz"
    np.savez_compressed(path, x=x,
                        y=np.full(len(x), metadata["activity_id"] - 1, dtype=np.int64),
                        window_start_ns=starts,
                        session_id=np.full(len(x), metadata["session_id"]),
                        participant_id=np.full(len(x), metadata["participant_id"]))
    summary.update({
        "preprocessing_version": CONTRACT_VERSION,
        "metadata": metadata,
        "metadata_sha256": sha256(metadata_path),
        "events_sha256": sha256(events_path),
        "normalization_sha256": sha256(normalization_path),
        "artifact_sha256": sha256(path),
        "final_test_explicitly_enabled": allow_final_test,
    })
    write_json(output / "recording-summary.json", summary)
    return summary
