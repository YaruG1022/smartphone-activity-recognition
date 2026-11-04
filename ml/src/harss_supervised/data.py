"""Verify training/validation lineage without opening test signal arrays."""

from dataclasses import dataclass
import json
from pathlib import Path

from harss_data.dataset import Windows, load_prepared, validate_split
from harss_data.normalization import validate_statistics
from harss_data.source import sha256
from harss_data.spec import CHANNELS, CLASSES, CONTRACT_VERSION, DATASET_SHA256, WRAPPER_SHA256


@dataclass
class TrainingData:
    train: Windows
    validation: Windows
    manifest: dict
    provenance: dict


def load_training_data(directory: Path) -> TrainingData:
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "data-manifest-v1"
            or manifest.get("preprocessing_version") != CONTRACT_VERSION
            or manifest.get("channels") != list(CHANNELS)
            or manifest.get("classes") != list(CLASSES)
            or manifest.get("source_archive_sha256") not in (DATASET_SHA256, WRAPPER_SHA256)):
        raise ValueError("Unexpected data manifest or preprocessing contract")
    validate_split(manifest["split"])
    paths = {name: directory / name for name in
             ("train.npz", "validation.npz", "normalization.json")}
    for name, path in paths.items():
        if sha256(path) != manifest["artifacts"][name]:
            raise ValueError(f"Training artifact hash mismatch: {name}")
    stats = json.loads(paths["normalization.json"].read_text(encoding="utf-8"))
    validate_statistics(stats)
    if set(stats["fit_subjects"]) != set(manifest["split"]["train"]):
        raise ValueError("Normalization fit subjects differ from the training split")
    partitions = {}
    for name in ("train", "validation"):
        data = load_prepared(directory / f"{name}.npz")
        if (set(data.subject_id) != set(manifest["split"][name])
                or len(data.x) != manifest["partitions"][name]["windows"]):
            raise ValueError(f"Unexpected prepared {name} subjects or row count")
        partitions[name] = data
    if (set(partitions["train"].subject_id) & set(partitions["validation"].subject_id)
            or set(partitions["train"].source_row) & set(partitions["validation"].source_row)):
        raise ValueError("Training and validation overlap")
    if stats["fit_window_count"] != len(partitions["train"].x):
        raise ValueError("Normalization fit count differs from the training set")
    return TrainingData(partitions["train"], partitions["validation"], manifest, {
        "manifest_sha256": sha256(manifest_path),
        "training_artifact_hashes": {name: sha256(path) for name, path in paths.items()},
        "source_archive_sha256": manifest["source_archive_sha256"],
        "split_config_sha256": manifest["split_config_sha256"],
        "preprocessing_version": CONTRACT_VERSION,
        "test_arrays_opened": False,
        "test_scores_computed": False,
    })
