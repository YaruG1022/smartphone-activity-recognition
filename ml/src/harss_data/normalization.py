"""Training-only channel statistics and the shared numerical transform."""

import numpy as np

from .dataset import Windows, validate_windows
from .spec import CHANNELS, CONTRACT_VERSION, OFFICIAL_TRAIN_SUBJECTS, STANDARD_GRAVITY


def fit_training(data: Windows, expected_subjects: list[int]) -> dict:
    validate_windows(data)
    if (not set(expected_subjects) <= set(OFFICIAL_TRAIN_SUBJECTS)
            or set(data.subject_id) != set(expected_subjects)):
        raise ValueError("Statistics may only be fitted on the declared training subjects")
    values = np.asarray(data.x, dtype=np.float64)
    mean = values.mean(axis=(0, 1))
    std = values.std(axis=(0, 1), ddof=0)
    return {
        "preprocessing_version": CONTRACT_VERSION,
        "channels": list(CHANNELS),
        "fit_partition": "train",
        "fit_subjects": sorted(expected_subjects),
        "fit_window_count": len(values),
        "fit_time_point_count": len(values) * values.shape[1],
        "ddof": 0,
        "minimum_std": 1e-8,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "scale": np.where(std < 1e-8, 1.0, std).tolist(),
    }


def validate_statistics(stats: dict) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(stats, dict):
        raise ValueError("Normalization must be a JSON object")
    if (stats.get("preprocessing_version") != CONTRACT_VERSION
            or stats.get("channels") != list(CHANNELS)
            or stats.get("fit_partition") != "train"
            or stats.get("ddof") != 0
            or stats.get("minimum_std") != 1e-8):
        raise ValueError("Unsupported normalization contract")
    mean, std, scale = (np.asarray(stats.get(key), dtype=np.float64)
                        for key in ("mean", "std", "scale"))
    if any(a.shape != (6,) or not np.isfinite(a).all() for a in (mean, std, scale)):
        raise ValueError("Normalization requires six finite means/stds/scales")
    if (std < 0).any() or not np.array_equal(scale, np.where(std < 1e-8, 1.0, std)):
        raise ValueError("Normalization scale does not match its declared standard deviation")
    return mean, scale


def standardize(values: np.ndarray, stats: dict) -> np.ndarray:
    mean, scale = validate_statistics(stats)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim not in (2, 3) or values.shape[-1] != 6 or not np.isfinite(values).all():
        raise ValueError("Expected finite six-channel signal values")
    with np.errstate(over="raise", invalid="raise"):
        output = ((values - mean) / scale).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Standardized signal is not representable as finite float32")
    return output


def canonical_units(values: np.ndarray) -> np.ndarray:
    """Convert Android accelerometer m/s^2 and gyroscope rad/s to v1 units."""
    values = np.array(values, dtype=np.float64, copy=True)
    if values.ndim not in (2, 3) or values.shape[-1] != 6 or not np.isfinite(values).all():
        raise ValueError("Expected finite six-channel Android values")
    values[..., :3] /= STANDARD_GRAVITY
    return values
