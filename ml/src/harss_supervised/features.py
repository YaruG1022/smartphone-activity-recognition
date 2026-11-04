"""Thirty statistical features derived from the same standardized windows."""

import numpy as np

from harss_data.spec import CHANNELS

STATISTICS = ("mean", "std", "min", "max", "rms")
FEATURE_NAMES = tuple(f"{channel}:{stat}" for channel in CHANNELS for stat in STATISTICS)


def statistical_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (128, 6) or not np.isfinite(x).all():
        raise ValueError("Features require finite standardized [N,128,6] windows")
    values = np.stack((x.mean(axis=1), x.std(axis=1, ddof=0),
                       x.min(axis=1), x.max(axis=1), np.sqrt((x * x).mean(axis=1))), axis=-1)
    return values.reshape(len(x), 30)
