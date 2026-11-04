"""Use a fixed six-class order for every metric and confusion matrix."""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from harss_data.spec import CLASSES


def classification_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or not len(y_true) or y_pred.shape != y_true.shape:
        raise ValueError("Expected matching, nonempty label vectors")
    for values in (y_true, y_pred):
        if not np.issubdtype(values.dtype, np.integer) or not np.isin(values, range(6)).all():
            raise ValueError("Labels must be integers in 0..5")
    labels = list(range(6))
    report = classification_report(y_true, y_pred, labels=labels, target_names=CLASSES,
                                    output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "per_class": {name: {"precision": float(report[name]["precision"]),
                               "recall": float(report[name]["recall"]),
                               "f1": float(report[name]["f1-score"]),
                               "support": int(report[name]["support"])} for name in CLASSES},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
