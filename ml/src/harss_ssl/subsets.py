"""Choose reproducible label budgets, stratified by subject and activity."""

import hashlib

import numpy as np

from harss_data.spec import CLASSES


def labeled_subsets(data, percentages, seed):
    rng = np.random.default_rng(seed)
    strata = np.unique(np.column_stack((data.subject_id, data.y)), axis=0)
    first = []
    remaining = []
    priorities = []
    for subject, label in strata:
        members = rng.permutation(np.flatnonzero((data.subject_id == subject) & (data.y == label)))
        first.append(members[0])
        remaining.append(members[1:])
        priorities.append(np.arange(1, len(members)) / len(members))
    # Guarantee one window per stratum. Rank later windows by their fractional
    # position within each shuffled stratum to keep larger budgets proportional.
    order = np.argsort(np.concatenate(priorities), kind="stable")
    ranked = np.concatenate((first, np.concatenate(remaining)[order]))
    result = {}
    for percent in percentages:
        # Round half up to a whole window and take a nested prefix.
        count = (len(data.x) * percent + 50) // 100
        if count < len(strata):
            raise ValueError(f"{percent}% label budget cannot cover all subject/activity strata")
        indices = np.sort(ranked[:count])
        result[percent] = data.take(indices)
    return result


def subset_summary(data, percent, seed):
    rows = data.source_row.astype("<i8", copy=False)
    return {
        "seed": seed,
        "label_percent": percent,
        "training_windows": len(data.x),
        "source_rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
        "class_counts": {name: int(np.count_nonzero(data.y == index))
                         for index, name in enumerate(CLASSES)},
        "subject_counts": {str(subject): int(np.count_nonzero(data.subject_id == subject))
                           for subject in np.unique(data.subject_id)},
    }
