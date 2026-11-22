import unittest

import numpy as np
import torch

from harss_data.dataset import Windows
from harss_ssl.pretraining import span_mask
from harss_ssl.subsets import labeled_subsets


class SelfSupervisedTest(unittest.TestCase):
    def test_mask_covers_two_nonoverlapping_spans(self):
        first = span_mask(64, 128, 16, torch.Generator().manual_seed(570))
        repeat = span_mask(64, 128, 16, torch.Generator().manual_seed(570))
        self.assertTrue(torch.equal(first, repeat))
        self.assertGreater(len(first.unique(dim=0)), 1)
        for row in first.numpy():
            changes = np.diff(np.r_[False, row, False].astype(int))
            lengths = (np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)).tolist()
            self.assertIn(lengths, ([16, 16], [32]))

    def test_label_budgets_are_nested_with_all_subject_activity_strata(self):
        subjects = np.repeat(np.arange(1, 5), 72)
        labels = np.tile(np.repeat(np.arange(6), 12), 4)
        rows = np.arange(288)
        windows = Windows(np.zeros((288, 128, 6), dtype=np.float32), labels, subjects, rows)
        selected = labeled_subsets(windows, (10, 25, 100), 570)
        repeated = labeled_subsets(windows, (10, 25, 100), 570)
        other_seed = labeled_subsets(windows, (10, 25, 100), 571)
        strata = set(zip(subjects, labels))
        self.assertEqual([len(subset.x) for subset in selected.values()], [29, 72, 288])
        self.assertTrue(set(selected[10].source_row) < set(selected[25].source_row))
        self.assertEqual(selected[100].source_row.tolist(), rows.tolist())
        self.assertFalse(np.array_equal(selected[10].source_row, other_seed[10].source_row))
        for budget, subset in selected.items():
            self.assertEqual(set(zip(subset.subject_id, subset.y)), strata)
            np.testing.assert_array_equal(subset.source_row, repeated[budget].source_row)


if __name__ == "__main__":
    unittest.main()
