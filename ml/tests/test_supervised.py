"""Supervised boundaries, independent expected metrics, and portable exports."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from harss_data.dataset import Windows
from harss_data.normalization import fit_training, standardize
from harss_data.prepare import write_json
from harss_data.source import sha256
from harss_data.spec import CHANNELS, CLASSES, CONTRACT_VERSION, WRAPPER_SHA256
from harss_supervised.data import TrainingData, load_training_data
from harss_supervised.experiment import aggregate_runs, export_check, load_config, validate_config
from harss_supervised.features import statistical_features
from harss_supervised.metrics import classification_metrics
from harss_supervised.models import build_model
from harss_supervised.training import load_checkpoint, set_seed, train_logistic

REPO = Path(__file__).resolve().parents[2]
CONFIG = load_config(REPO / "ml/configs/supervised-v1.json")


def fake_windows(subjects):
    ids = np.repeat(subjects, 6).astype(np.int64)
    labels = np.tile(np.arange(6, dtype=np.int64), len(subjects))
    rng = np.random.default_rng(23)
    x = (rng.normal(0, 0.01, (len(ids), 128, 6)) + labels[:, None, None]).astype(np.float32)
    return Windows(x, labels, ids, np.arange(len(ids), dtype=np.int64))


class SupervisedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_seed(570, 2)

    def test_statistical_features_have_known_order_and_values(self):
        x = np.zeros((1, 128, 6), dtype=np.float32)
        x[0, :, 0] = np.arange(128)
        x[0, :, 1] = -3
        f = statistical_features(x)
        self.assertEqual(f.shape, (1, 30))
        np.testing.assert_allclose(f[0, :5], [63.5, np.std(np.arange(128)), 0, 127,
                                            np.sqrt(np.mean(np.arange(128, dtype=float)**2))])
        np.testing.assert_array_equal(f[0, 5:10], [-3, 0, -3, -3, 3])

    def test_fixed_six_class_metrics_match_hand_computation(self):
        result = classification_metrics(np.arange(6), np.array([0, 0, 2, 2, 4, 5]))
        self.assertAlmostEqual(result["accuracy"], 4 / 6)
        self.assertAlmostEqual(result["macro_f1"], 5 / 9)
        self.assertEqual(result["per_class"]["SITTING"]["recall"], 0)
        self.assertEqual(np.asarray(result["confusion_matrix"]).sum(), 6)

    def test_missing_predictions_still_use_all_six_classes(self):
        result = classification_metrics(np.array([0, 0]), np.array([0, 0]))
        self.assertAlmostEqual(result["macro_f1"], 1 / 6)
        with self.assertRaises(ValueError):
            classification_metrics(np.array([6]), np.array([0]))

    def test_configuration_cannot_enable_test_scoring(self):
        for key, value in (("test_scoring", True), ("seeds", [570, 570]),
                           ("learning_rate", float("nan")), ("device", "cuda"),
                           ("dropout", 1.0), ("epochs", 0)):
            with self.subTest(key=key):
                config = copy.deepcopy(CONFIG)
                config[key] = value
                with self.assertRaises(ValueError):
                    validate_config(config)

    def test_all_neural_interfaces_and_gradients(self):
        for name in ("cnn", "gru", "cnn_gru"):
            with self.subTest(name=name):
                model = build_model(name, CONFIG)
                x = torch.randn(4, 128, 6)
                scores = model(x)
                self.assertEqual(scores.shape, (4, 6))
                loss = torch.nn.functional.cross_entropy(scores, torch.arange(4))
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None), 0)
                self.assertEqual(model.eval()(x[:1]).shape, (1, 6))

    def test_checkpoint_and_export_round_trip_for_all_neural_models(self):
        small = {**CONFIG, "conv_channels": [4, 8], "gru_hidden": 8}
        windows = np.random.default_rng(7).normal(size=(2, 128, 6)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            for name in ("cnn", "gru", "cnn_gru"):
                with self.subTest(name=name):
                    directory = Path(temporary) / name
                    directory.mkdir()
                    model = build_model(name, small).eval()
                    torch.save({"model": name, "config": small,
                                "state_dict": model.state_dict()}, directory / "model.pt")
                    restored, _ = load_checkpoint(directory / "model.pt")
                    with torch.inference_mode():
                        np.testing.assert_array_equal(model(torch.from_numpy(windows)).numpy(),
                                                      restored(torch.from_numpy(windows)).numpy())
                    result = export_check({"model": name, "seed": 570}, directory, windows)
                    self.assertTrue(result["passed"])

    def test_data_loader_does_not_open_test_arrays(self):
        split = json.loads((REPO / "ml/configs/subjects-v1.json").read_text())
        training, validation = fake_windows(split["train"]), fake_windows(split["validation"])
        stats = fit_training(training, split["train"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, data in (("train", training), ("validation", validation)):
                np.savez_compressed(root / f"{name}.npz", x=standardize(data.x, stats),
                                    y=data.y, subject_id=data.subject_id, source_row=data.source_row)
            # These source row IDs must refer to disjoint official-training rows.
            validation.source_row += len(training.x)
            np.savez_compressed(root / "validation.npz", x=standardize(validation.x, stats),
                                y=validation.y, subject_id=validation.subject_id,
                                source_row=validation.source_row)
            write_json(root / "normalization.json", stats)
            (root / "test.npz").write_bytes(b"POISON: opening this file as NPZ must fail")
            write_json(root / "manifest.json", {
                "schema_version": "data-manifest-v1", "preprocessing_version": CONTRACT_VERSION,
                "channels": list(CHANNELS), "classes": list(CLASSES), "split": split,
                "source_archive_sha256": WRAPPER_SHA256, "split_config_sha256": "synthetic-test-only",
                "artifacts": {name: sha256(root / name) for name in
                              ("train.npz", "validation.npz", "normalization.json")},
                "partitions": {"train": {"windows": len(training.x)},
                               "validation": {"windows": len(validation.x)}},
            })
            loaded = load_training_data(root)
            self.assertFalse(loaded.provenance["test_arrays_opened"])
            self.assertFalse(loaded.provenance["test_scores_computed"])
            with (root / "train.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_training_data(root)

    def test_feature_scaler_uses_training_features_only(self):
        training = fake_windows([1, 3])
        validation = fake_windows([5])
        validation.x += 1000
        data = TrainingData(training, validation, {}, {})
        with tempfile.TemporaryDirectory() as temporary:
            result = train_logistic(data, CONFIG, Path(temporary))
            with np.load(Path(temporary) / "model.npz", allow_pickle=False) as saved:
                np.testing.assert_allclose(saved["feature_mean"],
                                           statistical_features(training.x).mean(axis=0))
            self.assertTrue(result["checkpoint_reload_matches"])

    def test_single_deterministic_fit_has_no_invented_seed_dispersion(self):
        runs = []
        for name in CONFIG["models"]:
            values = [0.8] if name == "logistic_regression" else [0.7, 0.8, 0.9]
            for value in values:
                runs.append({"model": name, "parameter_count": 100,
                             "validation": {"accuracy": value, "macro_f1": value},
                             "training_seconds": 1, "desktop_inference": {"median_ms": 1}})
        rows = aggregate_runs(runs)
        self.assertIsNone(rows[0]["validation_macro_f1_sample_std"])
        self.assertAlmostEqual(rows[1]["validation_macro_f1_sample_std"], 0.1)


if __name__ == "__main__":
    unittest.main()
