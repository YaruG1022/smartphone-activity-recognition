"""Independently check saved predictions, then publish aggregate evidence."""

import argparse
import json
import os
from pathlib import Path
import platform
import statistics

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "ml/outputs/matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import numpy as np

from harss_data.prepare import write_json
from harss_data.source import sha256
from harss_supervised.data import load_training_data
from harss_supervised.experiment import training_code_hash
from harss_supervised.reporting import package_exports, publish_report


def validate_results(report, data, output):
    if report["status"] != "complete" or report["test_scores_computed"]:
        raise ValueError("Only completed validation-only results can be published")
    if report["provenance"]["data"] != data.provenance:
        raise ValueError("Results do not match the current training/validation data")
    if report["provenance"]["training_code_sha256"] != training_code_hash():
        raise ValueError("Results do not match the current training implementation")
    config = report["config"]
    expected = {(name, seed) for name in config["models"]
                for seed in (config["seeds"][:1] if name == "logistic_regression" else config["seeds"])}
    if len(report["runs"]) != len(expected) or {(r["model"], r["seed"]) for r in report["runs"]} != expected:
        raise ValueError("Missing or duplicate formal fits")
    for run in report["runs"]:
        folder = output / f"{run['model']}-seed{run['seed']}"
        for name, digest in run["artifacts"].items():
            if sha256(folder / name) != digest:
                raise ValueError(f"Changed run artifact: {folder.name}/{name}")
        with np.load(folder / "validation-predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["y_true"], data.validation.y)
            np.testing.assert_array_equal(saved["subject_id"], data.validation.subject_id)
            np.testing.assert_array_equal(saved["source_row"], data.validation.source_row)
            np.testing.assert_array_equal(saved["y_pred"], saved["logits"].argmax(axis=1))
            matrix = np.zeros((6, 6), dtype=np.int64)
            np.add.at(matrix, (saved["y_true"], saved["y_pred"]), 1)
        np.testing.assert_array_equal(matrix, run["validation"]["confusion_matrix"])
        diagonal = np.diag(matrix)
        accuracy = float(diagonal.sum() / matrix.sum())
        denom = matrix.sum(0) + matrix.sum(1)
        class_f1 = np.divide(2 * diagonal, denom, out=np.zeros(6), where=denom > 0)
        np.testing.assert_allclose(accuracy, run["validation"]["accuracy"], atol=1e-12)
        np.testing.assert_allclose(class_f1.mean(), run["validation"]["macro_f1"], atol=1e-12)
        for i, name in enumerate(report["classes"]):
            np.testing.assert_allclose(diagonal[i] / matrix[i].sum(),
                                       run["validation"]["per_class"][name]["recall"], atol=1e-12)
        if run["stochastic_training"]:
            history = run["history"]
            best = int(np.argmax([item["validation_macro_f1"] for item in history]))
            if (run["best_epoch"] != history[best]["epoch"]
                    or run["epochs_run"] != len(history) or len(history) > config["epochs"]):
                raise ValueError("Checkpoint selection does not match the declared rule")
            np.testing.assert_allclose(history[best]["validation_macro_f1"], class_f1.mean(), atol=1e-12)
    for row in report["aggregate"]:
        group = [r for r in report["runs"] if r["model"] == row["model"]]
        for metric in ("accuracy", "macro_f1"):
            values = [run["validation"][metric] for run in group]
            np.testing.assert_allclose(statistics.mean(values), row[f"validation_{metric}_mean"], atol=1e-12)
            if len(values) > 1:
                np.testing.assert_allclose(statistics.stdev(values), row[f"validation_{metric}_sample_std"], atol=1e-12)
            elif row[f"validation_{metric}_sample_std"] is not None:
                raise ValueError("A single fit cannot have across-seed dispersion")
    expected_windows = np.stack([data.train.x[np.flatnonzero(data.train.y == i)[0]] for i in range(6)])
    for check in report["export_checks"]:
        folder = output / f"{check['model']}-seed{check['seed']}"
        for name, digest in check["artifacts"].items():
            if sha256(folder / name) != digest:
                raise ValueError("Export artifact changed")
        replay = json.loads((folder / "export-replay.json").read_text(encoding="utf-8"))
        np.testing.assert_array_equal(replay["input"], expected_windows)
        np.testing.assert_allclose(replay["exported_logits"], replay["reference_logits"], atol=1e-4, rtol=1e-3)
        np.testing.assert_array_equal(np.argmax(replay["exported_logits"], axis=1),
                                      np.argmax(replay["reference_logits"], axis=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "ml/data/prepared/uci-har-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "ml/outputs/supervised-v1")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/experiments/supervised-v1")
    args = parser.parse_args()
    report = json.loads((args.output / "results.json").read_text(encoding="utf-8"))
    data = load_training_data(args.data)
    validate_results(report, data, args.output)
    bundles = package_exports(report, args.data, args.output)
    publish_report(report, args.report)
    write_json(args.report / "export-manifests.json", {"exports": bundles})
    write_json(args.report / "validation.json", {
        "formal_fits_checked": len(report["runs"]),
        "saved_prediction_metrics_independently_recomputed": True,
        "prediction_labels_subjects_and_rows_match_validation": True,
        "aggregate_means_and_sample_stds_independently_checked": True,
        "checkpoint_selection_rule_checked": True,
        "all_local_artifact_hashes_checked": True,
        "export_replay_inputs_verified_as_training_only": True,
        "test_arrays_opened": False, "test_scores_computed": False,
        "training_code_sha256": training_code_hash(),
        "hardware": {"processor": platform.processor(), "machine": platform.machine(),
                     "logical_cpus": os.cpu_count(), "training_cpu_threads": report["config"]["cpu_threads"]},
    })
    print(f"Verified {len(report['runs'])} fits and published validation evidence to {args.report}")


if __name__ == "__main__":
    main()
