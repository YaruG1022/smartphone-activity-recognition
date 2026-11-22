"""Evaluate the official UCI HAR partition using a validation-only freeze."""

import argparse
from functools import partial
import json
from pathlib import Path
import platform
from statistics import mean, stdev
from tempfile import TemporaryDirectory

import numpy as np
import sklearn
import torch

from freeze_desktop_experiment import DEFAULT_PROTOCOL, ROOT, frozen_text_sha256, select_from_validation
from harss_data.dataset import load_prepared
from harss_data.prepare import write_json
from harss_data.source import sha256
from harss_data.spec import CHANNELS, CLASSES, OFFICIAL_TEST_SUBJECTS, WINDOW_SIZE
from harss_supervised.data import load_training_data
from harss_supervised.features import FEATURE_NAMES, statistical_features
from harss_supervised.metrics import classification_metrics
from harss_supervised.training import load_checkpoint, predict_neural


METRICS = ("accuracy", "macro_f1")


def validate_selection(freeze: dict, protocol: dict) -> None:
    seeds = protocol["seeds"]
    expected = {
        (f"supervised/{model}/seed{seed}", "supervised", model, seed, None, None)
        for model in protocol["supervised_models"]
        for seed in (seeds[:1] if model == "logistic_regression" else seeds)
    } | {
        (f"self_supervised/{method}/{percent}/seed{seed}", "self_supervised",
         protocol["self_supervised_model"], seed, percent, method)
        for seed in seeds for percent in protocol["label_percents"] for method in protocol["methods"]
    }
    runs = freeze["runs"]
    actual = {
        (run["id"], run["group"], run["model"], run["seed"], run.get("label_percent"), run.get("method"))
        for run in runs
    }
    if len(runs) != len(expected) or actual != expected:
        raise ValueError("The freeze must list the full supervised and paired self-supervised grids")
    rule = protocol["selection"]
    ranking = select_from_validation(runs, rule["eligible_candidates"])
    if (freeze["selection"] != {"rule": rule, "validation_ranking": ranking}
            or freeze["selected_run_id"] != ranking[0]["best_run_id"]
            or freeze["evaluation"] != protocol["evaluation"]):
        raise ValueError("The freeze does not match the declared validation selection and test protocol")


def check_frozen_file(relative: str, digest: str, test_path: Path, hash_file=sha256) -> Path:
    path = (ROOT / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(ROOT) or path == test_path:
        raise ValueError(f"Expected a repository-relative non-test artifact: {relative}")
    if hash_file(path) != digest:
        raise ValueError(f"Changed frozen artifact: {relative}")
    return path


def logistic_predict(weights: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    features = (statistical_features(x) - weights["feature_mean"]) / weights["feature_scale"]
    return features @ weights["coef"].T + weights["intercept"]


def load_logistic(path: Path):
    features, classes = len(FEATURE_NAMES), len(CLASSES)
    shapes = {
        "coef": (classes, features), "intercept": (classes,),
        "feature_mean": (features,), "feature_scale": (features,),
    }
    with np.load(path, allow_pickle=False) as archive:
        weights = {name: archive[name] for name in shapes}
        correct_order = np.array_equal(archive["classes"], np.arange(classes))
    if (not correct_order
            or any(weights[name].shape != shape or not np.isfinite(weights[name]).all()
                   for name, shape in shapes.items())
            or np.any(weights["feature_scale"] <= 0)):
        raise ValueError(f"Invalid frozen logistic regression parameters: {path}")
    return partial(logistic_predict, weights)


def verify_inputs(freeze_path: Path, data_dir: Path) -> tuple[dict, dict, dict, str]:
    freeze_digest = sha256(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (freeze.get("schema_version") != "official-test-freeze-v1"
            or freeze.get("selection_partition") != "validation"
            or freeze.get("test_scores_computed") is not False
            or DEFAULT_PROTOCOL.relative_to(ROOT).as_posix() not in freeze.get("frozen_files", {})):
        raise ValueError("Expected a completed validation-only freeze with artifact hashes")
    test_path = (data_dir / "test.npz").resolve()
    for relative, digest in freeze["frozen_files"].items():
        check_frozen_file(relative, digest, test_path, hash_file=frozen_text_sha256)
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    validate_selection(freeze, protocol)

    prepared = load_training_data(data_dir)
    if prepared.provenance["manifest_sha256"] != freeze["data_manifest_sha256"]:
        raise ValueError("The prepared data manifest does not match the freeze")

    predictors = {}
    checkpoint_metadata = {}
    for run in freeze["runs"]:
        digest = run["sha256"]
        path = check_frozen_file(run["checkpoint"], digest, test_path)
        if digest not in predictors:
            if run["model"] == "logistic_regression":
                predictors[digest] = load_logistic(path)
                checkpoint_metadata[digest] = {"model": run["model"], "seed": run["seed"]}
            else:
                model, checkpoint = load_checkpoint(path)
                if (checkpoint["data_provenance"] != prepared.provenance
                        or checkpoint["class_order"] != list(CLASSES)
                        or checkpoint["input_shape"] != [1, WINDOW_SIZE, len(CHANNELS)]):
                    raise ValueError(f"Checkpoint contract or data provenance mismatch: {run['id']}")
                predictors[digest] = partial(predict_neural, model)
                checkpoint_metadata[digest] = checkpoint
        metadata = checkpoint_metadata[digest]
        fields = ("model", "seed", "best_epoch", "method", "label_percent")
        if any(metadata.get(field) != run.get(field) for field in fields):
            raise ValueError(f"Checkpoint metadata differs from the freeze: {run['id']}")
    return freeze, prepared.manifest, predictors, freeze_digest


def load_official_test(data_dir: Path, manifest: dict):
    path = data_dir / "test.npz"
    info = manifest["partitions"]["test"]
    if sha256(path) != manifest["artifacts"]["test.npz"]:
        raise ValueError("The official test artifact differs from the prepared data manifest")
    data = load_prepared(path)
    class_counts = dict(zip(CLASSES, np.bincount(data.y, minlength=len(CLASSES)).tolist()))
    if (info["official_partition"] != "test" or info["file"] != path.name
            or list(data.x.shape) != info["shape"] or len(data.x) != 2947 or info["windows"] != 2947
            or set(data.subject_id) != set(OFFICIAL_TEST_SUBJECTS)
            or sorted(set(data.subject_id)) != sorted(info["subjects"])
            or class_counts != info["class_counts"]
            or not np.array_equal(data.source_row, np.arange(len(data.x)))):
        raise ValueError("The prepared test rows, classes or subjects do not match the official partition")
    return data


def aggregate(runs: list[dict]) -> list[dict]:
    groups = {}
    for run in runs:
        key = (run["group"], run["model"], run.get("label_percent"), run.get("method"))
        groups.setdefault(key, []).append(run)
    rows = []
    for (group, model, percent, method), members in groups.items():
        row = {
            "group": group, "model": model, "label_percent": percent,
            "method": method, "n_runs": len(members),
        }
        for metric in METRICS:
            values = [run["test"][metric] for run in members]
            row[f"test_{metric}_mean"] = mean(values)
            row[f"test_{metric}_sample_std"] = stdev(values) if len(values) > 1 else None
        rows.append(row)
    return rows


def paired_differences(runs: list[dict]) -> list[dict]:
    scores = {
        (run["label_percent"], run["seed"], run["method"]): run["test"]
        for run in runs if run["group"] == "self_supervised"
    }
    percentages = sorted({percent for percent, _, _ in scores})
    seeds = sorted({seed for _, seed, _ in scores})
    rows = []
    for percent in percentages:
        per_seed = []
        for seed in seeds:
            pretrained = scores[percent, seed, "pretrained"]
            scratch = scores[percent, seed, "scratch"]
            difference = {metric: pretrained[metric] - scratch[metric] for metric in METRICS}
            per_seed.append({"seed": seed, **difference})
        row = {"label_percent": percent, "pretrained_minus_scratch": per_seed}
        for metric in METRICS:
            values = [entry[metric] for entry in per_seed]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_sample_std"] = stdev(values)
        rows.append(row)
    return rows


def score_runs(freeze: dict, predictors: dict, data) -> tuple[list[dict], dict]:
    predictions = {"y_true": data.y, "subject_id": data.subject_id, "source_row": data.source_row}
    runs = []
    metrics = {}
    for run in freeze["runs"]:
        digest = run["sha256"]
        prediction_key = f"logits_{digest}"
        if digest not in metrics:
            logits = predictors[digest](data.x)
            if logits.shape != (len(data.x), len(CLASSES)) or not np.isfinite(logits).all():
                raise ValueError(f"Invalid test logits from {run['id']}")
            predictions[prediction_key] = logits
            metrics[digest] = classification_metrics(data.y, logits.argmax(axis=1))
        runs.append({**run, "test": metrics[digest], "prediction_key": prediction_key})
    return runs, predictions


def evaluate(freeze_path: Path, data_dir: Path, output: Path | None) -> None:
    if output is not None and output.exists():
        raise FileExistsError(f"Test evaluation requires a new output directory: {output}")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    freeze, manifest, predictors, freeze_digest = verify_inputs(freeze_path, data_dir)
    if output is None:
        print(f"Verified {len(freeze['runs'])} frozen runs without opening official test arrays.")
        return

    data = load_official_test(data_dir, manifest)
    runs, predictions = score_runs(freeze, predictors, data)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".harss-test-", dir=output.parent) as temp:
        stage = Path(temp) / "results"
        stage.mkdir()
        predictions_path = stage / "test-predictions.npz"
        np.savez_compressed(predictions_path, **predictions)
        write_json(stage / "test-results.json", {
            "schema_version": "official-test-results-v1",
            "status": "complete",
            "evaluation_partition": "test",
            "test_scores_computed": True,
            "selection_partition": freeze["selection_partition"],
            "selected_run_id": freeze["selected_run_id"],
            "freeze_sha256": freeze_digest,
            "data_manifest_sha256": freeze["data_manifest_sha256"],
            "test_artifact_sha256": manifest["artifacts"]["test.npz"],
            "predictions_sha256": sha256(predictions_path),
            "classes": list(CLASSES),
            "test_partition": manifest["partitions"]["test"],
            "runtime": {
                "device": "cpu", "cpu_threads": 2, "python": platform.python_version(),
                "numpy": np.__version__, "pytorch": str(torch.__version__),
                "scikit_learn": sklearn.__version__,
            },
            "distinct_checkpoints_scored": len(predictors),
            "runs": runs,
            "aggregate": aggregate(runs),
            "paired_differences": paired_differences(runs),
        })
        stage.rename(output)
    print(f"Scored {len(predictors)} checkpoints for {len(runs)} frozen runs on {len(data.x)} windows.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "ml/data/prepared/uci-har-v1")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path, help="New directory for official test results")
    action.add_argument("--verify-only", action="store_true", help="Verify the freeze without reading test arrays")
    args = parser.parse_args()
    evaluate(args.freeze.resolve(), args.data.resolve(), args.output.resolve() if args.output else None)


if __name__ == "__main__":
    main()
