"""Select the final desktop classifier from validation results and freeze test inputs."""

import argparse
from hashlib import sha256 as hash_bytes
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from harss_data.source import sha256
from harss_data.spec import CLASSES
from harss_ssl.experiment import code_hash as ssl_code_hash
from harss_supervised.data import load_training_data
from harss_supervised.experiment import training_code_hash
from harss_supervised.metrics import classification_metrics


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "ml/configs/official-test-v1.json"
SSL_CONFIG = ROOT / "ml/configs/ssl-v1.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_text_sha256(path):
    """Hash text with canonical newlines across Git checkouts."""
    return hash_bytes(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def relative_to_root(path):
    return path.resolve().relative_to(ROOT).as_posix()


def verify_validation(record, directory, data):
    """Recompute the selection metrics on the already-fixed validation windows."""
    predictions = directory / "validation-predictions.npz"
    if sha256(predictions) != record["artifacts"][predictions.name]:
        raise ValueError(f"Validation prediction hash mismatch: {directory}")
    with np.load(predictions, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["y_true"], data.y)
        np.testing.assert_array_equal(arrays["subject_id"], data.subject_id)
        np.testing.assert_array_equal(arrays["source_row"], data.source_row)
        logits = arrays["logits"]
        if logits.shape != (len(data.y), len(CLASSES)) or not np.isfinite(logits).all():
            raise ValueError(f"Invalid validation logits: {directory}")
        predicted = logits.argmax(axis=1)
        np.testing.assert_array_equal(arrays["y_pred"], predicted)
    metrics = classification_metrics(data.y, predicted)
    for key in ("accuracy", "macro_f1"):
        if not math.isclose(metrics[key], record["validation"][key], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Validation {key} mismatch: {directory}")
    return {key: metrics[key] for key in ("accuracy", "macro_f1")}


def collect_runs(group, results_path, expected_ids, data, ssl_model=None):
    results = read_json(results_path)
    if (results.get("status") != "complete"
            or results.get("evaluation_partition") != "validation"
            or results.get("test_scores_computed") is not False
            or results["provenance"]["data"] != data.provenance):
        raise ValueError(f"Expected a complete validation-only run on the current split: {results_path}")
    if group == "supervised" and results["provenance"]["training_code_sha256"] != training_code_hash():
        raise ValueError("Saved supervised results were produced by different training code")
    if group == "self_supervised":
        provenance = results["provenance"]
        if (results.get("schema_version") != "ssl-results-v1"
                or provenance["training_code_sha256"] != ssl_code_hash()
                or provenance["config_sha256"] != sha256(SSL_CONFIG)):
            raise ValueError("Saved self-supervised results use different code or configuration")

    runs = {}
    for record in results["runs"]:
        model, seed = record["model"], record["seed"]
        directory_name = f"{model}-seed{seed}"
        candidate = f"supervised/{model}"
        fields = {}
        if group == "self_supervised":
            if model != ssl_model:
                raise ValueError(f"Unexpected classifier in self-supervised comparison: {model}")
            fields = {"method": record["method"], "label_percent": record["label_percent"]}
            candidate = f"self_supervised/{fields['method']}/{fields['label_percent']}"
            directory_name = record["directory"]
        run_id = f"{candidate}/seed{seed}"
        if run_id in runs or run_id not in expected_ids:
            raise ValueError(f"Unexpected or duplicate validation run: {run_id}")
        directory = (results_path.parent / directory_name).resolve()
        if directory.parent != results_path.parent.resolve():
            raise ValueError(f"Run directory must be under the experiment output: {directory}")
        filename = "model.npz" if model == "logistic_regression" else "model.pt"
        checkpoint = directory / filename
        checksum = sha256(checkpoint)
        if checksum != record["artifacts"][filename]:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
        runs[run_id] = {
            "id": run_id,
            "group": group,
            "model": model,
            "seed": seed,
            **fields,
            "checkpoint": relative_to_root(checkpoint),
            "sha256": checksum,
            "parameter_count": record["parameter_count"],
            "best_epoch": record["best_epoch"],
            "validation": verify_validation(record, directory, data.validation),
        }
    if runs.keys() != expected_ids:
        raise ValueError(f"Missing validation runs: {sorted(expected_ids - runs.keys())}")
    return list(runs.values())


def select_from_validation(runs, candidates):
    summaries = []
    for candidate in candidates:
        group = [run for run in runs if run["id"].rsplit("/", 1)[0] == candidate]
        sizes = {run["parameter_count"] for run in group}
        if len(sizes) != 1:
            raise ValueError(f"Expected a fixed architecture for {candidate}")
        finalist = min(group, key=lambda run: (-run["validation"]["macro_f1"], run["seed"]))
        summaries.append({
            "candidate": candidate,
            "mean_validation_macro_f1": mean(run["validation"]["macro_f1"] for run in group),
            "parameter_count": sizes.pop(),
            "best_run_id": finalist["id"],
        })
    return sorted(summaries, key=lambda row: (
        -row["mean_validation_macro_f1"], row["parameter_count"], row["candidate"],
    ))


def freeze(data_dir, supervised_dir, ssl_dir):
    protocol = read_json(DEFAULT_PROTOCOL)
    seeds = protocol["seeds"]
    supervised_ids = {
        f"supervised/{model}/seed{seed}"
        for model in protocol["supervised_models"]
        for seed in (seeds[:1] if model == "logistic_regression" else seeds)
    }
    ssl_ids = {
        f"self_supervised/{method}/{percent}/seed{seed}"
        for method in protocol["methods"]
        for percent in protocol["label_percents"]
        for seed in seeds
    }
    data = load_training_data(data_dir)
    supervised_results = supervised_dir / "results.json"
    ssl_results = ssl_dir / "results.json"
    runs = collect_runs("supervised", supervised_results, supervised_ids, data)
    runs.extend(collect_runs(
        "self_supervised", ssl_results, ssl_ids, data, protocol["self_supervised_model"],
    ))
    ranking = select_from_validation(runs, protocol["selection"]["eligible_candidates"])
    frozen_files = [
        DEFAULT_PROTOCOL, ROOT / "ml/configs/supervised-v1.json", SSL_CONFIG,
        ROOT / "ml/configs/subjects-v1.json", Path(__file__),
        ROOT / "ml/scripts/evaluate_frozen_test.py", supervised_results, ssl_results,
        *sorted((ROOT / "ml/src").glob("harss_*/*.py")),
    ]
    return {
        "schema_version": "official-test-freeze-v1",
        "selection_partition": "validation",
        "test_scores_computed": False,
        "data_manifest_sha256": data.provenance["manifest_sha256"],
        "frozen_files": {relative_to_root(path): frozen_text_sha256(path) for path in frozen_files},
        "selected_run_id": ranking[0]["best_run_id"],
        "selection": {"rule": protocol["selection"], "validation_ranking": ranking},
        "evaluation": protocol["evaluation"],
        "runs": runs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "ml/data/prepared/uci-har-v1")
    parser.add_argument("--supervised", type=Path, default=ROOT / "ml/outputs/supervised-v1")
    parser.add_argument("--ssl", type=Path, default=ROOT / "ml/outputs/ssl-v1")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Refusing to overwrite an existing freeze: {args.output}")
    result = freeze(args.data, args.supervised, args.ssl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Frozen {len(result['runs'])} runs; validation finalist: {result['selected_run_id']}")


if __name__ == "__main__":
    main()
