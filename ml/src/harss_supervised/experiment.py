"""A resumable, validation-only comparison with a frozen configuration."""

import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
import sklearn
import torch
from threadpoolctl import threadpool_limits

from harss_data.prepare import write_json
from harss_data.source import sha256
from .data import load_training_data
from .training import (
    artifact_hashes, load_checkpoint, sanity_checks, set_seed, train_logistic, train_neural,
)


def validate_config(config: dict) -> None:
    if not isinstance(config, dict) or config.get("version") != "supervised-v1":
        raise ValueError("Unsupported comparison configuration")
    if (config.get("models") != ["logistic_regression", "cnn", "gru", "cnn_gru"]
            or config.get("device") != "cpu" or config.get("test_scoring") is not False
            or config.get("selection_metric") != "validation_macro_f1"
            or config.get("tie_break") != "earliest_epoch"):
        raise ValueError("This protocol requires all four models and validation-only CPU scoring")
    seeds = config.get("seeds")
    if (not isinstance(seeds, list) or len(seeds) < 2
            or any(type(s) is not int or s < 0 or s >= 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Use at least two distinct integer seeds")
    for key in ("cpu_threads", "epochs", "patience", "batch_size", "gru_hidden", "logistic_max_iter"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "gradient_clip_norm", "logistic_C"):
        value = config.get(key)
        if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("dropout", "weight_decay"):
        value = config.get(key)
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key}")
    if config["dropout"] >= 1:
        raise ValueError("dropout must be below 1")
    channels = config.get("conv_channels")
    if not isinstance(channels, list) or len(channels) != 2 or any(type(c) is not int or c <= 0 for c in channels):
        raise ValueError("conv_channels must contain two positive integers")


def load_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def training_code_hash() -> str:
    digest = hashlib.sha256()
    for name in ("data.py", "features.py", "models.py", "metrics.py", "training.py", "experiment.py"):
        path = Path(__file__).parent / name
        digest.update(name.encode())
        digest.update(path.read_text(encoding="utf-8").encode())
    return digest.hexdigest()


def aggregate_runs(runs: list[dict]) -> list[dict]:
    rows = []
    for name in ("logistic_regression", "cnn", "gru", "cnn_gru"):
        group = [run for run in runs if run["model"] == name]
        row = {"model": name, "n_runs": len(group),
               "parameter_count": group[0]["parameter_count"]}
        for metric in ("accuracy", "macro_f1"):
            values = [run["validation"][metric] for run in group]
            row[f"validation_{metric}_mean"] = float(np.mean(values))
            row[f"validation_{metric}_sample_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        row["training_seconds_mean"] = float(np.mean([r["training_seconds"] for r in group]))
        row["desktop_median_ms_mean"] = float(np.mean([r["desktop_inference"]["median_ms"] for r in group]))
        rows.append(row)
    return rows


def export_check(run: dict, directory: Path, windows: np.ndarray) -> dict:
    model, _ = load_checkpoint(directory / "model.pt")
    path = directory / "model.pt2"
    # Non-strict export traces the CPU tensor operations used by GRU as well.
    program = torch.export.export(model, (torch.from_numpy(windows[:1]),), strict=False)
    torch.export.save(program, path)
    restored = torch.export.load(path).module()
    with torch.inference_mode():
        reference = torch.cat([model(torch.from_numpy(w[None])) for w in windows]).numpy()
        actual = torch.cat([restored(torch.from_numpy(w[None])) for w in windows]).numpy()
    np.testing.assert_allclose(actual, reference, atol=1e-4, rtol=1e-3)
    np.testing.assert_array_equal(actual.argmax(1), reference.argmax(1))
    write_json(directory / "export-replay.json", {
        "input": windows.tolist(), "reference_logits": reference.tolist(),
        "exported_logits": actual.tolist(), "predicted_classes": actual.argmax(1).tolist(),
        "purpose": "Training-window CPU export parity; not ground-truth accuracy",
    })
    return {"model": run["model"], "seed": run["seed"], "passed": True,
            "format": "torch.export .pt2", "input_shape": [1, 128, 6],
            "training_windows_checked": len(windows),
            "maximum_absolute_logit_error": float(np.max(np.abs(actual - reference))),
            "artifacts": artifact_hashes(directory, ["model.pt2", "export-replay.json"]),
            "scope": "CPU reload only; Android runtime compatibility is untested"}


def run_comparison(data_dir: Path, config_path: Path, output: Path) -> dict:
    config, output = load_config(config_path), Path(output)
    data = load_training_data(data_dir)
    software = {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                "torch": torch.__version__, "platform": platform.platform()}
    signature = {"config": config, "data": data.provenance,
                 "training_code_sha256": training_code_hash(), "software": software}
    output.mkdir(parents=True, exist_ok=True)
    signature_path = output / "signature.json"
    if signature_path.exists():
        if json.loads(signature_path.read_text(encoding="utf-8")) != signature:
            raise ValueError("Output belongs to a different code/config/data/environment; use a new output directory")
    elif any(output.iterdir()):
        raise ValueError("Output contains files without a comparison signature")
    else:
        write_json(signature_path, signature)
    set_seed(config["seeds"][0], config["cpu_threads"])
    with threadpool_limits(limits=config["cpu_threads"]):
        sanity_path = output / "sanity.json"
        if sanity_path.exists():
            sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
        else:
            sanity = sanity_checks(data, config)
            write_json(sanity_path, sanity)
        runs = []
        for name in config["models"]:
            seeds = config["seeds"][:1] if name == "logistic_regression" else config["seeds"]
            for seed in seeds:
                directory = output / f"{name}-seed{seed}"
                directory.mkdir(exist_ok=True)
                result_path = directory / "result.json"
                if result_path.exists():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    for filename, digest in result["artifacts"].items():
                        if sha256(directory / filename) != digest:
                            raise ValueError(f"Completed run artifact changed: {directory.name}/{filename}")
                    print(f"Verified completed run: {directory.name}", flush=True)
                else:
                    result = (train_logistic(data, config, directory) if name == "logistic_regression"
                              else train_neural(name, seed, data, config, directory))
                    write_json(result_path, result)
                    print(f"Completed {directory.name}: validation macro-F1={result['validation']['macro_f1']:.4f}", flush=True)
                runs.append(result)
        rows = aggregate_runs(runs)
        ranking = sorted(rows, key=lambda row: (-row["validation_macro_f1_mean"],
                                               row["parameter_count"], row["model"]))
        neural_ranking = [row for row in ranking if row["model"] != "logistic_regression"]
        selected = {}
        for name in ("cnn", "gru", "cnn_gru"):
            selected[name] = sorted([r for r in runs if r["model"] == name],
                                    key=lambda r: (-r["validation"]["macro_f1"], r["seed"]))[0]
        indices = [int(np.flatnonzero(data.train.y == i)[0]) for i in range(6)]
        exports = []
        for name, run in selected.items():
            directory = output / f"{name}-seed{run['seed']}"
            export_path = directory / "export-check.json"
            if export_path.exists():
                check = json.loads(export_path.read_text(encoding="utf-8"))
                for filename, digest in check["artifacts"].items():
                    if sha256(directory / filename) != digest:
                        raise ValueError(f"Export artifact changed: {filename}")
            else:
                check = export_check(run, directory, data.train.x[indices])
                write_json(export_path, check)
            exports.append(check)
        report = {
            "schema_version": "supervised-results-v1", "status": "complete",
            "evaluation_partition": "validation", "test_scores_computed": False,
            "config": config, "provenance": signature,
            "data_counts": {"train": len(data.train.x), "validation": len(data.validation.x)},
            "classes": list(data.manifest["classes"]), "runs": runs,
            "aggregate": rows, "sanity_checks": sanity, "export_checks": exports,
            "selection": {
                "rule": "Highest mean validation macro-F1, then fewer parameters, then model name",
                "best_model_by_mean_validation_macro_f1": ranking[0]["model"],
                "best_neural_model_by_mean_validation_macro_f1": neural_ranking[0]["model"],
                "export_seed_selection": "Highest validation macro-F1 within architecture, then lower seed",
            },
            "limitations": [
                "One fixed five-subject validation group; seed dispersion is not a confidence interval over new users.",
                "One configuration per architecture, not an exhaustive tuning study; parameter counts differ.",
                "Logistic regression has one deterministic fit; no across-seed uncertainty is estimated for it.",
                "Validation selects epochs and models, so its scores are development estimates, not final test results.",
                "CPU desktop timings exclude sensor preprocessing and do not establish phone performance.",
                "No self-supervised training, distillation, or Android inference was run in this module.",
            ],
        }
        write_json(output / "results.json", report)
        return report
