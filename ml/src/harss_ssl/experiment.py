"""Run and resume a matched, validation-only SSL comparison by training seed."""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import platform

from filelock import FileLock
import filelock
import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits
import torch

from harss_data.prepare import write_json
from harss_data.source import sha256
from harss_data.spec import WINDOW_SIZE
from harss_supervised.data import load_training_data
from harss_supervised.experiment import training_code_hash as supervised_code_hash
from harss_supervised.training import artifact_hashes

from .classification import train_classifier
from .pretraining import pretrain
from .subsets import labeled_subsets, subset_summary


METHODS = ("scratch", "pretrained")


def read_record(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_record(path, value):
    temporary = path.with_name(f".{path.name}.tmp")
    write_json(temporary, value)
    temporary.replace(path)


def load_config(path):
    config = read_record(path)
    if (config.get("version") != "ssl-v1" or config.get("model") != "cnn"
            or config.get("selection_metric") != "validation_macro_f1"
            or config.get("tie_break") != "earliest_epoch" or config.get("test_scoring") is not False):
        raise ValueError("This protocol requires validation-only CNN comparison")
    seeds = config["seeds"]
    if (not isinstance(seeds, list) or len(seeds) < 2
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Use at least two distinct 32-bit seeds")
    percentages = config["label_percents"]
    if (not isinstance(percentages, list) or not percentages
            or any(type(value) is not int or not 1 <= value <= 100 for value in percentages)
            or percentages != sorted(set(percentages)) or percentages[-1] != 100):
        raise ValueError("Use increasing integer percentages ending at 100")
    settings = config["classifier"]
    for key in ("cpu_threads", "epochs", "patience", "batch_size"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "gradient_clip_norm", "dropout", "weight_decay"):
        value = settings[key]
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid classifier {key}")
    if not settings["learning_rate"] or not settings["gradient_clip_norm"] or settings["dropout"] >= 1:
        raise ValueError("Classifier learning rate and gradient clip must be positive, with dropout below one")
    channels = settings["conv_channels"]
    if len(channels) != 2 or any(type(value) is not int or value <= 0 for value in channels):
        raise ValueError("CNN needs two positive channel widths")
    pretraining = config["pretraining"]
    if (type(pretraining["epochs"]) is not int or pretraining["epochs"] <= 0
            or type(pretraining["span_length"]) is not int
            or not 1 <= pretraining["span_length"] <= WINDOW_SIZE // 2):
        raise ValueError("Pretraining needs a positive epoch count and two spans that fit in a window")
    return config


def code_hash():
    digest = hashlib.sha256()
    for name in ("pretraining.py", "classification.py", "subsets.py", "experiment.py"):
        path = Path(__file__).parent / name
        digest.update(name.encode())
        digest.update(path.read_text(encoding="utf-8").encode())
    return digest.hexdigest()


def completed_run(path, artifacts):
    if not path.exists():
        return None
    record = read_record(path)
    if artifact_hashes(path.parent, artifacts) != record["artifacts"]:
        raise ValueError(f"Completed run artifacts changed: {path.parent.name}")
    print(f"Verified completed {path.parent.name}", flush=True)
    return record


def run_seed(data_dir, output, provenance, seed):
    with FileLock(str(output / f".seed{seed}.lock"), timeout=0):
        data = load_training_data(data_dir)
        if data.provenance != provenance["data"]:
            raise ValueError("Training data changed during comparison")
        config = provenance["config"]
        subsets = labeled_subsets(data.train, config["label_percents"], seed)
        subset_records = [subset_summary(subset, percent, seed) for percent, subset in subsets.items()]
        subset_path = output / f"subsets-seed{seed}.json"
        write_record(subset_path, {
            "seed": seed,
            "subsets": [{**record, "source_rows": subsets[record["label_percent"]].source_row.tolist()}
                        for record in subset_records],
        })
        summaries = {record["label_percent"]: record for record in subset_records}

        pretrain_dir = output / f"pretraining-seed{seed}"
        pretrain_dir.mkdir(exist_ok=True)
        pretrain_path = pretrain_dir / "result.json"
        with threadpool_limits(limits=config["classifier"]["cpu_threads"]):
            pretraining = completed_run(pretrain_path, ["encoder.pt"])
            if pretraining is None:
                pretraining = pretrain(data.train.x, seed, config, pretrain_dir)
                write_record(pretrain_path, pretraining)
            encoder = pretrain_dir / "encoder.pt"

            runs = []
            for percent, subset in subsets.items():
                for method in METHODS:
                    directory = output / f"{method}-labels{percent:03d}-seed{seed}"
                    directory.mkdir(exist_ok=True)
                    run_path = directory / "result.json"
                    run = completed_run(run_path, ["model.pt", "validation-predictions.npz"])
                    subset_digest = summaries[percent]["source_rows_sha256"]
                    encoder_digest = pretraining["artifacts"]["encoder.pt"] if method == "pretrained" else None
                    if run is None:
                        run = train_classifier(data, subset, seed, config["classifier"], directory,
                                               method=method, percent=percent,
                                               encoder_path=encoder if method == "pretrained" else None)
                        run["labeled_source_rows_sha256"] = subset_digest
                        run["pretrained_encoder_sha256"] = encoder_digest
                        write_record(run_path, run)
                    elif (run["labeled_source_rows_sha256"] != subset_digest
                          or run["pretrained_encoder_sha256"] != encoder_digest):
                        raise ValueError(f"Completed run used different training inputs: {directory.name}")
                    runs.append(run)
                    print(f"Completed {directory.name} validation_macro_f1={run['validation']['macro_f1']:.4f}",
                          flush=True)
        return pretraining, subset_records, runs


def aggregate_runs(runs, percentages, seeds):
    index = {(run["label_percent"], run["method"], run["seed"]): run for run in runs}
    rows, paired = [], []
    for percent in percentages:
        for method in METHODS:
            group = [index[percent, method, seed] for seed in seeds]
            row = {"label_percent": percent, "method": method, "n_runs": len(group),
                   "training_windows": group[0]["training_windows"]}
            for metric in ("accuracy", "macro_f1"):
                values = [run["validation"][metric] for run in group]
                row[f"validation_{metric}_mean"] = float(np.mean(values))
                row[f"validation_{metric}_sample_std"] = float(np.std(values, ddof=1))
            rows.append(row)

        per_seed = []
        for seed in seeds:
            scratch = index[percent, "scratch", seed]
            pretrained = index[percent, "pretrained", seed]
            if pretrained["labeled_source_rows_sha256"] != scratch["labeled_source_rows_sha256"]:
                raise ValueError("Matched training arms used different labeled windows")
            per_seed.append({"seed": seed, **{
                metric: pretrained["validation"][metric] - scratch["validation"][metric]
                for metric in ("accuracy", "macro_f1")
            }})
        row = {"label_percent": percent, "pretrained_minus_scratch": per_seed}
        for metric in ("accuracy", "macro_f1"):
            values = [run[metric] for run in per_seed]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sample_std"] = float(np.std(values, ddof=1))
        paired.append(row)
    return rows, paired


def run_comparison(data_dir: Path, config_path: Path, output: Path, *, workers: int = 1) -> dict:
    config = load_config(config_path)
    if type(workers) is not int or not 1 <= workers <= len(config["seeds"]):
        raise ValueError("Worker count must be between one and the number of seeds")
    data = load_training_data(data_dir)
    provenance = {
        "config": config,
        "config_sha256": sha256(config_path),
        "data": data.provenance,
        "training_code_sha256": code_hash(),
        "supervised_training_code_sha256": supervised_code_hash(),
        "parallel_seed_workers": workers,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                     "torch": torch.__version__, "filelock": filelock.__version__,
                     "platform": platform.platform()},
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.with_name(f".{output.name}.lock")), timeout=0):
        output.mkdir(exist_ok=True)
        signature = output / "signature.json"
        if signature.exists():
            if read_record(signature) != provenance:
                raise ValueError("Output belongs to a different code/config/data/environment; use a new directory")
        elif any(path.name != ".signature.json.tmp" for path in output.iterdir()):
            raise ValueError("Output contains existing files without an experiment signature")
        else:
            write_record(signature, provenance)

        jobs = [(data_dir.resolve(), output, provenance, seed) for seed in config["seeds"]]
        if workers == 1:
            results = [run_seed(*job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
                futures = [executor.submit(run_seed, *job) for job in jobs]
                results = [future.result() for future in futures]
        runs = [run for _, _, seed_runs in results for run in seed_runs]
        aggregate, paired = aggregate_runs(runs, config["label_percents"], config["seeds"])
        result = {
            "schema_version": "ssl-results-v1",
            "status": "complete",
            "evaluation_partition": "validation",
            "test_scores_computed": False,
            "config": config,
            "provenance": provenance,
            "data_counts": {"train": len(data.train.x), "validation": len(data.validation.x)},
            "classes": list(data.manifest["classes"]),
            "pretraining": [pretrained for pretrained, _, _ in results],
            "subsets": [subset for _, subsets, _ in results for subset in subsets],
            "runs": runs,
            "aggregate": aggregate,
            "paired_differences": paired,
        }
        write_record(output / "results.json", result)
    return result
