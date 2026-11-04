"""Figures and aggregate tables from observed validation results."""

import csv
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np

from harss_data.prepare import write_json
from harss_data.source import sha256
from harss_data.spec import CHANNELS, CONTRACT_VERSION

MODEL_NAMES = {"logistic_regression": "Logistic regression", "cnn": "CNN",
               "gru": "GRU", "cnn_gru": "CNN–GRU"}
COLORS = {"logistic_regression": "#64748b", "cnn": "#2563eb",
          "gru": "#8b5cf6", "cnn_gru": "#ea580c"}
SHORT_CLASSES = ["Walk", "Upstairs", "Downstairs", "Sit", "Stand", "Lie"]


def comparison_figure(report):
    rows = report["aggregate"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for ax, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1")):
        for i, row in enumerate(rows):
            name = row["model"]
            mean = row[f"validation_{metric}_mean"]
            std = row[f"validation_{metric}_sample_std"]
            ax.bar(i, mean, color=COLORS[name], width=0.62, alpha=0.85)
            if std is not None:
                ax.errorbar(i, mean, yerr=std, color="#111827", capsize=5, lw=1.5)
            values = [r["validation"][metric] for r in report["runs"] if r["model"] == name]
            offsets = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0]
            ax.scatter(i + np.asarray(offsets), values, s=25, color="#111827", zorder=4)
            ax.text(i, mean + (std or 0) + 0.025, f"{mean:.3f}", ha="center", fontsize=10)
        ax.set_xticks(range(4), ["Logistic\nn=1", "CNN\nn=3", "GRU\nn=3", "CNN–GRU\nn=3"])
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylabel(f"Validation {title.lower()}")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(title)
    fig.suptitle("Validation performance · one fixed five-subject group", fontsize=14)
    return fig


def confusion_figure(report):
    seed = report["config"]["seeds"][0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), layout="constrained")
    for ax, name in zip(axes.flat, MODEL_NAMES):
        run = next(r for r in report["runs"] if r["model"] == name and r["seed"] == seed)
        counts = np.asarray(run["validation"]["confusion_matrix"])
        fractions = counts / counts.sum(axis=1, keepdims=True)
        im = ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                if counts[i, j]:
                    ax.text(j, i, f"{fractions[i,j]:.1%}", ha="center", va="center",
                            fontsize=9, color="white" if fractions[i, j] > 0.55 else "#172554")
        ax.set_xticks(range(6), SHORT_CLASSES, rotation=30, ha="right")
        ax.set_yticks(range(6), SHORT_CLASSES)
        ax.set_xlabel("Predicted activity")
        ax.set_ylabel("True activity")
        label = "single fit" if name == "logistic_regression" else f"seed {seed}"
        ax.set_title(f"{MODEL_NAMES[name]} · {label}")
    fig.colorbar(im, ax=list(axes.flat), shrink=0.7, label="Fraction within true class")
    fig.suptitle("Validation confusion · predeclared first seed, row-normalized", fontsize=14)
    return fig


def learning_figure(report):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained", sharey=True)
    seed_colors = ["#2563eb", "#ea580c", "#16a34a"]
    for ax, name in zip(axes, ("cnn", "gru", "cnn_gru")):
        group = [r for r in report["runs"] if r["model"] == name]
        for run, color in zip(group, seed_colors):
            epochs = [item["epoch"] for item in run["history"]]
            f1 = [item["validation_macro_f1"] for item in run["history"]]
            ax.plot(epochs, f1, color=color, lw=1.6, label=f"Seed {run['seed']}")
            ax.scatter(run["best_epoch"], run["validation"]["macro_f1"],
                       color=color, marker="o", s=25, zorder=4)
        ax.set_title(MODEL_NAMES[name])
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, report["config"]["epochs"])
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Validation macro-F1")
    axes[-1].legend(loc="lower right", frameon=False)
    fig.suptitle("Development learning curves · circles mark selected epochs", fontsize=14)
    return fig


def summary_table(report):
    return [{"model": MODEL_NAMES[r["model"]], "n_runs": r["n_runs"],
             "validation_accuracy_mean": r["validation_accuracy_mean"],
             "validation_accuracy_sample_std": r["validation_accuracy_sample_std"],
             "validation_macro_f1_mean": r["validation_macro_f1_mean"],
             "validation_macro_f1_sample_std": r["validation_macro_f1_sample_std"],
             "parameters": r["parameter_count"],
             "mean_training_seconds": r["training_seconds_mean"],
             "mean_desktop_median_ms": r["desktop_median_ms_mean"]} for r in report["aggregate"]]


def package_exports(report, data_directory: Path, output_directory: Path):
    """Add versioned metadata and training normalization to local CPU exports."""
    statistics = Path(data_directory) / "normalization.json"
    expected = report["provenance"]["data"]["training_artifact_hashes"]["normalization.json"]
    if sha256(statistics) != expected:
        raise ValueError("Export normalization differs from the trained input transform")
    bundles = []
    for check in report["export_checks"]:
        folder = Path(output_directory) / f"{check['model']}-seed{check['seed']}"
        if sha256(folder / "model.pt2") != check["artifacts"]["model.pt2"]:
            raise ValueError("Export bytes changed before packaging")
        run = next(r for r in report["runs"]
                   if r["model"] == check["model"] and r["seed"] == check["seed"])
        shutil.copy2(statistics, folder / "normalization.json")
        manifest = {
            "model_version": f"supervised-v1-{check['model']}-seed{check['seed']}",
            "model_file": "model.pt2", "model_sha256": check["artifacts"]["model.pt2"],
            "format": "torch.export .pt2", "runtime": {"torch": report["provenance"]["software"]["torch"],
                                                        "device": "cpu", "android_validated": False},
            "preprocessing_version": CONTRACT_VERSION,
            "input": {"logical_name": "sensor_window", "shape": [1, 128, 6],
                      "dtype": "float32", "channels": list(CHANNELS), "standardized": True},
            "output": {"logical_name": "activity_logits", "shape": [1, 6],
                       "dtype": "float32", "semantics": "logits", "classes": report["classes"]},
            "normalization_file": "normalization.json", "normalization_sha256": expected,
            "checkpoint_file": "model.pt", "checkpoint_sha256": run["artifacts"]["model.pt"],
            "selected_epoch": run["best_epoch"], "seed": run["seed"],
            "training_configuration": report["config"],
            "training_code_sha256": report["provenance"]["training_code_sha256"],
            "data_provenance": report["provenance"]["data"],
        }
        write_json(folder / "export-manifest.json", manifest)
        bundles.append(manifest)
    return bundles


def publish_report(report, directory: Path):
    """Write aggregate evidence only; no checkpoints or source windows."""
    if report["status"] != "complete" or report["test_scores_computed"]:
        raise ValueError("Expected a completed validation-only comparison")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "results.json", report)
    table = summary_table(report)
    with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for name, function in (("comparison", comparison_figure),
                           ("confusion", confusion_figure), ("learning", learning_figure)):
        fig = function(report)
        fig.savefig(directory / f"{name}.png", dpi=160)
        svg_path = directory / f"{name}.svg"
        fig.savefig(svg_path, metadata={"Date": None})
        # Matplotlib leaves spaces at SVG path line endings; keep clean text diffs.
        svg_path.write_text("\n".join(line.rstrip() for line in
                                     svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
                            encoding="utf-8")
        plt.close(fig)
    def score(mean, std):
        return f"{mean:.4f}" if std is None else f"{mean:.4f} ± {std:.4f}"
    lines = ["# Supervised development results", "",
             "These are validation results on five fixed subjects, not final test or phone results.", "",
             "The input split, preprocessing, architectures, training budget, and selection rules are",
             "defined in [the protocol](../../supervised-module.md). Three seeds are reported for each",
             "neural architecture; logistic regression is one deterministic fit. Dispersion is the",
             "sample standard deviation across training seeds, not a confidence interval over people.", "",
             "| Model | Runs | Validation accuracy | Validation macro-F1 | Parameters | Mean training (s) | Desktop median (ms) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in table:
        accuracy = score(row['validation_accuracy_mean'], row['validation_accuracy_sample_std'])
        f1 = score(row['validation_macro_f1_mean'], row['validation_macro_f1_sample_std'])
        lines.append(f"| {row['model']} | {row['n_runs']} | {accuracy} | {f1} | {row['parameters']:,} | "
                     f"{row['mean_training_seconds']:.2f} | {row['mean_desktop_median_ms']:.3f} |")
    selected = report['selection']['best_model_by_mean_validation_macro_f1']
    ordered = sorted(report["aggregate"], key=lambda r: -r["validation_macro_f1_mean"])
    gap = ordered[0]["validation_macro_f1_mean"] - ordered[1]["validation_macro_f1_mean"]
    group = [r for r in report['runs'] if r['model'] == selected]
    recalls = {name: float(np.mean([r['validation']['per_class'][name]['recall'] for r in group]))
               for name in report['classes']}
    weakest = sorted(recalls, key=recalls.get)[:2]
    lines += ["", f"The highest mean validation macro-F1 belongs to **{MODEL_NAMES[selected]}**.",
              "This is a development choice for the current split and budget, not evidence of",
              "general superiority or statistical significance.", "",
              f"The mean macro-F1 difference from {MODEL_NAMES[ordered[1]['model']]} is {gap:.4f}.",
              "Read this difference together with the seed dispersion in the table; the current",
              "comparison does not establish a reliable advantage on new subjects.", "",
              f"Its two lowest mean class recalls are {weakest[0]} ({recalls[weakest[0]]:.3f}) and "
              f"{weakest[1]} ({recalls[weakest[1]]:.3f}). The full per-seed matrices and class scores",
              "are in [results.json](results.json).", "",
              "![Validation performance](comparison.png)", "",
              "Bars show means; points show individual runs; neural error bars show ±1 sample",
              "standard deviation. The logistic baseline has one fit and no seed error bar.", "",
              "![Validation confusion](confusion.png)", "",
              "These row-normalized matrices use the predeclared first seed (570), rather than",
              "selecting a favorable seed for the figure. Each neural result has two additional",
              "seeds in the machine-readable record.", "",
              "![Learning curves](learning.png)", "",
              "Each curve is one run. Circles identify the highest-validation-macro-F1 epoch;",
              "shorter curves reflect the shared early-stopping rule.", "",
              "All selected checkpoints passed reload checks. Each neural architecture also passed",
              "a saved-and-reloaded `.pt2` CPU export check on six training windows. Export seed",
              "selection is recorded separately and is based on validation, not test scores.", "",
              "Desktop times exclude sensor preprocessing and are not phone measurements.",
              "Elapsed training times also reflect different stopping epochs and variable desktop",
              "workload during the session. Treat them as run provenance, not a controlled",
              "architecture-speed ranking.",
              "The official test set remains unscored. No self-supervised, distillation, or",
              "Android experiment is included. Checkpoints and full predictions stay in ignored",
              "`ml/outputs/supervised-v1/`; rerun the notebook to recreate them.", ""]
    (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return table
