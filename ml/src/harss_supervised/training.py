"""Training, checkpoint round trips, and desktop-only timing."""

import random
import time
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from harss_data.prepare import write_json
from harss_data.source import sha256
from harss_data.spec import CLASSES
from .features import statistical_features
from .metrics import classification_metrics
from .models import build_model


def set_seed(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def benchmark(predict, window: np.ndarray) -> dict:
    for _ in range(10):
        predict(window)
    milliseconds = []
    for _ in range(50):
        start = time.perf_counter_ns()
        predict(window)
        milliseconds.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "device": "cpu", "batch_size": 1, "warmup_calls": 10, "timed_calls": 50,
        "median_ms": float(np.median(milliseconds)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "scope": "Desktop prediction from an already-standardized window; no sensor preprocessing or phone measurement",
    }


def artifact_hashes(directory, names) -> dict:
    return {name: sha256(directory / name) for name in names}


def train_logistic(data, config, directory) -> dict:
    start = time.perf_counter()
    features = statistical_features(data.train.x)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(
        C=config["logistic_C"], max_iter=config["logistic_max_iter"],
        solver="lbfgs", tol=1e-6, random_state=config["seeds"][0],
    ))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(features, data.train.y)
    training_seconds = time.perf_counter() - start
    scaler, estimator = pipeline.steps[0][1], pipeline.steps[1][1]
    if not np.array_equal(estimator.classes_, np.arange(6)):
        raise ValueError("Logistic class order differs from the model contract")
    logits = pipeline.decision_function(statistical_features(data.validation.x))
    predicted = logits.argmax(axis=1)
    path = directory / "model.npz"
    np.savez_compressed(path, coef=estimator.coef_, intercept=estimator.intercept_,
                        feature_mean=scaler.mean_, feature_scale=scaler.scale_,
                        classes=estimator.classes_)
    with np.load(path, allow_pickle=False) as saved:
        weights = {name: saved[name] for name in saved.files}
    def restored_predict(x):
        features = (statistical_features(x) - weights["feature_mean"]) / weights["feature_scale"]
        return features @ weights["coef"].T + weights["intercept"]
    restored = restored_predict(data.validation.x)
    np.testing.assert_allclose(restored, logits, atol=1e-10, rtol=1e-10)
    np.testing.assert_array_equal(restored.argmax(axis=1), predicted)
    np.savez_compressed(directory / "validation-predictions.npz", y_true=data.validation.y,
                        y_pred=predicted, logits=logits, subject_id=data.validation.subject_id,
                        source_row=data.validation.source_row)
    return {
        "model": "logistic_regression", "seed": config["seeds"][0],
        "stochastic_training": False, "feature_count": 30,
        "parameter_count": int(estimator.coef_.size + estimator.intercept_.size),
        "training_seconds": training_seconds, "epochs_run": None, "best_epoch": None,
        "solver_iterations": estimator.n_iter_.tolist(),
        "validation": classification_metrics(data.validation.y, predicted),
        "desktop_inference": benchmark(restored_predict, data.validation.x[:1]),
        "checkpoint_bytes": path.stat().st_size, "checkpoint_reload_matches": True,
        "history": [],
        "artifacts": artifact_hashes(directory, ["model.npz", "validation-predictions.npz"]),
    }


@torch.inference_mode()
def predict_neural(model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    outputs = [model(torch.from_numpy(x[i:i + batch_size])).cpu().numpy()
               for i in range(0, len(x), batch_size)]
    logits = np.concatenate(outputs)
    if logits.shape != (len(x), 6) or not np.isfinite(logits).all():
        raise ValueError("Classifier produced invalid logits")
    return logits


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["model"], checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


def train_neural(name, seed, data, config, directory) -> dict:
    set_seed(seed, config["cpu_threads"])
    model = build_model(name, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    loader = DataLoader(TensorDataset(torch.from_numpy(data.train.x), torch.from_numpy(data.train.y)),
                        batch_size=config["batch_size"], shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    history, best_state, best_f1, best_epoch, stale = [], None, -1.0, 0, 0
    start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is nonfinite")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"],
                                     error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
        logits = predict_neural(model, data.validation.x)
        metrics = classification_metrics(data.validation.y, logits.argmax(axis=1))
        val_loss = float(nn.functional.cross_entropy(torch.from_numpy(logits),
                                                     torch.from_numpy(data.validation.y)))
        improved = metrics["macro_f1"] > best_f1 + 1e-12
        if improved:
            best_f1, best_epoch, stale = metrics["macro_f1"], epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": total_loss / len(data.train.x),
                        "validation_loss": val_loss, "validation_accuracy": metrics["accuracy"],
                        "validation_macro_f1": metrics["macro_f1"],
                        "seconds": time.perf_counter() - epoch_start})
        if epoch == 1 or epoch % 5 == 0 or stale >= config["patience"]:
            print(f"{name} seed={seed} epoch={epoch} val_macro_f1={metrics['macro_f1']:.4f}", flush=True)
        if stale >= config["patience"]:
            break
    training_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    logits = predict_neural(model, data.validation.x)
    predicted = logits.argmax(axis=1)
    path = directory / "model.pt"
    torch.save({"model": name, "config": config, "seed": seed, "best_epoch": best_epoch,
                "state_dict": best_state, "input_shape": [1, 128, 6],
                "class_order": list(CLASSES), "data_provenance": data.provenance}, path)
    restored, _ = load_checkpoint(path)
    actual = predict_neural(restored, data.validation.x)
    np.testing.assert_array_equal(actual, logits)
    metrics = classification_metrics(data.validation.y, predicted)
    if abs(metrics["macro_f1"] - best_f1) > 1e-12:
        raise ValueError("Restored checkpoint differs from the selected validation epoch")
    np.savez_compressed(directory / "validation-predictions.npz", y_true=data.validation.y,
                        y_pred=predicted, logits=logits, subject_id=data.validation.subject_id,
                        source_row=data.validation.source_row)
    write_json(directory / "history.json", {"epochs": history})
    return {
        "model": name, "seed": seed, "stochastic_training": True,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_seconds": training_seconds, "epochs_run": len(history), "best_epoch": best_epoch,
        "validation": metrics,
        "desktop_inference": benchmark(lambda x: predict_neural(restored, x), data.validation.x[:1]),
        "checkpoint_bytes": path.stat().st_size, "checkpoint_reload_matches": True,
        "history": history,
        "artifacts": artifact_hashes(directory, ["model.pt", "validation-predictions.npz", "history.json"]),
    }


def sanity_checks(data, config) -> dict:
    """Memorize 24 training-only windows, separate from comparison runs."""
    indices = np.concatenate([np.flatnonzero(data.train.y == i)[:4] for i in range(6)])
    x = torch.from_numpy(data.train.x[indices])
    y = torch.from_numpy(data.train.y[indices])
    checks = {}
    for name in ("cnn", "gru", "cnn_gru"):
        set_seed(config["seeds"][0], config["cpu_threads"])
        model = build_model(name, {**config, "dropout": 0.0})
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        initial_loss = float(nn.functional.cross_entropy(model(x), y).detach())
        for step in range(1, 201):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            if step % 10 == 0:
                model.eval()
                with torch.inference_mode():
                    scores = model(x)
                    accuracy = float((scores.argmax(1) == y).float().mean())
                    final_loss = float(nn.functional.cross_entropy(scores, y))
                if accuracy >= 0.999 and final_loss < 0.1:
                    break
        if accuracy < 0.95 or final_loss >= initial_loss:
            raise ValueError(f"{name} failed the training-only memorization check")
        checks[name] = {"windows": len(indices), "steps": step,
                       "initial_loss": initial_loss, "final_loss": final_loss,
                       "training_accuracy": accuracy, "dropout_for_check": 0.0,
                       "purpose": "Training mechanics only; not a generalization result"}
        print(f"Training-only sanity check passed: {name}, {step} steps", flush=True)
    return checks
