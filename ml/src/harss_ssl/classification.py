"""Use identical classification training with scratch or pretrained encoders."""

import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from harss_data.spec import CLASSES
from harss_supervised.metrics import classification_metrics
from harss_supervised.models import build_model
from harss_supervised.training import artifact_hashes, load_checkpoint, predict_neural, set_seed


def train_classifier(data, subset, seed, settings, directory, *, method, percent, encoder_path=None):
    set_seed(seed, settings["cpu_threads"])
    model = build_model("cnn", settings)
    if encoder_path is not None:
        model.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu", weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    loader = DataLoader(TensorDataset(torch.from_numpy(subset.x), torch.from_numpy(subset.y)),
                        batch_size=settings["batch_size"], shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    history = []
    best_state, best_f1, best_epoch, stale = None, -1.0, 0, 0
    start = time.perf_counter()
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Classification loss is nonfinite")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"],
                                     error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
        logits = predict_neural(model, data.validation.x)
        scores = classification_metrics(data.validation.y, logits.argmax(axis=1))
        val_loss = float(nn.functional.cross_entropy(torch.from_numpy(logits),
                                                     torch.from_numpy(data.validation.y)))
        if scores["macro_f1"] > best_f1 + 1e-12:
            best_f1, best_epoch, stale = scores["macro_f1"], epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / len(subset.x),
            "validation_loss": val_loss,
            "validation_accuracy": scores["accuracy"],
            "validation_macro_f1": scores["macro_f1"],
        })
        if epoch == 1 or epoch % 5 == 0 or stale >= settings["patience"] or epoch == settings["epochs"]:
            print(f"{directory.name} epoch={epoch} val_macro_f1={scores['macro_f1']:.4f}", flush=True)
        if stale >= settings["patience"]:
            break
    seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    scores = predict_neural(model, data.validation.x)
    predictions = scores.argmax(axis=1)
    metrics = classification_metrics(data.validation.y, predictions)
    if abs(metrics["macro_f1"] - best_f1) > 1e-12:
        raise ValueError("Restored checkpoint differs from the selected validation epoch")
    checkpoint = directory / "model.pt"
    torch.save({"model": "cnn", "config": settings, "seed": seed, "best_epoch": best_epoch,
                "state_dict": best_state, "input_shape": [1, 128, 6],
                "class_order": list(CLASSES), "data_provenance": data.provenance,
                "method": method, "label_percent": percent}, checkpoint)
    restored, _ = load_checkpoint(checkpoint)
    np.testing.assert_array_equal(predict_neural(restored, data.validation.x), scores)
    np.savez_compressed(directory / "validation-predictions.npz", y_true=data.validation.y,
                        y_pred=predictions, logits=scores, subject_id=data.validation.subject_id,
                        source_row=data.validation.source_row)
    return {
        "model": "cnn",
        "method": method,
        "label_percent": percent,
        "seed": seed,
        "directory": directory.name,
        "training_windows": len(subset.x),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_seconds": seconds,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "history": history,
        "validation": metrics,
        "checkpoint_reload_matches": True,
        "artifacts": artifact_hashes(directory, ["model.pt", "validation-predictions.npz"]),
    }
