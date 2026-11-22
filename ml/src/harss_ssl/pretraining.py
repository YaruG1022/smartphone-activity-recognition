"""Recover masked signal spans using only the training subjects' windows."""

import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from harss_data.spec import CHANNELS
from harss_supervised.models import build_model
from harss_supervised.training import artifact_hashes, set_seed


def span_mask(batch_size, steps, span_length, generator):
    """Sample two nonoverlapping spans uniformly over valid start pairs."""
    positions = steps - 2 * span_length + 2
    if span_length < 1 or positions < 2:
        raise ValueError("Two masked spans must fit in each window")
    first = torch.randint(positions, (batch_size,), generator=generator)
    second = torch.randint(positions - 1, (batch_size,), generator=generator)
    second += (second >= first).long()
    left = torch.minimum(first, second)[:, None]
    right = (torch.maximum(first, second) + span_length - 1)[:, None]
    indices = torch.arange(steps)
    return ((indices >= left) & (indices < left + span_length)) | (
        (indices >= right) & (indices < right + span_length)
    )


def pretrain(train_signals, seed, config, directory):
    classifier = config["classifier"]
    settings = config["pretraining"]
    set_seed(seed, classifier["cpu_threads"])
    encoder = build_model("cnn", classifier).encoder
    first, last = classifier["conv_channels"]
    decoder = nn.Sequential(
        nn.ConvTranspose1d(last, first, kernel_size=4, stride=4),
        nn.ReLU(),
        nn.Conv1d(first, len(CHANNELS), kernel_size=3, padding=1),
    )
    network = nn.Sequential(encoder, decoder)
    optimizer = torch.optim.AdamW(network.parameters(), lr=classifier["learning_rate"],
                                  weight_decay=classifier["weight_decay"])
    loader = DataLoader(torch.from_numpy(train_signals), batch_size=classifier["batch_size"],
                        shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed))
    # Separate mask and shuffle streams retain reproducibility if loader behavior changes.
    mask_generator = torch.Generator().manual_seed(seed ^ 0xA5A5A5A5)
    history = []
    started = time.perf_counter()
    network.train()
    for epoch in range(1, settings["epochs"] + 1):
        total_loss = 0.0
        for signals in loader:
            mask = span_mask(len(signals), signals.shape[1], settings["span_length"], mask_generator)
            inputs = signals.masked_fill(mask[:, :, None], 0.0)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = network(inputs).transpose(1, 2)
            loss = nn.functional.mse_loss(reconstructed[mask], signals[mask])
            if not torch.isfinite(loss):
                raise ValueError("Masked reconstruction loss is nonfinite")
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), classifier["gradient_clip_norm"],
                                     error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(signals)
        mean_loss = total_loss / len(train_signals)
        history.append({"epoch": epoch, "masked_train_mse": mean_loss})
        if epoch == 1 or epoch % 5 == 0 or epoch == settings["epochs"]:
            print(f"masked reconstruction seed={seed} epoch={epoch} train_mse={mean_loss:.4f}", flush=True)
    seconds = time.perf_counter() - started
    torch.save(encoder.state_dict(), directory / "encoder.pt")
    return {
        "seed": seed,
        "training_windows": len(train_signals),
        "training_seconds": seconds,
        "history": history,
        "artifacts": artifact_hashes(directory, ["encoder.pt"]),
    }
