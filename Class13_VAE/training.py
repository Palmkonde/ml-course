import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from models import Autoencoder, VAE, reconstruction_loss, vae_loss

METRICS = ("total", "reconstruction", "kl")


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    latent_dim: int = 16
    hidden_dim: int = 256
    batch_size: int = 256
    epochs: int = 15
    learning_rate: float = 0.001
    beta: float = 1.0
    quick: bool = False


def setup(settings):
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mnist(train=True):
    return MNIST("data", train=train, download=True, transform=ToTensor())


def split_train(dataset, settings, include_support=False):
    rng = np.random.default_rng(settings.seed)
    indices = rng.permutation(len(dataset))
    validation, support, representation = np.split(indices, [5000, 10000 if include_support else 5000])
    if settings.quick:
        validation = validation[:512]
        representation = representation[:2048]
    return tuple(Subset(dataset, part.tolist()) for part in (representation, validation, support))


def loader(dataset, settings, shuffle=False):
    generator = torch.Generator().manual_seed(settings.seed)
    return DataLoader(dataset, batch_size=settings.batch_size, shuffle=shuffle, generator=generator)


def vae_batch_loss(model, images, labels, settings):
    condition = labels.to(images.device) if model.conditional else None
    logits, mu, logvar = model(images, condition)
    if logits.shape != images.shape or mu.shape != logvar.shape or mu.shape != (len(images), model.latent_dim):
        raise RuntimeError("Unexpected VAE output shapes.")
    return vae_loss(logits, images, mu, logvar, settings.beta)


def autoencoder_batch_loss(model, images, _labels, _settings):
    logits = model(images)
    if logits.shape != images.shape:
        raise RuntimeError("Unexpected autoencoder output shape.")
    return (reconstruction_loss(logits, images),)


def run_epoch(model, batches, settings, device, optimizer=None, *, batch_loss=vae_batch_loss, metrics=METRICS):
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(len(metrics), device=device)
    count = 0
    with torch.set_grad_enabled(training):
        for images, labels in batches:
            images = images.to(device)
            losses = batch_loss(model, images, labels, settings)
            values = torch.stack(losses).detach()
            if not torch.isfinite(values).all():
                raise FloatingPointError("Non-finite training or validation loss.")
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses[0].backward()
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Missing or non-finite gradient.")
                optimizer.step()
            totals += values * len(images)
            count += len(images)
    return dict(zip(metrics, (totals / count).tolist(), strict=True))


def fit_or_load(settings, train, validation, device, *, conditional):
    model = VAE(settings.latent_dim, settings.hidden_dim, conditional)
    kind = "conditional" if conditional else "unconditional"
    return _fit_or_load(model, settings, train, validation, device, kind, vae_batch_loss, METRICS)


def fit_autoencoder(settings, train, validation, device):
    model = Autoencoder(settings.latent_dim, settings.hidden_dim)
    return _fit_or_load(
        model, settings, train, validation, device,
        "autoencoder", autoencoder_batch_loss, ("reconstruction",),
    )


def _fit_or_load(model, settings, train, validation, device, kind, batch_loss, metrics):
    model.to(device)
    mode = "quick" if settings.quick else "full"
    signature = {"settings": asdict(settings), "split_version": 1, "model_version": 1}
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12]
    path = Path("runs") / f"{kind}-{mode}-{digest}.pt"
    if path.exists() and os.getenv("VAE_RETRAIN") != "1":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded {path} (epoch {checkpoint['epoch']})")
        return model.eval(), checkpoint["history"]

    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    train_batches = loader(train, settings, shuffle=True)
    val_batches = loader(validation, settings)
    history = []
    best_score = float("inf")
    for epoch in range(1, settings.epochs + 1):
        train_metrics = run_epoch(model, train_batches, settings, device, optimizer, batch_loss=batch_loss, metrics=metrics)
        # Use a fixed validation noise stream for comparable epoch scores.
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            torch.random.default_generator.manual_seed(settings.seed + 1)
            if device.type == "cuda":
                torch.cuda.manual_seed(settings.seed + 1)
            val_metrics = run_epoch(model, val_batches, settings, device, batch_loss=batch_loss, metrics=metrics)
        history.append((train_metrics, val_metrics))
        print(f"Epoch {epoch:2d} | train {train_metrics} | validation {val_metrics}")
        if val_metrics[metrics[0]] < best_score:
            best_score = val_metrics[metrics[0]]
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    path.parent.mkdir(exist_ok=True)
    torch.save({"model": best_state, "history": history, "epoch": best_epoch, "config": signature}, path)
    model.load_state_dict(best_state)
    print(f"Saved {path} (epoch {best_epoch})")
    return model.eval(), history
