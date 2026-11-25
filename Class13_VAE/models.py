import torch
from torch import nn
from torch.nn import functional as F

PIXELS = 28 * 28
CLASSES = 10


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(PIXELS, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, PIXELS),
        )

    def encode(self, images):
        return self.encoder(images.flatten(1))

    def decode(self, latent):
        return self.decoder(latent).reshape(-1, 1, 28, 28)

    def forward(self, images):
        return self.decode(self.encode(images))


class VAE(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=256, conditional=False):
        super().__init__()
        self.conditional = conditional
        self.latent_dim = latent_dim
        condition_size = CLASSES if conditional else 0
        self.encoder = nn.Sequential(nn.Linear(PIXELS + condition_size, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, PIXELS),
        )

    def _condition(self, features, labels):
        if self.conditional:
            if labels is None:
                raise ValueError("Conditional VAE requires class labels.")
            return torch.cat((features, F.one_hot(labels, CLASSES).to(features.dtype)), dim=1)
        if labels is not None:
            raise ValueError("Unconditional VAE takes no labels.")
        return features

    def encode(self, images, labels=None):
        hidden = self.encoder(self._condition(images.flatten(1), labels))
        return self.mu(hidden), self.logvar(hidden)

    def decode(self, latent, labels=None):
        return self.decoder(self._condition(latent, labels)).reshape(-1, 1, 28, 28)

    def forward(self, images, labels=None):
        mu, logvar = self.encode(images, labels)
        latent = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        return self.decode(latent, labels), mu, logvar


def reconstruction_loss(logits, images):
    per_pixel = F.binary_cross_entropy_with_logits(logits, images, reduction="none")
    return per_pixel.flatten(1).sum(1).mean()


def vae_loss(logits, images, mu, logvar, beta=1.0):
    reconstruction = reconstruction_loss(logits, images)
    kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(1).mean()
    return reconstruction + beta * kl, reconstruction, kl
