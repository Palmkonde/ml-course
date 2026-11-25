import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from models import CLASSES
from training import loader


@torch.inference_mode()
def extract(model, dataset, settings, device):
    embeddings, pixels, labels = [], [], []
    model.eval()
    for images, targets in loader(dataset, settings):
        mu, _ = model.encode(images.to(device))
        embeddings.append(mu.cpu().numpy())
        pixels.append(images.flatten(1).numpy())
        labels.append(targets.numpy())
    return tuple(np.concatenate(parts) for parts in (embeddings, pixels, labels))


def support_draws(labels, shots, repeats, seed):
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(labels == digit) for digit in range(CLASSES)]
    return {
        k: np.stack([
            np.concatenate([rng.choice(pool, k, replace=False) for pool in pools])
            for _ in range(repeats)
        ])
        for k in shots
    }


def prototype_predict(support, labels, query):
    prototypes = np.stack([support[labels == digit].mean(0) for digit in range(CLASSES)])
    return cdist(query, prototypes, metric="sqeuclidean").argmin(axis=1)


def logistic_predict(support, labels, query, regularization):
    classifier = LogisticRegression(C=regularization, solver="lbfgs", max_iter=2000)
    with threadpool_limits(limits=1):
        return classifier.fit(support, labels).predict(query)


def select_regularization(support, labels, validation, validation_labels, draws, candidates):
    scores = {
        c: np.mean([
            np.mean(logistic_predict(support[ids], labels[ids], validation, c) == validation_labels)
            for ids in draws
        ])
        for c in candidates
    }
    return max(scores, key=scores.get), scores


def evaluate(support, query, draws, regularization):
    support_latent, support_pixels, support_labels = support
    query_latent, query_pixels, query_labels = query
    results, predictions = {}, {}
    for k, repetitions in draws.items():
        predictions[k] = {
            "Raw pixel prototype": [],
            "VAE prototype": [],
            "VAE logistic regression": [],
        }
        for ids in repetitions:
            labels = support_labels[ids]
            by_method = predictions[k]
            by_method["Raw pixel prototype"].append(prototype_predict(support_pixels[ids], labels, query_pixels))
            by_method["VAE prototype"].append(prototype_predict(support_latent[ids], labels, query_latent))
            by_method["VAE logistic regression"].append(
                logistic_predict(support_latent[ids], labels, query_latent, regularization[k])
            )
        results[k] = {
            name: (np.asarray(values) == query_labels).mean(axis=1)
            for name, values in predictions[k].items()
        }
    return results, predictions
