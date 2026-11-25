import matplotlib.pyplot as plt
import numpy as np

from training import METRICS


def image_grid(images, title, row_labels=None, column_labels=None):
    rows, columns = images.shape[:2]
    fig, axes = plt.subplots(rows, columns, figsize=(max(4, columns * 0.85), rows * 0.85 + 0.7), squeeze=False)
    for (row, column), axis in np.ndenumerate(axes):
        axis.imshow(images[row, column].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axis.set(xticks=[], yticks=[])
        for spine in axis.spines.values():
            spine.set_visible(False)
        if column == 0 and row_labels is not None:
            axis.set_ylabel(row_labels[row])
        if row == 0 and column_labels is not None:
            axis.set_title(column_labels[column], fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def learning_curves(history, mode, *, metrics=METRICS, title="VAE learning curves"):
    fig, axes = plt.subplots(1, len(metrics), figsize=(max(5, len(metrics) * 3.7), 3), squeeze=False)
    axes = axes[0]
    epochs = range(1, len(history) + 1)
    for metric, axis in zip(metrics, axes, strict=True):
        for index, label in enumerate(("Training", "Validation")):
            axis.plot(epochs, [entry[index][metric] for entry in history], marker=".", label=label)
        axis.set(title=metric.capitalize(), xlabel="Epoch", ylabel="Mean loss per image")
    axes[0].legend()
    fig.suptitle(f"{title} | {mode}")
    fig.tight_layout()
    plt.show()
