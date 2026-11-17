"""Figures for the SVD representation."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from manifest import TISSUE_CLASSES
from svd_model import components_for_variance

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
CONTEXT_POINTS = "#c9c8c2"
SERIES = ["#2a78d6", "#eb6834"]  # categorical slots 1 and 2

VARIANCE_THRESHOLDS = [0.80, 0.90, 0.95, 0.99]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "text.color": TEXT,
    "font.size": 10,
    "axes.titlesize": 11,
    "legend.frameon": False,
})


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


def _as_image(vector, size):
    return vector.reshape(size, size, 3)


def plot_explained_variance(model, size, path):
    """Cumulative explained variance (left) and per-component variance (right)."""

    ratio = model.explained_variance_ratio
    k = np.arange(1, len(ratio) + 1)
    cumulative = np.cumsum(ratio) * 100

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    left.plot(k, cumulative, color=SERIES[0], linewidth=2)
    left.set_xlim(0, len(ratio) + 1)
    left.set_ylim(min(cumulative[0], 50) - 5, 100)
    left.set_xlabel("Number of components (k)")
    left.set_ylabel("Cumulative explained variance (%)")
    left.set_title("Variance captured by the first k components", loc="left")

    reached = components_for_variance(ratio, VARIANCE_THRESHOLDS)
    for threshold, count in reached.items():
        if count is None:
            continue
        y = cumulative[count - 1]
        left.plot(count, y, "o", markersize=7, color=SERIES[0],
                  markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        left.annotate(f"{threshold:.0%} at k = {count}", (count, y),
                      xytext=(8, -12), textcoords="offset points",
                      color=TEXT_SECONDARY, fontsize=9)

    right.plot(k, ratio * 100, color=SERIES[0], linewidth=2)
    right.set_yscale("log")
    right.set_xlim(0, len(ratio) + 1)
    right.set_xlabel("Component index")
    right.set_ylabel("Explained variance (%, log scale)")
    right.set_title("Variance captured by each component", loc="left")

    fig.suptitle(
        f"Randomized SVD on {size}x{size} RGB pixels, fit on {model.n_train:,} training images",
        x=0.01, ha="left", fontsize=12, color=TEXT,
    )
    fig.tight_layout()
    _save(fig, path)


def plot_components(model, size, path, count=15):
    """The mean training image followed by the leading components as images."""

    count = min(count, model.n_components)
    columns = 8
    rows = int(np.ceil((count + 1) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(1.6 * columns, 1.85 * rows),
                             gridspec_kw={"hspace": 0.35})

    for ax in axes.flat:
        ax.axis("off")

    axes.flat[0].imshow(_as_image(model.mean, size), interpolation="nearest")
    axes.flat[0].set_title("Mean image", fontsize=9)

    for i in range(count):
        component = model.components[i]
        # Map each component to [0, 1] symmetrically so mid-gray means zero.
        scaled = 0.5 + component / (2 * np.abs(component).max())
        ax = axes.flat[i + 1]
        ax.imshow(_as_image(scaled, size), interpolation="nearest")
        ax.set_title(f"C{i + 1} ({model.explained_variance_ratio[i]:.1%})", fontsize=9)

    fig.suptitle(
        "Mean training image and leading SVD components (mid-gray = 0)",
        x=0.01, ha="left", fontsize=12,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.84, wspace=0.08)
    _save(fig, path)


def plot_reconstructions(model, pixels, labels, size, path, ks, seed):
    """One validation image per class, reconstructed from the first k components."""

    rng = np.random.default_rng(seed)
    ks = [k for k in ks if k <= model.n_components]
    columns = ["Original"] + [f"k = {k}" for k in ks]

    fig, axes = plt.subplots(
        len(TISSUE_CLASSES), len(columns),
        figsize=(1.5 * len(columns), 1.5 * len(TISSUE_CLASSES)),
        squeeze=False,
    )

    for row, label in enumerate(TISSUE_CLASSES):
        index = rng.choice(np.nonzero(labels == label)[0])
        original = pixels[index:index + 1]
        features = model.transform(original)

        images = [original[0] / 255]
        for k in ks:
            images.append(np.clip(model.inverse_transform(features, k)[0], 0, 1))

        for col, image in enumerate(images):
            ax = axes[row, col]
            ax.imshow(_as_image(image, size), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(columns[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(label, fontsize=10, weight="bold", color=TEXT)

    fig.suptitle(
        f"Validation images rebuilt from k components ({size}x{size})",
        x=0.01, ha="left", fontsize=12,
    )
    fig.tight_layout()
    _save(fig, path)


def plot_class_scatter(features, labels, path, seed, max_points=4000):
    """Small multiples of component 1 vs 2: each panel highlights one class."""

    rng = np.random.default_rng(seed)
    x, y = features[:, 0], features[:, 1]

    context = np.arange(len(labels))
    if len(context) > max_points:
        context = rng.choice(context, max_points, replace=False)

    pad_x = (x.max() - x.min()) * 0.04
    pad_y = (y.max() - y.min()) * 0.04
    fig, axes = plt.subplots(3, 3, figsize=(10, 9.4), sharex=True, sharey=True)

    for ax, label in zip(axes.flat, TISSUE_CLASSES):
        members = np.nonzero(labels == label)[0]
        shown = members if len(members) <= max_points else rng.choice(members, max_points, replace=False)

        ax.scatter(x[context], y[context], s=4, color=CONTEXT_POINTS, linewidths=0, rasterized=True)
        ax.scatter(x[shown], y[shown], s=9, color=SERIES[0], alpha=0.7,
                   edgecolors=SURFACE, linewidths=0.3, rasterized=True)
        ax.set_title(f"{label}  (n = {len(members):,})", loc="left")
        ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
        ax.set_ylim(y.min() - pad_y, y.max() + pad_y)

    for ax in axes[-1]:
        ax.set_xlabel("Component 1")
    for ax in axes[:, 0]:
        ax.set_ylabel("Component 2")

    fig.suptitle(
        "Training images on the first two SVD components (blue = class, gray = all classes)",
        x=0.01, ha="left", fontsize=12,
    )
    fig.tight_layout()
    _save(fig, path)


def plot_benchmark(table, path):
    """Fit time and peak memory against component count, one line per resolution."""

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [
        (left, "fit_seconds", "Fit time (s)", "Randomized SVD fit time"),
        (right, "fit_peak_mb", "Peak NumPy memory during fit (MB)", "Peak memory during fit"),
    ]

    sizes = sorted(table["size"].unique())
    for ax, column, ylabel, title in panels:
        for color, size in zip(SERIES, sizes):
            rows = table[table["size"] == size].sort_values("n_components")
            ax.plot(rows["n_components"], rows[column], color=color, linewidth=2,
                    marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=f"{size}x{size}")
            last = rows.iloc[-1]
            ax.annotate(f"{size}x{size}", (last["n_components"], last[column]),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        color=TEXT_SECONDARY, fontsize=9)
        ax.set_xscale("log", base=2)
        ticks = sorted(table["n_components"].unique())
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_xlabel("Number of components (k)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left")

    n_train = int(table["n_train"].iloc[0])
    fig.suptitle(f"Resource use when fitting on {n_train:,} training images",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    _save(fig, path)
