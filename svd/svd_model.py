"""Fit randomized SVD on training pixels and apply it to any split.

The model is mean-centered truncated SVD (equivalent to PCA without whitening):

    features = (x / 255 - train_mean) @ components.T

It is fitted with scikit-learn's randomized solver and stored as plain NumPy
arrays, so teammates can load and apply it without pickle or a matching
scikit-learn version.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


def to_unit_float(pixels):
    """Scale uint8 pixels to float32 values in [0, 1]."""

    return pixels.astype(np.float32) / np.float32(255)


@dataclass
class SvdModel:
    mean: np.ndarray  # (n_features,) mean training pixel vector, in [0, 1] units
    components: np.ndarray  # (n_components, n_features) right singular vectors
    singular_values: np.ndarray  # (n_components,)
    explained_variance: np.ndarray  # (n_components,) variance of each feature column
    explained_variance_ratio: np.ndarray  # (n_components,) share of total train variance
    n_train: int

    @property
    def n_components(self):
        return self.components.shape[0]

    def transform(self, pixels, batch_size=10_000):
        """Project uint8 pixel rows onto the components, in memory-bounded batches."""

        out = np.empty((pixels.shape[0], self.n_components), dtype=np.float32)
        for start in range(0, pixels.shape[0], batch_size):
            batch = to_unit_float(pixels[start:start + batch_size]) - self.mean
            out[start:start + batch_size] = batch @ self.components.T
        return out

    def inverse_transform(self, features, n_components=None):
        """Reconstruct [0, 1] pixel rows from the first `n_components` features."""

        k = n_components or self.n_components
        return features[:, :k] @ self.components[:k] + self.mean

    def save(self, path):
        np.savez(
            path,
            mean=self.mean,
            components=self.components,
            singular_values=self.singular_values,
            explained_variance=self.explained_variance,
            explained_variance_ratio=self.explained_variance_ratio,
            n_train=self.n_train,
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            return cls(
                mean=f["mean"],
                components=f["components"],
                singular_values=f["singular_values"],
                explained_variance=f["explained_variance"],
                explained_variance_ratio=f["explained_variance_ratio"],
                n_train=int(f["n_train"]),
            )


def fit_svd(train_pixels, n_components, seed):
    """Fit randomized SVD on uint8 training pixels only."""

    n_samples, n_features = train_pixels.shape
    if not 0 < n_components <= min(n_samples, n_features):
        raise ValueError(
            f"n_components must be in [1, {min(n_samples, n_features)}], got {n_components}"
        )

    # copy=False lets PCA center the float matrix in place, halving peak memory.
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=seed,
        copy=False,
    )
    pca.fit(to_unit_float(train_pixels))

    return SvdModel(
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        singular_values=pca.singular_values_.astype(np.float32),
        explained_variance=pca.explained_variance_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        n_train=n_samples,
    )


def fit_on_train_split(pixels, split, n_components, seed):
    """Fit on the rows whose split is "train"; val and test rows are never seen."""

    is_train = np.asarray(split) == "train"
    if not is_train.any():
        raise ValueError("No training rows to fit on")
    return fit_svd(pixels[is_train], n_components, seed)


def components_for_variance(explained_variance_ratio, thresholds):
    """Return {threshold: smallest k reaching it, or None if never reached}."""

    cumulative = np.cumsum(explained_variance_ratio)
    result = {}
    for t in thresholds:
        reached = np.nonzero(cumulative >= t)[0]
        result[t] = int(reached[0]) + 1 if reached.size else None
    return result
