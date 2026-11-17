"""Downsample images and stack them into a pixel matrix (one row per image).

Each image is converted to RGB, downsampled with box averaging (each output pixel
is the mean of a block of input pixels, the same method as the October
preprocessing check-in), and flattened in (row, column, channel) order. A 32x32
image becomes a vector of 32 * 32 * 3 = 3,072 uint8 values.

Pixel matrices are cached as .npz files keyed by the manifest so repeated runs
skip the slow image decoding step.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from PIL import Image

from manifest import DATA_ROOT


def downsample(image, size):
    """Return a (size, size, 3) uint8 array for a PIL image."""

    rgb = image.convert("RGB")
    small = rgb.resize((size, size), Image.Resampling.BOX)
    return np.asarray(small, dtype=np.uint8)


def load_vector(relative_path, size):
    """Read one image (path relative to data/) and return its flattened pixels."""

    with Image.open(DATA_ROOT / relative_path) as image:
        return downsample(image, size).reshape(-1)


def build_pixel_matrix(relative_paths, size, workers=None):
    """Return a (n_images, size * size * 3) uint8 matrix, rows in input order."""

    workers = workers or os.cpu_count() or 1
    n_features = size * size * 3
    matrix = np.empty((len(relative_paths), n_features), dtype=np.uint8)

    load = partial(load_vector, size=size)
    if workers == 1:
        rows = map(load, relative_paths)
        for i, row in enumerate(rows):
            matrix[i] = row
        return matrix

    chunksize = max(1, len(relative_paths) // (workers * 8))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(load, relative_paths, chunksize=chunksize)):
            matrix[i] = row
    return matrix


def load_pixel_matrix(manifest, size, cache_path, workers=None):
    """Return the pixel matrix for `manifest`, using `cache_path` when it matches."""

    keys = (manifest["dataset"] + "/" + manifest["id"]).to_numpy(dtype=str)

    if cache_path.exists():
        with np.load(cache_path) as cached:
            if cached["X"].shape[1] == size * size * 3 and np.array_equal(cached["key"], keys):
                print(f"Loaded cached pixels: {cache_path}")
                return cached["X"]
        print(f"Cache does not match the manifest, rebuilding: {cache_path}")

    print(f"Downsampling {len(manifest):,} images to {size}x{size}...")
    matrix = build_pixel_matrix(manifest["path"].tolist(), size, workers)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X=matrix, key=keys)
    print(f"Saved pixel cache: {cache_path}")
    return matrix
