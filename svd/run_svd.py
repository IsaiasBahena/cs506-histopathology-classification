"""Fit randomized SVD on training pixels and write SVD features for every split.

Usage (from the svd/ directory):
    python run_svd.py --sample 1pct --size 32 --components 256
    python run_svd.py --sample full --size 32 --components 256

Outputs, where <tag> is "<size>px":
    outputs/<sample>/svd_<tag>_features.csv   features for train, val, and test (gitignored)
    outputs/<sample>/svd_<tag>_model.npz      fitted mean and components (gitignored)
    results/<sample>/svd_<tag>_params.json    parameters, shapes, timings, versions
    results/<sample>/explained_variance_<tag>.csv
    results/<sample>/*.png                    figures
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import sklearn

import figures
from manifest import SPLITS, TISSUE_CLASSES, build_manifest
from pixels import load_pixel_matrix
from svd_model import components_for_variance, fit_on_train_split

SVD_ROOT = Path(__file__).resolve().parent
SEED = 506
RECONSTRUCTION_KS = [8, 32, 128, 256]


def output_dirs(sample):
    outputs = SVD_ROOT / "outputs" / sample
    results = SVD_ROOT / "results" / sample
    outputs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return outputs, results


def feature_names(n_components):
    return [f"svd_{i:03d}" for i in range(n_components)]


def write_features(manifest, features, path):
    values = pd.DataFrame(features, columns=feature_names(features.shape[1]))
    table = pd.concat([manifest.reset_index(drop=True), values], axis=1)
    table.to_csv(path, index=False, float_format="%.7g")
    print(f"Saved features: {path} ({table.shape[0]:,} rows x {features.shape[1]} features)")


def write_explained_variance(model, path):
    ratio = model.explained_variance_ratio.astype(np.float64)
    table = pd.DataFrame({
        "component": np.arange(1, len(ratio) + 1),
        "singular_value": model.singular_values,
        "explained_variance": model.explained_variance,
        "explained_variance_ratio": ratio,
        "cumulative_explained_variance_ratio": np.cumsum(ratio),
    })
    table.to_csv(path, index=False, float_format="%.7g")
    print(f"Saved explained variance table: {path}")


def run(sample, size, n_components, workers):
    outputs, results = output_dirs(sample)
    tag = f"{size}px"

    manifest = build_manifest(sample)
    counts = manifest.groupby(["split", "label"]).size().unstack(fill_value=0)
    print("\nImages per split and class:")
    print(counts.reindex(index=SPLITS, columns=TISSUE_CLASSES).to_string())

    pixels = load_pixel_matrix(manifest, size, outputs / f"pixels_{tag}.npz", workers)
    split = manifest["split"].to_numpy()
    labels = manifest["label"].to_numpy()
    is_train = split == "train"

    # Fit on training rows only; val and test are only ever transformed.
    print(f"\nFitting randomized SVD: {is_train.sum():,} x {pixels.shape[1]:,} -> {n_components} components")
    start = time.perf_counter()
    model = fit_on_train_split(pixels, split, n_components, SEED)
    fit_seconds = time.perf_counter() - start
    print(f"Fit took {fit_seconds:.2f} s")

    start = time.perf_counter()
    features = model.transform(pixels)
    transform_seconds = time.perf_counter() - start

    write_features(manifest, features, outputs / f"svd_{tag}_features.csv")
    model.save(outputs / f"svd_{tag}_model.npz")
    write_explained_variance(model, results / f"explained_variance_{tag}.csv")

    reached = components_for_variance(model.explained_variance_ratio, figures.VARIANCE_THRESHOLDS)
    print("\nComponents needed to reach a share of training variance:")
    for threshold, count in reached.items():
        print(f"  {threshold:.0%}: {count if count else f'not reached within {n_components}'}")

    params = {
        "sample": sample,
        "image_size": size,
        "pixel_order": "row, column, channel (RGB); values scaled to [0, 1] before centering",
        "downsampling": "PIL box averaging",
        "n_pixel_features": int(pixels.shape[1]),
        "n_components": n_components,
        "feature_columns": f"svd_000 .. svd_{n_components - 1:03d}",
        "method": "mean-centered randomized SVD (sklearn PCA, svd_solver='randomized', no whitening)",
        "fit_split": "train",
        "random_state": SEED,
        "images_per_split": {s: int((split == s).sum()) for s in SPLITS},
        "total_explained_variance_ratio": float(model.explained_variance_ratio.sum()),
        "components_for_variance": {f"{t:.2f}": c for t, c in reached.items()},
        "fit_seconds": round(fit_seconds, 3),
        "transform_seconds_all_splits": round(transform_seconds, 3),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "Pillow": PIL.__version__,
        },
    }
    params_path = results / f"svd_{tag}_params.json"
    params_path.write_text(json.dumps(params, indent=2) + "\n")
    print(f"Saved parameters: {params_path}")

    figures.plot_explained_variance(model, size, results / f"explained_variance_{tag}.png")
    figures.plot_components(model, size, results / f"components_{tag}.png")
    is_val = split == "val"
    figures.plot_reconstructions(
        model, pixels[is_val], labels[is_val], size,
        results / f"reconstructions_{tag}.png", RECONSTRUCTION_KS, SEED,
    )
    figures.plot_class_scatter(
        features[is_train], labels[is_train], results / f"class_scatter_{tag}.png", SEED,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", choices=["1pct", "full"], default="1pct")
    parser.add_argument("--size", type=int, default=32, help="downsampled image side length")
    parser.add_argument("--components", type=int, default=256, help="number of SVD components to keep")
    parser.add_argument("--workers", type=int, default=None, help="image loading processes (default: all CPUs)")
    args = parser.parse_args()
    run(args.sample, args.size, args.components, args.workers)


if __name__ == "__main__":
    main()
