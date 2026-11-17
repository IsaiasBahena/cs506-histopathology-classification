"""Benchmark randomized SVD fit time and memory across resolutions and component counts.

Usage (from the svd/ directory):
    python benchmark.py --sample 1pct
    python benchmark.py --sample full --sizes 32 64 --components 16 32 64 128 256

Writes results/<sample>/svd_benchmark.csv and results/<sample>/svd_benchmark.png.

Memory is the peak of NumPy array allocations during the fit, measured with
tracemalloc. It includes the float32 copy of the training matrix and SVD work
arrays, but not BLAS-internal buffers, so treat it as a close lower bound.
"""

import argparse
import time
import tracemalloc

import numpy as np
import pandas as pd

import figures
from manifest import build_manifest
from pixels import load_pixel_matrix
from run_svd import SEED, output_dirs
from svd_model import fit_svd


def measure_fit(train_pixels, n_components, repeats):
    """Return (median fit seconds, peak MB) for one configuration."""

    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        fit_svd(train_pixels, n_components, SEED)
        seconds.append(time.perf_counter() - start)

    tracemalloc.start()
    fit_svd(train_pixels, n_components, SEED)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return float(np.median(seconds)), peak / 1e6


def run(sample, sizes, component_counts, repeats, workers):
    outputs, results = output_dirs(sample)
    manifest = build_manifest(sample)
    is_train = (manifest["split"] == "train").to_numpy()

    rows = []
    for size in sizes:
        pixels = load_pixel_matrix(manifest, size, outputs / f"pixels_{size}px.npz", workers)
        train = np.ascontiguousarray(pixels[is_train])
        val = pixels[(manifest["split"] == "val").to_numpy()]
        max_k = min(train.shape)

        for k in component_counts:
            if k > max_k:
                print(f"Skipping {size}x{size}, k = {k}: more than {max_k} components is impossible")
                continue
            fit_seconds, peak_mb = measure_fit(train, k, repeats)

            model = fit_svd(train, k, SEED)
            start = time.perf_counter()
            model.transform(val)
            transform_seconds = time.perf_counter() - start

            rows.append({
                "size": size,
                "n_pixel_features": train.shape[1],
                "n_components": k,
                "n_train": train.shape[0],
                "train_matrix_mb_uint8": round(train.nbytes / 1e6, 2),
                "train_matrix_mb_float32": round(train.size * 4 / 1e6, 2),
                "fit_seconds": round(fit_seconds, 4),
                "fit_peak_mb": round(peak_mb, 2),
                "val_transform_seconds": round(transform_seconds, 4),
                "n_val": val.shape[0],
                "explained_variance": round(float(model.explained_variance_ratio.sum()), 4),
            })
            print(f"{size}x{size}  k = {k:4d}  fit {fit_seconds:8.3f} s  peak {peak_mb:9.1f} MB  "
                  f"variance {rows[-1]['explained_variance']:.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(results / "svd_benchmark.csv", index=False)
    print(f"Saved benchmark table: {results / 'svd_benchmark.csv'}")
    figures.plot_benchmark(table, results / "svd_benchmark.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", choices=["1pct", "full"], default="1pct")
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--components", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--repeats", type=int, default=3, help="timed fits per configuration (median reported)")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    run(args.sample, args.sizes, args.components, args.repeats, args.workers)


if __name__ == "__main__":
    main()
