"""Extract handcrafted features from histopathology images."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from rgb_histogram import extract_rgb_histogram
from glcm import extract_glcm, get_glcm_feature_names


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
METADATA_PATH = DATA_ROOT / "full" / "image_metadata.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

METADATA_COLUMNS = ["id", "path", "label", "dataset"]


def get_feature_config(method, bins):
    """Return the extractor, feature names, and configuration."""

    if method == "rgb_histogram":
        if bins <= 0 or bins > 256 or 256 % bins != 0:
            raise ValueError(
                "RGB histogram bins must be a positive divisor of 256."
            )

        feature_names = [
            f"{channel}_hist_{i:03d}"
            for channel in ("r", "g", "b")
            for i in range(bins)
        ]

        def extractor(image):
            return extract_rgb_histogram(image, bins=bins)

        return extractor, feature_names, {"bins": bins}

    if method == "glcm":
        feature_names = get_glcm_feature_names()

        def extractor(image):
            return extract_glcm(image)

        return extractor, feature_names, {
            "levels": 32,
            "distances": [1, 2, 4],
            "angles_degrees": [0, 45, 90, 135],
            "symmetric": True,
            "normed": True,
        }

    raise ValueError(f"Unsupported feature method: {method}")


def load_metadata():
    """Load and validate the image metadata."""

    metadata = pd.read_csv(METADATA_PATH)

    missing = set(METADATA_COLUMNS) - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    if metadata[METADATA_COLUMNS].isna().any().any():
        raise ValueError("Metadata contains missing required values.")

    if metadata.duplicated(subset=["dataset", "id"]).any():
        raise ValueError("Duplicate dataset/image identifiers found.")

    if metadata["path"].duplicated().any():
        raise ValueError("Duplicate image paths found.")

    return metadata[METADATA_COLUMNS].reset_index(drop=True)


def extract_features(method, bins=16, batch_size=1000):
    """Extract features in resumable batches."""

    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    extractor, feature_names, parameters = get_feature_config(
        method, bins
    )

    metadata = load_metadata()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suffix = f"_{bins}" if method == "rgb_histogram" else ""
    run_name = f"{method}{suffix}"

    output_path = OUTPUT_DIR / f"{run_name}.csv"
    progress_path = OUTPUT_DIR / f"{run_name}.progress.json"

    total = len(metadata)
    feature_count = len(feature_names)
    expected_columns = METADATA_COLUMNS + feature_names

    print(f"Method: {method}")
    print(f"Images: {total:,}")
    print(f"Features per image: {feature_count}")
    print(f"Batch size: {batch_size:,}")
    print(f"Output: {output_path}")
    print()

    start_index = 0
    elapsed_previous = 0.0

    if progress_path.exists():
        if not output_path.exists():
            raise RuntimeError(
                "Progress file exists, but output CSV is missing."
            )

        progress = json.loads(progress_path.read_text())

        if (
            progress["method"] != method
            or progress["parameters"] != parameters
            or progress["total_images"] != total
        ):
            raise RuntimeError(
                "Saved progress does not match current configuration."
            )

        saved = pd.read_csv(output_path)

        if list(saved.columns) != expected_columns:
            raise RuntimeError("Saved CSV columns do not match.")

        if len(saved) != progress["completed_images"]:
            raise RuntimeError(
                "Saved CSV row count does not match progress."
            )

        if not saved[METADATA_COLUMNS].equals(
            metadata.iloc[:len(saved)].reset_index(drop=True)
        ):
            raise RuntimeError(
                "Saved CSV metadata does not match current metadata."
            )

        start_index = len(saved)
        elapsed_previous = progress["elapsed_seconds"]

        print(f"Resuming from image {start_index:,}")

    elif output_path.exists():
        raise FileExistsError(
            f"Output already exists without progress file: {output_path}"
        )

    start_time = time.perf_counter()

    with tqdm(
        total=total,
        initial=start_index,
        desc=f"Extracting {method}",
        unit="image"
    ) as progress_bar:

        for start in range(start_index, total, batch_size):
            end = min(start + batch_size, total)
            batch_metadata = metadata.iloc[start:end]

            batch_features = np.empty(
                (len(batch_metadata), feature_count),
                dtype=np.float32
            )

            for i, row in enumerate(
                batch_metadata.itertuples(index=False)
            ):
                # Metadata uses Windows-style relative paths.
                relative_path = Path(
                    *str(row.path).replace("\\", "/").split("/")
                )
                image_path = DATA_ROOT / relative_path

                try:
                    with Image.open(image_path) as image:
                        if image.mode != "RGB":
                            raise ValueError(
                                f"Expected RGB, got {image.mode}"
                            )

                        vector = np.asarray(
                            extractor(image),
                            dtype=np.float32
                        )

                    if vector.shape != (feature_count,):
                        raise ValueError(
                            f"Expected {feature_count} features, "
                            f"got shape {vector.shape}"
                        )

                    if not np.all(np.isfinite(vector)):
                        raise ValueError("Non-finite feature values.")

                    if method == "rgb_histogram":
                        if np.any(vector < 0):
                            raise ValueError("Negative histogram values.")

                        channel_sums = vector.reshape(3, bins).sum(axis=1)

                        if not np.allclose(channel_sums, 1.0, atol=1e-5):
                            raise ValueError(
                                "RGB histogram channels are not normalized."
                            )

                    batch_features[i] = vector

                except Exception as exc:
                    raise RuntimeError(
                        f"Feature extraction failed for "
                        f"{row.dataset}/{row.id}: {image_path}"
                    ) from exc

            feature_df = pd.DataFrame(
                batch_features,
                columns=feature_names
            )

            batch_result = pd.concat(
                [
                    batch_metadata.reset_index(drop=True),
                    feature_df
                ],
                axis=1
            )

            batch_result.to_csv(
                output_path,
                mode="a" if start > 0 else "w",
                header=(start == 0),
                index=False
            )

            elapsed = (
                elapsed_previous
                + time.perf_counter()
                - start_time
            )

            progress_data = {
                "method": method,
                "parameters": parameters,
                "total_images": total,
                "completed_images": end,
                "feature_count": feature_count,
                "elapsed_seconds": elapsed
            }

            progress_path.write_text(
                json.dumps(progress_data, indent=2)
            )

            progress_bar.update(end - start)

    elapsed_total = (
        elapsed_previous
        + time.perf_counter()
        - start_time
    )

    print()
    print("Extraction complete.")
    print(f"Output: {output_path}")
    print(f"Rows: {total:,}")
    print(f"Columns: {len(expected_columns)}")
    print(f"Total runtime: {elapsed_total / 60:.2f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract handcrafted histopathology features."
    )

    parser.add_argument(
        "--method",
        choices=["rgb_histogram", "glcm"],
        default="rgb_histogram"
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=16
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000
    )

    args = parser.parse_args()

    extract_features(
        method=args.method,
        bins=args.bins,
        batch_size=args.batch_size
    )
