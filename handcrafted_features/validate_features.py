"""Validate extracted handcrafted histopathology features."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from glcm import get_glcm_feature_names


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

EXPECTED_DATASETS = {
    "NCT-CRC-HE-100K": 100000,
    "CRC-VAL-HE-7K": 7180,
}

METADATA_COLUMNS = ["id", "path", "label", "dataset"]


def get_validation_config(method, bins):
    if method == "rgb_histogram":
        if bins <= 0 or bins > 256 or 256 % bins != 0:
            raise ValueError("Bins must be a positive divisor of 256.")

        feature_names = [
            f"{channel}_hist_{i:03d}"
            for channel in ("r", "g", "b")
            for i in range(bins)
        ]

        return OUTPUT_DIR / f"rgb_histogram_{bins}.csv", feature_names

    if method == "glcm":
        return OUTPUT_DIR / "glcm.csv", get_glcm_feature_names()

    raise ValueError(f"Unsupported method: {method}")


def validate_features(method, bins=16):
    output_path, feature_names = get_validation_config(method, bins)

    print(f"Method: {method}")
    print(f"File: {output_path}")

    df = pd.read_csv(output_path)

    expected_columns = METADATA_COLUMNS + feature_names
    assert list(df.columns) == expected_columns, (
        "Unexpected CSV columns or column ordering."
    )

    expected_rows = sum(EXPECTED_DATASETS.values())
    assert df.shape == (expected_rows, len(expected_columns)), (
        f"Unexpected dataset shape: {df.shape}"
    )

    print("\nDataset shape:", df.shape)

    print("\nDataset distribution:")
    print(df["dataset"].value_counts())

    print("\nClass distribution:")
    print(pd.crosstab(df["dataset"], df["label"]))

    assert df["dataset"].value_counts().to_dict() == EXPECTED_DATASETS, (
        "Unexpected dataset distribution."
    )

    assert df["path"].is_unique, "Duplicate image paths found."

    missing_values = int(df.isna().sum().sum())
    print("\nMissing values:", missing_values)
    assert missing_values == 0, "Missing values found."

    features = df[feature_names].to_numpy()

    nonfinite_count = int((~np.isfinite(features)).sum())
    print("Non-finite features:", nonfinite_count)
    assert nonfinite_count == 0, "Non-finite feature values found."

    print("Duplicate image paths:", df["path"].duplicated().sum())

    if method == "rgb_histogram":
        assert np.all(features >= 0), (
            "Negative histogram values found."
        )

        channel_sums = features.reshape(-1, 3, bins).sum(axis=2)

        print("\nHistogram normalization:")
        print("Minimum channel sum:", channel_sums.min())
        print("Maximum channel sum:", channel_sums.max())

        assert np.allclose(channel_sums, 1.0, atol=1e-5), (
            "Histogram channels are not normalized."
        )

    elif method == "glcm":
        print("\nGLCM feature ranges:")

        for prop in (
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
            "asm",
        ):
            columns = [
                name for name in feature_names
                if f"_{prop}_" in name
            ]

            values = df[columns].to_numpy()

            minimum = values.min()
            maximum = values.max()

            print(f"{prop:15s} min={minimum:.6f} max={maximum:.6f}")

            if prop in ("contrast", "dissimilarity"):
                assert np.all(values >= -1e-6), (
                    f"Negative {prop} values found."
                )

            elif prop in ("homogeneity", "energy", "asm"):
                assert np.all(
                    (values >= -1e-6) & (values <= 1.0 + 1e-6)
                ), f"Invalid {prop} values found."

            elif prop == "correlation":
                assert np.all(
                    (values >= -1.0 - 1e-6)
                    & (values <= 1.0 + 1e-6)
                ), "Invalid correlation values found."

        # Check that features vary across images.
        feature_std = features.std(axis=0)
        constant_features = [
            feature_names[i]
            for i in np.where(feature_std < 1e-10)[0]
        ]

        print("\nConstant GLCM features:", len(constant_features))

        if constant_features:
            print("Constant columns:", constant_features)

        assert not constant_features, (
            "Some GLCM features are constant across all images."
        )

    print("\nAll validation checks passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate extracted handcrafted features."
    )
    parser.add_argument(
        "--method",
        choices=["rgb_histogram", "glcm"],
        required=True,
    )
    parser.add_argument("--bins", type=int, default=16)

    args = parser.parse_args()

    validate_features(method=args.method, bins=args.bins)
