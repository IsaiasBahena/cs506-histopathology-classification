
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

FEATURE_FILES = {
    "rgb_histogram": "rgb_histogram_16.csv",
    "glcm": "glcm.csv",
    "lbp": "lbp.csv",
    "gabor": "gabor.csv",
}

METADATA_COLUMNS = ["id", "path", "label", "dataset"]

EXPECTED_ROWS = 107180
EXPECTED_FEATURES = 162


def validate_combined():
    combined_path = OUTPUT_DIR / "combined_features.csv"

    print(f"Validating: {combined_path}")

    combined = pd.read_csv(combined_path)

    feature_columns = [
        column
        for column in combined.columns
        if column not in METADATA_COLUMNS
    ]

    print(f"\nDataset shape: {combined.shape}")
    print(f"Feature count: {len(feature_columns)}")

    assert combined.shape == (
        EXPECTED_ROWS,
        len(METADATA_COLUMNS) + EXPECTED_FEATURES,
    ), "Unexpected combined dataset shape."

    assert combined.columns.is_unique, "Duplicate column names."
    assert not combined["path"].duplicated().any(), (
        "Duplicate image paths."
    )

    assert not combined.isna().any().any(), (
        "Missing values in combined dataset."
    )

    assert np.isfinite(
        combined[feature_columns].to_numpy()
    ).all(), "Non-finite feature values."

    print("\nDataset distribution:")
    print(combined["dataset"].value_counts())

    print("\nClass distribution:")
    print(pd.crosstab(combined["dataset"], combined["label"]))

    print("\nChecking individual feature families...")

    for method, filename in FEATURE_FILES.items():
        source = pd.read_csv(OUTPUT_DIR / filename)

        source_features = [
            column
            for column in source.columns
            if column not in METADATA_COLUMNS
        ]

        assert set(source_features).issubset(combined.columns), (
            f"{method}: missing feature columns."
        )

        merged = combined[
            ["path"] + source_features
        ].merge(
            source[["path"] + source_features],
            on="path",
            suffixes=("_combined", "_source"),
            validate="one_to_one",
        )

        assert len(merged) == EXPECTED_ROWS, (
            f"{method}: image mismatch."
        )

        combined_values = merged[
            [f"{column}_combined" for column in source_features]
        ].to_numpy()

        source_values = merged[
            [f"{column}_source" for column in source_features]
        ].to_numpy()

        assert np.allclose(
            combined_values,
            source_values,
            rtol=1e-6,
            atol=1e-8,
        ), f"{method}: feature values differ."

        print(
            f"  {method}: {len(source_features)} features verified"
        )

    print("\nAll combined feature validation checks passed!")


if __name__ == "__main__":
    validate_combined()
