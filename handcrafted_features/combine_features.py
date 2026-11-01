
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

EXPECTED_FEATURE_COUNTS = {
    "rgb_histogram": 48,
    "glcm": 36,
    "lbp": 54,
    "gabor": 24,
}

EXPECTED_ROWS = 107180


def combine_features():
    combined = None

    for method, filename in FEATURE_FILES.items():
        path = OUTPUT_DIR / filename

        print(f"\nLoading {method}: {path}")

        df = pd.read_csv(path)

        feature_columns = [
            column
            for column in df.columns
            if column not in METADATA_COLUMNS
        ]

        assert len(df) == EXPECTED_ROWS, (
            f"{method}: unexpected row count."
        )

        assert len(feature_columns) == EXPECTED_FEATURE_COUNTS[method], (
            f"{method}: unexpected feature count."
        )

        assert not df["path"].duplicated().any(), (
            f"{method}: duplicate image paths."
        )

        assert not df[METADATA_COLUMNS].isna().any().any(), (
            f"{method}: missing metadata."
        )

        assert not df[feature_columns].isna().any().any(), (
            f"{method}: missing feature values."
        )

        assert np.isfinite(
            df[feature_columns].to_numpy()
        ).all(), (
            f"{method}: non-finite feature values."
        )

        print(f"Rows: {len(df):,}")
        print(f"Features: {len(feature_columns)}")

        if combined is None:
            combined = df
            continue

        # Verify that every image has consistent metadata.
        metadata_check = combined[METADATA_COLUMNS].merge(
            df[METADATA_COLUMNS],
            on="path",
            how="outer",
            suffixes=("_combined", "_new"),
            indicator=True,
            validate="one_to_one",
        )

        assert (metadata_check["_merge"] == "both").all(), (
            f"{method}: image paths do not match."
        )

        for column in ["id", "label", "dataset"]:
            assert (
                metadata_check[f"{column}_combined"]
                == metadata_check[f"{column}_new"]
            ).all(), (
                f"{method}: inconsistent {column} values."
            )

        # Join features using image paths, not row positions.
        combined = combined.merge(
            df[["path"] + feature_columns],
            on="path",
            how="inner",
            validate="one_to_one",
        )

    expected_columns = (
        len(METADATA_COLUMNS)
        + sum(EXPECTED_FEATURE_COUNTS.values())
    )

    assert len(combined) == EXPECTED_ROWS
    assert len(combined.columns) == expected_columns
    assert combined.columns.is_unique
    assert not combined["path"].duplicated().any()

    feature_columns = [
        column
        for column in combined.columns
        if column not in METADATA_COLUMNS
    ]

    assert np.isfinite(
        combined[feature_columns].to_numpy()
    ).all()

    output_path = OUTPUT_DIR / "combined_features.csv"

    combined.to_csv(output_path, index=False)

    print("\nCombination complete.")
    print(f"Output: {output_path}")
    print(f"Rows: {len(combined):,}")
    print(f"Columns: {len(combined.columns)}")
    print(f"Features: {len(feature_columns)}")

    print("\nFeature groups:")
    for method, count in EXPECTED_FEATURE_COUNTS.items():
        print(f"  {method}: {count}")


if __name__ == "__main__":
    combine_features()
