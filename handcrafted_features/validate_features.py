from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "rgb_histogram_16.csv"
)

df = pd.read_csv(OUTPUT_PATH)

feature_columns = [
    f"{channel}_hist_{i:03d}"
    for channel in ("r", "g", "b")
    for i in range(16)
]

features = df[feature_columns].to_numpy()

print("Dataset shape:", df.shape)
print("\nDataset distribution:")
print(df["dataset"].value_counts())

print("\nClass distribution:")
print(pd.crosstab(df["dataset"], df["label"]))

print("\nMissing values:", df.isna().sum().sum())
print("Non-finite features:", (~np.isfinite(features)).sum())
print("Duplicate image paths:", df["path"].duplicated().sum())

channel_sums = features.reshape(-1, 3, 16).sum(axis=2)

print("\nHistogram normalization:")
print("Minimum channel sum:", channel_sums.min())
print("Maximum channel sum:", channel_sums.max())

assert df.shape == (107180, 52)
assert df["dataset"].value_counts().to_dict() == {
    "NCT-CRC-HE-100K": 100000,
    "CRC-VAL-HE-7K": 7180,
}
assert df["path"].is_unique
assert df.isna().sum().sum() == 0
assert np.all(np.isfinite(features))
assert np.all(features >= 0)
assert np.allclose(channel_sums, 1.0, atol=1e-5)

print("\nAll validation checks passed!")
