"""Build the image manifest (one row per image, with its split) for the SVD pipeline.

Split policy (shared with the rest of the group):
- NCT-CRC-HE-100K is divided into "train" and "val" (80/20, stratified by label).
- CRC-VAL-HE-7K is the independent "test" set and is never used for fitting.

For the full dataset the split comes from Person 1's files, data/full/train.csv and
data/full/val.csv. The 1% sample has no split files, so the same split call
(same function, test size, seed, and stratification) is applied to the sample.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

TISSUE_CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
DEV_DATASET = "NCT-CRC-HE-100K"
TEST_DATASET = "CRC-VAL-HE-7K"
SPLITS = ["train", "val", "test"]

# Same parameters as data/create_train_val_split.py
SPLIT_TEST_SIZE = 0.2
SPLIT_SEED = 4266457447

SAMPLES = {
    "1pct": {"metadata": DATA_ROOT / "sample_1pct" / "image_metadata_1pct.csv"},
    "full": {
        "metadata": DATA_ROOT / "full" / "image_metadata.csv",
        "train": DATA_ROOT / "full" / "train.csv",
        "val": DATA_ROOT / "full" / "val.csv",
    },
}

MANIFEST_COLUMNS = ["id", "path", "label", "dataset", "split"]
KEY = ["dataset", "id"]


def normalize_path(path):
    """Return a forward-slash path relative to data/, whatever OS wrote it."""

    path = str(path).replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def _load_metadata(path):
    metadata = pd.read_csv(path)

    required = ["id", "path", "label", "dataset"]
    missing = set(required) - set(metadata.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if metadata[required].isna().any().any():
        raise ValueError(f"{path}: missing required values")
    if metadata.duplicated(subset=KEY).any():
        raise ValueError(f"{path}: duplicate (dataset, id) rows")

    unknown = set(metadata["label"]) - set(TISSUE_CLASSES)
    if unknown:
        raise ValueError(f"{path}: unknown labels {sorted(unknown)}")
    datasets = set(metadata["dataset"])
    if not datasets <= {DEV_DATASET, TEST_DATASET}:
        raise ValueError(f"{path}: unknown datasets {sorted(datasets)}")

    metadata = metadata[required].copy()
    metadata["path"] = metadata["path"].map(normalize_path)
    return metadata


def split_dev_set(dev_metadata):
    """Reproduce data/create_train_val_split.py on the given NCT rows."""

    train, val = train_test_split(
        dev_metadata,
        test_size=SPLIT_TEST_SIZE,
        random_state=SPLIT_SEED,
        stratify=dev_metadata["label"],
    )
    return train, val


def assign_splits(metadata, train, val):
    """Label each metadata row train/val/test and check the split is clean.

    `train` and `val` only need (dataset, id) columns. Every NCT row must be in
    exactly one of them, and no CRC-VAL-HE-7K row may appear in either.
    """

    train_keys = set(map(tuple, train[KEY].to_numpy()))
    val_keys = set(map(tuple, val[KEY].to_numpy()))

    if train_keys & val_keys:
        raise ValueError(f"{len(train_keys & val_keys)} images are in both train and val")
    for name, keys in [("train", train_keys), ("val", val_keys)]:
        leaked = {k for k in keys if k[0] != DEV_DATASET}
        if leaked:
            raise ValueError(f"{len(leaked)} non-{DEV_DATASET} images found in {name}")

    keys = list(map(tuple, metadata[KEY].to_numpy()))
    dev_keys = {k for k in keys if k[0] == DEV_DATASET}
    unassigned = dev_keys - train_keys - val_keys
    if unassigned:
        raise ValueError(f"{len(unassigned)} {DEV_DATASET} images are in neither train nor val")
    unknown = (train_keys | val_keys) - dev_keys
    if unknown:
        raise ValueError(f"{len(unknown)} split images are missing from the metadata")

    def split_of(key):
        if key[0] == TEST_DATASET:
            return "test"
        return "train" if key in train_keys else "val"

    manifest = metadata.copy()
    manifest["split"] = [split_of(k) for k in keys]
    manifest["split"] = pd.Categorical(manifest["split"], categories=SPLITS, ordered=True)
    manifest = manifest.sort_values(["split", "dataset", "label", "id"]).reset_index(drop=True)
    manifest["split"] = manifest["split"].astype(str)
    return manifest[MANIFEST_COLUMNS]


def build_manifest(sample):
    """Return the manifest for a sample name ("1pct" or "full")."""

    if sample not in SAMPLES:
        raise ValueError(f"Unknown sample {sample!r}, expected one of {sorted(SAMPLES)}")
    paths = SAMPLES[sample]

    metadata = _load_metadata(paths["metadata"])

    if "train" in paths:
        train = pd.read_csv(paths["train"])
        val = pd.read_csv(paths["val"])
    else:
        train, val = split_dev_set(metadata[metadata["dataset"] == DEV_DATASET])

    return assign_splits(metadata, train, val)
