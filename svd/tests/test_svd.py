"""Unit tests for the SVD pipeline. They use synthetic data and need no dataset."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.decomposition import PCA

from manifest import DEV_DATASET, TEST_DATASET, assign_splits, normalize_path, split_dev_set
from pixels import downsample
from svd_model import SvdModel, components_for_variance, fit_on_train_split, fit_svd, to_unit_float


# ---------- downsampling ----------

@pytest.mark.parametrize("size, length", [(32, 3_072), (64, 12_288)])
def test_downsample_shape_and_vector_length(size, length):
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8))
    small = downsample(image, size)
    assert small.shape == (size, size, 3)
    assert small.dtype == np.uint8
    assert small.reshape(-1).shape == (length,)


def test_downsample_box_averages_blocks():
    # 4x4 image made of four 2x2 solid blocks -> each output pixel is that block's color
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    pixels[:2, :2] = [10, 20, 30]
    pixels[:2, 2:] = [40, 50, 60]
    pixels[2:, :2] = [70, 80, 90]
    pixels[2:, 2:] = [100, 110, 120]
    small = downsample(Image.fromarray(pixels), 2)
    assert small.tolist() == [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]]


def test_downsample_converts_to_rgb():
    gray = Image.fromarray(np.full((8, 8), 200, dtype=np.uint8), mode="L")
    small = downsample(gray, 4)
    assert small.shape == (4, 4, 3)
    assert (small == 200).all()


# ---------- manifest and splits ----------

def test_normalize_path_handles_windows_paths():
    assert normalize_path(r"full\NCT-CRC-HE-100K\ADI\ADI-A.tif") == "full/NCT-CRC-HE-100K/ADI/ADI-A.tif"
    assert normalize_path("./sample_1pct/x.tif") == "sample_1pct/x.tif"


def _metadata(n_dev=50, n_test=10):
    labels = ["ADI", "TUM"]
    rows = [
        {"id": f"D{i}.tif", "path": f"full/{DEV_DATASET}/D{i}.tif", "label": labels[i % 2], "dataset": DEV_DATASET}
        for i in range(n_dev)
    ] + [
        {"id": f"T{i}.tif", "path": f"full/{TEST_DATASET}/T{i}.tif", "label": labels[i % 2], "dataset": TEST_DATASET}
        for i in range(n_test)
    ]
    return pd.DataFrame(rows)


def test_assign_splits_puts_every_image_in_exactly_one_split():
    metadata = _metadata()
    train, val = split_dev_set(metadata[metadata["dataset"] == DEV_DATASET])
    manifest = assign_splits(metadata, train, val)

    assert len(manifest) == len(metadata)
    assert manifest["split"].value_counts().to_dict() == {"train": 40, "val": 10, "test": 10}
    assert (manifest.loc[manifest["split"] == "test", "dataset"] == TEST_DATASET).all()
    assert (manifest.loc[manifest["split"] != "test", "dataset"] == DEV_DATASET).all()


def test_split_is_stratified():
    metadata = _metadata(n_dev=100)
    train, val = split_dev_set(metadata[metadata["dataset"] == DEV_DATASET])
    assert val["label"].value_counts().to_dict() == {"ADI": 10, "TUM": 10}


def test_assign_splits_rejects_overlap():
    metadata = _metadata()
    dev = metadata[metadata["dataset"] == DEV_DATASET]
    with pytest.raises(ValueError, match="both train and val"):
        assign_splits(metadata, dev.iloc[:30], dev.iloc[25:])


def test_assign_splits_rejects_missing_dev_images():
    metadata = _metadata()
    dev = metadata[metadata["dataset"] == DEV_DATASET]
    with pytest.raises(ValueError, match="neither train nor val"):
        assign_splits(metadata, dev.iloc[:30], dev.iloc[30:45])


def test_assign_splits_rejects_test_images_in_train():
    metadata = _metadata()
    dev = metadata[metadata["dataset"] == DEV_DATASET]
    test = metadata[metadata["dataset"] == TEST_DATASET]
    with pytest.raises(ValueError, match=f"non-{DEV_DATASET} images found in train"):
        assign_splits(metadata, pd.concat([dev.iloc[:40], test.iloc[:1]]), dev.iloc[40:])


# ---------- SVD model ----------

@pytest.fixture
def pixel_data():
    rng = np.random.default_rng(1)
    # low-rank structure plus noise, like correlated neighboring pixels
    basis = rng.normal(size=(5, 48))
    def make(n):
        values = 128 + rng.normal(size=(n, 5)) @ basis * 15 + rng.normal(size=(n, 48)) * 3
        return np.clip(values, 0, 255).astype(np.uint8)
    return make(300), make(80)


def test_fit_ignores_val_and_test_rows(pixel_data):
    train, held_out = pixel_data
    pixels = np.vstack([train, held_out])
    split = np.array(["train"] * len(train) + ["val"] * 40 + ["test"] * 40)

    model = fit_on_train_split(pixels, split, 10, seed=0)
    np.testing.assert_allclose(model.mean, to_unit_float(train).mean(axis=0), rtol=1e-5)
    assert model.n_train == len(train)

    # Replacing every val/test pixel must not change the fitted model at all.
    changed = pixels.copy()
    changed[len(train):] = 255 - changed[len(train):]
    other = fit_on_train_split(changed, split, 10, seed=0)
    np.testing.assert_array_equal(model.mean, other.mean)
    np.testing.assert_array_equal(model.components, other.components)


def test_fit_requires_training_rows(pixel_data):
    train, _ = pixel_data
    with pytest.raises(ValueError, match="No training rows"):
        fit_on_train_split(train, np.array(["val"] * len(train)), 5, seed=0)


def test_does_not_modify_input(pixel_data):
    train, _ = pixel_data
    before = train.copy()
    fit_svd(train, 10, seed=0)
    np.testing.assert_array_equal(train, before)


def test_transform_matches_sklearn_pca(pixel_data):
    train, val = pixel_data
    model = fit_svd(train, 10, seed=0)
    reference = PCA(n_components=10, svd_solver="full").fit(to_unit_float(train))

    # Only the 5 signal components are uniquely defined; the remaining noise
    # components have near-equal variance, so any rotation of them is valid.
    ours = model.transform(val)[:, :5]
    theirs = reference.transform(to_unit_float(val))[:, :5]
    signs = np.sign((ours * theirs).sum(axis=0))  # components are defined up to sign
    np.testing.assert_allclose(ours, theirs * signs, atol=1e-3)
    np.testing.assert_allclose(model.explained_variance_ratio, reference.explained_variance_ratio_, rtol=0.05)


def test_transform_batches_give_same_result(pixel_data):
    train, val = pixel_data
    model = fit_svd(train, 8, seed=0)
    np.testing.assert_allclose(model.transform(val, batch_size=7), model.transform(val), rtol=1e-6)


def test_full_rank_reconstruction_is_exact(pixel_data):
    train, _ = pixel_data
    model = fit_svd(train, 48, seed=0)
    rebuilt = model.inverse_transform(model.transform(train))
    np.testing.assert_allclose(rebuilt, to_unit_float(train), atol=1e-4)


def test_explained_variance_is_sorted_and_bounded(pixel_data):
    train, _ = pixel_data
    model = fit_svd(train, 20, seed=0)
    ratio = model.explained_variance_ratio
    assert (np.diff(ratio) <= 1e-7).all()
    assert 0 < ratio.sum() <= 1 + 1e-6


def test_rejects_impossible_component_counts(pixel_data):
    train, _ = pixel_data
    with pytest.raises(ValueError):
        fit_svd(train, 49, seed=0)
    with pytest.raises(ValueError):
        fit_svd(train, 0, seed=0)


def test_save_and_load_round_trip(pixel_data, tmp_path):
    train, val = pixel_data
    model = fit_svd(train, 10, seed=0)
    model.save(tmp_path / "model.npz")
    loaded = SvdModel.load(tmp_path / "model.npz")
    np.testing.assert_array_equal(loaded.transform(val), model.transform(val))
    assert loaded.n_train == model.n_train


def test_components_for_variance():
    ratio = np.array([0.5, 0.3, 0.1, 0.05])
    assert components_for_variance(ratio, [0.5, 0.8, 0.9, 0.99]) == {0.5: 1, 0.8: 2, 0.9: 3, 0.99: None}
