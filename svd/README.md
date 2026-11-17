# SVD Pixel Representation

This directory builds the pixel-based image representation for the project.
Each image is downsampled, flattened into a pixel vector, and reduced to a small number of features with randomized Singular Value Decomposition (SVD).
These features are compared against the handcrafted features in `handcrafted_features/` using the same classifiers and splits.

## Session setup

From the `svd/` directory:

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

The pipeline reads the data produced by `data/` (see `data/README.md`):

- `--sample 1pct` reads `data/sample_1pct/image_metadata_1pct.csv`.
- `--sample full` reads `data/full/image_metadata.csv`, `data/full/train.csv`, and `data/full/val.csv`.

## Usage

```
python run_svd.py --sample 1pct --size 32 --components 256
python benchmark.py --sample 1pct
python -m pytest
```

Swap `1pct` for `full` to run on the full dataset.
The first run decodes every image and caches the pixel matrix in `outputs/`, so later runs start in seconds.

## Splits and leakage

| Split | Source | Used for |
|---|---|---|
| `train` | NCT-CRC-HE-100K, 80% | Fitting the SVD (and later, the classifiers) |
| `val` | NCT-CRC-HE-100K, 20% | Choosing the number of components and model settings |
| `test` | CRC-VAL-HE-7K | Final evaluation only |

The SVD is fitted on `train` rows only.
`val` and `test` images are only ever transformed with the already-fitted model.
`tests/test_svd.py` checks this by replacing every `val`/`test` pixel and confirming the fitted model does not change.

For the full dataset, the split comes from Person 1's `data/full/train.csv` and `data/full/val.csv`.
The 1% sample has no split files, so the pipeline applies the same split call to the sample (same function, 20% validation, seed `4266457447`, stratified by label).
1% sample results are for development only and are not final results.

## Method

1. **Downsample.** Each 224x224 image is converted to RGB and resized to 32x32 (or 64x64) with box averaging, so every output pixel is the mean of a block of original pixels.
   This is the same downsampling as the October preprocessing check-in (`preprocessing/`), and produces identical pixels.
2. **Flatten.** Each image becomes a vector in (row, column, channel) order: 32 x 32 x 3 = 3,072 values (12,288 at 64x64).
3. **Scale and center.** Pixels are divided by 255, and the mean training image is subtracted.
4. **Randomized SVD.** scikit-learn's `PCA(svd_solver="randomized")` finds the top `k` directions of variation in the centered training matrix.
   On the 1% sample, the randomized result matches an exact SVD to within 0.1 percentage points of explained variance for every k up to 256.
5. **Project.** Each image's features are `(pixels / 255 - train_mean) @ components.T`.
   Features are not whitened: feature 1 has the largest variance and later features have progressively less.
   Classifiers that are sensitive to scale (logistic regression, SVM) should apply a `StandardScaler` fitted on `train` rows only.

Components are ordered by explained variance, so the first `k` columns of the saved features are the `k`-component representation.
There is no need to refit to try a smaller `k`.

## Outputs

Large files go to `outputs/<sample>/`, which is gitignored:

| File | Contents |
|---|---|
| `svd_32px_features.csv` | One row per image: `id, path, label, dataset, split, svd_000 ... svd_255` |
| `svd_32px_model.npz` | Fitted `mean`, `components`, `singular_values`, `explained_variance`, `explained_variance_ratio`, `n_train` |
| `pixels_32px.npz` | Cached uint8 pixel matrix `X` and the matching `key` (`dataset/id`) per row |

Small results go to `results/<sample>/`:

| File | Contents |
|---|---|
| `svd_32px_params.json` | Parameters, split sizes, variance thresholds, timings, and software versions |
| `explained_variance_32px.csv` | Per-component and cumulative explained variance |
| `explained_variance_32px.png` | Cumulative and per-component explained variance |
| `components_32px.png` | The mean training image and the leading components shown as images |
| `reconstructions_32px.png` | One validation image per class rebuilt from k = 8, 32, 128, 256 components |
| `class_scatter_32px.png` | Training images on components 1 and 2, one panel per class |
| `svd_benchmark.csv`, `svd_benchmark.png` | Fit time and peak memory by resolution and number of components |

### Joining with other feature sets

Join on `(dataset, id)`, not on `path` and not on row order.
`path` depends on the operating system that wrote it (the handcrafted feature CSVs were written on Windows and use `\`), so a join on `path` across machines silently matches nothing.
This pipeline writes `path` with forward slashes.

```python
import pandas as pd

svd = pd.read_csv("svd/outputs/full/svd_32px_features.csv")
k = 64
columns = ["dataset", "id", "label", "split"] + [f"svd_{i:03d}" for i in range(k)]
train = svd.loc[svd["split"] == "train", columns]
```

To apply the fitted model to new images without refitting:

```python
from svd_model import SvdModel

model = SvdModel.load("outputs/full/svd_32px_model.npz")
features = model.transform(pixels_uint8)  # (n_images, 3072) uint8 -> (n_images, 256)
```

## Results (full dataset, 32x32)

The SVD was fitted on the 80,000 `train` images.
Figures and tables are in `results/full/`.

### What the components capture

- The first component alone explains 48% of the pixel variance.
  It is nearly uniform across the image, so it measures overall brightness and stain intensity.
- The second and tenth components are also uniform across the image: they are color shifts (more or less pink, more or less blue).
- Nearly all other leading components are smooth wave patterns (stripes and checkerboards at different angles and frequencies).
  This is what SVD produces when image content has no fixed position: tissue structures can appear anywhere in a patch, so the best pixel-level basis looks like a Fourier basis.
- 80% of the variance is reached at k = 158, and 256 components capture 85%.
  An exact eigendecomposition of the training covariance shows 90% needs 421 components, 95% needs 683, and 99% needs 1,063.
  The randomized SVD matches the exact explained variance to within 0.05 percentage points at k = 256.
- Validation images rebuilt from 8 components keep only the average color.
  32 components add coarse layout, and 128 to 256 components recover recognizable structure such as glands in NORM and fat cell outlines in ADI.

### Class separation

In the first two components, ADI (bright) and most of BACK (dark) form separate groups at opposite ends of component 1.
The other seven classes overlap heavily, so classifiers need more than two components.

### Comparison with handcrafted features (preliminary)

This is a quick check, not the project's model evaluation (that belongs to the supervised classification workstream).
Every feature set uses the same classifier (`StandardScaler` + `LogisticRegression`, default settings), is fitted on the 80,000 `train` images, and is scored on the 20,000 `val` images.
The test set was not used.

| Features | # features | Val macro F1 |
|---|---:|---:|
| RGB histogram (baseline) | 48 | 0.782 |
| GLCM | 36 | 0.738 |
| LBP | 54 | 0.822 |
| Gabor | 24 | 0.643 |
| All handcrafted | 162 | 0.947 |
| SVD, k = 8 | 8 | 0.465 |
| SVD, k = 16 | 16 | 0.500 |
| SVD, k = 32 | 32 | 0.502 |
| SVD, k = 64 | 64 | 0.498 |
| SVD, k = 256 | 256 | 0.497 |
| All handcrafted + SVD, k = 32 | 194 | 0.947 |

- Pixel SVD features reach about 0.50 macro F1, well below the 0.78 color histogram baseline.
- Performance stops improving after k = 16.
  Components beyond the first few encode where texture happens to sit in a patch, which carries no information about the tissue class.
- Adding SVD features to the handcrafted features does not change macro F1.
- SVD does best on ADI (0.83 F1) and LYM (0.74), which differ from other tissue in overall brightness and color.
  It does worst on STR (0.15) and NORM (0.18), which the handcrafted features classify at 0.87 and 0.91.
- Handcrafted features summarize the whole patch regardless of position (histograms, texture statistics), which matches how tissue type is defined.

### Development run (1% sample)

The same pipeline on the 1% sample (796 training images) found the same pattern, with two differences caused by the small sample.
Variance looked more concentrated (80% at k = 88), and components beyond about 128 added speckle noise to validation reconstructions rather than detail.
Both effects disappear with 80,000 training images.

## Runtime and memory (full dataset)

Measured by `benchmark.py --sample full` on 80,000 training images (16 CPU cores, 13 GB RAM, WSL2).
Fit time is the median of 3 fits.
Full results are in `results/full/svd_benchmark.csv` and `results/full/svd_benchmark.png`.

| Resolution | k | Fit time | Peak memory during fit | Explained variance |
|---|---:|---:|---:|---:|
| 32x32 | 16 | 2.8 s | 1.01 GB | 61.8% |
| 32x32 | 64 | 3.2 s | 1.06 GB | 71.8% |
| 32x32 | 256 | 6.7 s | 1.24 GB | 84.8% |
| 64x64 | 16 | 8.9 s | 3.96 GB | 52.1% |
| 64x64 | 64 | 10.1 s | 4.01 GB | 60.8% |
| 64x64 | 256 | 19.6 s | 4.20 GB | 72.9% |

- Peak memory is dominated by the float32 training matrix (0.98 GB at 32x32, 3.93 GB at 64x64).
  The pipeline keeps pixels as uint8 and only converts the training rows to float32, which PCA then centers in place, so no second full-size copy is made.
- Transforming the 20,000 validation images takes under 0.5 s at either resolution, in batches of 10,000.
- Decoding and downsampling all 107,180 images takes a few minutes on 16 processes and is cached after the first run.
- 64x64 needs four times the memory and three times the fit time, and captures less of its (larger) variance at the same k.
  Since the classification comparison shows pixel SVD is limited by position dependence rather than resolution, 32x32 remains the default.
- Randomized SVD is not the bottleneck at this scale: even 64x64 with 256 components fits in 20 seconds.
  Memory is the binding constraint: the 64x64 benchmark process peaked at 7.2 GB of RAM including the cached pixel matrices, which is too tight for an 8 GB machine.
