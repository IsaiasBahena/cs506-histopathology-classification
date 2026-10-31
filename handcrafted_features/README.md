
# Handcrafted Feature Extraction

This directory contains scripts for extracting handcrafted image features from colorectal histopathology images.

The goal is to represent histopathology images using classical computer vision features and evaluate their effectiveness for supervised classification and unsupervised clustering.

## Datasets

Feature extraction uses two datasets:

| Dataset | Images | Purpose |
|---|---:|---|
| NCT-CRC-HE-100K | 100,000 | Model development |
| CRC-VAL-HE-7K | 7,180 | Independent final evaluation |

All images are 224 × 224 RGB histopathology patches belonging to nine tissue classes: ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, and TUM.

Image paths and labels are obtained from `data/full/image_metadata.csv`.

## Implemented Features

### RGB Color Histograms

RGB histograms capture the distribution of pixel intensities in each color channel.

The current configuration uses:

- 16 bins per channel
- 3 channels (red, green, blue)
- 48 features per image
- Independent normalization of each channel histogram so its values sum to 1

RGB histograms describe color distributions but do not capture spatial relationships between pixels.

### Gray-Level Co-occurrence Matrix (GLCM)

GLCM features capture spatial relationships between neighboring grayscale pixel intensities. Unlike RGB histograms, which describe color distributions, GLCM measures texture characteristics such as contrast, uniformity, and intensity dependence.

The current configuration uses:

- **Input:** Original 224 × 224 images converted to grayscale
- **Quantization:** 32 gray levels
- **Distances:** 1, 2, and 4 pixels
- **Angles:** 0°, 45°, 90°, and 135°
- **Symmetric GLCM:** Enabled
- **GLCM normalization:** Enabled
- **Properties:** Contrast, dissimilarity, homogeneity, energy, correlation, and angular second moment (ASM)
- **Aggregation:** Mean and standard deviation across the four angles

For each image, 12 GLCMs are constructed using three distances and four angles.

Each of the six texture properties is summarized using its mean and standard deviation across angles for each distance.

The resulting feature vector contains:

3 distances × 6 properties × 2 statistics = **36 features per image**

Energy and ASM are mathematically related, but both are retained in the initial representation for comparison.

### Combined Feature Representations

The currently implemented feature extractors produce:

| Feature Family | Features per Image |
|---|---:|
| RGB histograms | 48 |
| GLCM texture | 36 |
| **Total** | **84** |

These features are currently stored in separate CSV files. They can be combined using image identifiers and dataset information to create an 84-dimensional feature representation.

The combined feature matrix has not yet been generated or validated.

## Directory Structure

```text
handcrafted_features/
├── README.md
├── requirements.txt
├── extract_features.py
├── rgb_histogram.py
├── glcm.py
├── validate_features.py
└── outputs/
```

- `extract_features.py`: Reads metadata, processes images in batches, and saves feature matrices.
- `rgb_histogram.py`: Calculates normalized RGB histograms for individual images.
- `glcm.py`: Computes GLCM texture statistics and generates feature names.
- `validate_features.py`: Validates extracted handcrafted feature matrices, including RGB histograms and GLCM texture features.
- `outputs/`: Stores generated feature matrices and extraction progress files. This directory is excluded from Git.

## Installation

From the repository root, install the required dependencies:

```bash
pip install -r handcrafted_features/requirements.txt
```

## Extracting Features

### RGB Histogram Feature Extraction

Run the RGB histogram extractor from the repository root:

```bash
python handcrafted_features/extract_features.py --method rgb_histogram --bins 16 --batch-size 1000
```

The script processes all images listed in the metadata and writes the results to:

```text
handcrafted_features/outputs/rgb_histogram_16.csv
```

It also creates:

```text
handcrafted_features/outputs/rgb_histogram_16.progress.json
```

The progress file records the extraction configuration, completed image count, and elapsed time.

### GLCM Feature Extraction

To extract GLCM texture features from all images, run:

```bash
python handcrafted_features/extract_features.py --method glcm --batch-size 1000
```

The script saves the extracted features to:

```text
handcrafted_features/outputs/glcm.csv
```

Extraction progress is recorded in:

```text
handcrafted_features/outputs/glcm.progress.json
```

The output contains 107,180 rows and 40 columns: four metadata columns and 36 GLCM features.

### Checkpointing

Features are saved after each batch. If extraction is interrupted, rerunning the same command attempts to resume from the last saved checkpoint.

The script checks the existing output and progress metadata before resuming. An interruption during CSV writing may require manual recovery.

## Output Format

### RGB Histogram Output

The RGB histogram CSV contains 107,180 rows and 52 columns:

| Columns | Description |
|---|---|
| `id` | Image identifier |
| `path` | Relative image path |
| `label` | Ground-truth tissue class |
| `dataset` | Source dataset |
| `r_hist_000` through `r_hist_015` | Red-channel histogram |
| `g_hist_000` through `g_hist_015` | Green-channel histogram |
| `b_hist_000` through `b_hist_015` | Blue-channel histogram |

Each row represents one image.

The extracted features can be joined with the project's training and validation metadata using image identifiers and dataset membership.

### GLCM Output

The GLCM CSV contains the same four metadata columns as the RGB histogram output:

- `id`
- `path`
- `label`
- `dataset`

The remaining 36 columns contain GLCM texture statistics.

Column names follow the pattern:

```text
glcm_d{distance}_{property}_{statistic}
```

Examples:

```text
glcm_d1_contrast_mean
glcm_d1_contrast_std
glcm_d2_homogeneity_mean
glcm_d2_homogeneity_std
glcm_d4_correlation_mean
glcm_d4_correlation_std
```

Each image is represented by one row.

## Validation

### RGB Histogram Validation

To validate the saved RGB histogram features, run:

```bash
python handcrafted_features/validate_features.py --method rgb_histogram
```

The validation checks include:

- Expected number of images and columns
- Dataset and class distributions
- Missing values
- Non-finite feature values
- Duplicate image paths
- Nonnegative histogram values
- Per-channel histogram normalization

### Completed RGB Histogram Extraction

| Metric | Result |
|---|---:|
| Images processed | 107,180 |
| Features per image | 48 |
| Output columns | 52 |
| Extraction time | 25.89 minutes |
| Average processing rate | 69 images/second |
| Missing values | 0 |
| Non-finite features | 0 |
| Duplicate image paths | 0 |
| Validation | All checks passed |

These measurements were obtained using 16 bins per RGB channel and a batch size of 1,000 images.

All extracted RGB histogram features satisfied the expected numerical constraints, including nonnegative values and normalization of each color channel.

### GLCM Validation

Run the GLCM validation script from the repository root:

```bash
python handcrafted_features/validate_features.py --method glcm
```

The validation checks include:

- Expected dataset dimensions and class distributions
- Missing values and non-finite feature values
- Duplicate image paths
- Valid numerical ranges for GLCM properties
- Constant features across the dataset

### Completed GLCM Extraction

| Metric | Result |
|---|---:|
| Images processed | 107,180 |
| Features per image | 36 |
| Output columns | 40 |
| Extraction time | 25.93 minutes |
| Average processing rate | 68.89 images/second |
| Missing values | 0 |
| Non-finite features | 0 |
| Duplicate image paths | 0 |
| Constant features | 0 |
| Validation | All checks passed |

The observed GLCM feature ranges were:

| Property | Minimum | Maximum |
|---|---:|---:|
| Contrast | 0.000856 | 100.785520 |
| Dissimilarity | 0.000279 | 7.758544 |
| Homogeneity | 0.000055 | 0.998579 |
| Energy | 0.000007 | 0.994302 |
| Correlation | -0.074638 | 0.998739 |
| ASM | 0.000001 | 0.988637 |

These ranges include both the mean and standard deviation features for each property.

All extracted GLCM features satisfied their expected numerical constraints.

The validation also confirmed that none of the 36 GLCM feature columns were constant across the dataset.

These checks confirm the structural and numerical validity of the feature matrices, but classification experiments are still required to evaluate their predictive usefulness.

## Modeling Considerations

Feature extraction is performed independently for each image and does not use tissue labels to calculate features.

The NCT-CRC-HE-100K dataset is used for model development, while CRC-VAL-HE-7K is reserved for independent final evaluation.

Any learned preprocessing transformations, including feature standardization, must be fitted only on training data and then applied to validation and test data.

The project's current internal training/validation split is image-level rather than patient-level because patient identifiers are not available in the training metadata. This may produce optimistic internal validation estimates and should be considered when interpreting results.

Different feature families may have substantially different numerical scales. Standardization should therefore be considered when combining feature representations, particularly for distance-based classifiers and clustering algorithms.

## Planned Features

Future implementations may include:

- Local Binary Patterns (LBP)
- Multi-scale and multi-orientation Gabor features
- Hematoxylin and eosin stain-specific features

These representations will be evaluated individually and in combination using consistent data splits and downstream models.
