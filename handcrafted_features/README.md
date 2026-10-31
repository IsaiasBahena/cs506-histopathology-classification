
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

Additional texture and stain-based feature extractors will be added in subsequent development.

## Directory Structure

```text
handcrafted_features/
├── README.md
├── requirements.txt
├── extract_features.py
├── rgb_histogram.py
├── validate_features.py
└── outputs/
```

- `extract_features.py`: Reads metadata, processes images in batches, and saves feature matrices.
- `rgb_histogram.py`: Calculates normalized RGB histograms for individual images.
- `validate_features.py`: Validates the generated 16-bin RGB histogram CSV.
- `outputs/`: Stores generated feature matrices and extraction progress files. This directory is excluded from Git.

## Installation

From the repository root, install the required dependencies:

```bash
pip install -r handcrafted_features/requirements.txt
```

## Extracting Features

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

### Checkpointing

Features are saved after each batch. If extraction is interrupted, rerunning the same command attempts to resume from the last saved checkpoint.

The script checks the existing output and progress metadata before resuming. An interruption during CSV writing may require manual recovery.

## Output Format

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

## Validation

To validate the saved RGB histogram features, run:

```bash
python handcrafted_features/validate_features.py
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
| Validation | All checks passed |

These measurements were obtained using 16 bins per RGB channel and a batch size of 1,000 images.

## Modeling Considerations

Feature extraction is performed independently for each image and does not use tissue labels to calculate features.

The NCT-CRC-HE-100K dataset is used for model development, while CRC-VAL-HE-7K is reserved for independent final evaluation.

Any learned preprocessing transformations, including feature standardization, must be fitted only on training data and then applied to validation and test data.

The project's current internal training/validation split is image-level rather than patient-level because patient identifiers are not available in the training metadata. This may produce optimistic internal validation estimates and should be considered when interpreting results.

## Planned Features

Future implementations may include:

- Gray-Level Co-occurrence Matrix (GLCM) texture features
- Local Binary Patterns (LBP)
- Multi-scale and multi-orientation Gabor features
- Hematoxylin and eosin stain-specific features

These representations will be evaluated individually and in combination using consistent data splits and downstream models.
