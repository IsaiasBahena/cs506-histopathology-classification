
# Image Quality Validation

This directory contains the image-quality validation pipeline for the **Classical Machine Learning for Colorectal Histopathology Image Classification** project.

The pipeline verifies dataset integrity, identifies technical image-quality problems, detects duplicate images, and validates metadata consistency before feature extraction and model training.

## Directory Structure

```text
image_validation/
├── validate_images.py
├── requirements.txt
├── README.md
└── reports/
    ├── image_quality_summary.json
    └── image_quality_issues.csv
```

## Datasets

The pipeline validates two colorectal histopathology image datasets:

| Dataset | Images |
|---|---:|
| NCT-CRC-HE-100K | 100,000 |
| CRC-VAL-HE-7K | 7,180 |
| **Total** | **107,180** |

Both datasets contain nine tissue categories:

- **ADI:** Adipose tissue
- **BACK:** Background
- **DEB:** Debris
- **LYM:** Lymphocytes
- **MUC:** Mucus
- **MUS:** Smooth muscle
- **NORM:** Normal colon mucosa
- **STR:** Cancer-associated stroma
- **TUM:** Colorectal adenocarcinoma epithelium

All images are expected to have dimensions of 224 × 224 pixels and use RGB color mode.

## Installation

The pipeline requires Python and the following packages:

- pandas
- Pillow
- tqdm

Install the dependencies from the repository root:

```bash
python -m pip install -r image_validation/requirements.txt
```

The standard-library modules `pathlib`, `datetime`, `hashlib`, and `json` do not require separate installation.

## Usage

The pipeline expects the metadata file at:

```text
data/full/image_metadata.csv
```

Image paths are resolved relative to the repository's `data/` directory.

From the repository root, run:

```bash
python image_validation/validate_images.py
```

The script automatically performs all validation checks, displays a summary in the terminal, and generates JSON and CSV reports.

## Validation Methodology

The pipeline performs nine categories of validation.

### 1. Metadata Consistency

Verifies that:

- Tissue labels belong to the nine expected categories.
- Dataset names match the two supported datasets.
- Image paths follow the expected directory structure.
- Filenames agree with metadata IDs.
- Filename prefixes agree with tissue labels.

### 2. Missing Metadata

Checks for missing values in the following required columns:

`id`, `path`, `label`, `dataset`, `width`, `height`, and `color_mode`.

The pipeline also verifies that all required columns exist.

### 3. Duplicate Metadata

Detects:

- Duplicate image paths.
- Duplicate image IDs within the same dataset.

### 4. Image Readability

Uses Pillow to open and fully decode each image.

Images that cannot be successfully decoded are flagged.

### 5. File Access

Identifies file-access errors encountered when reading images for file-level hashing.

### 6. Image Dimensions

Verifies that each image:

- Has dimensions of 224 × 224 pixels.
- Matches the dimensions recorded in its metadata.

### 7. Color Mode

Verifies that each image:

- Uses RGB color mode.
- Matches the color mode recorded in its metadata.

### 8. Exact File Duplicates

Calculates SHA-256 hashes of the original image-file bytes.

Images with matching hashes are flagged as exact duplicates.

### 9. Pixel-Identical Duplicates

Calculates SHA-256 hashes using decoded pixel values, image dimensions, and color mode.

This detects images with identical decoded content even when their underlying file representations differ.

## Validation Results

The complete validation pipeline was executed on all 107,180 images.

| Validation Check | Issues Detected |
|---|---:|
| Metadata consistency | 0 |
| Missing metadata values | 0 |
| Duplicate metadata records | 0 |
| Unreadable images | 0 |
| File access errors | 0 |
| Invalid dimensions | 0 |
| Invalid color modes | 0 |
| Exact file duplicates | 0 |
| Pixel-identical duplicates | 0 |

**All 107,180 images passed the implemented validation checks.**

## Generated Reports

The pipeline automatically generates two files inside `image_validation/reports/`.

### `image_quality_summary.json`

Contains:

- UTC validation timestamp.
- Total number of images.
- Number of images per dataset.
- Number of detected issues in each validation category.

### `image_quality_issues.csv`

Contains three columns:

| Column | Description |
|---|---|
| `issue_type` | Validation category |
| `path` | Image path associated with the issue |
| `details` | Additional information about the detected problem |

If no issues are detected, the CSV contains only the column headers.

Both reports are overwritten whenever the pipeline runs.

## Data Cleaning Decisions

No images were removed or modified because no technical or structural issues were detected by the implemented validation checks.

The original datasets were preserved unchanged.

If future validation runs identify problems, affected images should be flagged and reviewed before deciding whether to exclude or correct them.

## Limitations

The pipeline validates technical integrity and metadata consistency, but it does not:

- Verify that tissue labels are biologically correct.
- Detect near-duplicate images with similar but nonidentical pixel values.
- Assess staining quality, blur, or other subjective visual-quality characteristics.
- Establish patient-level independence or rule out patient-level data leakage.
- Independently inventory image files that are absent from the metadata.

These limitations should be considered when interpreting the validation results and conducting subsequent classification experiments.
