
# Handcrafted Feature Extraction

This directory contains scripts for extracting, combining, validating, and analyzing handcrafted image features from colorectal histopathology images.

The goal is to represent histopathology images using classical computer vision features and evaluate their effectiveness for supervised classification and unsupervised clustering.

## Datasets

Feature extraction uses two datasets:

| Dataset | Images | Purpose |
|---|---:|---|
| NCT-CRC-HE-100K | 100,000 | Model development |
| CRC-VAL-HE-7K | 7,180 | Independent final evaluation |
| **Total** | **107,180** | |

All images are 224 × 224 RGB histopathology patches belonging to nine tissue classes: ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, and TUM.

Image paths and labels are obtained from `data/full/image_metadata.csv`.

## Implemented Features

### RGB Color Histograms

RGB histograms capture the distribution of pixel intensities in each color channel.

The current configuration uses:

- **Bins:** 16 per channel
- **Channels:** Red, green, and blue
- **Features:** 48 per image
- **Normalization:** Each channel histogram independently normalized to sum to 1

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

### Local Binary Patterns (LBP)

Local Binary Patterns describe local image texture by comparing the intensity of each pixel with the intensities of neighboring pixels.

For each center pixel, neighboring intensities are compared against the center intensity. Neighbors with intensities greater than or equal to the center receive a binary value of 1, while neighbors with lower intensities receive 0.

The resulting binary pattern characterizes the local intensity structure surrounding the pixel.

The current implementation uses multi-scale, rotation-invariant uniform LBP.

**Configuration:**

- **Input:** Original 224 × 224 images converted to grayscale
- **Method:** Uniform LBP (`uniform`)
- **Neighborhoods:** Circular neighborhoods
- **Scales:** Three combinations of neighboring points and radii
- **Aggregation:** Normalized histograms of LBP codes
- **Normalization:** Each scale's histogram independently normalized to sum to 1

The three scales are:

| Neighbors (P) | Radius (R) | Histogram Bins |
|---:|---:|---:|
| 8 | 1 | 10 |
| 16 | 2 | 18 |
| 24 | 3 | 26 |
| **Total** | | **54** |

The number of histogram bins for uniform LBP is `P + 2`.

Uniform LBP groups local binary patterns according to their structure, with patterns containing more than two circular binary transitions grouped into a single nonuniform category.

The `uniform` method is rotation-invariant, allowing equivalent local patterns under rotation to share the same representation.

For each scale:

1. Compute an LBP code for every pixel in the grayscale image.
2. Construct a histogram counting the frequency of each LBP code.
3. Normalize the histogram so its values sum to 1.
4. Append the normalized histogram to the feature vector.

The three histograms are concatenated into a single feature vector:

10 + 18 + 26 = **54 features per image**

Using multiple radii allows the representation to capture local texture patterns at different spatial scales.

Unlike GLCM, which summarizes statistical relationships between pairs of grayscale intensities, LBP describes local arrangements of relative pixel intensities.

LBP features are computed independently of tissue labels and do not explicitly identify biological structures.

### Gabor Texture Features

Gabor filters capture image texture by measuring responses to patterns at specific spatial frequencies and orientations.

A Gabor filter combines a sinusoidal wave with a Gaussian envelope. The sinusoidal component detects spatial intensity variations at a particular frequency and orientation, while the Gaussian envelope localizes the filter response.

Unlike GLCM and LBP, which characterize intensity relationships and local binary patterns, Gabor filters measure the strength and spatial variation of oriented texture responses.

The current implementation uses a bank of Gabor filters with multiple frequencies and orientations.

**Configuration:**

- **Input:** Original 224 × 224 images converted to grayscale
- **Intensity normalization:** Grayscale intensities converted to floating-point values and divided by 255
- **Frequencies:** 0.05, 0.10, and 0.20 cycles/pixel
- **Orientations:** 0°, 45°, 90°, and 135°
- **Filter type:** Complex-valued Gabor kernels
- **Convolution:** FFT-based convolution using `scipy.signal.fftconvolve`
- **Boundary handling:** Zero-padding
- **Output:** Magnitude of the complex filter response
- **Aggregation:** Mean and standard deviation of the magnitude response

The filter bank contains:

3 frequencies × 4 orientations = **12 Gabor filters**

For each filter, the implementation computes the real and imaginary response maps using FFT-based convolution.

The response magnitude is calculated as:

`magnitude = sqrt(real^2 + imaginary^2)`

Two statistics are then extracted from the magnitude response:

- **Mean:** Average magnitude across all image pixels
- **Standard deviation:** Spatial variation in the magnitude response

The resulting feature vector contains:

3 frequencies × 4 orientations × 2 statistics = **24 features per image**

#### FFT-Based Implementation

The implementation uses `skimage.filters.gabor_kernel` to construct the Gabor kernels and `scipy.signal.fftconvolve` to perform convolution.

All 12 Gabor kernels are generated once when the module is imported and reused for subsequent images.

For each image:

1. Convert the RGB image to grayscale.
2. Convert grayscale intensities to floating-point values in the range [0, 1].
3. Apply each Gabor kernel using FFT-based convolution.
4. Compute the magnitude of the complex response.
5. Calculate the mean and standard deviation of the magnitude.
6. Concatenate the resulting statistics into a 24-dimensional feature vector.

The FFT implementation was adopted because the initial spatial-convolution implementation was computationally expensive for the full dataset, particularly at lower frequencies.

The original implementation had an estimated runtime of approximately 19 hours, while the completed FFT-based extraction required 209.56 minutes.

The FFT implementation uses zero-padding at image boundaries, whereas the original `skimage.filters.gabor` implementation used reflective boundary handling by default.

Therefore, the FFT-based features are not necessarily numerically identical to the original implementation. All saved Gabor features were extracted using the FFT-based implementation with consistent boundary handling.

The Gaussian envelope parameters use the defaults provided by `skimage.filters.gabor_kernel`, including its default bandwidth.

### Combined Feature Representations

The implemented feature extractors produce:

| Feature Family | Features per Image |
|---|---:|
| RGB histograms | 48 |
| GLCM texture | 36 |
| LBP texture | 54 |
| Gabor texture | 24 |
| **Total** | **162** |

Each feature family is stored in a separate CSV file.

The four feature matrices have also been combined into a single dataset containing all 162 handcrafted features per image.

The combined dataset contains:

- 107,180 images
- 4 metadata columns
- 162 feature columns
- 166 columns in total

The combination uses image paths to align feature vectors across the four source datasets.

The individual feature files are retained so that each representation can be evaluated independently.

The combined dataset supports experiments with individual feature families and selected combinations without requiring repeated feature extraction.

The combined feature matrix has been generated and independently validated.

## Directory Structure

```text
handcrafted_features/
├── README.md
├── requirements.txt
├── extract_features.py
├── rgb_histogram.py
├── glcm.py
├── lbp.py
├── gabor.py
├── combine_features.py
├── validate_features.py
├── validate_combined.py
├── visualize_features.py
├── figures/
│   ├── class_distributions/
│   │   ├── rgb_class_distributions.png
│   │   ├── glcm_class_distributions.png
│   │   ├── lbp_class_distributions.png
│   │   └── gabor_class_distributions.png
│   ├── correlations/
│   │   ├── rgb_correlation.png
│   │   ├── glcm_correlation.png
│   │   ├── lbp_correlation.png
│   │   ├── gabor_correlation.png
│   │   └── highly_correlated_features.csv
│   └── feature_analysis/
│       ├── feature_anova_scores.csv
│       └── top_features_anova.png
└── outputs/
    ├── rgb_histogram_16.csv
    ├── rgb_histogram_16.progress.json
    ├── glcm.csv
    ├── glcm.progress.json
    ├── lbp.csv
    ├── lbp.progress.json
    ├── gabor.csv
    ├── gabor.progress.json
    └── combined_features.csv
```

### Script Descriptions

- `extract_features.py`: Reads metadata, processes images in batches, and saves feature matrices.
- `rgb_histogram.py`: Calculates normalized RGB histograms for individual images.
- `glcm.py`: Computes GLCM texture statistics and generates feature names.
- `lbp.py`: Computes multi-scale uniform LBP histograms and generates feature names.
- `gabor.py`: Computes multi-frequency, multi-orientation Gabor texture features using FFT-based convolution.
- `combine_features.py`: Combines the four extracted feature matrices into a single dataset, aligning images by path and verifying metadata consistency.
- `validate_features.py`: Validates extracted RGB histogram, GLCM, LBP, and Gabor feature matrices.
- `validate_combined.py`: Independently validates the combined feature matrix against the four original source files.
- `visualize_features.py`: Generates class-wise feature distributions, correlation heatmaps, correlation summaries, and ANOVA feature rankings.
- `figures/`: Stores generated exploratory analysis figures and summary CSV files.
- `outputs/`: Stores generated feature matrices and extraction progress files. This directory is excluded from Git.

## Installation

From the repository root, install the required dependencies:

```bash
pip install -r handcrafted_features/requirements.txt
```

The implementation uses NumPy, pandas, Pillow, scikit-image, SciPy, Matplotlib, Seaborn, and scikit-learn.

The Gabor implementation requires SciPy for FFT-based convolution and scikit-image for Gabor kernel generation.

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

### LBP Feature Extraction

To extract multi-scale uniform LBP features from all images, run:

```bash
python handcrafted_features/extract_features.py --method lbp --batch-size 1000
```

The script saves the extracted features to:

```text
handcrafted_features/outputs/lbp.csv
```

Extraction progress is recorded in:

```text
handcrafted_features/outputs/lbp.progress.json
```

The output contains 107,180 rows and 58 columns: four metadata columns and 54 LBP features.

Each image is represented by three normalized LBP histograms corresponding to the configured neighborhood scales.

### Gabor Feature Extraction

To extract multi-frequency, multi-orientation Gabor features from all images, run:

```bash
python handcrafted_features/extract_features.py --method gabor --batch-size 1000
```

The script saves the extracted features to:

```text
handcrafted_features/outputs/gabor.csv
```

Extraction progress is recorded in:

```text
handcrafted_features/outputs/gabor.progress.json
```

The output contains 107,180 rows and 28 columns: four metadata columns and 24 Gabor features.

Each image is represented by the mean and standard deviation of the magnitude responses from 12 Gabor filters.

### Combining Feature Matrices

After extracting all four feature families, run the combination script from the repository root:

```bash
python handcrafted_features/combine_features.py
```

The script reads:

```text
handcrafted_features/outputs/rgb_histogram_16.csv
handcrafted_features/outputs/glcm.csv
handcrafted_features/outputs/lbp.csv
handcrafted_features/outputs/gabor.csv
```

The combined dataset is saved to:

```text
handcrafted_features/outputs/combined_features.csv
```

The output contains 107,180 rows and 166 columns: four metadata columns and 162 handcrafted features.

The script performs the following checks:

- Expected number of images in each source file
- Expected number of features for each feature family
- Missing metadata and feature values
- Non-finite feature values
- Duplicate image paths
- Consistent image identifiers across source files
- Consistent tissue labels and dataset membership
- Unique column names in the combined dataset
- Expected final dataset dimensions

Feature matrices are joined using image paths rather than row positions.

This ensures that feature vectors from different extractors correspond to the same image even if the source CSV files have different row orders.

The four original CSV files are retained.

No feature scaling, normalization, or dimensionality reduction is performed during combination.

These transformations should be handled by downstream modeling pipelines to prevent data leakage.

**Completed combination:**

| Metric | Result |
|---|---:|
| Images combined | 107,180 |
| RGB histogram features | 48 |
| GLCM features | 36 |
| LBP features | 54 |
| Gabor features | 24 |
| Total features | 162 |
| Metadata columns | 4 |
| Total columns | 166 |
| Combination | All checks passed |

### Checkpointing

Features are saved after each batch. If extraction is interrupted, rerunning the same command attempts to resume from the last saved checkpoint.

The script checks the existing output and progress metadata before resuming. An interruption during CSV writing may require manual recovery.

If the feature extraction implementation or its configuration changes, existing output files should not be mixed with results from the new implementation.

The combination script operates on completed feature CSV files and does not require the original image files.

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

### LBP Output

The LBP CSV contains 107,180 rows and 58 columns.

The first four columns contain image metadata:

- `id`
- `path`
- `label`
- `dataset`

The remaining 54 columns contain normalized LBP histogram values.

Column names follow the pattern:

```text
lbp_p{neighbors}_r{radius}_hist_{bin}
```

Examples:

```text
lbp_p8_r1_hist_000
lbp_p8_r1_hist_001
...
lbp_p8_r1_hist_009

lbp_p16_r2_hist_000
...
lbp_p16_r2_hist_017

lbp_p24_r3_hist_000
...
lbp_p24_r3_hist_025
```

The features are organized into three consecutive groups:

| Feature Group | Number of Columns | Histogram Bin Indices |
|---|---:|---|
| P=8, R=1 | 10 | 0–9 |
| P=16, R=2 | 18 | 0–17 |
| P=24, R=3 | 26 | 0–25 |

Each group is independently normalized so that its histogram values sum to approximately 1.

Each image is represented by one row.

### Gabor Output

The Gabor CSV contains 107,180 rows and 28 columns.

The first four columns contain image metadata:

- `id`
- `path`
- `label`
- `dataset`

The remaining 24 columns contain Gabor response magnitude statistics.

Column names follow the pattern:

```text
gabor_f{frequency}_theta{orientation}_{statistic}
```

Examples:

```text
gabor_f0.05_theta0_mean
gabor_f0.05_theta0_std
gabor_f0.05_theta45_mean
gabor_f0.05_theta45_std
gabor_f0.10_theta90_mean
gabor_f0.10_theta90_std
gabor_f0.20_theta135_mean
gabor_f0.20_theta135_std
```

For each frequency and orientation, two features are stored:

- `mean`: Mean magnitude of the Gabor response
- `std`: Standard deviation of the Gabor response magnitude

The feature ordering follows the configured frequencies, orientations, and statistics.

Unlike RGB and LBP histograms, Gabor features are not normalized to sum to 1.

Each image is represented by one row.

### Combined Feature Output

The combined CSV contains 107,180 rows and 166 columns.

The first four columns contain image metadata:

- `id`
- `path`
- `label`
- `dataset`

The remaining 162 columns contain the concatenated handcrafted features.

The feature columns are organized into four consecutive groups:

| Feature Group | Number of Columns | Column Prefix |
|---|---:|---|
| RGB histograms | 48 | `r_hist_`, `g_hist_`, `b_hist_` |
| GLCM | 36 | `glcm_` |
| LBP | 54 | `lbp_` |
| Gabor | 24 | `gabor_` |
| **Total** | **162** | |

The combined dataset preserves the original feature names.

The feature columns are not standardized or normalized together.

This allows downstream experiments to select individual feature families or combinations without modifying the original extracted values.

#### Selecting Feature Families

The combined dataset supports different feature configurations.

For example, a classifier can be evaluated using:

- RGB histograms only (48 features)
- GLCM only (36 features)
- LBP only (54 features)
- Gabor only (24 features)
- RGB + GLCM (84 features)
- RGB + LBP (102 features)
- GLCM + LBP + Gabor (114 features)
- All four feature families (162 features)

Feature groups can be selected using their column prefixes.

For example:

```python
import pandas as pd

df = pd.read_csv(
    "handcrafted_features/outputs/combined_features.csv"
)

metadata_columns = ["id", "path", "label", "dataset"]

rgb_columns = [
    column
    for column in df.columns
    if column.startswith(("r_hist_", "g_hist_", "b_hist_"))
]

glcm_columns = [
    column
    for column in df.columns
    if column.startswith("glcm_")
]

lbp_columns = [
    column
    for column in df.columns
    if column.startswith("lbp_")
]

gabor_columns = [
    column
    for column in df.columns
    if column.startswith("gabor_")
]

# Individual feature families
X_rgb = df[rgb_columns]
X_glcm = df[glcm_columns]
X_lbp = df[lbp_columns]
X_gabor = df[gabor_columns]

# Selected combinations
X_rgb_glcm = df[rgb_columns + glcm_columns]

X_texture = df[
    glcm_columns + lbp_columns + gabor_columns
]

# All handcrafted features
feature_columns = [
    column
    for column in df.columns
    if column not in metadata_columns
]

X_all = df[feature_columns]

# Tissue labels
y = df["label"]
```

These examples only select feature columns. They do not perform training, scaling, or evaluation.

The dataset split and any learned preprocessing transformations should be handled separately by the modeling pipeline.

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

**Completed RGB Histogram Extraction:**

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

**Completed GLCM Extraction:**

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

### LBP Validation

Run the LBP validation script from the repository root:

```bash
python handcrafted_features/validate_features.py --method lbp
```

The validation checks include:

- Expected dataset dimensions and class distributions
- Missing values
- Non-finite feature values
- Duplicate image paths
- Nonnegative histogram values
- Independent normalization of all three LBP histograms
- Constant features across the dataset

For each image, the validator checks that the 10-bin, 18-bin, and 26-bin histograms independently sum to approximately 1.

A numerical tolerance is used to account for floating-point rounding errors.

**Completed LBP Extraction:**

| Metric | Result |
|---|---:|
| Images processed | 107,180 |
| Features per image | 54 |
| Output columns | 58 |
| Extraction time | 121.48 minutes |
| Average processing rate | 14.71 images/second |
| Missing values | 0 |
| Non-finite features | 0 |
| Duplicate image paths | 0 |
| Constant features | 0 |
| Validation | All checks passed |

The observed histogram normalization ranges were:

| Configuration | Histogram Bins | Minimum Sum | Maximum Sum |
|---|---:|---:|---:|
| P=8, R=1 | 10 | 0.9999999295 | 1.0000000681 |
| P=16, R=2 | 18 | 0.9999999371 | 1.0000000620 |
| P=24, R=3 | 26 | 0.9999999270 | 1.0000000717 |

All three LBP histograms were normalized correctly within the configured numerical tolerance.

The small deviations from exactly 1 are consistent with floating-point rounding.

The validation confirmed that none of the 54 LBP feature columns were constant across the dataset.

### Gabor Validation

Run the Gabor validation script from the repository root:

```bash
python handcrafted_features/validate_features.py --method gabor
```

The validation checks include:

- Expected dataset dimensions and class distributions
- Missing values
- Non-finite feature values
- Duplicate image paths
- Nonnegative Gabor feature values
- Constant features across the dataset
- Observed minimum and maximum values for each Gabor feature

Unlike normalized histograms, Gabor response statistics are not expected to sum to 1.

**Completed Gabor Extraction:**

| Metric | Result |
|---|---:|
| Images processed | 107,180 |
| Filters per image | 12 |
| Features per image | 24 |
| Output columns | 28 |
| Extraction time | 209.56 minutes |
| Average processing rate | 8.52 images/second |
| Missing values | 0 |
| Non-finite features | 0 |
| Duplicate image paths | 0 |
| Constant features | 0 |
| Minimum feature value | 0.00025538384 |
| Maximum feature value | 0.059361417 |
| Validation | All checks passed |

The observed Gabor feature ranges were:

| Feature | Minimum | Maximum |
|---|---:|---:|
| `gabor_f0.05_theta0_mean` | 0.001437 | 0.059361 |
| `gabor_f0.05_theta0_std` | 0.004802 | 0.039766 |
| `gabor_f0.05_theta45_mean` | 0.000426 | 0.048069 |
| `gabor_f0.05_theta45_std` | 0.000889 | 0.036345 |
| `gabor_f0.05_theta90_mean` | 0.001987 | 0.054254 |
| `gabor_f0.05_theta90_std` | 0.006225 | 0.036754 |
| `gabor_f0.05_theta135_mean` | 0.000430 | 0.047362 |
| `gabor_f0.05_theta135_std` | 0.000842 | 0.030258 |
| `gabor_f0.10_theta0_mean` | 0.000941 | 0.045644 |
| `gabor_f0.10_theta0_std` | 0.003744 | 0.034563 |
| `gabor_f0.10_theta45_mean` | 0.000419 | 0.047321 |
| `gabor_f0.10_theta45_std` | 0.000817 | 0.029861 |
| `gabor_f0.10_theta90_mean` | 0.001230 | 0.047471 |
| `gabor_f0.10_theta90_std` | 0.004865 | 0.039919 |
| `gabor_f0.10_theta135_mean` | 0.000450 | 0.044195 |
| `gabor_f0.10_theta135_std` | 0.000810 | 0.031351 |
| `gabor_f0.20_theta0_mean` | 0.000556 | 0.042753 |
| `gabor_f0.20_theta0_std` | 0.002647 | 0.029374 |
| `gabor_f0.20_theta45_mean` | 0.000255 | 0.048922 |
| `gabor_f0.20_theta45_std` | 0.000512 | 0.030936 |
| `gabor_f0.20_theta90_mean` | 0.000787 | 0.045174 |
| `gabor_f0.20_theta90_std` | 0.003849 | 0.032605 |
| `gabor_f0.20_theta135_mean` | 0.000275 | 0.044432 |
| `gabor_f0.20_theta135_std` | 0.000512 | 0.029829 |

All extracted Gabor features were finite and nonnegative.

The validation confirmed that none of the 24 Gabor feature columns were constant across the dataset.

The observed differences in feature ranges reflect variations in response magnitude across frequencies, orientations, and images.

These numerical checks establish the structural and numerical consistency of the extracted Gabor features but do not establish their predictive usefulness.

### Combined Feature Validation

After generating the combined feature matrix, run:

```bash
python handcrafted_features/validate_combined.py
```

The script independently validates the combined dataset against the four original feature CSV files.

The validation checks include:

- Expected dataset dimensions
- Expected total feature count
- Unique column names
- Missing values
- Non-finite feature values
- Duplicate image paths
- Dataset and class distributions
- Presence of all original feature columns
- Matching image paths across source and combined datasets
- Numerical agreement between combined and original feature values

For each feature family, the validator joins the combined dataset with the original feature CSV using image paths.

It then compares the corresponding feature values using:

- Relative tolerance: `rtol=1e-6`
- Absolute tolerance: `atol=1e-8`

This verifies that the combination process preserved the extracted feature values.

**Completed Combined Feature Validation:**

| Metric | Result |
|---|---:|
| Images | 107,180 |
| Total columns | 166 |
| Metadata columns | 4 |
| Feature columns | 162 |
| NCT-CRC-HE-100K images | 100,000 |
| CRC-VAL-HE-7K images | 7,180 |
| Missing values | 0 |
| Non-finite feature values | 0 |
| Duplicate image paths | 0 |
| RGB histogram features verified | 48 |
| GLCM features verified | 36 |
| LBP features verified | 54 |
| Gabor features verified | 24 |
| **Validation** | **All checks passed** |

All 162 features were successfully verified against their original source files.

The combined dataset preserves the original feature values and image associations.

No feature misalignment or numerical inconsistencies were detected within the configured validation tolerances.

### Overall Extraction Summary

| Feature Family | Features | Extraction Time | Validation |
|---|---:|---:|---|
| RGB histograms | 48 | 25.89 minutes | Passed |
| GLCM | 36 | 25.93 minutes | Passed |
| LBP | 54 | 121.48 minutes | Passed |
| Gabor | 24 | 209.56 minutes | Passed |
| **Total** | **162** | **382.86 minutes** | **All passed** |

The four feature families have been extracted and validated independently across all 107,180 images.

The combined extraction time was approximately 6 hours and 23 minutes.

The four feature matrices were subsequently merged into a single 162-dimensional representation.

The combined dataset was independently validated against the original feature files, with all checks passing.

These checks confirm the structural and numerical validity of the feature matrices, but classification experiments are still required to evaluate their predictive usefulness.

## Exploratory Feature Analysis

Exploratory analysis was performed to investigate class-wise feature distributions, correlations between features, and the ability of individual features to distinguish tissue classes.

All exploratory analyses used the **100,000 images from NCT-CRC-HE-100K**.

The independent CRC-VAL-HE-7K dataset was excluded from exploratory analysis to preserve it for final evaluation.

### Visualization Pipeline

The `visualize_features.py` script generates:

- Class-wise boxplots for representative RGB, GLCM, LBP, and Gabor features
- Pearson correlation heatmaps for each feature family
- A CSV containing highly correlated feature pairs
- One-way ANOVA F-statistics for all 162 features
- A bar chart of the 20 highest-ranked features by ANOVA F-statistic

Run from the repository root:

```bash
python handcrafted_features/visualize_features.py
```

The correlation threshold can also be configured:

```bash
python handcrafted_features/visualize_features.py --correlation-threshold 0.95
```

The default correlation threshold is 0.95.

Generated figures and analysis results are saved under:

```text
handcrafted_features/figures/
```

### Class-Wise Feature Distributions

Representative features from each family were visualized using boxplots across the nine tissue classes.

The boxplots show the median, interquartile range, and whiskers for each feature within each class.

Outliers are hidden in the plots to improve readability.

Three representative features were selected from each family.

#### RGB Histograms

The selected RGB histogram features show differences in color distributions across tissue classes.

Notable observations include:

- BACK generally exhibits low values in the selected histogram bins.
- STR shows relatively high values in selected green and blue intensity bins.
- MUC exhibits a relatively high median in the selected blue histogram bin.
- Considerable overlap exists between several tissue classes.

These findings suggest that global color distributions contain class-dependent information.

However, an individual histogram-bin value represents the proportion of pixels within an intensity interval, not the average intensity of the entire color channel.

Generated figure:

```text
figures/class_distributions/rgb_class_distributions.png
```

#### GLCM

The selected GLCM features show differences in local grayscale texture characteristics.

Notable observations include:

- BACK exhibits relatively low GLCM contrast.
- BACK exhibits substantially higher homogeneity than most tissue classes.
- LYM and TUM exhibit relatively high contrast values.
- NORM exhibits relatively high correlation values in the selected feature.
- Several tissue classes have overlapping distributions.

The high homogeneity of BACK is consistent with its comparatively uniform grayscale appearance.

The observed differences indicate that GLCM captures meaningful variation in local texture across tissue categories.

Generated figure:

```text
figures/class_distributions/glcm_class_distributions.png
```

#### Local Binary Patterns

Selected LBP histogram bins show differences in local texture pattern frequencies.

Notable observations include:

- BACK generally exhibits low values in the selected LBP bins.
- LYM exhibits relatively distinct distributions in selected bins.
- Some LBP bins show relatively narrow distributions for LYM.
- Considerable overlap exists among several other tissue classes.

These observations suggest that local binary texture patterns differ across tissue categories.

However, individual LBP histogram bins do not completely separate all nine classes.

Generated figure:

```text
figures/class_distributions/lbp_class_distributions.png
```

#### Gabor Features

Selected Gabor features show differences in oriented texture response magnitudes.

Notable observations include:

- LYM exhibits relatively high Gabor response magnitudes.
- BACK exhibits consistently low responses in the selected features.
- NORM exhibits relatively high responses for some lower-frequency filters.
- Several other tissue classes exhibit overlapping distributions.

These observations suggest that oriented spatial texture responses capture class-dependent differences in tissue appearance.

Generated figure:

```text
figures/class_distributions/gabor_class_distributions.png
```

#### Summary of Class-Wise Distributions

The four feature families capture different aspects of histopathology image appearance.

The class-wise distributions reveal meaningful differences between certain tissue categories, particularly BACK and LYM.

However, considerable overlap remains between multiple classes.

No single selected feature provides complete separation across all nine tissue classes.

These findings motivate evaluating multiple handcrafted features together rather than relying on individual descriptors.

The representative features were selected for exploratory visualization and were not chosen using supervised feature selection.

### Feature Correlation Analysis

Pearson correlation coefficients were calculated between the 162 handcrafted features using the NCT-CRC-HE-100K development dataset.

Separate correlation heatmaps were generated for each feature family.

The analysis also identified pairs of features with absolute Pearson correlation greater than or equal to 0.95.

#### Highly Correlated Feature Pairs

| Feature Family | Number of Features | Highly Correlated Pairs |
|---|---:|---:|
| RGB | 48 | 2 |
| GLCM | 36 | 26 |
| LBP | 54 | 31 |
| Gabor | 24 | 0 |
| **Total** | **162** | **59** |

All 59 highly correlated pairs belonged to the same feature family.

No cross-family feature pairs exceeded the selected absolute correlation threshold.

This result suggests that substantial redundancy exists within some feature families.

However, the absence of cross-family correlations above 0.95 does not establish that the feature families contain independent predictive information.

#### RGB Correlations

The RGB correlation heatmap exhibits structured relationships between neighboring intensity bins and across color channels.

Correlations within each channel are partly influenced by the fact that histogram bins are normalized to sum to 1.

Some neighboring bins exhibit positive correlations, while other bins exhibit negative relationships.

These patterns reflect relationships among the color distributions of the images.

#### GLCM Correlations

The GLCM correlation heatmap exhibits strong positive and negative correlations between several texture statistics.

Features calculated at different pixel distances are particularly strongly correlated.

For example:

| Feature 1 | Feature 2 | Absolute Correlation |
|---|---|---:|
| `glcm_d1_energy_mean` | `glcm_d2_energy_mean` | 0.999767 |
| `glcm_d1_asm_mean` | `glcm_d2_asm_mean` | 0.999622 |
| `glcm_d2_energy_mean` | `glcm_d4_energy_mean` | 0.998752 |
| `glcm_d1_dissimilarity_mean` | `glcm_d2_dissimilarity_mean` | 0.998457 |
| `glcm_d1_homogeneity_mean` | `glcm_d2_homogeneity_mean` | 0.997752 |

These results indicate that some GLCM statistics measured at different distances contain highly redundant information.

However, different GLCM properties can still describe distinct aspects of texture.

#### LBP Correlations

The LBP correlation heatmap exhibits block structures corresponding to the three neighborhood configurations.

Some histogram bins are strongly correlated across different scales.

For example, `lbp_p16_r2_hist_017` and `lbp_p24_r3_hist_025` have an absolute correlation of approximately 0.993.

These relationships suggest that certain local binary patterns are related across neighborhood sizes.

Other bins exhibit weaker or negative correlations, indicating variation in the captured texture information.

#### Gabor Correlations

The Gabor correlation heatmap exhibits predominantly positive correlations between filter-response features.

Correlation strengths vary across frequencies, orientations, and response statistics.

No Gabor feature pairs exceeded the absolute correlation threshold of 0.95.

However, this does not imply that the Gabor features are uncorrelated.

Several features still exhibit moderate or strong positive correlations below the selected threshold.

#### Interpretation of Feature Redundancy

The correlation analysis identified substantial redundancy within GLCM and LBP representations.

Highly correlated features were retained in the combined feature matrix.

Removing correlated features may reduce dimensionality, but it does not necessarily improve classification or clustering performance.

Downstream experiments can compare the full representation against a correlation-filtered representation to determine whether feature reduction is beneficial.

Any correlation-based feature filtering used in predictive modeling should be fitted using training data only.

#### Generated Correlation Outputs

```text
figures/correlations/
├── rgb_correlation.png
├── glcm_correlation.png
├── lbp_correlation.png
├── gabor_correlation.png
└── highly_correlated_features.csv
```

The CSV contains:

- `feature_1`
- `feature_2`
- `absolute_correlation`

The file is sorted by descending absolute correlation.

### Feature Discriminativeness Analysis

One-way ANOVA was used to evaluate differences in individual feature values across the nine tissue classes.

The analysis was performed on all 162 handcrafted features using the NCT-CRC-HE-100K development dataset.

For each feature, the ANOVA F-statistic compares between-class variation against within-class variation:

```text
F = Between-class mean square / Within-class mean square
```

Higher F-statistics indicate stronger differences between class means relative to within-class variability.

ANOVA evaluates each feature independently and does not measure the incremental predictive contribution of a feature when combined with others.

#### Highest-Ranked Feature per Family

| Feature Family | Highest-Ranked Feature | F-statistic |
|---|---|---:|
| RGB | `g_hist_012` | 11,280.81 |
| GLCM | `glcm_d4_homogeneity_mean` | 53,565.36 |
| LBP | `lbp_p8_r1_hist_002` | 58,266.41 |
| Gabor | `gabor_f0.10_theta45_mean` | 24,926.87 |

The highest-ranked individual feature was `lbp_p8_r1_hist_002`, with an F-statistic of approximately 58,266.

The highest-ranked GLCM feature was `glcm_d4_homogeneity_mean`, with an F-statistic of approximately 53,565.

#### Top 20 Features

| Rank | Feature | F-statistic | Family |
|---:|---|---:|---|
| 1 | `lbp_p8_r1_hist_002` | 58,266.41 | LBP |
| 2 | `glcm_d4_homogeneity_mean` | 53,565.36 | GLCM |
| 3 | `glcm_d1_energy_mean` | 42,774.78 | GLCM |
| 4 | `glcm_d2_homogeneity_mean` | 41,184.17 | GLCM |
| 5 | `glcm_d2_energy_mean` | 40,600.59 | GLCM |
| 6 | `glcm_d4_energy_mean` | 35,946.65 | GLCM |
| 7 | `lbp_p8_r1_hist_004` | 35,404.50 | LBP |
| 8 | `lbp_p8_r1_hist_008` | 35,114.57 | LBP |
| 9 | `glcm_d1_homogeneity_mean` | 34,848.36 | GLCM |
| 10 | `glcm_d1_asm_std` | 31,613.80 | GLCM |
| 11 | `glcm_d4_asm_std` | 26,835.54 | GLCM |
| 12 | `gabor_f0.10_theta45_mean` | 24,926.87 | Gabor |
| 13 | `gabor_f0.10_theta0_mean` | 23,386.66 | Gabor |
| 14 | `gabor_f0.10_theta135_mean` | 22,904.78 | Gabor |
| 15 | `glcm_d4_dissimilarity_mean` | 22,897.54 | GLCM |
| 16 | `lbp_p16_r2_hist_011` | 22,134.18 | LBP |
| 17 | `glcm_d1_asm_mean` | 21,937.66 | GLCM |
| 18 | `lbp_p8_r1_hist_003` | 21,030.90 | LBP |
| 19 | `glcm_d2_asm_mean` | 20,174.71 | GLCM |
| 20 | `lbp_p24_r3_hist_014` | 19,962.72 | LBP |

#### Distribution of Top-Ranked Features

Among the 20 highest-ranked features:

| Feature Family | Features in Top 20 |
|---|---:|
| GLCM | 11 |
| LBP | 6 |
| Gabor | 3 |
| RGB | 0 |
| **Total** | **20** |

GLCM and LBP dominate the highest-ranked individual features.

These findings suggest that several grayscale texture descriptors exhibit strong class-dependent mean differences in the dataset.

However, the rankings do not establish that GLCM or LBP will necessarily produce the highest classification accuracy.

Highly ranked features may also be strongly correlated with one another.

RGB features may still provide complementary predictive information when combined with texture descriptors.

#### Statistical Significance

The ANOVA analysis also reports p-values.

Many features have reported p-values of `0.0` because extremely small values underflow to zero in floating-point calculations.

These values should not be interpreted as mathematically zero probabilities.

With 100,000 images, even relatively small differences can produce highly significant p-values.

The F-statistics are therefore used primarily to compare the relative strength of class-dependent mean differences rather than to establish predictive performance.

#### Generated ANOVA Outputs

```text
figures/feature_analysis/
├── feature_anova_scores.csv
└── top_features_anova.png
```

The CSV contains:

- `feature`
- `f_score`
- `p_value`
- `family`

All 162 features are ranked by descending F-statistic.

The PNG visualizes the 20 highest-ranked features, colored by feature family.

### Exploratory Analysis Summary

The exploratory analysis produced three main findings.

**1. Handcrafted features capture class-dependent differences.**

Class-wise distributions reveal meaningful differences in color and texture characteristics, particularly for BACK and LYM.

However, individual features exhibit substantial overlap across several tissue classes.

**2. Some feature families contain substantial redundancy.**

The correlation analysis identified 59 feature pairs with absolute Pearson correlation of at least 0.95.

Most highly correlated pairs belonged to GLCM and LBP.

These results motivate evaluating whether feature reduction improves downstream performance.

**3. Texture features dominate the highest individual ANOVA rankings.**

GLCM and LBP account for 17 of the 20 highest-ranked features.

This suggests that grayscale texture descriptors exhibit strong differences in feature means across tissue classes.

However, these exploratory findings do not establish which feature representation produces the best classification or clustering results.

That requires downstream experiments using consistent training and evaluation procedures.

### Methodological Considerations

The exploratory analysis was performed on the complete NCT-CRC-HE-100K development dataset.

These results are intended for descriptive analysis and project documentation.

For downstream supervised classification, feature selection based on ANOVA scores must be fitted using training data only.

When cross-validation is used, feature selection must be performed independently within each training fold.

The same principle applies to feature scaling and data-driven correlation filtering.

The CRC-VAL-HE-7K dataset remains reserved for independent final evaluation.

## Modeling Considerations

Feature extraction is performed independently for each image and does not use tissue labels to calculate features.

The NCT-CRC-HE-100K dataset is used for model development, while CRC-VAL-HE-7K is reserved for independent final evaluation.

Any learned preprocessing transformations, including feature standardization, must be fitted only on training data and then applied to validation and test data.

The project's current internal training/validation split is image-level rather than patient-level because patient identifiers are not available in the training metadata. This may produce optimistic internal validation estimates and should be considered when interpreting results.

Different feature families may have substantially different numerical scales. Standardization should therefore be considered when combining feature representations, particularly for distance-based classifiers and clustering algorithms.

The four feature families describe different aspects of histopathology image appearance:

- **RGB histograms:** Global color intensity distributions
- **GLCM:** Statistical relationships between grayscale pixel intensities
- **LBP:** Distributions of local binary texture patterns
- **Gabor:** Oriented texture responses at multiple spatial frequencies

GLCM, LBP, and Gabor are all grayscale texture descriptors, but they capture different aspects of image structure.

The current Gabor representation retains orientation-specific features, while the GLCM implementation aggregates texture statistics across orientations.

These representations should be evaluated individually and in combination to determine whether combining color and texture information improves classification or clustering performance.

### Individual and Combined Feature Experiments

The combined feature matrix supports experiments using individual feature families and different combinations.

Evaluating feature families independently allows the project to compare their relative effectiveness.

For example, RGB histograms can be evaluated separately from GLCM, LBP, and Gabor features to determine whether color or texture information is more informative for tissue classification.

Combining feature families allows models to use multiple types of image information simultaneously.

Potential experiments include:

- Individual RGB, GLCM, LBP, and Gabor representations
- Color and texture combinations
- All grayscale texture features combined
- All 162 handcrafted features combined

More features do not necessarily improve model performance.

Combining representations may introduce redundant or noisy features, so the combined representation should be evaluated empirically.

Feature selection or dimensionality reduction may also be considered if the combined representation contains redundant or highly correlated features.

### Supervised Classification

The handcrafted feature matrices can be used as inputs to classical machine learning classifiers, including:

- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines

The same training and evaluation splits should be used when comparing feature representations.

Model selection, hyperparameter tuning, and learned preprocessing should use only the model-development data.

CRC-VAL-HE-7K should remain reserved for independent final evaluation.

### Unsupervised Clustering

The feature matrices can also be used for unsupervised clustering experiments.

Potential methods include:

- K-Means
- K-Means++
- Hierarchical clustering
- DBSCAN

Clustering quality can be compared across feature families and selected combinations.

Because clustering methods often depend on distances between feature vectors, feature scaling should be considered carefully.

Dimensionality reduction and visualization methods such as PCA, t-SNE, and UMAP may help explore the structure of the extracted representations.

### Sharing Generated Features

All generated feature CSV files are stored in `handcrafted_features/outputs/`.

This directory is excluded from Git because the generated feature matrices are large data artifacts rather than source code.

The extraction, combination, validation, and visualization scripts are maintained in the repository, but the generated feature matrices must be shared separately.

For downstream modeling, teammates can use the combined CSV without repeating the original image feature extraction.

The original individual feature CSVs can also be shared when experiments require separate feature representations.

The combined dataset contains the same original features, so individual feature families can be selected directly using their column names.

## Planned Features

Future implementations may include:

- Hematoxylin and eosin stain-specific features

These representations may be evaluated individually and in combination using consistent data splits and downstream models.
