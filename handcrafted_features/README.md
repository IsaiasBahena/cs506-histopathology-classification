
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
- **Normalization:** Each scale's histogram is independently normalized to sum to 1

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

The currently implemented feature extractors produce:

| Feature Family | Features per Image |
|---|---:|
| RGB histograms | 48 |
| GLCM texture | 36 |
| LBP texture | 54 |
| Gabor texture | 24 |
| **Total** | **162** |

These features are currently stored in separate CSV files.

They can be combined using image identifiers and dataset information to create a 162-dimensional feature representation.

The combined feature matrix has not yet been generated or validated.

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
├── validate_features.py
└── outputs/
```

- `extract_features.py`: Reads metadata, processes images in batches, and saves feature matrices.
- `rgb_histogram.py`: Calculates normalized RGB histograms for individual images.
- `glcm.py`: Computes GLCM texture statistics and generates feature names.
- `lbp.py`: Computes multi-scale uniform LBP histograms and generates feature names.
- `gabor.py`: Computes multi-frequency, multi-orientation Gabor texture features using FFT-based convolution.
- `validate_features.py`: Validates extracted handcrafted feature matrices, including RGB histograms, GLCM, LBP, and Gabor features.
- `outputs/`: Stores generated feature matrices and extraction progress files. This directory is excluded from Git.

## Installation

From the repository root, install the required dependencies:

```bash
pip install -r handcrafted_features/requirements.txt
```

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

### Checkpointing

Features are saved after each batch. If extraction is interrupted, rerunning the same command attempts to resume from the last saved checkpoint.

The script checks the existing output and progress metadata before resuming. An interruption during CSV writing may require manual recovery.

If the feature extraction implementation or its configuration changes, existing output files should not be mixed with results from the new implementation.

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

### Completed LBP Extraction

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

### Completed Gabor Extraction

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

These checks confirm the structural and numerical validity of the feature matrices, but classification experiments are still required to evaluate their predictive usefulness.

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

Feature selection or dimensionality reduction may also be considered if the combined representation contains redundant or highly correlated features.

## Planned Features

Future implementations may include:

- Hematoxylin and eosin stain-specific features

These representations will be evaluated individually and in combination using consistent data splits and downstream models.
