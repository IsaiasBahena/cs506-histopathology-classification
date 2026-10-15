# Classical Machine Learning For Colorectal Histopathology Image Classification

## Project Description

Histopathology images are used to examine tissue structure and identify abnormalities associated with diseases like cancer. Histopathology images contain many measurable visual characteristics like color, texture, and tissue structure. These can be analyzed using classical machine learning methods.

In this project, we will investigate whether quantitative features extracted from colorectal histopathology images can be used to distinguish different tissue types using classical machine learning methods. We will use the NCT-CRC-HE-100K dataset, which contains 100,000 colorectal histopathology image patches belonging to nine tissue categories.

The main focus of the project will be the relationship between image representation and classification performance. We will compare different methods of converting histopathology images into numerical features. Our initial approaches will include handcrafted features describing visual properties such as color and texture, as well as lower-dimensional representations derived from image pixel data using Singular Value Decomposition (SVD). These representations will then be used to train and evaluate classification methods covered in the course.

We will also explore whether unsupervised clustering methods can identify meaningful groups of histopathology images based only on their extracted features and examine how these groups relate to the known tissue categories.

The project will follow the complete data science lifecycle by collecting the dataset from a public source, validating and cleaning the images, extracting quantitative features, visualizing patterns in the data, training and evaluating machine learning models, and interpreting the resulting performance and failure cases.

## Project Goals

The primary goal of this project is to develop and evaluate a reproducible machine learning pipeline for classifying colorectal histopathology images into nine tissue categories using quantitative image features and classical machine learning methods. We will investigate the following research questions:

1. How accurately can classical machine learning models classify colorectal histopathology images into their corresponding tissue categories? We will establish two baselines: a majority-class classifier and a logistic regression classifier trained using color histogram features. Our goal is for our best-performing model and feature representation to improve macro F1 score by at least 5 percentage points over the color histogram and logistic regression baseline on the internal validation set.
2. How does image representation affect classification performance? We will compare handcrafted features, including color histograms, Gray-Level Co-occurrence Matrix (GLCM) texture features, and Local Binary Patterns (LBP), with lower-dimensional pixel-based representations obtained using SVD. We will evaluate these representations using the same classification models and evaluation procedures to determine how feature selection affects performance.
3. To what extent do clusters produced from extracted image features correspond to the known colorectal tissue categories? We will evaluate agreement between cluster assignments and ground-truth tissue labels using Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).

Classification performance will be evaluated using metrics like accuracy, precision, recall, and macro F1 score. Confusion matrices will also be used to investigate class-specific errors and determine which tissue categories are the most difficult for the models to distinguish.

## Data Collection

We will use the publicly available NCT-CRC-HE-100K colorectal histopathology dataset on Zenodo. The dataset contains 100,000 non-overlapping image patches extracted from hematoxylin and eosin (H&E) stained histopathology slides of colorectal cancer and normal tissue. Each image is 224x224 pixels and belongs to one of nine tissue categories: adipose tissue, background, debris, lymphocytes, mucus, smooth muscle, normal colon mucosa, cancer-associated stroma, or colorectal adenocarcinoma epithelium.

We will also collect the associated CRC-VAL-HE-7K dataset, which contains 7,180 histopathology image patches from patients who do not overlap with those represented in NCT-CRC-HE-100K. We plan to reserve this independent dataset for final evaluation in order to test how well the resulting models generalize to data from patients not represented during model development.

We will implement a Python data acquisition script that downloads the required dataset archives from Zenodo, verifies the downloaded files, extracts the archives, and organizes the images into a directory structure. The final README will document the original data source and provide the instructions necessary to reproduce the data collection process.

After collection, we will validate the dataset by checking image readability, dimensions, color channels, tissue labels, class distributions, and potential duplicates. Any identified issues and resulting cleaning decisions will be documented and incorporated into the reproducible data-processing pipeline.

## Modeling Plan

Our initial modeling approach will compare two different image representations:

1. Handcrafted image features: We will extract color histograms to capture color distributions, GLCM texture statistics to describe spatial relationships between pixel intensities, and LBP features to capture local texture patterns. We will experiment with these feature types individually and in combination. 
2. Pixel-based representations with dimensionality reduction using SVD: Applying SVD directly to all 100,000 images at their original 224x224 RGB resolution would require substantial memory and computational resources. To address this, we will initially downsample images to 32x32 pixels and use randomized SVD to obtain lower-dimensional representations. We will experiment with different numbers of retained components and consider larger image resolutions or smaller subsets depending on available computational resources. SVD will be fitted using only the training data, and the learned transformation will then be applied to the validation and independent test datasets.

For supervised classification, we will establish majority-class and color histogram with logistic regression baselines. We will then experiment with classical machine learning methods such as logistic regression, decision trees, and support vector machines. We will compare the performance of these methods across the different image representations using consistent training and validation splits. 

We also plan to apply unsupervised methods such as K-means/K-means++ and hierarchical clustering to the extracted features. This will allow us to investigate whether the feature representations naturally organize histopathology images into groups that correspond to known tissue categories. We will evaluate clustering results using ARI and NMI.

Because some algorithms may be computationally expensive on the full dataset, we may use reproducible subsets for selected experiments, particularly hierarchical clustering and nonlinear SVMs. When comparing feature representations or models, we will use the same image subsets to ensure fair comparisons. 

## Visualization Plan

Visualizations will be used throughout the project to explore the dataset, understand the extracted features, and interpret model results. These include:

- Representative histopathology images from each tissue category
- Tissue class distributions
- Distributions of extracted image features
- SVD explained variance across different numbers of components
- Low-dimensional visualizations of the image feature spaces
- Clustering results compared with known tissue categories
- Confusion matrices showing classification errors
- Comparisons of classification performance across models, feature representations, and baselines

## Timeline

### Weeks 1-2: Data Collection, Validation, and Exploration

- Set up the GitHub repository and project environment
- Implement the reproducible data acquisition process
- Download and organize NCT-CRC-HE-100K and CRC-VAL-HE-7K
- Implement data validation and cleaning checks
- Examine tissue class distributions and dataset characteristics
- Produce initial exploratory visualizations
- Establish reproducible training and validation splits

### Weeks 3-4: Feature Extraction and Representation

- Implement color histogram, GLCM, and LBP feature extraction
- Experiment with individual handcrafted feature types and combinations
- Develop the pixel-based image representation using downsampled images, initially at the 32x32 resolution
- Apply randomized SVD to produce lower-dimensional representations while managing memory requirements
- Experiment with different numbers of SVD components and analyze explained variance
- Produce visualizations comparing tissue classes in the different feature spaces
- Establish majority-class and color histogram with logistic regression baselines

### Weeks 5-6: Modeling and Evaluation

- Train and tune selected classical classification models
- Compare model performance across feature representations and against the established baselines
- Evaluate performance using accuracy, precision, recall, macro F1 score, and confusion matrices
- Determine whether the models achieve the target improvement of at least 5 percentage points in macro F1 over the logistic regression baseline
- Perform initial clustering experiments using K-means/K-means++ and hierarchical clustering
- Evaluate clustering results using ARI and NMI
- Analyze which tissue classes are most difficult to distinguish and investigate common failure cases

### Week 7: Final Experiments and Interpretation

- Complete additional experiments motivated by preliminary results
- Evaluate finalized models using the independent CRC-VAL-HE-7K dataset
- Finalize clustering and classification comparisons
- Produce final-quality visualizations
- Analyze limitations and interpret the main findings

### Week 8: Reproducibility, Report, and Presentation

- Finalize the Makefile and reproducible build and execution process
- Add tests for important data-processing and feature-extraction functionality
- Configure the GitHub workflow to run automated tests
- Reproduce the final results using the documented pipeline
- Complete the README final report
- Prepare and record the final presentation

## Scope and Fallback Plan

The core project consists of reproducible data collection and validation, handcrafted image feature extraction, SVD-based image representations, supervised classification, model evaluation, and data visualization.

Unsupervised clustering will serve as a secondary analysis after the core classification pipeline is functional. Additional analyses, such as experimenting with other clustering methods, image similarity and retrieval, or features extracted from a pretrained image model, will be considered if enough time remains.

If the original scope is too ambitious, we will prioritize color histogram and texture-based handcrafted features, downsampled images with randomized SVD, and a smaller set of supervised classification models. We may also use reproducible subsets of the dataset to reduce computational requirements while retaining the complete data collection, cleaning, feature extraction, visualization, training, and evaluation pipeline.