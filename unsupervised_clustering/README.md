# K-Means++ Clustering Results

This analysis explores the clustering of various feature sets to see how they align with given tissue labels.  Clusters are created using K-Means++ based off various subsets of the features. Afterward, the known labels are used to evaluate and describe the clusters created.

## Before running

Before running, must create the combined feature csv file by following the instructions in the README in `handcrafted_features/`. This script expects the resulting file at:

```text
handcrafted_features/combined_features.csv
```

## Setup

Within the unsupervisedLearning folder, download the requirements using the following command.

```text
pip install -r requirements.txt
```

