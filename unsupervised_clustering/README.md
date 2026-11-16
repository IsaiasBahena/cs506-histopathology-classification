# K-Means Clustering Results

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

##  Results

Results are saved under `unsupervisedLearning/clusteringResults/`. Each feature-set subfolder contains:

- `composition_counts.csv` — counts of images from each known label in each cluster.
- `composition_percent.csv` — percentage of each cluster belonging to each known label.
- `pca.png` — side-by-side PCA plots colored by known labels and K-Means clusters.

The main `clusteringResults/` folder contains `clustering_scores.csv`, with the feature counts, ARI, NMI, and silhouette score for each feature set, sorted by ARI.

## Evaluation

ARI (Adjusted Rand Index) and NMI (Normalized Mutual Information) compare the clusters with the known labels; higher scores indicate greater alignment. The silhouette score measures cluster separation in feature space. The silhouette score is estimated using a random sample of up to 5,000 rows to reduce computation time.

PCA plots project the standardized features into two dimensions for visualization. On the left is the clustering of known tissue labels, and on the right is the clustering using k-means++ on the feature set.

## Results

| Feature set | Number of features | ARI | NMI | Silhouette |
|---|---:|---:|---:|---:|
| Texture (GLCM + LBP + Gabor; excludes RGB) | 114 | 0.270 | 0.442 | 0.228 |
| Gabor | 24 | 0.239 | 0.376 | 0.242 |
| All handcrafted | 162 | 0.232 | 0.412 | 0.173 |
| GLCM | 36 | 0.199 | 0.344 | 0.257 |
| LBP | 54 | 0.198 | 0.369 | 0.311 |
| RGB | 48 | 0.179 | 0.351 | 0.237 |
| RGB + GLCM | 84 | 0.175 | 0.366 | 0.167 |

Texture features had the highest ARI and NMI, while LBP had the highest silhouette score. However, while the high silhouette score and the PCA image shows clear distinction of clustering for LBP, it is not necessarily accurate clustering. While texture features appear to perform best for alignment of known tissue labels, these results are exploratory and intended to provide context and evaluation of feature clustering before using supervised validation.

For score comparison, see `clustering_scores.csv`. For individual feature-sets, each set is listed under clusteringResults with there associated composition tables and PCA plots.
