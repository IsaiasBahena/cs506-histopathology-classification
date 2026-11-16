from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("handcrafted_features/combined_features.csv")
label_column = "label"
metadata_columns = ["id", "path", "label", "dataset"]
y = df[label_column]

#defining various sets of feature columns
rgb_columns = [
    column for column in df.columns
    if column.startswith(("r_hist_", "g_hist_", "b_hist_"))
]
glcm_columns = [
    column for column in df.columns
    if column.startswith("glcm_")
]
lbp_columns = [
    column for column in df.columns
    if column.startswith("lbp_")
]
gabor_columns = [
    column for column in df.columns
    if column.startswith("gabor_")
]
#putting feature sets together based off previous cols
feature_sets = {
    "RGB": rgb_columns,
    "GLCM": glcm_columns,
    "LBP": lbp_columns,
    "Gabor": gabor_columns,
    "RGB + GLCM": rgb_columns + glcm_columns,
    "Texture": glcm_columns + lbp_columns + gabor_columns, 
}
all_feature_columns = [
    column for column in df.columns
    if column not in metadata_columns
]
feature_sets["All Features"] = all_feature_columns
output_dir = Path("unsupervisedLearning/clusteringResults")
output_dir.mkdir(parents=True, exist_ok=True)
results = []
for name, columns in feature_sets.items():
    print(f"processing {name}, number of individual features: {len(columns)}")
    X = df[columns].apply(pd.to_numeric, errors="coerce")
    X_scaled = StandardScaler().fit_transform(X) #remove mean and unit variance
    kmeans = KMeans(
        init="k-means++", #using k-means++
        n_clusters=9,
        random_state=5,
        n_init=20, #running 20 times
    )
    cluster_ids = kmeans.fit_predict(X_scaled)
    #numeric scores
    ARI = adjusted_rand_score(y, cluster_ids)
    NMI = normalized_mutual_info_score(y, cluster_ids)
    #silhouette score
    n_cluster_labels = len(np.unique(cluster_ids))
    sample_size = min(5000, len(X_scaled)) #using random sample of 5,000 rows
    if 2 <= n_cluster_labels < len(X_scaled):
        try:
            silhouette = silhouette_score(
                X_scaled,
                cluster_ids,
                sample_size=sample_size,
                random_state=5,
            )
        except ValueError:
            #if sample rows don't include every cluster could happen
            silhouette = np.nan
    else:
        silhouette = np.nan
        
    results.append({
        "feature_set": name,
        "num features": len(columns),
        "ARI": ARI, #Adjusted Rand Index
        "NMI": NMI, #Normalized Mutual Information (range )
        "silhouette": silhouette,
    })
    # Cluster-composition table: rows are clusters; columns are true labels.
    labels = y.astype(str)
    composition_counts = pd.crosstab(
        pd.Series(cluster_ids, name="cluster"),
        pd.Series(labels, name="true_label"),
    )

    # Each row sums to 100%; values show the label percentages within each cluster.
    composition_percent = (
        composition_counts
        .div(composition_counts.sum(axis=1), axis=0)
        .mul(100)
        .round(1)
    )
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
    # each feature set has its own folder
    feature_output_dir = output_dir / safe_name
    feature_output_dir.mkdir(parents=True, exist_ok=True)

    composition_counts.to_csv(
        feature_output_dir / "composition_counts.csv"
    )
    composition_percent.to_csv(
        feature_output_dir / "composition_percent.csv"
    )

    print(f"ARI: {ARI:.4f} | NMI: {NMI:.4f} | Silhouette: {silhouette:.4f}")
    print("Cluster composition (% of each cluster):")
    print(composition_percent.to_string())

    # Project the scaled features to 2D for visualization.
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.sum() * 100

    # Create side-by-side plots: known labels and discovered clusters.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def plot_categories(ax, categories, title):
        categories = pd.Series(categories).astype(str).to_numpy()
        category_values = pd.unique(categories)
        category_to_code = {
            value: index for index, value in enumerate(category_values)
        }
        codes = np.array([category_to_code[value] for value in categories])

        cmap = plt.get_cmap("tab10", len(category_values))
        scatter = ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=codes,
            cmap=cmap,
            vmin=-0.5,
            vmax=len(category_values) - 0.5,
            s=12,
            alpha=0.7,
            rasterized=True,
        )

        colorbar = fig.colorbar(
            scatter,
            ax=ax,
            ticks=np.arange(len(category_values)),
        )
        colorbar.ax.set_yticklabels(category_values)
        ax.set_title(title)
        ax.set_xlabel("Principal component 1")
        ax.set_ylabel("Principal component 2")

    plot_categories(axes[0], labels, "Known tissue labels")
    plot_categories(axes[1], cluster_ids, "K-means clusters")

    fig.suptitle(
        f"{name} | ARI={ARI:.3f}, NMI={NMI:.3f} "
        f"| PCA variance explained={explained:.1f}%"
    )
    fig.tight_layout()
    fig.savefig(
        feature_output_dir / "pca.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


# Display all feature-set scores, highest ARI first.
results_df = pd.DataFrame(results).sort_values("ARI", ascending=False)

print("\nSummary:")
print(results_df.to_string(index=False))

results_df.to_csv(output_dir / "clustering_scores.csv", index=False)
print(f"\nTables and plots saved in: {output_dir.resolve()}")