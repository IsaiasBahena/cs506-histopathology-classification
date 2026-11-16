
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import f_classif


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "outputs" / "combined_features.csv"
FIGURE_DIR = BASE_DIR / "figures"

METADATA_COLUMNS = ["id", "path", "label", "dataset"]

CLASSES = [
    "ADI", "BACK", "DEB", "LYM", "MUC",
    "MUS", "NORM", "STR", "TUM",
]

FEATURE_GROUPS = {
    "rgb": ("r_hist_", "g_hist_", "b_hist_"),
    "glcm": ("glcm_",),
    "lbp": ("lbp_",),
    "gabor": ("gabor_",),
}

# Representative features for class-wise comparison.
SELECTED_FEATURES = {
    "RGB": [
        "r_hist_004",
        "g_hist_008",
        "b_hist_012",
    ],
    "GLCM": [
        "glcm_d1_contrast_mean",
        "glcm_d2_homogeneity_mean",
        "glcm_d4_correlation_mean",
    ],
    "LBP": [
        "lbp_p8_r1_hist_000",
        "lbp_p16_r2_hist_008",
        "lbp_p24_r3_hist_012",
    ],
    "Gabor": [
        "gabor_f0.05_theta0_mean",
        "gabor_f0.10_theta90_mean",
        "gabor_f0.20_theta135_mean",
    ],
}


def load_features():
    print(f"Loading {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    # Reserve CRC-VAL-HE-7K for independent final evaluation.
    df = df[
        df["dataset"] == "NCT-CRC-HE-100K"
    ].copy()

    assert len(df) == 100000, (
        "Unexpected model-development dataset size."
    )

    assert set(df["label"]) == set(CLASSES), (
        "Unexpected tissue classes."
    )

    print(f"Loaded {len(df):,} images.")

    return df


def get_feature_columns(df, prefixes):
    return [
        column
        for column in df.columns
        if column.startswith(prefixes)
    ]


def plot_class_distributions(df):
    print("\nGenerating class-wise feature distributions...")

    output_dir = FIGURE_DIR / "class_distributions"
    output_dir.mkdir(parents=True, exist_ok=True)

    for group, features in SELECTED_FEATURES.items():
        missing = [
            feature
            for feature in features
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing {group} features: {missing}"
            )

        fig, axes = plt.subplots(
            1, len(features),
            figsize=(18, 5),
        )

        for ax, feature in zip(axes, features):
            sns.boxplot(
                data=df,
                x="label",
                y=feature,
                order=CLASSES,
                ax=ax,
                showfliers=False,
                color="steelblue",
            )

            ax.set_title(feature, fontsize=10)
            ax.set_xlabel("Tissue class")
            ax.set_ylabel("Feature value")
            ax.tick_params(axis="x", rotation=45)

        fig.suptitle(
            f"{group} Feature Distributions by Tissue Class",
            fontsize=14,
        )

        fig.tight_layout()

        output_path = output_dir / (
            f"{group.lower()}_class_distributions.png"
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved: {output_path}")


def plot_correlations(df):
    print("\nGenerating correlation heatmaps...")

    output_dir = FIGURE_DIR / "correlations"
    output_dir.mkdir(parents=True, exist_ok=True)

    for group, prefixes in FEATURE_GROUPS.items():
        columns = get_feature_columns(df, prefixes)

        if not columns:
            raise ValueError(
                f"No feature columns found for {group}."
            )

        correlation = df[columns].corr()

        fig, ax = plt.subplots(
            figsize=(12, 10)
        )

        sns.heatmap(
            correlation,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={"label": "Pearson correlation"},
            ax=ax,
        )

        ax.set_title(
            f"{group.upper()} Feature Correlations"
        )

        fig.tight_layout()

        output_path = output_dir / (
            f"{group}_correlation.png"
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved: {output_path}")


def summarize_correlations(df, threshold=0.95):
    print("\nSummarizing highly correlated features...")

    output_dir = FIGURE_DIR / "correlations"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS
    ]

    correlation = df[feature_columns].corr().abs()

    upper = correlation.where(
        np.triu(
            np.ones(correlation.shape, dtype=bool),
            k=1,
        )
    )

    pairs = (
        upper.stack()
        .reset_index()
    )

    pairs.columns = [
        "feature_1",
        "feature_2",
        "absolute_correlation",
    ]

    pairs = pairs[
        pairs["absolute_correlation"] >= threshold
    ].sort_values(
        "absolute_correlation",
        ascending=False,
    )

    output_path = (
        output_dir / "highly_correlated_features.csv"
    )

    pairs.to_csv(output_path, index=False)

    print(
        f"Feature pairs with |r| >= {threshold}: "
        f"{len(pairs):,}"
    )

    print(f"Saved: {output_path}")

    if not pairs.empty:
        print("\nTop 10 correlated pairs:")
        print(pairs.head(10).to_string(index=False))


def analyze_feature_discriminativeness(df):
    print("\nAnalyzing feature discriminativeness...")

    output_dir = FIGURE_DIR / "feature_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS
    ]

    X = df[feature_columns].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy()

    # One-way ANOVA across the nine tissue classes.
    f_scores, p_values = f_classif(X, y)

    results = pd.DataFrame({
        "feature": feature_columns,
        "f_score": f_scores,
        "p_value": p_values,
    })

    def identify_family(feature):
        for family, prefixes in FEATURE_GROUPS.items():
            if feature.startswith(prefixes):
                return family.upper()

        raise ValueError(f"Unknown feature family: {feature}")

    results["family"] = results["feature"].apply(
        identify_family
    )

    results = results.sort_values(
        "f_score",
        ascending=False,
    )

    output_path = output_dir / "feature_anova_scores.csv"
    results.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

    print("\nTop 20 features:")
    print(
        results.head(20).to_string(
            index=False
        )
    )

    print("\nTop feature per family:")
    for family in ["RGB", "GLCM", "LBP", "GABOR"]:
        family_results = results[
            results["family"] == family
        ]

        best = family_results.iloc[0]

        print(
            f"{family}: {best['feature']} "
            f"(F={best['f_score']:.2f})"
        )

    # Plot top 20 features.
    top_features = results.head(20).copy()

    plt.figure(figsize=(12, 8))

    sns.barplot(
        data=top_features,
        x="f_score",
        y="feature",
        hue="family",
        dodge=False,
        palette="Set2",
    )

    plt.title(
        "Top 20 Handcrafted Features by ANOVA F-Statistic"
    )
    plt.xlabel("ANOVA F-statistic")
    plt.ylabel("Feature")

    plt.tight_layout()

    plot_path = output_dir / "top_features_anova.png"

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {plot_path}")



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
    )

    args = parser.parse_args()

    if not 0 <= args.correlation_threshold <= 1:
        parser.error(
            "--correlation-threshold must be between 0 and 1."
        )

    sns.set_theme(style="whitegrid")

    df = load_features()

    plot_class_distributions(df)
    plot_correlations(df)

    summarize_correlations(
        df,
        threshold=args.correlation_threshold,
    )

    analyze_feature_discriminativeness(df)

    print("\nAll visualizations completed!")


if __name__ == "__main__":
    main()
