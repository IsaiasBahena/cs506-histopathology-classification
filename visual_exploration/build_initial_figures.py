from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

tissueClasses = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
state = 1 #in order to make reproducable and same subset, having a set state
data_dir = Path("../data")
metadata = pd.read_csv("../data/sample_1pct/image_metadata_1pct.csv") #getting data from the 1% metadata csv
output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)
metadata = metadata[metadata["label"].isin(tissueClasses)].copy()
manifest = metadata.sort_values("id")

#helper method to obtain n samples from each class
def sample_tissue_class(n):
    samples = []
    for label in tissueClasses:
        group = manifest[manifest["label"] == label]
        if not group.empty:
            samples.append(
                group.sample(n=min(n, len(group)), random_state=state)
            )
    return pd.concat(samples, ignore_index=True)

#build tissue_montage with samples from each tissue class
def build_montage():
    montage_rows = sample_tissue_class(n=3) #3 samples
    fig, axes = plt.subplots(
    nrows=len(tissueClasses), ncols=4,
    figsize=(9, 18),
    squeeze=False
    )
    for row, label in enumerate(tissueClasses):
        axes[row, 0].axis("off")
        axes[row, 0].text(0, 0.5, label, fontsize=16, weight="bold")
        examples = montage_rows[montage_rows["label"] == label]
        for col in range(3): 
            ax = axes[row, col + 1]
            ax.axis("off")
            if col < len(examples):
                image_path = data_dir / examples.iloc[col]["path"]
                with Image.open(image_path) as image:
                    ax.imshow(image.convert("RGB"))
    fig.suptitle("Visual Sampling of Each Tissue Class", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_dir / "tissue_montage.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

#average mean brightness
def build_mean_brightness_figure():
    feature_rows = sample_tissue_class(n=50)
    brightness_by_class = {label: [] for label in tissueClasses}
    for _, row in feature_rows.iterrows():
        image_path = data_dir / row["path"]
        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        brightness_by_class[row["label"]].append(float(pixels.mean()))
    values = [brightness_by_class[label] for label in tissueClasses]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(values, tick_labels=tissueClasses, showfliers=False)
    ax.set_title("Mean RGB intensity by tissue category")
    ax.set_xlabel("Tissue category")
    ax.set_ylabel("Mean pixel intensity (0–255)") #0 would be black (dark), and #255 full brightness white
    fig.tight_layout()
    fig.savefig(
        output_dir / "mean_brightness_by_tissue_class.png",
        dpi=160,
        bbox_inches="tight"
    )
    plt.close(fig)

#histogram color distribution based on intensity for 3 channels (red, blue green)
def build_color_intensity_distribution_figure():
    bins = np.arange(257)  # 256 intensity values, from 0 to 255
    channel_names = ["Red", "Green", "Blue"]
    channel_colors = ["red", "green", "blue"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    feature_rows = sample_tissue_class(50)
    #for each class, building an individual graph
    for ax, label in zip(axes.flat, tissueClasses):
        class_images = feature_rows[feature_rows["label"] == label]
        histograms = [[], [], []] #one distribution per color value (red, green, blue)
        for _, row in class_images.iterrows():
            image_path = data_dir / row["path"]
            with Image.open(image_path) as image:
                pixels = np.asarray(image.convert("RGB"))
            for colorChannel in range(3):
                counts, _ = np.histogram(
                    pixels[:, :, colorChannel],
                    bins=bins,
                    range=(0, 256)
                )
                histograms[colorChannel].append(counts / counts.sum())
        #for each color channel, creating a normalized histogram
        for colorChannel in range(3):
            if histograms[colorChannel]:
                average_histogram = np.mean(histograms[colorChannel], axis=0)
                ax.plot(
                    bins[:-1],
                    average_histogram,
                    color=channel_colors[colorChannel],
                    label=channel_names[colorChannel]
                )
        ax.set_title(label)
        ax.set_xlim(0, 255)
        ax.set_xlabel("Color Channel intensity (0–255)")
        ax.set_ylabel("Average proportion of pixels")
        ax.legend()
    fig.suptitle("Average RGB intensity distributions by tissue class")
    fig.tight_layout()
    fig.savefig(
        "figures/rgb_distributions_by_class.png",
        dpi=160,
        bbox_inches="tight"
    )
    plt.close(fig)

#each method will produce one figure png file under the figures folder
build_montage()
build_mean_brightness_figure()
build_color_intensity_distribution_figure()
