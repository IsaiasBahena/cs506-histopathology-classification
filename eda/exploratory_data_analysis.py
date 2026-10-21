import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tifffile

# Locate the project directory and metadata file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "sample_1pct" / "image_metadata_1pct.csv"

# Load image metadata
df = pd.read_csv(DATA_PATH)

# Use only the NCT dataset for exploratory analysis
df = df[df["dataset"] == "NCT-CRC-HE-100K"]

# Display basic information
print("Number of images:", len(df))
print("\nFirst five rows:")
print(df.head())

# Calculate class counts and percentages
class_counts = df["label"].value_counts().sort_index()
class_percentages = class_counts / len(df) * 100

# Create a summary table
class_summary = pd.DataFrame({
    "Count": class_counts,
    "Percentage": class_percentages.round(2)
})

print("\nClass distribution:")
print(class_summary)

# Create output directory
OUTPUT_DIR = PROJECT_ROOT / "eda" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot class distribution
plt.figure(figsize=(10, 6))

bars = plt.bar(class_counts.index, class_counts.values)

plt.title("Tissue Class Distribution (1% NCT Sample)")
plt.xlabel("Tissue Class")
plt.ylabel("Number of Images")

# Add counts above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        str(int(height)),
        ha="center"
    )

plt.tight_layout()

# Save figure and table
plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=300)
class_summary.to_csv(OUTPUT_DIR / "class_distribution.csv")

plt.close()

print("\nClass distribution figure and table saved successfully.")

print("DEBUG: Starting image statistics analysis")

# Calculate image brightness statistics
def calculate_image_statistics(image_path):
    image = tifffile.imread(image_path).astype(np.float32)

    # Mean intensity for each color channel
    mean_rgb = image.mean(axis=(0, 1))

    # Convert RGB values to perceived brightness
    brightness = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    )

    return {
        "mean_red": mean_rgb[0],
        "mean_green": mean_rgb[1],
        "mean_blue": mean_rgb[2],
        "mean_brightness": brightness.mean(),
        "brightness_std": brightness.std()
    }


# Analyze images in the NCT sample
image_statistics = []

for _, row in df.iterrows():
    image_path = PROJECT_ROOT / "data" / row["path"]

    stats = calculate_image_statistics(image_path)

    stats["image_id"] = row["id"]
    stats["label"] = row["label"]

    image_statistics.append(stats)

# Convert results into a table
stats_df = pd.DataFrame(image_statistics)

# Save image statistics
stats_df.to_csv(OUTPUT_DIR / "image_statistics.csv", index=False)

print("\nImage statistics calculated successfully.")
print(stats_df.head())

# Compare image brightness across tissue classes
brightness_data = [
    stats_df.loc[
        stats_df["label"] == label, "mean_brightness"
    ].values
    for label in class_counts.index
]

plt.figure(figsize=(11, 6))

plt.boxplot(
    brightness_data,
    tick_labels=class_counts.index,
    patch_artist=True,
    boxprops={"facecolor": "lightblue"},
    medianprops={"color": "darkred", "linewidth": 2}
)

plt.title("Brightness Distribution by Tissue Class (1% NCT Sample)")
plt.xlabel("Tissue Class")
plt.ylabel("Mean Image Brightness (0-255)")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "brightness_distribution.png", dpi=300)
plt.close()

print("\nBrightness distribution figure saved successfully.")

# Compare brightness variability across tissue classes
variability_data = [
    stats_df.loc[
        stats_df["label"] == label, "brightness_std"
    ].values
    for label in class_counts.index
]

plt.figure(figsize=(11, 6))

plt.boxplot(
    variability_data,
    tick_labels=class_counts.index,
    patch_artist=True,
    boxprops={"facecolor": "lightgreen"},
    medianprops={"color": "darkred", "linewidth": 2}
)

plt.title("Brightness Variability by Tissue Class (1% NCT Sample)")
plt.xlabel("Tissue Class")
plt.ylabel("Within-Image Brightness Standard Deviation")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "brightness_variability.png", dpi=300)
plt.close()

print("\nBrightness variability figure saved successfully.")