# Author: Nomi Anwar

from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROC_DIR = PROJECT_ROOT / "preprocessing"
OUTPUT_DIR = PREPROC_DIR / "output"
FIGURES_DIR = PREPROC_DIR / "figures"
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample_1pct"

tissue_classes = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
state = 1 #fixed seed so the same example images are chosen every run
original_size = 224
sizes = [64, 32] #downsampled resolutions to compare against the original
full_nct_count = 100000 #size of NCT-CRC-HE-100K, used to project memory needs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
metadata = pd.read_csv(SAMPLE_DIR / "image_metadata_1pct.csv")
metadata = metadata[((metadata["dataset"] == "NCT-CRC-HE-100K") & metadata["label"].isin(tissue_classes))]
manifest = metadata.sort_values("id").reset_index(drop=True)

# BOX averages each block of pixels
def downsample(image, size):
    return image.resize((size, size), Image.BOX)

# downsample all the images, flatten them to vectors and save one matrix per size
def build_pixel_matrices():
    vectors = {size: [] for size in sizes}
    for _, row in manifest.iterrows():
        with Image.open(DATA_DIR / row["path"]) as image:
            rgb = image.convert("RGB")
        for size in sizes:
            vector = np.asarray(downsample(rgb, size), dtype=np.uint8).reshape(-1)
            assert vector.shape[0] == size * size * 3
            vectors[size].append(vector)
    for size in sizes:
        x = np.stack(vectors[size])
        np.savez_compressed(
            OUTPUT_DIR / f"pixels_{size}.npz",
            X=x,
            image_id=manifest["id"].to_numpy(dtype=str),
            label=manifest["label"].to_numpy(dtype=str)
        )
        print(f"{size}x{size}: X shape {x.shape}, dtype {x.dtype}")

# one example per class shown at the original and each downsampled resolution
def build_comparison_figure():
    examples = pd.concat(
        [manifest[manifest["label"] == label].sample(n=1, random_state=state) for label in tissue_classes]
    )
    columns = [original_size] + sizes
    fig, axes = plt.subplots(
        nrows=len(tissue_classes), ncols=len(columns),
        figsize=(3 * len(columns), 2.6 * len(tissue_classes)),
        squeeze=False
    )
    for row, (_, example) in enumerate(examples.iterrows()):
        with Image.open(DATA_DIR / example["path"]) as image:
            rgb = image.convert("RGB")
        for col, size in enumerate(columns):
            ax = axes[row, col]
            shown = rgb if size == original_size else downsample(rgb, size)
            ax.imshow(shown, interpolation="nearest") # nearest keeps the pixelation visible for visualization
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{size}x{size}", fontsize=14)
            if col == 0:
                ax.set_ylabel(example["label"], fontsize=14, weight="bold")
    fig.suptitle("Original vs downsampled images by tissue class", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(FIGURES_DIR / "downsampling_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

def build_size_table():
    rows = []
    for size in [original_size] + sizes:
        features = size * size * 3
        rows.append({
            "resolution": f"{size}x{size}",
            "array_shape": f"({size}, {size}, 3)",
            "features": features,
            "bytes_per_image_uint8": features,
            "bytes_per_image_float32": features * 4,
            "sample_matrix_mb_float32": round(len(manifest) * features * 4 / 1e6, 1),
            "full_nct_gb_float32": round(full_nct_count * features * 4 / 1e9, 2)
        })
    table = pd.DataFrame(rows)
    table.to_csv(FIGURES_DIR / "representation_sizes.csv", index=False)
    print(table.to_string(index=False))
    average_kb = manifest["path"].apply(lambda p: os.path.getsize(DATA_DIR / p)).mean() / 1e3
    print(f"Average original .tif size on disk: {average_kb:.1f} KB")

build_pixel_matrices()
build_comparison_figure()
build_size_table()
