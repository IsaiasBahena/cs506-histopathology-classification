import csv
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FULL_DIR = DATA_DIR / "full"

def build_metadata():
    with open(FULL_DIR / "image_metadata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("id", "path", "label", "dataset", "width", "height",
        "color_mode"))
        for name in ("NCT-CRC-HE-100K", "CRC-VAL-HE-7K"):
            p = Path(FULL_DIR / name)
            for child in sorted(p.iterdir()):
                for file in sorted(child.iterdir()):
                    image = Image.open(file)
                    writer.writerow((file.name, file, child.name, p.name,
                    image.size[0], image.size[1], image.mode))

build_metadata()
