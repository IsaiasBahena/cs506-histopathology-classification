import csv
from pathlib import Path
from PIL import Image

def build_metadata():
    with open("full/image_metadata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("id", "path", "label", "dataset", "width", "height",
        "color_mode"))
        for name in ("NCT-CRC-HE-100K", "CRC-VAL-HE-7K"):
            p = Path("./full/" + name)
            for child in sorted(p.iterdir()):
                for file in sorted(child.iterdir()):
                    image = Image.open(file)
                    writer.writerow((file.name, file, child.name, p.name,
                    image.size[0], image.size[1], image.mode))

build_metadata()
