from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample_1pct"

def count_1pct():
    with open(SAMPLE_DIR / "count_1pct.txt", "w") as f:
        for name in ("NCT-CRC-HE-100K", "CRC-VAL-HE-7K"):
            total_count = 0
            p = Path(SAMPLE_DIR / name)
            f.write(name + " image count (1pct sample)\n")
            for child in sorted(p.iterdir()):
                count = 0
                for file in child.iterdir():
                    if file.is_file():
                        count += 1
                        total_count += 1
                f.write(f"{child.name}: {count}\n")
            f.write(f"Total: {total_count}\n\n")

count_1pct()
