from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FULL_DIR = DATA_DIR / "full"

def count():
    with open(FULL_DIR / "count_full.txt", "w") as f:
        for name in ("NCT-CRC-HE-100K", "CRC-VAL-HE-7K"):
            total_count = 0
            p = Path(FULL_DIR / name)
            f.write(name + " image count\n")
            for child in sorted(p.iterdir()):
                count = 0
                for file in child.iterdir():
                    if file.is_file():
                        count += 1
                        total_count += 1
                f.write(f"{child.name}: {count}\n")
            f.write(f"Total: {total_count}\n\n")
        
count()
