from pathlib import Path

def count():
    with open("full/count_full.txt", "w") as f:
        for name in ("NCT-CRC-HE-100K", "CRC-VAL-HE-7K"):
            total_count = 0
            p = Path("./full/" + name)
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
