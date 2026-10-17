from pathlib import Path

def count_train():
    total_count = 0
    classes = []
    p = Path("./full/NCT-CRC-HE-100K")
    with open("full/count_full.txt", "w") as f:
        f.write("Train image count\n")
        for child in sorted(p.iterdir()):
            count = 0
            for file in child.iterdir():
                if file.is_file():
                    count += 1
                    total_count += 1
            f.write(f"{child.name}: {count}\n")
        f.write(f"Total: {total_count}\n\n")
    
def count_test():
    total_count = 0
    classes = []
    p = Path("./full/CRC-VAL-HE-7K")
    with open("full/count_full.txt", "a") as f:
        f.write("Test image count\n")
        for child in sorted(p.iterdir()):
            count = 0
            for file in child.iterdir():
                if file.is_file():
                    count += 1
                    total_count += 1
            f.write(f"{child.name}: {count}\n")
        f.write(f"Total: {total_count}\n")

count_train()
count_test()
