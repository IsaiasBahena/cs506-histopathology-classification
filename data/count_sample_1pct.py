from pathlib import Path

def count_train_1pct():
    total_count = 0
    classes = []
    p = Path("./sample_1pct/NCT-CRC-HE-100K")
    with open("sample_1pct/count_1pct.txt", "w") as f:
        f.write("Image count of 1% train sample\n")
        for child in sorted(p.iterdir()):
            count = 0
            for file in child.iterdir():
                if file.is_file():
                    count += 1
                    total_count += 1
            f.write(f"{child.name}: {count}\n")
        f.write(f"Total: {total_count}\n\n")
    
def count_test_1pct():
    total_count = 0
    classes = []
    p = Path("./sample_1pct/CRC-VAL-HE-7K")
    with open("sample_1pct/count_1pct.txt", "a") as f:
        f.write("Image count of 1% test sample\n")
        for child in sorted(p.iterdir()):
            count = 0
            for file in child.iterdir():
                if file.is_file():
                    count += 1
                    total_count += 1
            f.write(f"{child.name}: {count}\n")
        f.write(f"Total: {total_count}\n")

count_train_1pct()
count_test_1pct()
