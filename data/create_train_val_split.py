import csv
import pandas
from pathlib import Path
from sklearn import model_selection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FULL_DIR = DATA_DIR / "full"

def create_train_val_split():
    data = pandas.read_csv(FULL_DIR / "image_metadata.csv")
    data = data[data["dataset"] == "NCT-CRC-HE-100K" ]
    classes = data["label"]
    split = model_selection.train_test_split(data, test_size=0.2, random_state=4266457447, stratify=classes)
    split[0].to_csv(FULL_DIR / "train.csv", index=False)
    split[1].to_csv(FULL_DIR / "val.csv", index=False)

create_train_val_split()
