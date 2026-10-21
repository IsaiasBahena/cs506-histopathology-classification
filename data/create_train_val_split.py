import csv
import pandas
from sklearn import model_selection

def create_train_val_split():
    data = pandas.read_csv("full/image_metadata.csv")
    data = data[data["dataset"] == "NCT-CRC-HE-100K" ]
    classes = data["label"]
    split = model_selection.train_test_split(data, test_size=0.2, random_state=4266457447, stratify=classes)
    split[0].to_csv("full/train.csv", index=False)
    split[1].to_csv("full/val.csv", index=False)

create_train_val_split()
