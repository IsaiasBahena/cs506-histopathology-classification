from zenodo_get import download
from zipfile import ZipFile

download("1214456", output_dir="./full", file_glob="*K.zip")
for name in ("CRC-VAL-HE-7K.zip", "NCT-CRC-HE-100K.zip"):
    with ZipFile("./full/" + name, "r") as zip:
        zip.extractall("./full")
