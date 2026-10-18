from zenodo_get import download
from zipfile import ZipFile

def extract():
    download("1214456", output_dir="./full", retry_attempts=99, timeout=300.0,
    file_glob="*K.zip")
    for name in ("CRC-VAL-HE-7K.zip", "NCT-CRC-HE-100K.zip"):
        with ZipFile("./full/" + name, "r") as zip:
            zip.extractall("./full")

extract()
