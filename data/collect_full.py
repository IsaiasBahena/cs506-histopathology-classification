from pathlib import Path
from zenodo_get import download
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FULL_DIR = DATA_DIR / "full"

def extract():
    download("1214456", output_dir=FULL_DIR, retry_attempts=99, timeout=300.0,
    file_glob="*K.zip")
    for name in ("CRC-VAL-HE-7K.zip", "NCT-CRC-HE-100K.zip"):
        with ZipFile(FULL_DIR / name, "r") as zip:
            zip.extractall(FULL_DIR)

extract()
