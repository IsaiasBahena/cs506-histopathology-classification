from pathlib import Path
from remotezip import RemoteZip

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def list_nct():
    url = "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip"
    with RemoteZip(url, initial_buffer_size=12*1024*1024) as zip:
        infolist = zip.infolist()
        with open(DATA_DIR / "100k_list.txt", "w") as f:
            for zip_info in infolist:
                f.write(zip_info.filename + "\n")

def list_val():
    url = "https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip"
    with RemoteZip(url, initial_buffer_size=1024*1024) as zip:
        infolist = zip.infolist()
        with open(DATA_DIR / "7k_list.txt", "w") as f:
            for zip_info in infolist:
                f.write(zip_info.filename + "\n")

list_nct()
list_val()
