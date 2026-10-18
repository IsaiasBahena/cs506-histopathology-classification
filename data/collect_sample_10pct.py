from itertools import chain
from remotezip import RemoteZip

def extract_nct_10pct():
    url = "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip"
    with RemoteZip(url, initial_buffer_size=12*1024*1024) as zip:
        infolist = zip.infolist()
        for zip_info in chain(infolist[:1042], infolist[10409:11466],
        infolist[20976:22128], infolist[32489:33645], infolist[44047:44937],
        infolist[52944:54298], infolist[66481:67358], infolist[75245:76290],
        infolist[85692:87124]):
            zip.extract(zip_info.filename, path="./sample_10pct")

def extract_val_10pct():
    url = "https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip"
    with RemoteZip(url, initial_buffer_size=1024*1024) as zip:
        infolist = zip.infolist()
        for zip_info in chain(infolist[:135], infolist[1340:1425],
        infolist[2188:2222], infolist[2528:2592], infolist[3163:3267],
        infolist[4199:4259], infolist[4792:4867], infolist[5534:5577],
        infolist[5956:6080]):
            zip.extract(zip_info.filename, path="./sample_10pct")

extract_nct_10pct()
extract_val_10pct()
