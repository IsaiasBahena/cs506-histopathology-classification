from itertools import chain
from remotezip import RemoteZip

def extract_train_1pct():
    url = "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip"
    with RemoteZip(url, initial_buffer_size=12*1024*1024) as zip:
        infolist = zip.infolist()
        for zip_info in chain(infolist[:106], infolist[10409:10515],
        infolist[20976:21092], infolist[32489:32605], infolist[44047:44136],
        infolist[52944:53080], infolist[66481:66569], infolist[75245:75350],
        infolist[85692:85836]):
            zip.extract(zip_info.filename, path="./sample_1pct")

def extract_test_1pct():
    url = "https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip"
    with RemoteZip(url, initial_buffer_size=1024*1024) as zip:
        infolist = zip.infolist()
        for zip_info in chain(infolist[:15], infolist[1340:1349],
        infolist[2188:2192], infolist[2528:2535], infolist[3163:3174],
        infolist[4199:4205], infolist[4792:4800], infolist[5534:5539],
        infolist[5956:5969]):
            zip.extract(zip_info.filename, path="./sample_1pct")

extract_train_1pct()
extract_test_1pct()
