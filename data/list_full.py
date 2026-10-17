from remotezip import RemoteZip

def list_train():
    url = "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip"
    with RemoteZip(url, initial_buffer_size=12*1024*1024) as zip:
        infolist = zip.infolist()
        with open("100k_list.txt", "w") as f:
            for zip_info in infolist:
                f.write(zip_info.filename + "\n")

def list_test():
    url = "https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip"
    with RemoteZip(url, initial_buffer_size=1024*1024) as zip:
        infolist = zip.infolist()
        with open("7k_list.txt", "w") as f:
            for zip_info in infolist:
                f.write(zip_info.filename + "\n")

list_train()
list_test()
