# Data collection
## Session setup
Every session, cd into the "data" directory and run
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
You can download a 1% sample or the full dataset.

## 1% sample
```
python3 collect_sample_1pct.py
```
This will create a "sample_1pct" directory containing sample NCT/VAL directories. These contain sample class directories. Each sample class directory contains the first 1% of its original images.

```
python3 count_sample_1pct.py
```
This will add "count_1pct.txt" to the "sample_1pct" directory. This contains class/total image counts for the sample NCT/VAL directories.

```
python3 build_metadata_1pct.py
```
This will add "image_metadata_1pct.csv" to the "sample_1pct" directory. This contains the id, path, class, dataset source, width, height, and color mode of every sample image.

## Full dataset
```
python3 collect_full.py
```
This will create a "full" directory containing NCT/VAL directories. These contain class directories.

```
python3 count_full.py
```
This will add "count_full.txt" to the "full" directory. This contains class/total image counts for the NCT/VAL directories.

```
python3 build_metadata_full.py
```
This will add "image_metadata.csv" to the "full" directory. This contains the id, path, class, dataset source, width, height, and color mode of every image.

```
python3 create_train_val_split.py
```
This will add "train.csv" and "val.csv" to the "full" directory. These are the 80% train, 20% validation sets created using a seeded, stratified split on the NCT directory. The VAL directory is reserved for final testing.

## Citations
David Völgyes. (2020, February 20). Zenodo_get: a downloader for Zenodo records (Version 3.1.0). Zenodo. https://doi.org/10.5281/zenodo.1261812

Kather, J. N., Halama, N., & Marx, A. (2018). 100,000 histological images of human colorectal cancer and healthy tissue (Version v0.1) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.1214456
