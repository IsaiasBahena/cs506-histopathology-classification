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
