# Data collection
## Setup
cd into the "data" directory and run
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

In the future, you can just run
```
source env/bin/activate
```
You can download a 1% sample or the full dataset.

## 1% sample
```
python3 collect_sample_1pct.py
```
This will create a "sample_1pct" directory containing sample train/test directories. These contain class directories. Each class directory contains the first 1% of its original images.

```
python3 count_sample_1pct.py
```
This will add "count_1pct.txt" to the "sample_1pct" directory. This contains class/total image counts for the sample train/test directories.

## Full dataset
```
python3 collect_full.py
```
This will create a "full" directory containing train/test directories. These contain class directories.

```
python3 count_full.py
```
This will add "count_full.txt" to the "full" directory. This contains class/total image counts for the train/test directories.