# Visual Exploration
## Follow Data Instructions
Before creating visual figures, you must follow the steps of the README in the data folder to install the required data.
Be sure to download the 1% sample, as this is what will be used in the initial figures.
## Setup
```
pip install -r requirements.txt
```
Within the visual_exploration folder, download the requirements by running the following command.
## Building Initial Figures
```
python3 build_initial_figures.py
```
Within the visual_exploration folder, run the python file. After doing so, if not already created, a folder named figures will be created under visual_exploration, 
containing the following images.
1. **tissue_montage.png**, an image representing 3 samples from each of the 9 tissue categories, created using a reproducable role to provide the same samples. 
2. **mean_brightness_by_tissue_class.png**, a graph displaying the mean brightness of each images by evaluating the mean pixel intensity
3. **rgb_distributions_by_class.png**, a graph for each of the 9 tissue classes, displaying a histogram for each channel (red, green blue)
