Nomi Preprocessing October Check In

run "downsampling_histopathology_images.py" to produce:
    figures folder containing:
        downsampling_comparison.png
        representation_sizes.csv

    output folder containing:
        pixels_32.npz
        pixels_64.npz
        Each one holds the downsampled images as the matrix SVD will use

┌──────────┬─────────────────────────────┬────────────────────────────────────────────────────────┐
│   Key    │            Shape            │                       What it is                       │
├──────────┼─────────────────────────────┼────────────────────────────────────────────────────────┤
│ X        │ (996, 3072) or (996, 12288) │ One row per image: the flattened 32×32 or 64×64 pixels │
├──────────┼─────────────────────────────┼────────────────────────────────────────────────────────┤
│ image_id │ (996,)                      │ Filename for each row, e.g. ADI-AAAMHQMK.tif           │
├──────────┼─────────────────────────────┼────────────────────────────────────────────────────────┤
│ label    │ (996,)                      │ Tissue class for each row, e.g. ADI                    │
└──────────┴─────────────────────────────┴────────────────────────────────────────────────────────┘
