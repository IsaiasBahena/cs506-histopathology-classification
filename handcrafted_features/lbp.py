import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern


LBP_SCALES = (
    (8, 1),
    (16, 2),
    (24, 3)
)

LBP_METHOD = "uniform"


def extract_lbp(image):
    """
    Extract multi-scale uniform LBP features from one RGB image.
    
    Parameters
    ----------
    image : PIL.Image.Image or numpy.ndarray
        Input RGB image.
        
    Returns
    -------
    numpy.ndarray
        LBP feature vector of shape (54,).
    """
    if isinstance(image, Image.Image):
        grayscale = np.asarray(image.convert("L"))
    else:
        grayscale = np.asarray(
            Image.fromarray(np.asarray(image)).convert("L")
        )

    features = []

    for neighbors, radius in LBP_SCALES:
        lbp = local_binary_pattern(
            grayscale,
            P=neighbors,
            R=radius,
            method=LBP_METHOD
        )

        num_bins = neighbors + 2

        histogram, _ = np.histogram(
            lbp,
            bins=np.arange(num_bins + 1)
        )

        histogram = histogram.astype(np.float32)
        histogram /= histogram.sum()

        features.extend(histogram)

    return np.asarray(features, dtype=np.float32)


def get_lbp_feature_names():
    """Return LBP feature names in extraction order."""
    names = []

    for neighbors, radius in LBP_SCALES:
        for bin_index in range(neighbors + 2):
            names.append(
                f"lbp_p{neighbors}_r{radius}_hist_{bin_index:03d}"
            )

    return names
