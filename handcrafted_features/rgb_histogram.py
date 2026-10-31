import numpy as np


def extract_rgb_histogram(image, bins=16):
    """
    Extract normalized RGB color histograms from an image.
    
    Parameters
    ----------
    image : PIL.Image.Image or numpy.ndarray
        Input RGB image.
    bins : int
        Number of histogram bins per color channel
        
    Returns
    -------
    numpy.ndarray
        Concatenated RGB histogram feature vector
        with shape (3 * bins,).
    """
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have three RGB channels.")

    histograms = []

    for channel in range(3):
        histogram, _ = np.histogram(
            image[:, :, channel],
            bins=bins,
            range=(0, 256)
        )

        histogram = histogram.astype(np.float32)
        histogram /= histogram.sum()

        histograms.append(histogram)

    return np.concatenate(histograms)
