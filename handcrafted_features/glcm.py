
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops


GLCM_DISTANCES = (1, 2, 4)
GLCM_ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)

GLCM_PROPERTIES = (
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "ASM",
)


def extract_glcm(image, levels=32):
    """
    Extract GLCM texture features from one RGB image.

    Parameters
    ----------
    image : PIL.Image.Image or numpy.ndarray
        Input RGB image.
    levels : int
        Number of quantized grayscale levels.

    Returns
    -------
    numpy.ndarray
        GLCM feature vector of shape (36,).
    """
    if not 2 <= levels <= 256:
        raise ValueError("GLCM levels must be between 2 and 256.")

    if isinstance(image, Image.Image):
        grayscale = np.asarray(image.convert("L"))
    else:
        rgb = Image.fromarray(np.asarray(image)).convert("L")
        grayscale = np.asarray(rgb)

    # Quantize grayscale intensities to [0, levels - 1].
    quantized = (
        grayscale.astype(np.uint16) * levels // 256
    ).astype(np.uint8)

    glcm = graycomatrix(
        quantized,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    # Calculate each property once for all distances and angles.
    property_values = {
        prop: graycoprops(glcm, prop)
        for prop in GLCM_PROPERTIES
    }

    features = []

    for distance_index in range(len(GLCM_DISTANCES)):
        for prop in GLCM_PROPERTIES:
            values = property_values[prop][distance_index]

            features.append(values.mean())
            features.append(values.std())

    return np.asarray(features, dtype=np.float32)


def get_glcm_feature_names():
    """Return feature names in extraction order."""
    names = []

    for distance in GLCM_DISTANCES:
        for prop in GLCM_PROPERTIES:
            name = prop.lower()

            names.append(f"glcm_d{distance}_{name}_mean")
            names.append(f"glcm_d{distance}_{name}_std")

    return names
