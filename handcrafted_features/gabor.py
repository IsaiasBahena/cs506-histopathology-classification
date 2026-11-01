
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from skimage.filters import gabor_kernel


GABOR_FREQUENCIES = (0.05, 0.10, 0.20)

GABOR_ORIENTATIONS = (
    0.0,
    np.pi / 4,
    np.pi / 2,
    3 * np.pi / 4,
)

GABOR_KERNELS = [
    (frequency, orientation, gabor_kernel(frequency, theta=orientation))
    for frequency in GABOR_FREQUENCIES
    for orientation in GABOR_ORIENTATIONS
]


def extract_gabor(image):
    """Extract multi-frequency, multi-orientation Gabor features."""
    if isinstance(image, Image.Image):
        grayscale = np.asarray(
            image.convert("L"), dtype=np.float32
        )
    else:
        grayscale = np.asarray(
            Image.fromarray(np.asarray(image)).convert("L"),
            dtype=np.float32,
        )

    grayscale /= 255.0

    features = []

    for frequency, orientation, kernel in GABOR_KERNELS:
        real = fftconvolve(
            grayscale,
            np.real(kernel),
            mode="same",
        )

        imaginary = fftconvolve(
            grayscale,
            np.imag(kernel),
            mode="same",
        )

        magnitude = np.hypot(real, imaginary)

        features.append(float(magnitude.mean()))
        features.append(float(magnitude.std()))

    return np.asarray(features, dtype=np.float32)


def get_gabor_feature_names():
    """Return Gabor feature names in extraction order."""
    names = []

    for frequency in GABOR_FREQUENCIES:
        for orientation_index, _ in enumerate(GABOR_ORIENTATIONS):
            prefix = (
                f"gabor_f{frequency:.2f}_"
                f"theta{orientation_index * 45}"
            )

            names.append(f"{prefix}_mean")
            names.append(f"{prefix}_std")

    return names
