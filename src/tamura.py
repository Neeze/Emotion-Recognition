import numpy as np
import cv2
from scipy.ndimage import sobel

def calculate_coarseness(image: np.ndarray) -> float:
    """
    Calculates the coarseness feature of an image using Tamura's method.

    Args:
        image (np.ndarray): The input grayscale image.

    Returns:
        float: The coarseness value of the image.
    """
    best_size = 1
    best_average = 0
    for size in range(1, 7):  # Iterate over different window sizes (1 to 6)
        shifted_right = np.roll(image, size, axis=1)
        shifted_down = np.roll(image, size, axis=0)

        average_diff_right = np.abs(image - shifted_right).mean()
        average_diff_down = np.abs(image - shifted_down).mean()

        average = (average_diff_right + average_diff_down) / 2

        if average > best_average:
            best_average = average
            best_size = size

    return best_size

def calculate_contrast(image: np.ndarray) -> float:
    """
    Calculates the contrast feature of an image using Tamura's method.

    Args:
        image (np.ndarray): The input grayscale image.

    Returns:
        float: The contrast value of the image.
    """
    # Calculate local standard deviation (alpha)
    alpha = np.std(image, ddof=1)  # ddof=1 for sample standard deviation

    # Calculate average intensity (mu)
    mu = np.mean(image)

    # Calculate contrast (fourth root of alpha / fourth root of mu)
    contrast = (alpha ** (1/4)) / (mu ** (1/4))

    return contrast

def calculate_directionality(image: np.ndarray) -> float:
    """
    Calculates the directionality feature of an image using Tamura's method.

    Args:
        image (np.ndarray): The input grayscale image.

    Returns:
        float: The directionality value of the image.
    """
    # Calculate gradients using Sobel operator
    gx = sobel(image, axis=1)
    gy = sobel(image, axis=0)

    # Calculate gradient magnitude and direction
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    gradient_direction = np.arctan2(gy, gx)

    # Quantize gradient direction into 16 bins
    bins = np.linspace(-np.pi, np.pi, 17)
    bin_indices = np.digitize(gradient_direction, bins) - 1

    # Calculate histogram of gradient directions
    hist, _ = np.histogram(bin_indices, bins=16, range=(0, 15), weights=gradient_magnitude)

    # Calculate directionality (sharpness of the histogram peak)
    directionality = 1 - (hist.max() / hist.sum())

    return directionality

def calculate_line_likeness(image: np.ndarray) -> float:
    """
    Calculates the line-likeness feature of an image using Tamura's method.

    Args:
        image (np.ndarray): The input grayscale image.

    Returns:
        float: The line-likeness value of the image.
    """
    # Implementation is complex and not included here
    raise NotImplementedError("Line-likeness calculation is not currently implemented.")

def calculate_regularity(image: np.ndarray) -> float:
    """
    Calculates the regularity feature of an image using Tamura's method.

    Args:
        image (np.ndarray): The input grayscale image.

    Returns:
        float: The regularity value of the image.
    """
    # Implementation is complex and not included here
    raise NotImplementedError("Regularity calculation is not currently implemented.")

def calculate_roughness(image: np.ndarray) -> float:
    """
    Calculates the roughness feature of an image using Tamura's method.

    Args:
        image (np.ndarray): The input grayscale image.

    Returns:
        float: The roughness value of the image.
    """
    coarseness = calculate_coarseness(image)
    contrast = calculate_contrast(image)
    roughness = coarseness + contrast
    return roughness
