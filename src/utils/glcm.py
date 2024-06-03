import numpy as np
from skimage.feature import graycomatrix, graycoprops

def calculate_energy(glcm: np.ndarray) -> np.ndarray:
    """
    Calculates the energy (angular second moment) feature from a GLCM.

    Args:
        glcm (np.ndarray): The gray-level co-occurrence matrix.

    Returns:
        np.ndarray: The energy values for each angle and distance combination in the GLCM.
    """
    return graycoprops(glcm, 'energy')


def calculate_contrast(glcm: np.ndarray) -> np.ndarray:
    """
    Calculates the contrast feature from a GLCM.

    Args:
        glcm (np.ndarray): The gray-level co-occurrence matrix.

    Returns:
        np.ndarray: The contrast values for each angle and distance combination in the GLCM.
    """
    return graycoprops(glcm, 'contrast')

def calculate_homogeneity(glcm: np.ndarray) -> np.ndarray:
    """
    Calculates the homogeneity (inverse difference moment) feature from a GLCM.

    Args:
        glcm (np.ndarray): The gray-level co-occurrence matrix.

    Returns:
        np.ndarray: The homogeneity values for each angle and distance combination in the GLCM.
    """
    return graycoprops(glcm, 'homogeneity')

def calculate_correlation(glcm: np.ndarray) -> np.ndarray:
    """
    Calculates the correlation feature from a GLCM.

    Args:
        glcm (np.ndarray): The gray-level co-occurrence matrix.

    Returns:
        np.ndarray: The correlation values for each angle and distance combination in the GLCM.
    """
    return graycoprops(glcm, 'correlation')