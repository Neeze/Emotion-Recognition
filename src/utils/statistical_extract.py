import cv2
import numpy as np
from scipy.stats import kurtosis, skew

class StatisticalFeatureExtractor:
    """
    A class for extracting statistical features from images.
    """

    def extract_features(self, image: np.ndarray) -> dict[str, float]:
        """
        Extracts various statistical features from an image.

        Args:
            image (np.ndarray): The input grayscale image as a NumPy array.

        Returns:
            dict[str, float]: A dictionary containing the calculated statistical features.
        """
        if len(image.shape) == 3:  # Check if image is RGB, convert to grayscale if necessary
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        features = {
            "mean": np.mean(image),
            "std_dev": np.std(image),
            "variance": np.var(image),
            "kurtosis": kurtosis(image, axis=None),
            "skewness": skew(image, axis=None),
            "min": np.min(image),
            "max": np.max(image),
            "median": np.median(image),
            # Add more features as needed (e.g., RMS, energy, etc.)
        }
        return features


if __name__ == "__main__":
    image_path = "demo/cat.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    extractor = StatisticalFeatureExtractor()
    features = extractor.extract_features(image)
    for key, value in features.items():
        print(f"{key}: {value}")