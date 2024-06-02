import cv2
import numpy as np
from scipy.stats import kurtosis, skew


class StatisticalFeatureExtractor:
    """
    A class for extracting statistical features from images.
    """

    def extract_features(self, image_path: str) -> dict[str, float]:
        """
        Extracts various statistical features from an image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            dict[str, float]: A dictionary containing the calculated statistical features.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at: {image_path}")

        features = {
            "mean": np.mean(image),
            "std_dev": np.std(image),
            "variance": np.var(image),
            "kurtosis": kurtosis(image, axis=None),
            "skewness": skew(image, axis=None),
            "min": np.min(image),
            "max": np.max(image),
            "median": np.median(image),
        }
        return features


if __name__ == "__main__":
    extractor = StatisticalFeatureExtractor()
    features = extractor.extract_features("demo/cat.jpg")
    for key, value in features.items():
        print(f"{key}: {value}")