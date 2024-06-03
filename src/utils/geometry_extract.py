import cv2
import numpy as np

class GeometricFeatureExtractor:
    """
    A class for extracting geometric features from binary images (e.g., masks or segmented objects).
    """

    def extract_features(self, image: np.ndarray) -> dict[str, float]:
        """
        Extracts various geometric features from a binary image.

        Args:
            image (np.ndarray): The input binary image as a NumPy array.

        Returns:
            dict[str, float]: A dictionary containing the calculated geometric features.
        """
        # Threshold to ensure binary image
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assume the largest contour is the object of interest
        largest_contour = max(contours, key=cv2.contourArea)

        features = {
            "area": cv2.contourArea(largest_contour),
            "perimeter": cv2.arcLength(largest_contour, True),
            "centroid": np.mean(largest_contour, axis=0).flatten(),
            # Additional features
            "equivalent_diameter": np.sqrt(4 * cv2.contourArea(largest_contour) / np.pi),
            "extent": cv2.contourArea(largest_contour) / (image.shape[0] * image.shape[1]),
            "solidity": cv2.contourArea(largest_contour) / cv2.contourArea(cv2.convexHull(largest_contour)),
        }

        # Calculate aspect ratio (width / height)
        x, y, w, h = cv2.boundingRect(largest_contour)
        features["aspect_ratio"] = float(w) / h if h != 0 else 0  # Avoid division by zero

        return features
    

if __name__ == "__main__":
    extractor = GeometricFeatureExtractor()
    image_path = "demo/cat.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    features = extractor.extract_features(image)
    for key, value in features.items():
        print(f"{key}: {value}")