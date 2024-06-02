import cv2
import numpy as np
from skimage.feature import graycomatrix
from glcm import calculate_energy, calculate_contrast, calculate_homogeneity, calculate_correlation
from tamura import calculate_coarseness, calculate_contrast, calculate_directionality, calculate_roughness

class TextureFeatureExtractor:
    """
    A class for extracting texture features from images using both GLCM and Tamura methods.
    """

    def extract_glcm_features(self, image_path: str, distances: list[int] = [1], angles: list[float] = [0]) -> dict[str, np.ndarray]:
        """
        Extracts GLCM (Gray-Level Co-occurrence Matrix) features from an image.

        Args:
            image_path (str): The path to the image file.
            distances (list[int], optional): List of pixel pair distances for GLCM calculation. Defaults to [1].
            angles (list[float], optional): List of angles (in radians) for GLCM calculation. Defaults to [0].

        Returns:
            dict[str, np.ndarray]: A dictionary containing the calculated GLCM features.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at: {image_path}")

        glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)

        features = {
            "energy": calculate_energy(glcm),
            "contrast": calculate_contrast(glcm),
            "homogeneity": calculate_homogeneity(glcm),
            "correlation": calculate_correlation(glcm)
        }
        return features

    def extract_tamura_features(self, image_path: str) -> dict[str, float]:
        """
        Extracts Tamura texture features from an image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            dict[str, float]: A dictionary containing the calculated Tamura features.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at: {image_path}")

        features = {
            "coarseness": calculate_coarseness(image),
            "contrast": calculate_contrast(image),
            "directionality": calculate_directionality(image),
            "roughness": calculate_roughness(image)
        }
        return features


if __name__=="__main__":
    texture_extractor = TextureFeatureExtractor()
    image_path = "demo/cat.jpg"

    # GLCM
    glcm_features = texture_extractor.extract_glcm_features(image_path)
    print("GLCM Features:")
    for feature, value in glcm_features.items():
        print(f"{feature}: {value}")

    # Tamura
    tamura_features = texture_extractor.extract_tamura_features(image_path)
    print("\nTamura Features:")
    for feature, value in tamura_features.items():
        print(f"{feature}: {value}")