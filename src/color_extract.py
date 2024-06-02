import cv2
import numpy as np

class ColorFeatureExtractor:
    """
    A class for extracting color features from images.
    """

    def color_moments(self, image_path: str) -> tuple[float, float, float]:
        """
        Calculates the first three color moments (mean, standard deviation, and skewness) of an image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            tuple[float, float, float]: A tuple containing the mean, standard deviation, and skewness of the image.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calculate mean
        mean = np.mean(image, axis=(0, 1))

        # Calculate standard deviation
        std_dev = np.std(image, axis=(0, 1))
        
        # Calculate skewness
        skewness = np.mean((((image - mean) / std_dev) ** 3), axis=(0, 1))

        return mean, std_dev, skewness

    def color_histogram(self, image_path: str, bins: int = 256) -> np.ndarray:
        """
        Calculates the color histogram of an image.

        Args:
            image_path (str): The path to the image file.
            bins (int, optional): The number of bins for the histogram. Defaults to 256.

        Returns:
            np.ndarray: The color histogram of the image.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def average_rgb(self, image_path: str) -> np.ndarray:
        """
        Calculates the average RGB value of an image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            np.ndarray: The average RGB value of the image.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        avg_color_per_row = np.average(image, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        return avg_color

if __name__ == "__main__":
    extractor = ColorFeatureExtractor()
    image_path = "demo/cat.jpg"
    mean, std_dev, skewness = extractor.color_moments(image_path)
    hist = extractor.color_histogram(image_path)
    avg_rgb = extractor.average_rgb(image_path)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Skewness: {skewness}")
    print(f"Color Histogram: {hist}")
    print(f"Average RGB: {avg_rgb}")