import cv2
import numpy as np

class ColorFeatureExtractor:
    """
    A class for extracting color features from images.
    """

    def color_moments(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the first three color moments (mean, standard deviation, and skewness) of an image.

        Args:
            image (np.ndarray): The input image as a NumPy array (RGB or BGR).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the mean, standard deviation, and skewness of the image.
        """
        if image.ndim != 3:
            raise ValueError("Input image must be a 3-dimensional array (RGB or BGR).")

        # Ensure image is in RGB format
        if image.shape[2] == 3:  # BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Calculate mean
        mean = np.mean(image, axis=(0, 1))

        # Calculate standard deviation
        std_dev = np.std(image, axis=(0, 1))

        # Calculate skewness
        skewness = np.mean((((image - mean) / std_dev) ** 3), axis=(0, 1))

        return mean, std_dev, skewness

    def color_histogram(self, image: np.ndarray, bins: int = 256) -> np.ndarray:
        """
        Calculates the color histogram of an image.

        Args:
            image (np.ndarray): The input image as a NumPy array (RGB or BGR).
            bins (int, optional): The number of bins for the histogram. Defaults to 256.

        Returns:
            np.ndarray: The color histogram of the image.
        """
        if image.ndim != 3:
            raise ValueError("Input image must be a 3-dimensional array (RGB or BGR).")

        # Ensure image is in RGB format
        if image.shape[2] == 3:  # BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def average_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Calculates the average RGB value of an image.

        Args:
            image (np.ndarray): The input image as a NumPy array (RGB or BGR).

        Returns:
            np.ndarray: The average RGB value of the image.
        """
        if image.ndim != 3:
            raise ValueError("Input image must be a 3-dimensional array (RGB or BGR).")

        # Ensure image is in RGB format
        if image.shape[2] == 3:  # BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        avg_color_per_row = np.average(image, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        return avg_color

if __name__ == "__main__":
    extractor = ColorFeatureExtractor()
    image_path = "demo/cat.jpg"
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    mean, std_dev, skewness = extractor.color_moments(image)
    hist = extractor.color_histogram(image)
    avg_rgb = extractor.average_rgb(image)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Skewness: {skewness}")
    print(f"Color Histogram: {hist}")
    print(f"Average RGB: {avg_rgb}")