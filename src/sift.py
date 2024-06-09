import cv2
from skimage.transform import resize
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def extract_features(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        return np.array([])
    return descriptors.flatten()  # Flatten the SIFT descriptors into a 1D array


def extract_features_and_labels(img_paths, labels, max_length=10000):
    with ThreadPoolExecutor() as executor:
        features = list(tqdm(executor.map(extract_features, img_paths), total=len(img_paths), desc="Extracting features"))

    max_length = max_length
    padded_features = [np.pad(feature, (0, max_length - len(feature)), 'constant') if feature.size > 0 else np.zeros(max_length) for feature in features]

    return np.array(padded_features), np.array(labels)


    