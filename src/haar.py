import cv2
from skimage.transform import resize
from skimage.feature import haar_like_feature
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    features = []
    # Multi-scale feature extraction
    for size in [16, 24, 32]:  
        resized_img = resize(img, (size, size))
        feature_types = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y']  # Haar types
        haar_features = haar_like_feature(resized_img, 0, 0, size, size, feature_types)
        features.extend(haar_features)
    return features

def extract_features_and_labels(img_paths, labels):
    with ThreadPoolExecutor() as executor:
        features = list(tqdm(executor.map(extract_features, img_paths), total=len(img_paths), desc="Extracting features"))
    return np.array(features), np.array(labels)