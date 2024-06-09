import os
import argparse
import numpy as np
import pandas as pd
from src.haar import extract_features_and_labels
from sklearn.neural_network import MLPClassifier


if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Emotion Recognition Model")
    parser.add_argument("--data", required=True, help="Path to the CSV data file")
    parser.add_argument("--save", default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--test", action="store_true", help="Path to the CSV test data file")
    args = parser.parse_args()

    # Data Loading
    data = pd.read_csv(args.data) 
    image_paths = data["image_path"].values
    
    # Convert labels to integers
    data["label"] = data["label"].astype(str)
    unique_labels = data["label"].unique()
    label_map = {label: i for i, label in enumerate(unique_labels)}
    reverse_label_map = {v: k for k, v in label_map.items()}
    data["label"] = data["label"].map(label_map)
    labels = data["label"].values

    print(f"Number of images: {len(image_paths)}")
    print(f"Number of labels: {len(labels)}")
    print(f"Unique labels: {unique_labels}")
    print(f"Model in use: MLPClassifier")
    
    X, y = extract_features_and_labels(image_paths, labels)

    # Model Training
    print("Training model...")
    model = MLPClassifier(activation='relu', solver='adam', verbose=True)
    model.fit(X, y)
    print("Model trained successfully!")
    
    # Save model parameters
    params = {
        'class': unique_labels,
        'params': model.get_params(),
        'coefs_': model.coefs_,
        'intercepts_': model.intercepts_,
        'n_layers_': model.n_layers_,
        'n_outputs_': model.n_outputs_,
        'output_activation_': model.out_activation_,
        'label_binarizer': model._label_binarizer
    }
    os.makedirs(args.save, exist_ok=True)
    np.save(os.path.join(args.save,'mlp_params.npy'), params)
    print("Model parameters saved successfully!")

    if args.test:
        test_data = pd.read_csv(args.test)
        test_image_paths = test_data["image_path"].values
        test_labels = test_data["label"].astype(str).map(label_map).values
        X_test, y_test = extract_features_and_labels(test_image_paths, test_labels)
        accuracy = model.score(X_test, y_test)
        print(f"Test accuracy: {accuracy}")