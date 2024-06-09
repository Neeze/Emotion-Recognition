import numpy as np
import argparse
from sklearn.neural_network import MLPClassifier
from src.haar import extract_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Emotion Recognition Model")
    parser.add_argument("--image", required=True, help="Path to the image")
    parser.add_argument("--model", required=True, help="Path to the model parameters .npy file")
    args = parser.parse_args()
    
    
    model = MLPClassifier() 
    loaded_params = np.load(args.model, allow_pickle=True).item()
    class_labels = loaded_params['class']
    model.set_params(**loaded_params['params'])
    model.coefs_ = loaded_params['coefs_']
    model.intercepts_ = loaded_params['intercepts_']
    model.n_layers_ = loaded_params['n_layers_']
    model.n_outputs_ = loaded_params['n_outputs_']
    model.out_activation_ = loaded_params['output_activation_']
    model._label_binarizer = loaded_params['label_binarizer']
    print("Model loaded successfully!")
    
    # Model Evaluation
    img_path = args.image
    new_features = extract_features(img_path)
    predicted_label = model.predict([new_features])[0]
    print(f"Predicted emotion for new image: {class_labels[predicted_label]}")
