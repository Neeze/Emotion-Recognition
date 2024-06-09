import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt



if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Emotion Recognition Model")
    parser.add_argument("--data", required=True, help="Path to the CSV data file")
    parser.add_argument("--save", default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--extractor", default='haar', help="Feature extractors ['haar', 'sift'] to use (default: haar)")
    parser.add_argument("--model", default='mlp', help="Model to use  ['mlp', 'svm', 'knn', 'rf'] (default: mlp)")
    parser.add_argument("--test", required=True, help="Path to the CSV test data file")
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
    print(f"Extractor in use: {args.extractor}")
    print(f"Model in use: {args.model}")
    
    # Feature Extraction Initialization
    print("Extracting features...")
    if args.extractor == 'haar':
        from src.haar import extract_features_and_labels
        X, y = extract_features_and_labels(image_paths, labels)
    elif args.extractor == 'sift':
        from src.sift import extract_features_and_labels
        X, y = extract_features_and_labels(image_paths, labels, max_length=8000)
    else:
        raise ValueError("Invalid feature extractor. Please choose from ['haar', 'sift']")
    
    # Model Initialization
    print("Initializing model...")
    if args.model == 'mlp':
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(160,), 
                            max_iter=1000,
                            learning_rate='adaptive', 
                            activation='relu', 
                            solver='adam', 
                            tol=1e-6,
                            verbose=True)
    elif args.model == 'svm':
        from sklearn.svm import SVC
        model = SVC(kernel='linear', 
                    C=1.0, 
                    verbose=True)
    elif args.model == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=2)
    elif args.model == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=160)
    else:
        raise ValueError("Invalid model type. Please choose from ['mlp', 'svm', 'knn', 'rf']")
    
    # Model Training
    print("Training model...")
    model.fit(X, y)
    print("Model trained successfully!")
    
    # Save model parameters
    if args.model =='mlp':
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
        np.save(os.path.join(args.save,f'mlp_params_{args.extractor}.npy'), params)
        print("Model parameters saved successfully!")

    # Model Evaluation
    if args.test:
        test_data = pd.read_csv(args.test)
        test_image_paths = test_data["image_path"].values
        test_labels = test_data["label"].astype(str).map(label_map).values
        
        if args.extractor == 'haar':
            X_test, y_test = extract_features_and_labels(test_image_paths, test_labels)
        elif args.extractor == 'sift':
            X_test, y_test = extract_features_and_labels(test_image_paths, test_labels, max_length=8000)
        accuracy = model.score(X_test, y_test)
        print(f"Test accuracy: {accuracy}")
        
        
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        # Display and Save Confusion Matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap=plt.cm.Blues)
        os.makedirs('evaluation', exist_ok=True)
        plt.title('Confusion Matrix')
        plt.savefig(f'evaluation/confusion_matrix_{args.model}_{args.extractor}.png') #save to file