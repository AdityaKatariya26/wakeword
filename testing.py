import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# Function to load features from the test set
def load_features(features_dir, target_length=300):
    X, y = [], []
    for label, class_dir in enumerate(["positive", "negative"]):
        class_path = os.path.join(features_dir, class_dir)
        for filename in os.listdir(class_path):
            if filename.endswith(".npy"):
                feature = np.load(os.path.join(class_path, filename))
                # Pad or trim to ensure consistent input shape
                if feature.shape[1] > target_length:
                    feature = feature[:, :target_length]
                elif feature.shape[1] < target_length:
                    padding = target_length - feature.shape[1]
                    feature = np.pad(feature, ((0, 0), (0, padding)), mode="constant")
                X.append(feature)
                y.append(label)
    return np.array(X), np.array(y)


# Function to evaluate the model
def evaluate_model(model_path, features_dir):
    print("Loading test data...")
    X_test, y_test = load_features(features_dir)
    X_test = X_test[..., np.newaxis]  # Add channel dimension

    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)

    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Predict labels for the test set
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


# Main function
def main():
    model_path = "models/mobilenet_wake_word.h5"  # Path to your trained model
    test_dir = "features/test"  # Path to your test features directory
    evaluate_model(model_path, test_dir)


if __name__ == "__main__":
    main()
