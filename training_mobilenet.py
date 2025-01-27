import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU,
    GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
import matplotlib.pyplot as plt


# MobileNet-inspired architecture
def build_mobilenet_like_model(input_shape, num_classes=1, dropout_rate=0.8, l2_weight=0.03):
    """
    Builds a lightweight MobileNet-inspired model with depthwise separable convolutions.
    """
    inputs = Input(shape=input_shape)

    # Initial Conv2D layer
    x = Conv2D(16, (3, 3), strides=(2, 2), padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2_weight))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Depthwise separable convolution blocks
    for filters in [32, 64, 128]:
        # Depthwise convolution (no regularization here)
        x = DepthwiseConv2D((3, 3), strides=(2, 2), padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Pointwise convolution (with kernel_regularizer)
        x = Conv2D(filters, (1, 1), strides=(1, 1), padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2_weight))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    # Global Average Pooling and Fully Connected Layers
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)  # Increased dropout to prevent overfitting
    outputs = Dense(num_classes, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(l2_weight))(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower initial learning rate
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# Function to pad or trim features to a fixed shape
def pad_or_trim_features(feature, target_length=300):
    if feature.shape[1] > target_length:
        feature = feature[:, :target_length]
    elif feature.shape[1] < target_length:
        padding = target_length - feature.shape[1]
        feature = np.pad(feature, ((0, 0), (0, padding)), mode="constant")
    return feature


# Preprocess and augment features
def load_features(features_dir, target_length=300, augment=False):
    X, y = [], []
    augmenter = Compose([
        AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.1, p=0.7),  # Increased noise variability
        TimeStretch(min_rate=0.7, max_rate=1.3, p=0.5),
        PitchShift(min_semitones=-5, max_semitones=5, p=0.5),
    ])

    for label, class_dir in enumerate(["positive", "negative"]):
        class_path = os.path.join(features_dir, class_dir)
        for filename in os.listdir(class_path):
            if filename.endswith(".npy"):
                feature = np.load(os.path.join(class_path, filename))
                feature = pad_or_trim_features(feature, target_length=target_length)

                # Apply augmentation
                if augment:
                    feature = augmenter(samples=feature.astype(np.float32), sample_rate=16000)

                X.append(feature)
                y.append(label)
    return np.array(X), np.array(y)


# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.title('Training and Validation Metrics')
    plt.grid(True)
    plt.show()


# Main training function
def main():
    # Paths to features
    train_dir = "features/train"
    val_dir = "features/validation"
    test_dir = "features/test"

    # Load data
    print("Loading training data with augmentation...")
    X_train, y_train = load_features(train_dir, augment=True)
    print("Loading validation data...")
    X_val, y_val = load_features(val_dir)
    print("Loading test data...")
    X_test, y_test = load_features(test_dir)

    # Add channel dimension
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Compute class weights
    y_train = y_train.astype(int)
    classes = np.unique(y_train).astype(int)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(enumerate(class_weights))

    # Build model
    input_shape = X_train.shape[1:]
    model = build_mobilenet_like_model(input_shape, dropout_rate=0.8, l2_weight=0.03)

    # Callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    csv_logger = CSVLogger("training_log.csv", append=True)

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr, csv_logger]
    )

    # Plot history
    plot_training_history(history)

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    model.save("models/mobilenet_wake_word.h5")
    print("Model saved to 'models/mobilenet_wake_word.h5'")


if __name__ == "__main__":
    main()
