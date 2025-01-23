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
import librosa
import matplotlib.pyplot as plt


# Build MobileNet-inspired architecture
def build_mobilenet_like_model(input_shape, num_classes=1, dropout_rate=0.6, l2_weight=0.01):
    inputs = Input(shape=input_shape)

    # Initial Conv2D layer
    x = Conv2D(16, (3, 3), strides=(2, 2), padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2_weight))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Depthwise separable convolution blocks
    for filters in [32, 64, 128]:
        x = DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2_weight))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, (1, 1), strides=(1, 1), padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2_weight))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    # Global Average Pooling and Fully Connected Layers
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(l2_weight))(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# Preprocess audio to extract MFCC features
def preprocess_audio(audio, target_length=300, n_mfcc=40, sample_rate=16000):
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # Convert to mono if stereo
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    if mfcc.shape[1] > target_length:
        mfcc = mfcc[:, :target_length]  # Trim
    else:
        padding = target_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode="constant")  # Pad
    return mfcc


# Load data and preprocess into MFCCs
def load_data(features_dir, target_length=300, sample_rate=16000, n_mfcc=40):
    X, y = [], []
    for label, class_dir in enumerate(["positive", "negative"]):
        class_path = os.path.join(features_dir, class_dir)
        for filename in os.listdir(class_path):
            if filename.endswith(".wav"):
                audio, _ = librosa.load(os.path.join(class_path, filename), sr=sample_rate)
                mfcc = preprocess_audio(audio, target_length=target_length, n_mfcc=n_mfcc, sample_rate=sample_rate)
                X.append(mfcc)
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
    # Paths to data
    train_dir = "features/train"
    val_dir = "features/validation"
    test_dir = "features/test"

    # Data parameters
    target_length = 300
    n_mfcc = 40
    sample_rate = 16000

    # Load data
    print("Loading training data...")
    X_train, y_train = load_data(train_dir, target_length, sample_rate, n_mfcc)
    y_train = y_train.astype(int)  # Ensure labels are integers

    print("Loading validation data...")
    X_val, y_val = load_data(val_dir, target_length, sample_rate, n_mfcc)

    print("Loading test data...")
    X_test, y_test = load_data(test_dir, target_length, sample_rate, n_mfcc)

    # Add channel dimension for Conv2D
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Debugging: Print unique labels in y_train
    print(f"Unique labels in y_train: {np.unique(y_train)}")
    print(f"Data type of y_train: {y_train.dtype}")

    # Ensure classes are integers
    classes = np.unique(y_train).astype(int)
    print(f"Classes for class_weight computation: {classes}")

    # Compute class weights
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(enumerate(class_weights))
    print(f"Computed class weights: {class_weights}")

    # Build the model
    input_shape = X_train.shape[1:]  # (40, T, 1)
    model = build_mobilenet_like_model(input_shape, dropout_rate=0.6, l2_weight=0.01)

    # Callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    csv_logger = CSVLogger("training_log.csv", append=True)

    # Train the model
    print("Training MobileNet-like model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr, csv_logger]
    )

    # Plot training history
    plot_training_history(history)

    # Evaluate the model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Classification report
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the model
    model.save("models/mobilenet_wake_word.h5")
    print("Model saved to 'models/mobilenet_wake_word.h5'")

if __name__ == "__main__":
    main()
