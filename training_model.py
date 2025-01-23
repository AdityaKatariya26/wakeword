import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU, Add,
    GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf


def pad_or_trim_spectrogram(spectrogram, target_length):
    """
    Pads or trims a spectrogram to the target length along the time axis.
    """
    if spectrogram.shape[1] > target_length:
        # Trim to target length
        return spectrogram[:, :target_length]
    else:
        # Pad with zeros to target length
        padding = target_length - spectrogram.shape[1]
        return np.pad(spectrogram, ((0, 0), (0, padding)), mode="constant")


def load_data(features_dir, target_length=300):
    """
    Loads spectrogram data and labels from the given directory, ensuring fixed length.
    """
    X, y = [], []
    for label, class_dir in enumerate(["positive", "negative"]):
        class_path = os.path.join(features_dir, class_dir)
        for npy_file in os.listdir(class_path):
            if npy_file.endswith(".npy"):
                spectrogram = np.load(os.path.join(class_path, npy_file))
                fixed_spectrogram = pad_or_trim_spectrogram(spectrogram, target_length)
                X.append(fixed_spectrogram)
                y.append(label)
    return np.array(X), np.array(y)


def residual_block(x, filters, kernel_size=(3, 3), stride=(1, 1), use_l2=True, l2_weight=0.01):
    """
    Defines a residual block with Conv2D layers and skip connections.
    """
    kernel_regularizer = tf.keras.regularizers.l2(l2_weight) if use_l2 else None

    # Shortcut connection
    shortcut = x
    if x.shape[-1] != filters:  # Apply 1x1 convolution if the filter count changes
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding="same",
                          kernel_regularizer=kernel_regularizer)(shortcut)

    # First convolution
    x = Conv2D(filters, kernel_size, strides=stride, padding="same",
               kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second convolution
    x = Conv2D(filters, kernel_size, strides=(1, 1), padding="same",
               kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)

    # Add skip connection
    x = Add()([shortcut, x])
    x = ReLU()(x)

    return x


def build_tcresnet(input_shape, num_classes=1, use_l2=True, l2_weight=0.01):
    """
    Builds the TCResNet model for binary classification.
    """
    inputs = Input(shape=input_shape)

    # Initial Conv2D layer
    x = Conv2D(16, (7, 7), strides=(2, 2), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Residual blocks
    x = residual_block(x, 16, use_l2=use_l2, l2_weight=l2_weight)
    x = residual_block(x, 32, use_l2=use_l2, l2_weight=l2_weight)
    x = residual_block(x, 64, use_l2=use_l2, l2_weight=l2_weight)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Fully connected layers
    x = Dropout(0.6)(x)  # Increased Dropout to prevent overfitting
    outputs = Dense(num_classes, activation="sigmoid")(x)  # Binary classification

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0005),  # Reduced learning rate
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    # Directories
    train_dir = "features/train"
    val_dir = "features/validation"
    test_dir = "features/test"

    # Fixed spectrogram length
    target_length = 300

    # Load datasets
    print("Loading training data...")
    X_train, y_train = load_data(train_dir, target_length)
    print("Loading validation data...")
    X_val, y_val = load_data(val_dir, target_length)
    print("Loading test data...")
    X_test, y_test = load_data(test_dir, target_length)

    # Add channel dimension (required for Conv2D)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Compute class weights to handle imbalance
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    # Build model
    input_shape = X_train.shape[1:]  # (40, T, 1)
    model = build_tcresnet(input_shape, use_l2=True, l2_weight=0.01)

    # Callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)

    # Train model
    print("Training TCResNet model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr]
    )

    # Plot training progress
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.show()

    # Evaluate model
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

    # Save model
    model.save("models/tcresnet_wake_word.h5")
    print("Model saved to 'models/tcresnet_wake_word.h5'")


if __name__ == "__main__":
    main()
