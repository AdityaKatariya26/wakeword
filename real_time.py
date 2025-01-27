import os
import numpy as np
import tensorflow as tf
import sounddevice as sd
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split

# Parameters
SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # Duration of audio in seconds
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION
MODEL_PATH = "models/mobilenet_wake_word.h5"
USER_SAMPLES_DIR = "user_samples"
FINE_TUNED_MODEL_PATH = "models/fine_tuned_model.h5"
MFCC_SHAPE = (40, 300, 1)

# Record User Inputs
def record_user_samples(num_samples=5, is_positive=True):
    """Record user's wake word inputs."""
    label = "positive" if is_positive else "negative"
    output_dir = os.path.join(USER_SAMPLES_DIR, label)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Recording {num_samples} {label} samples...")
    for i in range(num_samples):
        print(f"Recording sample {i + 1}/{num_samples}. Please say '{'Hey Ava' if is_positive else 'something else'}' (you have {CHUNK_DURATION} seconds)...")
        recording = sd.rec(frames=CHUNK_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()  # Wait until recording is finished
        sample_path = os.path.join(output_dir, f"sample_{i + 1}.wav")
        sf.write(sample_path, recording.flatten(), SAMPLE_RATE)
        print(f"Saved {label} sample {i + 1} to {sample_path}")
    print(f"{label.capitalize()} recording complete!")

# Preprocess Audio to Extract MFCC Features
def preprocess_audio_file(file_path):
    """Extract MFCC features from an audio file."""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=400, hop_length=160)
    if mfcc.shape[1] < MFCC_SHAPE[1]:
        mfcc = np.pad(mfcc, ((0, 0), (0, MFCC_SHAPE[1] - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :MFCC_SHAPE[1]]
    mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel dimension
    return mfcc

# Load and Preprocess Dataset
def load_user_dataset():
    """Load and preprocess user-provided samples."""
    X, y = [], []
    for label, class_dir in enumerate(["positive", "negative"]):
        class_path = os.path.join(USER_SAMPLES_DIR, class_dir)
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            if file_path.endswith(".wav"):
                mfcc = preprocess_audio_file(file_path)
                X.append(mfcc)
                y.append(label)
    return np.array(X), np.array(y)

# Fine-Tune the Model
def fine_tune_model(X, y, model_path, fine_tuned_model_path):
    """Fine-tune the pre-trained model with user data."""
    print("Loading pre-trained model...")
    model = tf.keras.models.load_model(model_path)
    print("Pre-trained model loaded.")

    # Compile the model for fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Split the user dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Fine-tuning the model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,  # Small number of epochs to prevent overfitting
        batch_size=8,
        verbose=1
    )

    # Save the fine-tuned model
    model.save(fine_tuned_model_path)
    print(f"Fine-tuned model saved to {fine_tuned_model_path}")
    return model

# Main Function
def main():
    # Step 1: Record Positive Samples
    record_user_samples(num_samples=5, is_positive=True)
    # Step 2: Record Negative Samples
    record_user_samples(num_samples=5, is_positive=False)
    
    # Step 3: Load and Preprocess User Dataset
    print("Loading and preprocessing user-provided data...")
    X, y = load_user_dataset()
    print(f"Loaded {len(X)} samples from user data.")

    # Step 4: Fine-Tune the Model
    fine_tuned_model = fine_tune_model(X, y, MODEL_PATH, FINE_TUNED_MODEL_PATH)

    # Step 5: Use the Fine-Tuned Model for Real-Time Detection
    print("\nReal-time detection is ready with the fine-tuned model!")
    # You can now use the fine_tuned_model for real-time detection.

if __name__ == "__main__":
    main()
