import os
import librosa
import numpy as np

# Paths to `data/` (raw .wav files) and `features/` (preprocessed .npy files)
data_dirs = {
    "train": "data/train",
    "validation": "data/validation",
    "test": "data/test"
}

feature_dirs = {
    "train": "features/train",
    "validation": "features/validation",
    "test": "features/test"
}

def validate_audio_files(base_dir, sample_rate=16000):
    """
    Validates the structure and content of .wav files in a given directory.
    """
    for split, dir_path in base_dir.items():
        print(f"\nValidating {split} dataset in {dir_path}...")

        for class_label in ["positive", "negative"]:
            class_dir = os.path.join(dir_path, class_label)
            if not os.path.exists(class_dir):
                print(f"[ERROR] Missing folder: {class_dir}")
                continue

            print(f"\nClass: {class_label}")
            file_count = 0
            valid_files = 0
            invalid_files = 0
            durations = []

            for filename in os.listdir(class_dir):
                if filename.endswith(".wav"):
                    file_count += 1
                    file_path = os.path.join(class_dir, filename)

                    try:
                        audio, sr = librosa.load(file_path, sr=sample_rate)
                        durations.append(len(audio) / sr)
                        valid_files += 1
                    except Exception as e:
                        print(f"[ERROR] Could not load {file_path}: {e}")
                        invalid_files += 1

            print(f"Total files: {file_count}")
            print(f"Valid files: {valid_files}")
            print(f"Invalid files: {invalid_files}")

            if durations:
                print(f"Average duration: {sum(durations) / len(durations):.2f} seconds")
                print(f"Min duration: {min(durations):.2f} seconds")
                print(f"Max duration: {max(durations):.2f} seconds")
            else:
                print(f"[WARNING] No valid audio files found in {class_dir}")


def validate_features(base_dir):
    """
    Validates the structure and content of .npy files in a given directory.
    """
    for split, dir_path in base_dir.items():
        print(f"\nValidating {split} dataset in {dir_path}...")

        for class_label in ["positive", "negative"]:
            class_dir = os.path.join(dir_path, class_label)
            if not os.path.exists(class_dir):
                print(f"[ERROR] Missing folder: {class_dir}")
                continue

            print(f"\nClass: {class_label}")
            file_count = 0
            valid_files = 0
            invalid_files = 0

            for filename in os.listdir(class_dir):
                if filename.endswith(".npy"):
                    file_count += 1
                    file_path = os.path.join(class_dir, filename)

                    try:
                        npy_data = np.load(file_path)
                        if npy_data.ndim != 2:  # Check for proper spectrogram shape
                            print(f"[WARNING] Invalid shape in {file_path}: {npy_data.shape}")
                        valid_files += 1
                    except Exception as e:
                        print(f"[ERROR] Could not load {file_path}: {e}")
                        invalid_files += 1

            print(f"Total files: {file_count}")
            print(f"Valid files: {valid_files}")
            print(f"Invalid files: {invalid_files}")


if __name__ == "__main__":
    print("Validating raw audio files in `data/`...")
    validate_audio_files(data_dirs)

    print("\nValidating preprocessed features in `features/`...")
    validate_features(feature_dirs)
