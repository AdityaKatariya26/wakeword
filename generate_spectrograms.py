import os
import librosa
import numpy as np
from pathlib import Path

def generate_log_mel_spectrogram(file_path, sample_rate=16000, n_fft=512, hop_length=160, n_mels=40):
    """
    Generates a Log-Mel Spectrogram for a given .wav file.
    """
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=sample_rate)

        # Generate Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        return log_mel_spec
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None

def process_dataset(input_dir, output_dir, sample_rate=16000, n_fft=512, hop_length=160, n_mels=40):
    """
    Processes a dataset folder, generates Log-Mel Spectrograms, and saves them as .npy files.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create the output directory if it doesn't exist
    for subfolder in input_dir.rglob("*"):
        if subfolder.is_dir():
            (output_dir / subfolder.relative_to(input_dir)).mkdir(parents=True, exist_ok=True)

    # Process all .wav files in the input directory
    for wav_file in input_dir.rglob("*.wav"):
        try:
            # Generate spectrogram
            spectrogram = generate_log_mel_spectrogram(
                wav_file, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
            )
            
            if spectrogram is not None:
                # Save as .npy
                output_path = output_dir / wav_file.relative_to(input_dir).with_suffix(".npy")
                np.save(output_path, spectrogram)
                print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")

def main():
    # Input and output directories
    input_dirs = {
        "train": "data/train",
        "validation": "data/validation",
        "test": "data/test"
    }
    output_base_dir = "features"

    for split, input_dir in input_dirs.items():
        output_dir = Path(output_base_dir) / split
        print(f"Processing {split} dataset...")
        process_dataset(input_dir, output_dir)

if __name__ == "__main__":
    main()
