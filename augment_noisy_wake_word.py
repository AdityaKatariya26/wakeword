import os
from pathlib import Path
import soundfile as sf
from audiomentations import Compose, AddBackgroundNoise, Gain

def augment_noisy_wake_words(input_folder, output_folder, noise_sources_folder, target_size=10000):
    """
    Augments noisy wake words to reach the target dataset size.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    noise_sources_folder = Path(noise_sources_folder)
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    # Define augmentation pipeline
    augment = Compose([
        AddBackgroundNoise(sounds_path=str(noise_sources_folder), p=0.9),  # Add realistic noise
        Gain(min_gain_db=-10, max_gain_db=5, p=0.5),                      # Adjust volume
    ])

    # Augment files
    current_size = len(list(output_folder.rglob("*.wav")))
    for file_path in input_folder.rglob("*.wav"):
        while current_size < target_size:
            try:
                audio, sr = sf.read(file_path)
                
                # Apply augmentations
                augmented_audio = augment(samples=audio, sample_rate=sr)
                
                # Save augmented audio
                output_file = output_folder / f"{file_path.stem}_aug_{current_size}.wav"
                sf.write(output_file, augmented_audio, sr)
                current_size += 1
                print(f"Augmented: {file_path} -> {output_file}")
            except Exception as e:
                print(f"Failed to augment {file_path}: {e}")
            if current_size >= target_size:
                break

def main():
    input_folder = "data/noisy_wake_words"
    output_folder = "data/augmented_noisy_wake_words"
    noise_sources_folder = "data/background_noise"
    augment_noisy_wake_words(input_folder, output_folder, noise_sources_folder)

if __name__ == "__main__":
    main()
