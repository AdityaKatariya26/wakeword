import os
from pathlib import Path
import soundfile as sf
from audiomentations import Compose, PitchShift, TimeStretch, Shift, AddBackgroundNoise

def augment_clear_wake_words(input_folder, output_folder, background_noise_folder, target_size=5000):
    """
    Augments clear wake words to reach the target dataset size.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    background_noise_folder = Path(background_noise_folder)
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    # Define augmentation pipeline
    augment = Compose([
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5),                 # Narrower pitch shifting
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),                      # Narrower time stretching
        Shift(max_shift=0.2, rollover=True, p=0.5),                          # Controlled time shifting
        AddBackgroundNoise(sounds_path=str(background_noise_folder), p=0.8), # Background noise
    ])

    # Augment files
    current_size = len(list(output_folder.rglob("*.wav")))
    for file_path in input_folder.rglob("*.wav"):
        while current_size < target_size:
            try:
                audio, sr = sf.read(file_path)
                
                # Apply augmentations
                augmented_audio = augment(samples=audio, sample_rate=sr)
                
                # Validate the augmented audio (duration match)
                if len(augmented_audio) != len(audio):
                    print(f"Skipped invalid augmentation for: {file_path}")
                    continue
                
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
    input_folder = "data/clear_wake_words"
    output_folder = "data/augmented_clear_wake_words"
    background_noise_folder = "data/background_noise"
    augment_clear_wake_words(input_folder, output_folder, background_noise_folder)

if __name__ == "__main__":
    main()
