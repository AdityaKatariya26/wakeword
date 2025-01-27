import os
from pathlib import Path
import shutil

def organize_positive_samples(augmented_clear_folder, augmented_noisy_folder, target_folder):
    """
    Combines augmented clear and noisy wake words into a single positive folder.
    """
    target_folder = Path(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    # Add augmented clear wake words
    for file in Path(augmented_clear_folder).rglob("*.wav"):
        shutil.copy(file, target_folder / file.name)

    # Add augmented noisy wake words
    for file in Path(augmented_noisy_folder).rglob("*.wav"):
        shutil.copy(file, target_folder / file.name)

    print(f"Positive samples organized into {target_folder}")

def main():
    augmented_clear_folder = "data/augmented_clear_wake_words"
    augmented_noisy_folder = "data/augmented_noisy_wake_words"
    target_folder = "data/positive"

    organize_positive_samples(augmented_clear_folder, augmented_noisy_folder, target_folder)

if __name__ == "__main__":
    main()
