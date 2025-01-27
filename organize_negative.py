import os
from pathlib import Path
import shutil

def organize_negative_samples(background_folder, google_folder, mozilla_folder, target_folder):
    """
    Combines background noise and negative speech into a single negative folder.
    """
    target_folder = Path(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    # Add background noise
    for file in Path(background_folder).rglob("*.wav"):
        shutil.copy(file, target_folder / file.name)

    # Add Google negative speech
    for file in Path(google_folder).rglob("*.wav"):
        shutil.copy(file, target_folder / file.name)

    # Add Mozilla negative speech
    for file in Path(mozilla_folder).rglob("*.wav"):
        shutil.copy(file, target_folder / file.name)

    print(f"Negative samples organized into {target_folder}")

def main():
    background_folder = "data/background_noise"
    google_folder = "data/negative/google"
    mozilla_folder = "data/negative/mozilla_wav"
    target_folder = "data/org_negative"

    organize_negative_samples(background_folder, google_folder, mozilla_folder, target_folder)

if __name__ == "__main__":
    main()
