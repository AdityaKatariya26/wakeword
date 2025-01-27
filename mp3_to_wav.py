import os
import librosa
import soundfile as sf
from pathlib import Path

def convert_mp3_to_wav(source_folder, target_folder):
    """
    Converts all MP3 files in the source_folder to WAV format and saves them in the target_folder.
    """
    source_folder = Path(source_folder)
    target_folder = Path(target_folder)
    if not target_folder.exists():
        target_folder.mkdir(parents=True)

    for mp3_file in source_folder.rglob("*.mp3"):
        try:
            wav_file = target_folder / f"{mp3_file.stem}.wav"
            print(f"Converting {mp3_file} to {wav_file}...")
            # Load MP3 file
            audio, sr = librosa.load(mp3_file, sr=None)
            # Save as WAV
            sf.write(wav_file, audio, sr)
            print(f"Converted: {mp3_file} -> {wav_file}")
        except Exception as e:
            print(f"Failed to convert {mp3_file}: {e}")

def main():
    source_folder = r"C:\Users\Hp\OneDrive\Desktop\fine tuning\data\negative\mozilla"  # Adjust path
    target_folder = r"C:\Users\Hp\OneDrive\Desktop\fine tuning\data\negative\mozilla_wav"  # Target folder
    convert_mp3_to_wav(source_folder, target_folder)

if __name__ == "__main__":
    main()
