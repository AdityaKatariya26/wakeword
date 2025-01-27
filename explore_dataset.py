import os
from pathlib import Path

def count_files_in_folders(base_dir):
    folder_counts = {}
    for root, _, files in os.walk(base_dir):
        folder_name = os.path.basename(root)
        wav_files = [file for file in files if file.endswith(".wav")]
        folder_counts[folder_name] = len(wav_files)
    return folder_counts

def main():
    base_data_dir = "data"  # Adjust the path if needed
    dataset_folders = ["clear_wake_words", "noisy_wake_words", "background_noise", "negative"]
    
    print("Dataset Distribution:")
    for dataset in dataset_folders:
        folder_path = os.path.join(base_data_dir, dataset)
        if os.path.exists(folder_path):
            counts = count_files_in_folders(folder_path)
            print(f"\n{dataset}:")
            for label, count in counts.items():
                print(f"  {label}: {count} files")
        else:
            print(f"{dataset} folder not found!")

if __name__ == "__main__":
    main()
