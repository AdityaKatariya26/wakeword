import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

def split_and_copy(files, train_folder, val_folder, test_folder, train_count, val_count, test_count):
    """
    Splits files into train, validation, and test sets, and copies them to the respective folders.
    """
    # Shuffle and limit files
    total_needed = train_count + val_count + test_count
    if len(files) > total_needed:
        files = files[:total_needed]  # Limit to required number

    # Split files
    train_files, temp_files = train_test_split(files, train_size=train_count, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=test_count / (val_count + test_count), random_state=42)

    # Copy files
    for file in train_files:
        shutil.copy(file, train_folder / file.name)
    for file in val_files:
        shutil.copy(file, val_folder / file.name)
    for file in test_files:
        shutil.copy(file, test_folder / file.name)

    return len(train_files), len(val_files), len(test_files)

def split_dataset(positive_dir, negative_dir, output_dir, split_plan):
    """
    Splits the dataset into train, validation, and test folders with balanced ratios.
    """
    positive_dir = Path(positive_dir)
    negative_dir = Path(negative_dir)
    output_dir = Path(output_dir)

    # Create train, validation, and test directories
    train_pos_dir = output_dir / "train" / "positive"
    val_pos_dir = output_dir / "validation" / "positive"
    test_pos_dir = output_dir / "test" / "positive"

    train_neg_dir = output_dir / "train" / "negative"
    val_neg_dir = output_dir / "validation" / "negative"
    test_neg_dir = output_dir / "test" / "negative"

    for folder in [train_pos_dir, val_pos_dir, test_pos_dir, train_neg_dir, val_neg_dir, test_neg_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    # Get all files
    positive_files = list(positive_dir.rglob("*.wav"))
    negative_files = list(negative_dir.rglob("*.wav"))

    # Debug: Print counts
    print(f"Found {len(positive_files)} positive files")
    print(f"Found {len(negative_files)} negative files")

    # Split and copy positive samples
    print("Splitting Positive Samples...")
    split_and_copy(
        positive_files, train_pos_dir, val_pos_dir, test_pos_dir,
        train_count=split_plan["train"]["positive"],
        val_count=split_plan["validation"]["positive"],
        test_count=split_plan["test"]["positive"]
    )

    # Split and copy negative samples
    print("Splitting Negative Samples...")
    split_and_copy(
        negative_files, train_neg_dir, val_neg_dir, test_neg_dir,
        train_count=split_plan["train"]["negative"],
        val_count=split_plan["validation"]["negative"],
        test_count=split_plan["test"]["negative"]
    )

    print("Dataset split completed successfully!")

def main(): 
    # Directories for positive and negative samples
    positive_dir = "data/positive"
    negative_dir = "data/org_negative"

    # Output directory for train, validation, and test splits
    output_dir = "data"

    # Split plan for dataset
    split_plan = {
        "train": {"positive": 12000, "negative": 16000},
        "validation": {"positive": 1500, "negative": 2000},
        "test": {"positive": 1500, "negative": 2000}
    }

    # Split the dataset
    split_dataset(positive_dir, negative_dir, output_dir, split_plan)

if __name__ == "__main__":
    main()
