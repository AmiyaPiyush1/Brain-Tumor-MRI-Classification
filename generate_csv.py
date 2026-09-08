"""
Generate train.csv from Brain Tumor MRI Dataset folder structure.

Expected dataset structure:
    dataset/
    └── Training/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/

Usage:
    python generate_csv.py
"""

import os
import pandas as pd

# Mapping from folder name to integer label
LABEL_MAP = {
    'notumor': 0,
    'glioma': 1,
    'meningioma': 2,
    'pituitary': 3
}

# Mapping from integer label to display name (for reference)
CLASS_NAMES = {
    0: 'No Tumor',
    1: 'Glioma',
    2: 'Meningioma',
    3: 'Pituitary'
}


def generate_csv(dataset_dir: str = 'dataset/Training', output_csv: str = 'dataset/train.csv'):
    """
    Scan the dataset folder and generate a CSV with columns:
        id_code  — filename without extension
        label    — integer class label (0=No Tumor, 1=Glioma, 2=Meningioma, 3=Pituitary)
    """
    records = []

    for folder_name, label in LABEL_MAP.items():
        folder_path = os.path.join(dataset_dir, folder_name)

        if not os.path.exists(folder_path):
            print(f"Warning: folder not found — {folder_path}")
            continue

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for filename in files:
            id_code = os.path.splitext(filename)[0]
            records.append({
                'id_code': id_code,
                'label': label
            })

        print(f"  {CLASS_NAMES[label]:12s} ({folder_name:10s}): {len(files):5d} images")

    df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\nCSV saved to: {output_csv}")
    print(f"Total samples: {len(df)}")
    print("\nClass distribution:")
    for label, count in df['label'].value_counts().sort_index().items():
        print(f"  {CLASS_NAMES[label]:12s} (label={label}): {count} images ({100*count/len(df):.1f}%)")

    return df


if __name__ == "__main__":
    print("Generating train.csv from Brain Tumor MRI Dataset...\n")
    df = generate_csv(
        dataset_dir='dataset/Training',
        output_csv='dataset/train.csv'
    )
    print("\nDone!")
