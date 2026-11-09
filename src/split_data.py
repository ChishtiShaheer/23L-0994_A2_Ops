"""
Task 2.1 Helper: Split Dataset into Train, Validation, and Test Lists
Author: 23L-0994
Project: 23L-0994_A2_Ops
Generates reproducible split text files (train.txt, val.txt, test.txt) with stratified sampling.
"""
import os
import random
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def create_splits(catalog_csv="reports/dataset_audit_catalog.csv", data_dir="data/raw", splits_dir="data/splits", seed=42):
    os.makedirs(splits_dir, exist_ok=True)

    if os.path.exists(catalog_csv):
        df = pd.read_csv(catalog_csv)
        valid_df = df[~df['is_corrupt']].copy()
    else:
        # Scan data dir
        records = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for root, _, files in os.walk(data_dir):
            for file in files:
                p = Path(root) / file
                if p.suffix.lower() in valid_extensions:
                    parent = p.parent.name.lower()
                    label = "AI-Generated" if ("ai" in parent or "fake" in parent or "gen" in parent) else "Real"
                    records.append({"file_path": str(p), "class": label})
        valid_df = pd.DataFrame(records)

    if len(valid_df) == 0:
        print("Error: No images found to create splits.")
        return

    # 70% Train, 15% Validation, 15% Test
    train_df, temp_df = train_test_split(
        valid_df,
        test_size=0.30,
        random_state=seed,
        stratify=valid_df['class']
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df['class']
    )

    def write_split(dataframe, filename):
        out_path = Path(splits_dir) / filename
        with open(out_path, 'w') as f:
            for _, row in dataframe.iterrows():
                f.write(f"{row['file_path']},{row['class']}\n")
        print(f"[+] Wrote {len(dataframe)} items to {out_path}")

    write_split(train_df, "train.txt")
    write_split(val_df, "val.txt")
    write_split(test_df, "test.txt")

    print("\nDataset split summary:")
    print(f"  Training Set   : {len(train_df)} samples ({len(train_df)/len(valid_df)*100:.1f}%)")
    print(f"  Validation Set : {len(val_df)} samples ({len(val_df)/len(valid_df)*100:.1f}%)")
    print(f"  Test Set       : {len(test_df)} samples ({len(test_df)/len(valid_df)*100:.1f}%)")

if __name__ == "__main__":
    create_splits()
