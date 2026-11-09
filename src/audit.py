"""
Task 1.2: Dataset Audit & Deduplication Script
Author: 23L-0994
Project: 23L-0994_A2_Ops
Walks dataset, computes SHA-256 hashes, verifies image integrity, detects duplicates,
and produces summary statistics per class.
"""
import os
import sys
import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image

def compute_sha256(filepath, chunk_size=65536):
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()

def audit_dataset(data_dir="data/raw", output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Error: Dataset directory '{data_dir}' not found.")
        print("Please place dataset in 'data/raw/ai_generated' (or 'data/raw/ai') and 'data/raw/real'.")
        sys.exit(1)

    print("=" * 70)
    print(f"Starting Dataset Audit on: {data_path.resolve()}")
    print("=" * 70)

    records = []
    corrupt_files = []
    seen_hashes = {}
    duplicates = []

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() not in valid_extensions:
                continue

            # Class label inferred from parent folder name
            parent_name = file_path.parent.name.lower()
            if 'ai' in parent_name or 'fake' in parent_name or 'generated' in parent_name:
                label = 'AI-Generated'
            elif 'real' in parent_name:
                label = 'Real'
            else:
                label = parent_name

            file_size_bytes = file_path.stat().st_size
            file_size_kb = file_size_bytes / 1024.0

            try:
                # Content hash
                file_hash = compute_sha256(file_path)

                # Deduplication check
                if file_hash in seen_hashes:
                    duplicates.append({
                        "file_1": seen_hashes[file_hash],
                        "file_2": str(file_path),
                        "hash": file_hash
                    })
                else:
                    seen_hashes[file_hash] = str(file_path)

                # Verify PIL read
                with Image.open(file_path) as img:
                    img.verify()

                # Re-open for metadata extraction
                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode
                    fmt = img.format

                records.append({
                    "file_path": str(file_path),
                    "filename": file_path.name,
                    "class": label,
                    "file_size_bytes": file_size_bytes,
                    "file_size_kb": file_size_kb,
                    "width": width,
                    "height": height,
                    "color_mode": mode,
                    "format": fmt,
                    "sha256": file_hash,
                    "is_corrupt": False
                })

            except Exception as e:
                corrupt_files.append({
                    "file_path": str(file_path),
                    "error": str(e)
                })
                records.append({
                    "file_path": str(file_path),
                    "filename": file_path.name,
                    "class": label,
                    "file_size_bytes": file_size_bytes,
                    "file_size_kb": file_size_kb,
                    "width": None,
                    "height": None,
                    "color_mode": None,
                    "format": None,
                    "sha256": None,
                    "is_corrupt": True
                })

    df = pd.DataFrame(records)
    csv_out = Path(output_dir) / "dataset_audit_catalog.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n[+] Full catalog saved to {csv_out}")

    # Summary table
    print("\n" + "=" * 70)
    print("DATASET AUDIT SUMMARY TABLE (Evidence for Task 1.2)")
    print("=" * 70)
    valid_df = df[~df['is_corrupt']]

    summary = []
    for cls_name, grp in valid_df.groupby('class'):
        summary.append({
            "Class": cls_name,
            "Total Images": len(grp),
            "Min Width": int(grp['width'].min()),
            "Median Width": int(grp['width'].median()),
            "Max Width": int(grp['width'].max()),
            "Min Height": int(grp['height'].min()),
            "Median Height": int(grp['height'].median()),
            "Max Height": int(grp['height'].max()),
            "Min Size (KB)": round(grp['file_size_kb'].min(), 2),
            "Median Size (KB)": round(grp['file_size_kb'].median(), 2),
            "Max Size (KB)": round(grp['file_size_kb'].max(), 2),
        })

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    print("-" * 70)
    print(f"Total Images Processed : {len(df)}")
    print(f"Corrupt / Unreadable   : {len(corrupt_files)}")
    print(f"Exact Duplicate Pairs  : {len(duplicates)}")
    print("=" * 70)

    summary_csv = Path(output_dir) / "audit_summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[+] Summary metrics exported to {summary_csv}")

    return df, summary_df

if __name__ == "__main__":
    audit_dataset()
