"""
RAF-DB -> fully training-ready split (still NO training happens here)
------------------------------------------------------------------------
Takes rafdb_metadata.csv (from prepare_rafdb_data.py) and applies the
remaining data engineering steps needed before any model touches this data:

  1. Duplicate/leakage check - hashes every image file and checks for
     (a) duplicates within train, (b) duplicates within test, and
     (c) the same image appearing in BOTH train and test (a leakage bug
     that would silently inflate reported accuracy).
  2. Stratified train/val split carved out of the original `train` rows
     only - `test` is never touched or split further.
  3. Per-sample class weights (inverse frequency) added to the train split,
     for use with a WeightedRandomSampler at training time - chosen over
     naive oversampling because it doesn't duplicate literal images on disk.
  4. Per-channel pixel mean/std computed from the TRAIN split ONLY (never
     val or test - computing normalization stats from data the model will
     be evaluated on is a leakage mistake).
  5. A documented augmentation config (not applied to disk - these get
     used as a live transform pipeline at training time).

Outputs (all inside --outdir):
    rafdb_train.csv          (with sample_weight column)
    rafdb_val.csv
    rafdb_test.csv           (untouched copy, for traceability)
    normalization_stats.json
    augmentation_config.json
    dedup_report.json

Usage:
    python prepare_rafdb_training_split.py --input data_prep/rafdb_metadata.csv --outdir data_prep/rafdb_ready
"""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

VAL_FRACTION = 0.15
RANDOM_SEED = 42


def hash_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def run_dedup_check(df: pd.DataFrame) -> dict:
    print("\nHashing all images for duplicate/leakage check (this takes a moment)...")
    df = df.copy()
    df["file_hash"] = df["image_path"].apply(hash_file)

    train_hashes = df.loc[df["split"] == "train", "file_hash"]
    test_hashes = df.loc[df["split"] == "test", "file_hash"]

    dupes_in_train = train_hashes.duplicated().sum()
    dupes_in_test = test_hashes.duplicated().sum()
    cross_leak = set(train_hashes) & set(test_hashes)

    report = {
        "duplicate_images_within_train": int(dupes_in_train),
        "duplicate_images_within_test": int(dupes_in_test),
        "images_leaked_between_train_and_test": len(cross_leak),
    }

    print(f"  Duplicates within train: {report['duplicate_images_within_train']}")
    print(f"  Duplicates within test:  {report['duplicate_images_within_test']}")
    print(f"  Train/test leakage (identical image in both): {report['images_leaked_between_train_and_test']}")
    if cross_leak:
        print("  WARNING: identical images found in both train and test - "
              "any evaluation accuracy is unreliable until this is resolved.")

    return report, df


def stratified_split(train_df: pd.DataFrame):
    train_split, val_split = train_test_split(
        train_df,
        test_size=VAL_FRACTION,
        stratify=train_df["emotion"],
        random_state=RANDOM_SEED,
    )
    print(f"\nStratified split of original train rows: "
          f"{len(train_split)} train / {len(val_split)} val")
    print("\nVal split class balance (should mirror train proportions):")
    print(val_split["emotion"].value_counts(normalize=True).round(3))
    return train_split.reset_index(drop=True), val_split.reset_index(drop=True)


def add_class_weights(train_split: pd.DataFrame) -> pd.DataFrame:
    counts = train_split["emotion"].value_counts()
    n_classes = len(counts)
    total = len(train_split)
    weight_per_class = {emo: total / (n_classes * cnt) for emo, cnt in counts.items()}
    train_split = train_split.copy()
    train_split["sample_weight"] = train_split["emotion"].map(weight_per_class)

    print("\nPer-class sample weights (inverse frequency, for WeightedRandomSampler):")
    for emo, w in sorted(weight_per_class.items(), key=lambda x: -x[1]):
        print(f"  {emo:<10} count={counts[emo]:<5} weight={w:.3f}")

    return train_split


def compute_normalization_stats(train_split: pd.DataFrame) -> dict:
    print(f"\nComputing per-channel pixel mean/std from {len(train_split)} TRAIN images only...")
    n_pixels = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)

    for path in train_split["image_path"]:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
        channel_sum += arr.sum(axis=(0, 1))
        channel_sq_sum += (arr ** 2).sum(axis=(0, 1))
        n_pixels += arr.shape[0] * arr.shape[1]

    mean = channel_sum / n_pixels
    var = (channel_sq_sum / n_pixels) - mean ** 2
    std = np.sqrt(np.clip(var, a_min=1e-8, a_max=None))

    stats = {
        "mean_rgb": mean.tolist(),
        "std_rgb": std.tolist(),
        "note": "Computed from TRAIN split only, on 0-1 scaled pixel values. "
                "Apply the SAME mean/std to val and test at training/inference time "
                "- never recompute stats from val/test.",
    }
    print(f"  mean (R,G,B): {[round(float(m), 4) for m in mean]}")
    print(f"  std  (R,G,B): {[round(float(s), 4) for s in std]}")
    return stats


def build_augmentation_config() -> dict:
    # Documented here, not applied to disk - these are meant to be read
    # by the training script's live transform pipeline (e.g. torchvision
    # transforms), applied only to the train split, never to val/test.
    return {
        "applies_to": "train split only - never val or test",
        "horizontal_flip_prob": 0.5,
        "rotation_degrees": 15,
        "brightness_jitter": 0.2,
        "contrast_jitter": 0.2,
        "note": "Standard augmentation applied uniformly across classes. Class "
                "imbalance is handled via sample_weight + WeightedRandomSampler "
                "instead of extra augmentation intensity on minority classes, "
                "to avoid over-representing a small number of literal images.",
    }


def main(input_path: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(input_path)

    dedup_report, df_hashed = run_dedup_check(df)
    with open(os.path.join(outdir, "dedup_report.json"), "w") as f:
        json.dump(dedup_report, f, indent=2)

    train_full = df_hashed[df_hashed["split"] == "train"].drop(columns=["file_hash"])
    test_df = df_hashed[df_hashed["split"] == "test"].drop(columns=["file_hash"])

    train_split, val_split = stratified_split(train_full)
    train_split = add_class_weights(train_split)

    norm_stats = compute_normalization_stats(train_split)
    with open(os.path.join(outdir, "normalization_stats.json"), "w") as f:
        json.dump(norm_stats, f, indent=2)

    aug_config = build_augmentation_config()
    with open(os.path.join(outdir, "augmentation_config.json"), "w") as f:
        json.dump(aug_config, f, indent=2)

    train_split.to_csv(os.path.join(outdir, "rafdb_train.csv"), index=False)
    val_split.to_csv(os.path.join(outdir, "rafdb_val.csv"), index=False)
    test_df.to_csv(os.path.join(outdir, "rafdb_test.csv"), index=False)

    print(f"\nSaved to {outdir}/:")
    print(f"  rafdb_train.csv   ({len(train_split)} rows, with sample_weight)")
    print(f"  rafdb_val.csv     ({len(val_split)} rows)")
    print(f"  rafdb_test.csv    ({len(test_df)} rows, untouched)")
    print(f"  normalization_stats.json")
    print(f"  augmentation_config.json")
    print(f"  dedup_report.json")
    print("\nNo model has been trained. No images resized/copied - paths only. "
          "This dataset is now fully training-ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to rafdb_metadata.csv")
    parser.add_argument("--outdir", default="rafdb_ready")
    args = parser.parse_args()
    main(args.input, args.outdir)