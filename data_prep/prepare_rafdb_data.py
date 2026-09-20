"""
RAF-DB -> training-ready metadata (verification + VAD mapping, NO training)
-----------------------------------------------------------------------------
RAF-DB images are already face-aligned, so heavy CV preprocessing isn't
needed. What IS needed before training:
  1. Verify every labeled image actually exists and opens correctly.
  2. Confirm image dimensions are consistent (resize plan if not).
  3. Report class balance (RAF-DB is known to be imbalanced).
  4. Map the 7 discrete emotion labels to VAD (valence/arousal/dominance)
     triples, since ArtSpeak's Behaviour Analysis Layer works in VAD space,
     not discrete emotion categories.

Output: a single metadata CSV with columns:
    image_path, split, label_id, emotion, valence, arousal, dominance

Usage:
    python prepare_rafdb_data.py --root /path/to/raf-db-dataset --out data_prep/rafdb_metadata.csv
"""

import argparse
import os

import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Official RAF-DB single-label emotion mapping
# ---------------------------------------------------------------------------
LABEL_TO_EMOTION = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
    7: "Neutral",
}

# ---------------------------------------------------------------------------
# Emotion -> VAD mapping.
# Values follow the commonly-cited circumplex/PAD-style figures used in
# affective computing literature (consistent in direction and rough
# magnitude with Russell 1980 and the NRC-VAD lexicon, Mohammad 2018).
# CITE YOUR CHOSEN SOURCE EXPLICITLY IN THE REPORT - exact numbers vary
# slightly study to study; what must stay consistent is the ORDERING
# (e.g. Happiness > Neutral > Sadness on valence).
# ---------------------------------------------------------------------------
EMOTION_TO_VAD = {
    "Surprise":  {"valence":  0.40, "arousal":  0.67, "dominance": -0.13},
    "Fear":      {"valence": -0.64, "arousal":  0.60, "dominance": -0.43},
    "Disgust":   {"valence": -0.60, "arousal":  0.35, "dominance":  0.11},
    "Happiness": {"valence":  0.81, "arousal":  0.51, "dominance":  0.46},
    "Sadness":   {"valence": -0.63, "arousal": -0.27, "dominance": -0.33},
    "Anger":     {"valence": -0.51, "arousal":  0.59, "dominance":  0.25},
    "Neutral":   {"valence":  0.00, "arousal":  0.00, "dominance":  0.00},
}

EXPECTED_SIZE = None  # set to e.g. (100, 100) once you've confirmed real sizes; None = just report


def verify_split(root: str, csv_path: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    records = []
    missing, corrupt, size_mismatches = 0, 0, 0
    sizes_seen = {}

    for _, row in df.iterrows():
        label_id = int(row["label"])
        image_name = row["image"]
        # RAF-DB layout: DATASET/<split>/<label>/<image_name>
        image_path = os.path.join(root, "DATASET", split, str(label_id), image_name)

        if not os.path.exists(image_path):
            missing += 1
            continue

        try:
            with Image.open(image_path) as im:
                size = im.size
                im.verify()  # raises if the file is truncated/corrupt
        except Exception:
            corrupt += 1
            continue

        sizes_seen[size] = sizes_seen.get(size, 0) + 1
        if EXPECTED_SIZE and size != EXPECTED_SIZE:
            size_mismatches += 1

        emotion = LABEL_TO_EMOTION.get(label_id)
        if emotion is None:
            print(f"  WARNING: unknown label_id {label_id} for {image_name}, skipping")
            continue
        vad = EMOTION_TO_VAD[emotion]

        records.append({
            "image_path": image_path,
            "split": split,
            "label_id": label_id,
            "emotion": emotion,
            "valence": vad["valence"],
            "arousal": vad["arousal"],
            "dominance": vad["dominance"],
        })

    print(f"\n[{split}] {len(df)} labeled rows -> {len(records)} verified images")
    print(f"  Missing files: {missing}")
    print(f"  Corrupt/unreadable files: {corrupt}")
    print(f"  Distinct image dimensions seen: {sizes_seen}")
    if EXPECTED_SIZE:
        print(f"  Images not matching expected size {EXPECTED_SIZE}: {size_mismatches}")

    return pd.DataFrame.from_records(records)


def main(root: str, out_path: str):
    train_df = verify_split(root, os.path.join(root, "train_labels.csv"), "train")
    test_df = verify_split(root, os.path.join(root, "test_labels.csv"), "test")

    full_df = pd.concat([train_df, test_df], ignore_index=True)

    print("\n--- Class balance (train) ---")
    print(train_df["emotion"].value_counts())
    print("\n--- Class balance (test) ---")
    print(test_df["emotion"].value_counts())

    imbalance_ratio = train_df["emotion"].value_counts().min() / train_df["emotion"].value_counts().max()
    print(f"\nTrain imbalance ratio (min class / max class): {imbalance_ratio:.3f}")
    if imbalance_ratio < 0.3:
        print("  -> SEVERE imbalance. Plan on class_weight='balanced' or oversampling minority "
              "classes (Fear/Disgust are typically the smallest in RAF-DB) before training.")

    full_df.to_csv(out_path, index=False)
    print(f"\nSaved metadata -> {out_path} ({len(full_df)} verified rows)")
    print("No model has been trained. No images have been resized or copied - "
          "this only verified integrity and attached VAD labels.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to raf-db-dataset folder")
    parser.add_argument("--out", default="rafdb_metadata.csv")
    args = parser.parse_args()
    main(args.root, args.out)