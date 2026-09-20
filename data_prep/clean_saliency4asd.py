"""
Saliency4ASD cleaning pipeline (v2 - matched to confirmed file layout)
-----------------------------------------------------------------------
Confirmed structure:
    TrainingData/
        ASD/ASD_scanpath_<id>.txt      (300 files, header: Idx, x, y, duration)
        TD/TD_scanpath_<id>.txt        (300 files, same header)
        Images/<id>.<ext>              (300 stimulus images)
        ASD_FixMaps/, TD_FixMaps/      (heatmap pngs - not used here)

Each .txt file has NO subject column. Fixations from multiple viewers appear
to be concatenated back-to-back within one file. This script detects viewer
boundaries by watching for the Idx column resetting to 0, and treats each
reset-to-reset chunk as one viewer's scanpath. If a file never resets, it's
treated as a single sequence.

IMPORTANT leakage note: this dataset has no persistent subject ID across
files, so "split by subject" isn't meaningful here. We split by IMAGE
instead (all rows for a given image_id go entirely into train, val, or
test) so no stimulus image is seen in both training and evaluation.

Usage:
    python clean_saliency4asd_v2.py --root /path/to/TrainingData --clean
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
MAX_FIXATION_DURATION_MS = 5000  # drop implausible outliers only


def find_image_dims(img_dir: str, image_id: str):
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(img_dir, image_id + ext)
        if os.path.exists(candidate):
            with Image.open(candidate) as im:
                return im.size  # (width, height)
    return None, None


def split_into_viewer_segments(df: pd.DataFrame) -> list:
    """Split a single file's rows into per-viewer chunks based on Idx resets."""
    reset_points = df.index[df["idx"] == 0].tolist()
    if len(reset_points) <= 1:
        return [df]  # single sequence, no resets (or only the first row)
    segments = []
    for i, start in enumerate(reset_points):
        end = reset_points[i + 1] if i + 1 < len(reset_points) else len(df)
        segments.append(df.iloc[start:end].reset_index(drop=True))
    return segments


def load_one_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]
    # expected columns after normalizing: idx, x, y, duration
    df = df.dropna()
    return df


def process_group(root: str, group: str, img_dir: str) -> list:
    records = []
    group_dir = os.path.join(root, group)
    files = sorted(glob.glob(os.path.join(group_dir, "*.txt")))
    print(f"[{group}] processing {len(files)} files...")

    for fpath in files:
        basename = os.path.splitext(os.path.basename(fpath))[0]
        # filenames look like ASD_scanpath_12 -> image_id "12"
        image_id = basename.split("_")[-1]

        df = load_one_file(fpath)
        if df.empty or not {"idx", "x", "y", "duration"}.issubset(df.columns):
            print(f"  WARNING: unexpected columns in {fpath}: {list(df.columns)}")
            continue

        df = df[df["duration"] <= MAX_FIXATION_DURATION_MS]
        if df.empty:
            continue

        width, height = find_image_dims(img_dir, image_id)

        segments = split_into_viewer_segments(df)

        for seg_idx, seg in enumerate(segments):
            seg = seg.copy()
            if width and height:
                seg = seg[(seg["x"] >= 0) & (seg["x"] <= width) & (seg["y"] >= 0) & (seg["y"] <= height)]
                if seg.empty:
                    continue
                seg["x_norm"] = seg["x"] / width
                seg["y_norm"] = seg["y"] / height
            else:
                # no matching image found - normalize within the segment itself
                x_range = seg["x"].max() - seg["x"].min()
                y_range = seg["y"].max() - seg["y"].min()
                seg["x_norm"] = (seg["x"] - seg["x"].min()) / x_range if x_range > 0 else 0.5
                seg["y_norm"] = (seg["y"] - seg["y"].min()) / y_range if y_range > 0 else 0.5

            scanpath_len = float(
                np.sqrt(np.diff(seg["x_norm"]) ** 2 + np.diff(seg["y_norm"]) ** 2).sum()
            ) if len(seg) > 1 else 0.0

            records.append(
                {
                    "group": group,
                    "image_id": image_id,
                    "viewer_segment": seg_idx,
                    "n_fixations": len(seg),
                    "total_dwell_ms": seg["duration"].sum(),
                    "mean_fixation_ms": seg["duration"].mean(),
                    "std_fixation_ms": seg["duration"].std() if len(seg) > 1 else 0.0,
                    "mean_x_norm": seg["x_norm"].mean(),
                    "mean_y_norm": seg["y_norm"].mean(),
                    "spatial_spread": float(seg["x_norm"].std() + seg["y_norm"].std()) if len(seg) > 1 else 0.0,
                    "scanpath_length": scanpath_len,
                }
            )

    return records


def clean(root: str, out_path: str):
    img_dir = os.path.join(root, "Images")
    all_records = []
    for group in ("ASD", "TD"):
        all_records.extend(process_group(root, group, img_dir))

    if not all_records:
        print("No records produced - check the folder path and file contents.")
        return

    out_df = pd.DataFrame.from_records(all_records).drop_duplicates()

    # --- split by IMAGE, not subject (see note at top of file) ---
    image_ids = sorted(out_df["image_id"].unique(), key=lambda x: int(x) if x.isdigit() else x)
    rng = np.random.default_rng(42)
    shuffled = list(image_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_imgs = set(shuffled[: int(n * 0.7)])
    val_imgs = set(shuffled[int(n * 0.7): int(n * 0.85)])
    out_df["split"] = out_df["image_id"].apply(
        lambda i: "train" if i in train_imgs else ("val" if i in val_imgs else "test")
    )

    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df)} rows -> {out_path}")
    print("\nRows per group:")
    print(out_df["group"].value_counts())
    print("\nRows per split:")
    print(out_df["split"].value_counts())
    print(f"\nUnique images: {out_df['image_id'].nunique()} (should be <= 300)")
    print(f"Viewer segments per image (ASD, mean): "
          f"{out_df[out_df['group']=='ASD'].groupby('image_id').size().mean():.1f}")
    print(f"Viewer segments per image (TD, mean): "
          f"{out_df[out_df['group']=='TD'].groupby('image_id').size().mean():.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to TrainingData folder")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--out", default="saliency4asd_clean.csv")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit(f"Root path not found: {args.root}")

    if args.clean:
        clean(args.root, args.out)
    else:
        print("Pass --clean to run the pipeline.")