"""
Saliency4ASD -> final training-ready feature CSV (NO training happens here)
-----------------------------------------------------------------------------
Takes the cleaned CSV (from clean_saliency4asd.py) and produces a second,
smaller CSV containing ONLY the columns that should feed into model
training later - with junk rows removed and skewed features transformed.

This script deliberately does nothing beyond filtering/feature prep so you
can review the output before any model touches it.

Usage:
    python prepare_training_data.py --input data_prep/saliency4asd_clean.csv --out data_prep/saliency4asd_features.csv
"""

import argparse

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# SCHEMA - what's kept vs dropped, and why
# ---------------------------------------------------------------------------
KEEP_AS_IS = [
    "group",          # target label (ASD / TD) - kept as text here, encoded later at training time
    "image_id",       # NOT a feature - kept only so training can group-split by image to avoid leakage
    "n_fixations",
    "mean_x_norm",
    "mean_y_norm",
    "spatial_spread",
    "scanpath_length",
]

# these are right-skewed (millisecond durations) - log1p makes their
# distribution far more usable for most models, and we keep the transformed
# version under a clear _log name rather than overwriting silently
LOG_TRANSFORM = {
    "total_dwell_ms": "total_dwell_ms_log",
    "mean_fixation_ms": "mean_fixation_ms_log",
    "std_fixation_ms": "std_fixation_ms_log",
}

# explicitly dropped and why:
#   viewer_segment  -> internal index from the cleaning step, not a real signal
#   split           -> was a placeholder from the cleaning script; the real
#                       train/val/test split should be regenerated at training
#                       time via GroupKFold on image_id, not reused from here
DROP_EXPLICITLY = ["viewer_segment", "split"]

MIN_FIXATIONS_PER_SEGMENT = 3  # below this, spread/scanpath shape is meaningless


def prepare(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns from {input_path}")
    print(f"Original columns: {list(df.columns)}")

    missing_expected = [c for c in KEEP_AS_IS + list(LOG_TRANSFORM.keys()) if c not in df.columns]
    if missing_expected:
        raise SystemExit(
            f"Input file is missing expected columns: {missing_expected}\n"
            f"Check it was produced by clean_saliency4asd.py and not modified."
        )

    # --- filter out low-quality rows ---
    before = len(df)
    df = df[df["n_fixations"] >= MIN_FIXATIONS_PER_SEGMENT].copy()
    print(f"\nDropped {before - len(df)} viewer-segments with fewer than "
          f"{MIN_FIXATIONS_PER_SEGMENT} fixations ({before} -> {len(df)} rows)")

    # --- log-transform skewed duration features ---
    for raw_col, log_col in LOG_TRANSFORM.items():
        df[log_col] = np.log1p(df[raw_col].clip(lower=0))

    # --- assemble final column set ---
    final_cols = KEEP_AS_IS + list(LOG_TRANSFORM.values())
    out_df = df[final_cols].dropna().reset_index(drop=True)

    dropped_cols = [c for c in df.columns if c not in final_cols]
    print(f"\nDropped columns (not used as features): {DROP_EXPLICITLY}")
    print(f"Raw duration columns replaced by log-transformed versions: {list(LOG_TRANSFORM.keys())}")
    print(f"\nFinal feature columns kept ({len(final_cols)}): {final_cols}")

    # --- summary for manual review before training ---
    print(f"\nFinal row count: {len(out_df)}")
    print("\nClass balance:")
    print(out_df["group"].value_counts())
    print(f"\nUnique images: {out_df['image_id'].nunique()}")
    print("\nFeature summary statistics:")
    print(out_df.drop(columns=["group", "image_id"]).describe().T)

    out_df.to_csv(output_path, index=False)
    print(f"\nSaved training-ready feature CSV -> {output_path}")
    print("\nNo model has been trained. Review the summary above before running the training script.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to saliency4asd_clean.csv")
    parser.add_argument("--out", default="saliency4asd_features.csv")
    args = parser.parse_args()
    prepare(args.input, args.out)