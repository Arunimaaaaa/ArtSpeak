import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "empatica4_clean.csv"
OUTPUT_FILE = "empatica4_features_final.csv"

WINDOW_SECONDS = 5

# Columns that identify the observation
ID_COLUMNS = [
    "subject_id",
    "time_bin",
    "time_sec",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("EMPATICA E4 FEATURE EXTRACTION")
print("=" * 70)

input_path = Path(INPUT_FILE)

if not input_path.exists():
    raise FileNotFoundError(
        f"\nCould not find {INPUT_FILE}\n"
        f"Current directory: {Path.cwd()}\n"
        "Make sure this script is inside the same folder as the CSV."
    )

df = pd.read_csv(input_path)

print(f"\nInput file: {INPUT_FILE}")
print(f"Input shape: {df.shape}")
print(f"Subjects: {df['subject_id'].nunique()}")


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "subject_id",
    "time_bin",
    "time_sec",

    "EDA_mean",
    "EDA_std",
    "EDA_min",
    "EDA_max",
    "EDA_count",

    "BVP_mean",
    "BVP_std",
    "BVP_min",
    "BVP_max",
    "BVP_count",

    "HR_mean",
    "HR_std",
    "HR_min",
    "HR_max",
    "HR_count",

    "TEMP_mean",
    "TEMP_std",
    "TEMP_min",
    "TEMP_max",
    "TEMP_count",

    "ACC_mean",
    "ACC_std",
    "ACC_min",
    "ACC_max",
    "ACC_count",

    "IBI_mean",
    "IBI_std",
    "IBI_count",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns:\n{missing_columns}"
    )


# ============================================================
# SORT DATA
# ============================================================

df = df.sort_values(
    ["subject_id", "time_bin"]
).reset_index(drop=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_divide(a, b):
    """
    Safely divide two pandas Series.
    Returns NaN when denominator is zero.
    """
    if np.isscalar(b):
        if b == 0:
            return pd.Series(np.nan, index=a.index)
        return a / b

    return a / b.replace(0, np.nan)


def rolling_mean(group, column, window=3):
    return group[column].rolling(
        window=window,
        min_periods=1
    ).mean()


def rolling_std(group, column, window=3):
    return group[column].rolling(
        window=window,
        min_periods=2
    ).std()


# ============================================================
# SIGNAL AVAILABILITY / QUALITY FEATURES
# ============================================================

print("\nCreating signal quality features...")


# Expected samples in each 5-second window
EXPECTED_COUNTS = {
    "EDA": 20,       # 4 Hz × 5 sec
    "BVP": 320,      # 64 Hz × 5 sec
    "HR": 5,         # 1 Hz × 5 sec
    "TEMP": 20,      # 4 Hz × 5 sec
    "ACC": 160,      # 32 Hz × 5 sec
}


for signal, expected in EXPECTED_COUNTS.items():

    count_col = f"{signal}_count"

    df[f"{signal}_availability"] = (
        df[count_col].fillna(0) / expected
    ).clip(0, 1)

    df[f"{signal}_complete"] = (
        df[count_col] >= expected
    ).astype(int)

    df[f"{signal}_missing"] = (
        expected - df[count_col].fillna(0)
    ).clip(lower=0)


# IBI is different because it contains irregular intervals
df["IBI_available"] = (
    df["IBI_count"].notna()
).astype(int)

df["IBI_interval_count"] = (
    df["IBI_count"].fillna(0)
)


# ============================================================
# EDA FEATURES
# ============================================================

print("Extracting EDA features...")

df["EDA_range"] = (
    df["EDA_max"] - df["EDA_min"]
)

df["EDA_cv"] = safe_divide(
    df["EDA_std"],
    df["EDA_mean"].abs()
)

df["EDA_relative_range"] = safe_divide(
    df["EDA_range"],
    df["EDA_mean"].abs()
)

# Change between consecutive windows
df["EDA_change"] = (
    df.groupby("subject_id")["EDA_mean"]
    .diff()
)

df["EDA_abs_change"] = (
    df["EDA_change"].abs()
)

df["EDA_change_rate"] = (
    safe_divide(
        df["EDA_change"],
        WINDOW_SECONDS
    )
)

# Percentage change
df["EDA_pct_change"] = (
    df.groupby("subject_id")["EDA_mean"]
    .pct_change()
    .replace([np.inf, -np.inf], np.nan)
)

# Rolling features
df["EDA_rolling_mean_3"] = (
    df.groupby("subject_id", group_keys=False)
    .apply(
        lambda g: rolling_mean(g, "EDA_mean", 3),
        include_groups=False
    )
    .reset_index(level=0, drop=True)
)

df["EDA_rolling_std_3"] = (
    df.groupby("subject_id", group_keys=False)
    .apply(
        lambda g: rolling_std(g, "EDA_mean", 3),
        include_groups=False
    )
    .reset_index(level=0, drop=True)
)


# ============================================================
# BVP FEATURES
# ============================================================

print("Extracting BVP features...")

df["BVP_range"] = (
    df["BVP_max"] - df["BVP_min"]
)

df["BVP_cv"] = safe_divide(
    df["BVP_std"],
    df["BVP_mean"].abs()
)

df["BVP_abs_mean"] = (
    df["BVP_mean"].abs()
)

df["BVP_change"] = (
    df.groupby("subject_id")["BVP_mean"]
    .diff()
)

df["BVP_abs_change"] = (
    df["BVP_change"].abs()
)

df["BVP_rolling_mean_3"] = (
    df.groupby("subject_id", group_keys=False)
    .apply(
        lambda g: rolling_mean(g, "BVP_mean", 3),
        include_groups=False
    )
    .reset_index(level=0, drop=True)
)

df["BVP_rolling_std_3"] = (
    df.groupby("subject_id", group_keys=False)
    .apply(
        lambda g: rolling_std(g, "BVP_mean", 3),
        include_groups=False
    )
    .reset_index(level=0, drop=True)
)


# ============================================================
# HEART RATE FEATURES
# ============================================================

print("Extracting HR features...")

df["HR_range"] = (
    df["HR_max"] - df["HR_min"]
)

df["HR_cv"] = safe_divide(
    df["HR_std"],
    df["HR_mean"]
)

df["HR_change"] = (
    df.groupby("subject_id")["HR_mean"]
    .diff()
)

df["HR_abs_change"] = (
    df["HR_change"].abs()
)

df["HR_change_rate"] = (
    safe_divide(
        df["HR_change"],
        WINDOW_SECONDS
    )
)

df["HR_pct_change"] = (
    df.groupby("subject_id")["HR_mean"]
    .pct_change()
    .replace([np.inf, -np.inf], np.nan)
)

df["HR_rolling_mean_3"] = (
    df.groupby("subject_id", group_keys=False)
    .apply(
        lambda g: rolling_mean(g, "HR_mean", 3),
        include_groups=False
    )
    .reset_index(level=0, drop=True)
)

df["HR_rolling_std_3"] = (
    df.groupby("subject_id", group_keys=False)
    .apply(
        lambda g: rolling_std(g, "HR_mean", 3),
        include_groups=False
    )
    .reset_index(level=0, drop=True)
)


# ============================================================
# HRV-RELATED FEATURES FROM IBI
# ============================================================

print("Extracting IBI / HRV features...")

# IBI is measured in seconds in your current dataset.

df["IBI_ms"] = (
    df["IBI_mean"] * 1000
)

df["IBI_std_ms"] = (
    df["IBI_std"] * 1000
)

df["IBI_cv"] = safe_divide(
    df["IBI_std"],
    df["IBI_mean"]
)

# Approximate heart rate derived from mean IBI.
# HR ≈ 60 / IBI(seconds)
df["IBI_derived_HR"] = safe_divide(
    60,
    df["IBI_mean"]
)

# Difference between device HR and IBI-derived HR
df["HR_IBI_difference"] = (
    df["HR_mean"] - df["IBI_derived_HR"]
)

df["HR_IBI_abs_difference"] = (
    df["HR_IBI_difference"].abs()
)

# IBI availability quality
df["IBI_density"] = (
    df["IBI_count"] / 10.0
).clip(0, 1)

# Approximate RMSSD from the information available.
#
# IMPORTANT:
# This is NOT true beat-to-beat RMSSD because the current
# dataset contains aggregated IBI statistics rather than
# individual IBI intervals.
#
# We therefore create a conservative proxy rather than
# falsely claiming it is exact RMSSD.

df["IBI_variability_proxy"] = (
    df["IBI_std_ms"]
)

# Change in IBI between windows
df["IBI_change"] = (
    df.groupby("subject_id")["IBI_mean"]
    .diff()
)

df["IBI_abs_change"] = (
    df["IBI_change"].abs()
)


# ============================================================
# TEMPERATURE FEATURES
# ============================================================

print("Extracting temperature features...")

df["TEMP_range"] = (
    df["TEMP_max"] - df["TEMP_min"]
)

df["TEMP_cv"] = safe_divide(
    df["TEMP_std"],
    df["TEMP_mean"].abs()
)

df["TEMP_change"] = (
    df.groupby("subject_id")["TEMP_mean"]
    .diff()
)

df["TEMP_abs_change"] = (
    df["TEMP_change"].abs()
)

df["TEMP_change_rate"] = (
    safe_divide(
        df["TEMP_change"],
        WINDOW_SECONDS
    )
)

df["TEMP_rolling_mean_3"] = (
    df.groupby("subject_id", group_keys=False)
    .apply(
        lambda g: rolling_mean(g, "TEMP_mean", 3),
        include_groups=False
    )
    .reset_index(level=0, drop=True)
)


# ============================================================
# ACCELEROMETER / MOVEMENT FEATURES
# ============================================================

print("Extracting accelerometer features...")

df["ACC_range"] = (
    df["ACC_max"] - df["ACC_min"]
)

df["ACC_cv"] = safe_divide(
    df["ACC_std"],
    df["ACC_mean"].abs()
)

df["ACC_change"] = (
    df.groupby("subject_id")["ACC_mean"]
    .diff()
)

df["ACC_abs_change"] = (
    df["ACC_change"].abs()
)

# Approximate movement intensity
df["ACC_movement_intensity"] = (
    df["ACC_std"]
)

# Range can also represent movement amplitude
df["ACC_movement_amplitude"] = (
    df["ACC_range"]
)

df["ACC_rolling_mean_3"] = (
    df.groupby("subject_id", group_keys=False)
    .apply(
        lambda g: rolling_mean(g, "ACC_mean", 3),
        include_groups=False
    )
    .reset_index(level=0, drop=True)
)

df["ACC_rolling_std_3"] = (
    df.groupby("subject_id", group_keys=False)
    .apply(
        lambda g: rolling_std(g, "ACC_mean", 3),
        include_groups=False
    )
    .reset_index(level=0, drop=True)
)


# ============================================================
# CROSS-SIGNAL FEATURES
# ============================================================

print("Creating cross-signal features...")


# HR + EDA relationship
df["EDA_HR_product"] = (
    df["EDA_mean"] * df["HR_mean"]
)

# Physiological activation proxy
#
# This is NOT a clinical stress score.
# It is simply a numerical feature combining normalized
# EDA and HR within each subject.

df["EDA_HR_ratio"] = safe_divide(
    df["EDA_mean"],
    df["HR_mean"]
)

# Movement + HR
df["HR_ACC_product"] = (
    df["HR_mean"] * df["ACC_std"]
)

# Movement + EDA
df["EDA_ACC_product"] = (
    df["EDA_mean"] * df["ACC_std"]
)


# ============================================================
# SUBJECT-WISE NORMALIZATION
# ============================================================

print("Creating subject-wise normalized features...")

#
# Subject normalization is very important for your project.
#
# Different people naturally have different:
#   resting HR
#   EDA level
#   temperature
#   movement patterns
#
# We therefore create z-score versions relative to each
# subject's own recording.
#

NORMALIZE_COLUMNS = [
    "EDA_mean",
    "EDA_std",
    "EDA_range",

    "BVP_mean",
    "BVP_std",
    "BVP_range",

    "HR_mean",
    "HR_std",
    "HR_range",

    "TEMP_mean",
    "TEMP_std",
    "TEMP_range",

    "ACC_mean",
    "ACC_std",
    "ACC_range",

    "IBI_mean",
    "IBI_std",
]


for col in NORMALIZE_COLUMNS:

    if col not in df.columns:
        continue

    subject_mean = (
        df.groupby("subject_id")[col]
        .transform("mean")
    )

    subject_std = (
        df.groupby("subject_id")[col]
        .transform("std")
    )

    df[f"{col}_z"] = (
        (df[col] - subject_mean)
        / subject_std.replace(0, np.nan)
    )


# ============================================================
# TEMPORAL FEATURES
# ============================================================

print("Creating temporal features...")

# Seconds since beginning of subject recording
df["session_time_min"] = (
    df["time_sec"] / 60.0
)

# Normalized session progress
df["session_progress"] = (
    df.groupby("subject_id")["time_sec"]
    .transform(
        lambda x: (
            x - x.min()
        ) / max(
            x.max() - x.min(),
            1
        )
    )
)


# ============================================================
# GLOBAL PHYSIOLOGICAL QUALITY SCORE
# ============================================================

print("Creating signal quality score...")

availability_columns = [
    "EDA_availability",
    "BVP_availability",
    "HR_availability",
    "TEMP_availability",
    "ACC_availability",
]

df["physiological_signal_quality"] = (
    df[availability_columns]
    .mean(axis=1)
)


# IBI is optional because it is naturally sparse.
df["cardiac_signal_quality"] = (
    0.7 * df["HR_availability"]
    + 0.3 * df["IBI_available"]
)


# ============================================================
# HANDLE INFINITE VALUES
# ============================================================

print("\nCleaning infinite values...")

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# MISSING VALUE STRATEGY
# ============================================================

print("Handling missing values...")


# ------------------------------------------------------------
# IMPORTANT:
#
# We DO NOT immediately fill every missing value with zero.
#
# Missing IBI means no IBI interval was available.
# Missing EDA/HR/TEMP means signal data was unavailable.
#
# We first preserve this information using availability
# columns, then use subject-wise interpolation/median where
# appropriate.
# ------------------------------------------------------------


SIGNAL_COLUMNS = [
    "EDA_mean",
    "EDA_std",
    "EDA_min",
    "EDA_max",

    "BVP_mean",
    "BVP_std",
    "BVP_min",
    "BVP_max",

    "HR_mean",
    "HR_std",
    "HR_min",
    "HR_max",

    "TEMP_mean",
    "TEMP_std",
    "TEMP_min",
    "TEMP_max",

    "ACC_mean",
    "ACC_std",
    "ACC_min",
    "ACC_max",

    "IBI_mean",
    "IBI_std",
]


# Interpolate short gaps within each subject.
#
# limit=2 means we only interpolate small gaps.
# We do NOT create long artificial signal segments.

for col in SIGNAL_COLUMNS:

    if col not in df.columns:
        continue

    df[col] = (
        df.groupby("subject_id")[col]
        .transform(
            lambda x: x.interpolate(
                method="linear",
                limit=2,
                limit_direction="both"
            )
        )
    )


# Fill remaining numeric missing values with GLOBAL medians
# only after interpolation.
#
# IBI receives special treatment below.

for col in SIGNAL_COLUMNS:

    if col not in df.columns:
        continue

    if col.startswith("IBI_"):
        continue

    median_value = df[col].median()

    df[col] = df[col].fillna(median_value)


# IBI should NOT be treated as zero.
#
# Since IBI is sparse, we use the median IBI only for model
# compatibility while preserving the original availability
# information in IBI_available and IBI_density.

ibi_median = df["IBI_mean"].median()

df["IBI_mean"] = (
    df["IBI_mean"].fillna(ibi_median)
)

df["IBI_std"] = (
    df["IBI_std"].fillna(
        df["IBI_std"].median()
    )
)


# ============================================================
# RE-CALCULATE FEATURES AFFECTED BY IMPUTATION
# ============================================================

df["IBI_ms"] = (
    df["IBI_mean"] * 1000
)

df["IBI_std_ms"] = (
    df["IBI_std"] * 1000
)

df["IBI_cv"] = safe_divide(
    df["IBI_std"],
    df["IBI_mean"]
)

df["IBI_derived_HR"] = safe_divide(
    60,
    df["IBI_mean"]
)

df["HR_IBI_difference"] = (
    df["HR_mean"] - df["IBI_derived_HR"]
)

df["HR_IBI_abs_difference"] = (
    df["HR_IBI_difference"].abs()
)

df["IBI_variability_proxy"] = (
    df["IBI_std_ms"]
)


# ============================================================
# FEATURE SANITY LIMITS
# ============================================================

print("Applying basic numerical sanity checks...")


# These are NOT diagnostic thresholds.
# They only prevent obviously impossible numerical values
# from damaging model training.

df["HR_mean"] = df["HR_mean"].clip(
    lower=30,
    upper=220
)

df["HR_min"] = df["HR_min"].clip(
    lower=30,
    upper=220
)

df["HR_max"] = df["HR_max"].clip(
    lower=30,
    upper=220
)

df["TEMP_mean"] = df["TEMP_mean"].clip(
    lower=15,
    upper=45
)


# IBI in seconds.
# Physiologically implausible extremes are clipped for
# numerical stability rather than interpreted as real values.

df["IBI_mean"] = df["IBI_mean"].clip(
    lower=0.25,
    upper=2.5
)

df["IBI_std"] = df["IBI_std"].clip(
    lower=0,
    upper=1
)


# ============================================================
# FINAL FEATURE LIST
# ============================================================

META_COLUMNS = [
    "subject_id",
    "time_bin",
    "time_sec",
]


FEATURE_COLUMNS = [
    # ----------------------------
    # Original EDA
    # ----------------------------
    "EDA_mean",
    "EDA_std",
    "EDA_min",
    "EDA_max",
    "EDA_count",

    # EDA engineered
    "EDA_range",
    "EDA_cv",
    "EDA_relative_range",
    "EDA_change",
    "EDA_abs_change",
    "EDA_change_rate",
    "EDA_pct_change",
    "EDA_rolling_mean_3",
    "EDA_rolling_std_3",

    # ----------------------------
    # BVP
    # ----------------------------
    "BVP_mean",
    "BVP_std",
    "BVP_min",
    "BVP_max",
    "BVP_count",
    "BVP_range",
    "BVP_cv",
    "BVP_abs_mean",
    "BVP_change",
    "BVP_abs_change",
    "BVP_rolling_mean_3",
    "BVP_rolling_std_3",

    # ----------------------------
    # HR
    # ----------------------------
    "HR_mean",
    "HR_std",
    "HR_min",
    "HR_max",
    "HR_count",
    "HR_range",
    "HR_cv",
    "HR_change",
    "HR_abs_change",
    "HR_change_rate",
    "HR_pct_change",
    "HR_rolling_mean_3",
    "HR_rolling_std_3",

    # ----------------------------
    # IBI / HRV
    # ----------------------------
    "IBI_mean",
    "IBI_std",
    "IBI_count",
    "IBI_ms",
    "IBI_std_ms",
    "IBI_cv",
    "IBI_derived_HR",
    "HR_IBI_difference",
    "HR_IBI_abs_difference",
    "IBI_variability_proxy",
    "IBI_change",
    "IBI_abs_change",

    # ----------------------------
    # Temperature
    # ----------------------------
    "TEMP_mean",
    "TEMP_std",
    "TEMP_min",
    "TEMP_max",
    "TEMP_count",
    "TEMP_range",
    "TEMP_cv",
    "TEMP_change",
    "TEMP_abs_change",
    "TEMP_change_rate",
    "TEMP_rolling_mean_3",

    # ----------------------------
    # Accelerometer
    # ----------------------------
    "ACC_mean",
    "ACC_std",
    "ACC_min",
    "ACC_max",
    "ACC_count",
    "ACC_range",
    "ACC_cv",
    "ACC_change",
    "ACC_abs_change",
    "ACC_movement_intensity",
    "ACC_movement_amplitude",
    "ACC_rolling_mean_3",
    "ACC_rolling_std_3",

    # ----------------------------
    # Cross-signal
    # ----------------------------
    "EDA_HR_product",
    "EDA_HR_ratio",
    "HR_ACC_product",
    "EDA_ACC_product",

    # ----------------------------
    # Temporal
    # ----------------------------
    "session_time_min",
    "session_progress",

    # ----------------------------
    # Quality
    # ----------------------------
    "physiological_signal_quality",
    "cardiac_signal_quality",

    # ----------------------------
    # Availability
    # ----------------------------
    "EDA_availability",
    "EDA_complete",
    "EDA_missing",

    "BVP_availability",
    "BVP_complete",
    "BVP_missing",

    "HR_availability",
    "HR_complete",
    "HR_missing",

    "TEMP_availability",
    "TEMP_complete",
    "TEMP_missing",

    "ACC_availability",
    "ACC_complete",
    "ACC_missing",

    "IBI_available",
    "IBI_interval_count",
    "IBI_density",
]


# Add subject-normalized features
for col in NORMALIZE_COLUMNS:

    if col in df.columns:
        FEATURE_COLUMNS.append(
            f"{col}_z"
        )


# Remove duplicates while preserving order
FEATURE_COLUMNS = list(
    dict.fromkeys(FEATURE_COLUMNS)
)


# ============================================================
# FINAL DATASET
# ============================================================

final_columns = (
    META_COLUMNS +
    FEATURE_COLUMNS
)

# Keep only columns that actually exist
final_columns = [
    col for col in final_columns
    if col in df.columns
]

final_df = df[final_columns].copy()


# ============================================================
# FINAL MISSING VALUE CHECK
# ============================================================

# Any remaining numeric NaNs are filled using column medians.
#
# This is primarily to ensure compatibility with standard
# ML pipelines. Availability indicators preserve information
# about whether the original signal was actually present.

numeric_columns = final_df.select_dtypes(
    include=[np.number]
).columns

for col in numeric_columns:

    if final_df[col].isna().any():

        median_value = final_df[col].median()

        if pd.isna(median_value):
            median_value = 0.0

        final_df[col] = (
            final_df[col].fillna(median_value)
        )


# ============================================================
# FINAL DUPLICATE CHECK
# ============================================================

duplicates = final_df.duplicated(
    subset=["subject_id", "time_bin"]
).sum()

print("\nDuplicate subject/time windows:", duplicates)

if duplicates > 0:
    print("WARNING: duplicate windows detected.")


# ============================================================
# SAVE
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("FINAL E4 FEATURE DATASET")
print("=" * 70)

print(f"Rows:              {len(final_df):,}")
print(f"Columns:           {len(final_df.columns):,}")
print(f"Subjects:          {final_df['subject_id'].nunique()}")
print(f"Output:            {OUTPUT_FILE}")

print("\nMissing values:")
missing_report = (
    final_df.isna()
    .sum()
    .sort_values(ascending=False)
)

missing_report = missing_report[
    missing_report > 0
]

if len(missing_report) == 0:
    print("  NONE")
else:
    print(missing_report.to_string())


print("\nFeature groups:")

groups = {
    "EDA": [c for c in final_df.columns if c.startswith("EDA")],
    "BVP": [c for c in final_df.columns if c.startswith("BVP")],
    "HR": [c for c in final_df.columns if c.startswith("HR")],
    "IBI/HRV": [c for c in final_df.columns if c.startswith("IBI")],
    "TEMP": [c for c in final_df.columns if c.startswith("TEMP")],
    "ACC": [c for c in final_df.columns if c.startswith("ACC")],
}

for name, columns in groups.items():
    print(f"  {name:<10}: {len(columns)} features")


print("\nSignal availability:")

for signal in [
    "EDA",
    "BVP",
    "HR",
    "TEMP",
    "ACC",
]:

    col = f"{signal}_availability"

    if col in final_df.columns:

        avg = final_df[col].mean() * 100

        print(
            f"  {signal:<5}: "
            f"{avg:.2f}% average availability"
        )


ibi_available = (
    final_df["IBI_available"].mean() * 100
)

print(
    f"  IBI  : "
    f"{ibi_available:.2f}% windows originally contained IBI"
)


print("\nRows per subject:")

print(
    final_df
    .groupby("subject_id")
    .size()
    .to_string()
)


print("\nFirst 5 rows:")

print(
    final_df.head().to_string()
)


print("\n" + "=" * 70)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print(
    f"\nFINAL FILE READY FOR NEXT ML STAGE:\n"
    f"  {OUTPUT_FILE}\n"
)