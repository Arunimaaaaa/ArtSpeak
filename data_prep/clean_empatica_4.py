import argparse
import glob
import os

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

KNOWN_RATES = {
    "EDA": 4.0,
    "TEMP": 4.0,
    "BVP": 64.0,
    "ACC": 32.0,
    "HR": 1.0,
}

RESAMPLE_WINDOW_SEC = 5

SIGNALS = ["EDA", "BVP", "HR", "TEMP", "ACC"]


# ============================================================
# NUMERIC CLEANING
# ============================================================

def clean_numeric_string(value):
    """
    Convert a messy value into a float.

    Examples:
        "2.1277501" -> 2.1277501
        "0.830302"  -> 0.830302
        "1.177.810" -> 1.17781
        "2.582.516" -> 2.582516

    Some files, especially subject_23/EDA.csv, contain
    multiple decimal points. We repair those values by
    keeping the first decimal point and removing the
    additional decimal points.
    """

    if pd.isna(value):
        return np.nan

    s = str(value).strip()

    if not s:
        return np.nan

    # Remove spaces
    s = s.replace(" ", "")

    # Normal numeric value
    try:
        return float(s)
    except ValueError:
        pass

    # --------------------------------------------------------
    # Repair values with multiple decimal points
    #
    # Example:
    #   1.177.810 -> 1.177810
    #   2.582.516 -> 2.582516
    # --------------------------------------------------------

    if s.count(".") > 1:

        parts = s.split(".")

        if all(part.isdigit() for part in parts):

            repaired = parts[0] + "." + "".join(parts[1:])

            try:
                return float(repaired)
            except ValueError:
                return np.nan

    # --------------------------------------------------------
    # Handle comma as decimal separator
    # --------------------------------------------------------

    if "," in s and "." not in s:

        try:
            return float(s.replace(",", "."))
        except ValueError:
            pass

    return np.nan


def convert_column_to_numeric(series):
    """
    Convert an entire pandas Series to numeric values.
    """

    return series.apply(clean_numeric_string)


# ============================================================
# TIMESTAMP DETECTION
# ============================================================

def looks_like_timestamp(value):
    """
    Empatica Unix timestamps are approximately in this range.
    """

    try:
        value = float(value)

        return 1e9 < value < 2e9

    except (ValueError, TypeError):

        return False


# ============================================================
# HEADER DETECTION
# ============================================================

def detect_header(path, signal_name):
    """
    Detect standard Empatica E4 two-line headers.

    Standard format:

        Unix start timestamp
        sampling frequency
        data...

    Returns:

        header_found
        rate
    """

    expected_rate = KNOWN_RATES[signal_name]

    try:

        with open(
            path,
            "r",
            errors="replace"
        ) as f:

            line1 = f.readline().strip()
            line2 = f.readline().strip()

        first_value = line1.split(",")[0].strip()
        second_value = line2.split(",")[0].strip()

        timestamp = clean_numeric_string(first_value)
        second = clean_numeric_string(second_value)

        if (
            not pd.isna(timestamp)
            and looks_like_timestamp(timestamp)
            and not pd.isna(second)
        ):

            relative_error = (
                abs(second - expected_rate)
                / expected_rate
            )

            if relative_error <= 0.15:

                return True, float(second)

    except Exception:
        pass

    return False, expected_rate


# ============================================================
# LOAD EDA / BVP / HR / TEMP / ACC
# ============================================================

def load_signal_file(path, signal_name):
    """
    Load one Empatica signal.

    Returns:

        dataframe with:
            t_sec
            value

        rate
        header_found
    """

    header_found, rate = detect_header(
        path,
        signal_name
    )

    skiprows = 2 if header_found else 0

    # --------------------------------------------------------
    # ACC
    # --------------------------------------------------------

    if signal_name == "ACC":

        df = pd.read_csv(
            path,
            header=None,
            skiprows=skiprows,
            names=["x", "y", "z"],
            dtype=str,
            engine="python",
        )

        df["x"] = convert_column_to_numeric(
            df["x"]
        )

        df["y"] = convert_column_to_numeric(
            df["y"]
        )

        df["z"] = convert_column_to_numeric(
            df["z"]
        )

        # Remove invalid ACC rows
        before = len(df)

        df = df.dropna(
            subset=["x", "y", "z"]
        ).copy()

        removed = before - len(df)

        if removed > 0:

            print(
                f"  {signal_name}: removed "
                f"{removed} invalid rows"
            )

        # ACC magnitude
        df["value"] = np.sqrt(
            df["x"] ** 2
            + df["y"] ** 2
            + df["z"] ** 2
        )

    # --------------------------------------------------------
    # OTHER SIGNALS
    # --------------------------------------------------------

    else:

        df = pd.read_csv(
            path,
            header=None,
            skiprows=skiprows,
            names=["raw_value"],
            dtype=str,
            engine="python",
        )

        before = len(df)

        df["value"] = convert_column_to_numeric(
            df["raw_value"]
        )

        df = df.dropna(
            subset=["value"]
        ).copy()

        removed = before - len(df)

        if removed > 0:

            print(
                f"  {signal_name}: repaired/removed "
                f"{removed} invalid numeric rows"
            )

    # --------------------------------------------------------
    # RESET INDEX
    # --------------------------------------------------------

    df = df.reset_index(drop=True)

    # --------------------------------------------------------
    # CREATE TIME AXIS
    # --------------------------------------------------------

    df["t_sec"] = (
        np.arange(len(df), dtype=float)
        / rate
    )

    result = df[
        ["t_sec", "value"]
    ].copy()

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print(
        f"  {signal_name}: "
        f"{len(result):,} samples"
    )

    if header_found:

        print(
            f"         detected header, "
            f"rate = {rate:.4f} Hz"
        )

    else:

        print(
            f"         no standard header detected, "
            f"using {rate:.4f} Hz"
        )

    return result, rate, header_found


# ============================================================
# LOAD IBI
# ============================================================

def load_ibi_file(path):
    """
    Load Empatica IBI.csv.

    Typical format:

        start_timestamp

        time_since_start, ibi_seconds

    IBI is event-based rather than regularly sampled.
    """

    try:

        with open(
            path,
            "r",
            errors="replace"
        ) as f:

            first_line = f.readline().strip()

        first_parts = first_line.split(",")

        first_value = clean_numeric_string(
            first_parts[0]
        )

        if (
            not pd.isna(first_value)
            and looks_like_timestamp(first_value)
        ):

            skiprows = 1

        else:

            skiprows = 0

    except Exception:

        skiprows = 0

    df = pd.read_csv(
        path,
        header=None,
        skiprows=skiprows,
        names=["t_sec", "ibi_sec"],
        dtype=str,
        engine="python",
    )

    df["t_sec"] = convert_column_to_numeric(
        df["t_sec"]
    )

    df["ibi_sec"] = convert_column_to_numeric(
        df["ibi_sec"]
    )

    df = df.dropna(
        subset=["t_sec", "ibi_sec"]
    ).copy()

    # --------------------------------------------------------
    # Remove physiologically unreasonable IBI values
    #
    # IBI is measured in seconds.
    # 0.2 sec = 300 BPM
    # 2.5 sec = 24 BPM
    # --------------------------------------------------------

    df = df[
        (df["ibi_sec"] > 0.2)
        & (df["ibi_sec"] < 2.5)
    ].copy()

    df = df.sort_values(
        "t_sec"
    )

    df = df.reset_index(
        drop=True
    )

    return df[
        ["t_sec", "ibi_sec"]
    ]


# ============================================================
# RESAMPLE SIGNAL INTO 5-SECOND WINDOWS
# ============================================================

def resample_to_grid(
    df,
    window_sec,
    signal_name
):
    """
    Aggregate signal values into fixed windows.

    Features:

        mean
        std
        min
        max
        count
    """

    if df.empty:

        return pd.DataFrame()

    data = df.copy()

    data["bin"] = np.floor(
        data["t_sec"]
        / window_sec
    ).astype(int)

    grouped = (
        data
        .groupby("bin")["value"]
        .agg(
            mean="mean",
            std="std",
            min="min",
            max="max",
            count="count",
        )
    )

    grouped.columns = [
        f"{signal_name}_{column}"
        for column in grouped.columns
    ]

    print(
        f"         {len(grouped):,} windows"
    )

    return grouped


# ============================================================
# PROCESS ONE SUBJECT
# ============================================================

def process_subject(
    subject_dir,
    subject_id
):

    print(
        f"\nProcessing {subject_id}..."
    )

    frames = []

    # --------------------------------------------------------
    # NORMAL SIGNALS
    # --------------------------------------------------------

    for signal in SIGNALS:

        path = os.path.join(
            subject_dir,
            f"{signal}.csv"
        )

        if not os.path.exists(path):

            print(
                f"  WARNING: "
                f"{signal}.csv not found"
            )

            continue

        try:

            df, rate, header_found = (
                load_signal_file(
                    path,
                    signal
                )
            )

            features = resample_to_grid(
                df,
                RESAMPLE_WINDOW_SEC,
                signal
            )

            if not features.empty:

                frames.append(
                    features
                )

        except Exception as e:

            print(
                f"  ERROR processing "
                f"{signal}.csv: {e}"
            )

    # --------------------------------------------------------
    # IBI
    # --------------------------------------------------------

    ibi_path = os.path.join(
        subject_dir,
        "IBI.csv"
    )

    if os.path.exists(ibi_path):

        try:

            ibi = load_ibi_file(
                ibi_path
            )

            print(
                f"  IBI: "
                f"{len(ibi):,} intervals"
            )

            if not ibi.empty:

                ibi["bin"] = np.floor(
                    ibi["t_sec"]
                    / RESAMPLE_WINDOW_SEC
                ).astype(int)

                ibi_features = (
                    ibi
                    .groupby("bin")["ibi_sec"]
                    .agg(
                        mean="mean",
                        std="std",
                        count="count",
                    )
                )

                ibi_features.columns = [
                    "IBI_mean",
                    "IBI_std",
                    "IBI_count",
                ]

                print(
                    f"        "
                    f"{len(ibi_features):,} windows"
                )

                frames.append(
                    ibi_features
                )

        except Exception as e:

            print(
                f"  ERROR processing "
                f"IBI.csv: {e}"
            )

    # --------------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------------

    if not frames:

        print(
            f"  WARNING: No usable data "
            f"for {subject_id}"
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # MERGE SIGNALS
    # --------------------------------------------------------

    merged = frames[0].copy()

    for frame in frames[1:]:

        merged = merged.join(
            frame,
            how="outer"
        )

    # --------------------------------------------------------
    # TIME INFORMATION
    # --------------------------------------------------------

    merged = (
        merged
        .reset_index()
        .rename(
            columns={
                "bin": "time_bin"
            }
        )
    )

    merged["subject_id"] = subject_id

    merged["time_sec"] = (
        merged["time_bin"]
        * RESAMPLE_WINDOW_SEC
    )

    # --------------------------------------------------------
    # COLUMN ORDER
    # --------------------------------------------------------

    identifier_columns = [
        "subject_id",
        "time_bin",
        "time_sec",
    ]

    feature_columns = [
        c
        for c in merged.columns
        if c not in identifier_columns
    ]

    merged = merged[
        identifier_columns
        + feature_columns
    ]

    return merged


# ============================================================
# QUALITY REPORT
# ============================================================

def quality_report(df):

    print(
        "\n==================================="
    )

    print(
        "DATA QUALITY REPORT"
    )

    print(
        "==================================="
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Subjects: "
        f"{df['subject_id'].nunique()}"
    )

    print(
        f"Duplicate rows: "
        f"{df.duplicated().sum()}"
    )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    print(
        "\nMissing percentage:"
    )

    missing = (
        df.isna()
        .mean()
        .mul(100)
        .round(2)
        .sort_values(
            ascending=False
        )
    )

    print(
        missing.to_string()
    )

    # --------------------------------------------------------
    # SIGNAL AVAILABILITY
    # --------------------------------------------------------

    print(
        "\nSignal availability:"
    )

    count_columns = [
        "EDA_count",
        "BVP_count",
        "HR_count",
        "TEMP_count",
        "ACC_count",
        "IBI_count",
    ]

    for column in count_columns:

        if column not in df.columns:

            continue

        expected = {
            "EDA_count": 20,
            "BVP_count": 320,
            "HR_count": 5,
            "TEMP_count": 20,
            "ACC_count": 160,
        }.get(column)

        if expected is not None:

            percentage = (
                df[column]
                .ge(expected)
                .mean()
                * 100
            )

            print(
                f"  {column}: "
                f"{percentage:.2f}% "
                f"full windows"
            )

        else:

            percentage = (
                df[column]
                .notna()
                .mean()
                * 100
            )

            print(
                f"  {column}: "
                f"{percentage:.2f}% "
                f"windows contain data"
            )


# ============================================================
# MAIN CLEANING FUNCTION
# ============================================================

def clean(root, out_path):

    subject_dirs = sorted(
        glob.glob(
            os.path.join(
                root,
                "subject_*"
            )
        )
    )

    print(
        f"Found "
        f"{len(subject_dirs)} "
        f"subject folders"
    )

    all_subjects = []

    # --------------------------------------------------------
    # PROCESS EVERY SUBJECT
    # --------------------------------------------------------

    for subject_dir in subject_dirs:

        subject_id = os.path.basename(
            subject_dir.rstrip(os.sep)
        )

        try:

            result = process_subject(
                subject_dir,
                subject_id
            )

            if not result.empty:

                all_subjects.append(
                    result
                )

        except Exception as e:

            print(
                f"\nERROR processing "
                f"{subject_id}: {e}"
            )

            # Do NOT stop the entire pipeline.
            continue

    # --------------------------------------------------------
    # CHECK OUTPUT
    # --------------------------------------------------------

    if not all_subjects:

        print(
            "\nNo data produced. "
            "Check the root path."
        )

        return

    # --------------------------------------------------------
    # COMBINE ALL SUBJECTS
    # --------------------------------------------------------

    final_df = pd.concat(
        all_subjects,
        ignore_index=True
    )

    # --------------------------------------------------------
    # REMOVE COMPLETELY EMPTY FEATURE ROWS
    # --------------------------------------------------------

    identifier_columns = [
        "subject_id",
        "time_bin",
        "time_sec",
    ]

    feature_columns = [
        c
        for c in final_df.columns
        if c not in identifier_columns
    ]

    before = len(final_df)

    final_df = final_df.dropna(
        how="all",
        subset=feature_columns
    )

    removed = (
        before
        - len(final_df)
    )

    print(
        f"\nRemoved "
        f"{removed:,} "
        f"completely empty rows"
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    final_df = final_df.sort_values(
        [
            "subject_id",
            "time_bin"
        ]
    )

    final_df = final_df.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # QUALITY REPORT
    # --------------------------------------------------------

    quality_report(
        final_df
    )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    output_directory = os.path.dirname(
        os.path.abspath(out_path)
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    final_df.to_csv(
        out_path,
        index=False,
        float_format="%.8f"
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print(
        "\n==================================="
    )

    print(
        "CLEANING COMPLETE"
    )

    print(
        "==================================="
    )

    print(
        f"Rows: {len(final_df):,}"
    )

    print(
        f"Subjects: "
        f"{final_df['subject_id'].nunique()}"
    )

    print(
        f"Output: {out_path}"
    )

    print(
        "\nColumns:"
    )

    for column in final_df.columns:

        print(
            f"  {column}"
        )

    print(
        "\nRows per subject:"
    )

    print(
        final_df[
            "subject_id"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# COMMAND LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Clean Empatica E4 "
            "subject data"
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        help=(
            "Path to subjects directory"
        )
    )

    parser.add_argument(
        "--out",
        default="empatica4_clean.csv",
        help=(
            "Output CSV path"
        )
    )

    args = parser.parse_args()

    clean(
        args.root,
        args.out
    )