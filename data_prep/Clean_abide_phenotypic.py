"""
ABIDE I + II phenotypic data -> clean, merged, training-ready CSV
---------------------------------------------------------------------
Scope note: this deliberately ignores the raw .nii imaging files bundled
in the ABIDE download. Your project's report scopes ABIDE as "phenotypic
behavioural baselines" - the four downstream layers never consume raw
MRI, so processing it would be out-of-scope work for no pipeline benefit.

What this script does with the two phenotypic CSVs:
  1. Strips whitespace from column names (ABIDE II ships "AGE_AT_SCAN "
     with a trailing space - a classic silent join-key bug if untouched).
  2. Maps the numeric codes to readable labels using the STANDARD ABIDE
     convention: DX_GROUP 1=Autism, 2=Control; SEX 1=Male, 2=Female.
     VERIFY this against any readme bundled with your specific download -
     it is the standard convention but not verified against your exact
     file's own documentation.
  3. Tags each row with its source cohort (ABIDE_I / ABIDE_II) - kept
     because these are different scan sites/protocols, which matters if
     you ever want to check for site effects.
  4. Checks for duplicate SUB_IDs (within each file and across both).
  5. Checks for missing/implausible ages.
  6. Concatenates into one clean file.

Honest limitation flagged in the output: this phenotypic file only has
demographics (age, sex, diagnosis) - not clinical behavioural scores
(ADOS/SRS/IQ). Treat it as covariate/stratification data, not a
standalone predictive dataset.

Usage:
    python clean_abide_phenotypic.py --abide1 /path/to/abide1_data.csv --abide2 /path/to/abide2_data.csv --out data_prep/abide_clean.csv
"""

import argparse

import pandas as pd

DX_GROUP_MAP = {1: "ASD", 2: "TD"}
SEX_MAP = {1: "Male", 2: "Female"}

PLAUSIBLE_AGE_RANGE = (2, 80)  # generous bounds just to catch data-entry errors


def load_and_standardize(path: str, cohort_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]  # fixes "AGE_AT_SCAN " -> "AGE_AT_SCAN"

    required = {"SUB_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing expected columns: {missing}")

    df = df.rename(columns={
        "SUB_ID": "sub_id",
        "DX_GROUP": "dx_group_raw",
        "AGE_AT_SCAN": "age",
        "SEX": "sex_raw",
    })

    unmapped_dx = set(df["dx_group_raw"].unique()) - set(DX_GROUP_MAP.keys())
    unmapped_sex = set(df["sex_raw"].unique()) - set(SEX_MAP.keys())
    if unmapped_dx:
        print(f"  WARNING [{cohort_label}]: unexpected DX_GROUP codes {unmapped_dx}, left unmapped")
    if unmapped_sex:
        print(f"  WARNING [{cohort_label}]: unexpected SEX codes {unmapped_sex}, left unmapped")

    df["diagnosis"] = df["dx_group_raw"].map(DX_GROUP_MAP)
    df["sex"] = df["sex_raw"].map(SEX_MAP)
    df["cohort"] = cohort_label

    dupes = df["sub_id"].duplicated().sum()
    if dupes:
        print(f"  WARNING [{cohort_label}]: {dupes} duplicate sub_id values within this file")

    bad_age = df[(df["age"] < PLAUSIBLE_AGE_RANGE[0]) | (df["age"] > PLAUSIBLE_AGE_RANGE[1])]
    if len(bad_age):
        print(f"  WARNING [{cohort_label}]: {len(bad_age)} rows with implausible age "
              f"(outside {PLAUSIBLE_AGE_RANGE}): {bad_age['age'].tolist()}")

    missing_vals = df[["dx_group_raw", "age", "sex_raw"]].isna().sum()
    if missing_vals.sum():
        print(f"  WARNING [{cohort_label}]: missing values found:\n{missing_vals[missing_vals > 0]}")

    return df[["sub_id", "cohort", "diagnosis", "age", "sex"]]


def main(abide1_path: str, abide2_path: str, out_path: str):
    print("Processing ABIDE I...")
    df1 = load_and_standardize(abide1_path, "ABIDE_I")
    print(f"  {len(df1)} rows loaded")

    print("\nProcessing ABIDE II...")
    df2 = load_and_standardize(abide2_path, "ABIDE_II")
    print(f"  {len(df2)} rows loaded")

    combined = pd.concat([df1, df2], ignore_index=True)

    cross_dupes = combined["sub_id"].duplicated().sum()
    print(f"\nCross-cohort duplicate sub_id check: {cross_dupes} found "
          f"{'(expected 0 - these are different scan sites)' if cross_dupes == 0 else '- INVESTIGATE, sub_id may not be globally unique'}")

    print(f"\nCombined total: {len(combined)} rows")
    print("\nDiagnosis balance:")
    print(combined["diagnosis"].value_counts())
    print("\nSex balance:")
    print(combined["sex"].value_counts())
    print("\nAge summary:")
    print(combined["age"].describe())
    print("\nRows per cohort:")
    print(combined["cohort"].value_counts())

    combined.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print("\nNOTE: this file contains demographics only (age/sex/diagnosis), "
          "not clinical behavioural scores (ADOS/SRS/IQ). Use as covariate/"
          "stratification data, not a standalone predictive dataset. "
          "Document this limitation in your report.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--abide1", required=True)
    parser.add_argument("--abide2", required=True)
    parser.add_argument("--out", default="abide_clean.csv")
    args = parser.parse_args()
    main(args.abide1, args.abide2, args.out)