"""
Empatica E4 dataset - full inspection pass across all subjects
------------------------------------------------------------------
This dataset has been shown to be INCONSISTENT: some signal files have
the standard 2-line header (timestamp, sample_rate), some don't, and IBI
sometimes has a text header and sometimes doesn't - even across subjects
in the same download. This script checks every subject/signal combination
and reports which pattern it follows, so cleaning decisions are based on
the real data rather than a guess from a handful of subjects.

Known nominal Empatica E4 sample rates (used whenever a header is absent):
    EDA=4Hz, BVP=64Hz, HR=1Hz, TEMP=4Hz, ACC=32Hz
IBI is event-based (irregular), not a fixed rate.

Usage:
    python inspect_empatica_e4.py --root /Users/akankshajadhav/Downloads/subjects
"""

import argparse
import glob
import os

NOMINAL_RATES = {"EDA": 4.0, "BVP": 64.0, "HR": 1.0, "TEMP": 4.0, "ACC": 32.0}
TIMESTAMP_THRESHOLD = 1e8  # anything bigger than this is almost certainly a unix timestamp, not a data value


def _try_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def detect_singlecol(path: str, signal: str) -> dict:
    """For EDA, BVP, HR, TEMP - one value per line."""
    with open(path) as f:
        line1 = f.readline().strip()
        line2 = f.readline().strip()

    v1 = _try_float(line1)
    v2 = _try_float(line2)
    has_header = v1 is not None and v1 > TIMESTAMP_THRESHOLD

    if has_header:
        rate = v2 if v2 is not None else NOMINAL_RATES[signal]
        return {"header": True, "timestamp": v1, "rate": rate, "header_lines": 2}
    else:
        return {"header": False, "timestamp": None, "rate": NOMINAL_RATES[signal], "header_lines": 0}


def detect_acc(path: str) -> dict:
    """ACC has 3 comma-separated values per line (x,y,z)."""
    with open(path) as f:
        line1 = f.readline().strip()
        line2 = f.readline().strip()

    fields1 = [x.strip() for x in line1.split(",")]
    v1 = _try_float(fields1[0]) if fields1 else None
    has_header = v1 is not None and v1 > TIMESTAMP_THRESHOLD

    if has_header:
        fields2 = [x.strip() for x in line2.split(",")]
        rate = _try_float(fields2[0]) if fields2 else NOMINAL_RATES["ACC"]
        return {"header": True, "timestamp": v1, "rate": rate or NOMINAL_RATES["ACC"], "header_lines": 2}
    else:
        return {"header": False, "timestamp": None, "rate": NOMINAL_RATES["ACC"], "header_lines": 0}


def detect_ibi(path: str) -> dict:
    """IBI: irregular events. Header (if present) looks like 'timestamp, IBI' (text)."""
    with open(path) as f:
        line1 = f.readline().strip()

    fields1 = [x.strip() for x in line1.split(",")]
    if len(fields1) >= 2:
        v1 = _try_float(fields1[0])
        v2 = _try_float(fields1[1])
        # header if second field is NOT a number (i.e. literally "IBI")
        has_header = v1 is not None and v2 is None
    else:
        has_header = False
        v1 = None

    return {"header": has_header, "timestamp": v1 if has_header else None, "rate": None,
             "header_lines": 1 if has_header else 0}


def inspect_subject(subject_dir: str) -> dict:
    result = {}
    for signal in ("EDA", "BVP", "HR", "TEMP"):
        path = os.path.join(subject_dir, f"{signal}.csv")
        if os.path.exists(path):
            result[signal] = detect_singlecol(path, signal)

    acc_path = os.path.join(subject_dir, "ACC.csv")
    if os.path.exists(acc_path):
        result["ACC"] = detect_acc(acc_path)

    ibi_path = os.path.join(subject_dir, "IBI.csv")
    if os.path.exists(ibi_path):
        result["IBI"] = detect_ibi(ibi_path)

    return result


def main(root: str):
    subject_dirs = sorted(glob.glob(os.path.join(root, "subject_*")))
    print(f"Found {len(subject_dirs)} subject folders.\n")

    # tally how many subjects follow each pattern, per signal
    pattern_counts = {sig: {"header": 0, "no_header": 0} for sig in
                       ("EDA", "BVP", "HR", "TEMP", "ACC", "IBI")}
    flagged_subjects = []

    for sdir in subject_dirs:
        subj_name = os.path.basename(sdir)
        result = inspect_subject(sdir)

        for sig, info in result.items():
            key = "header" if info["header"] else "no_header"
            pattern_counts[sig][key] += 1

        # flag subjects where different signals disagree on header presence
        header_flags = {sig: info["header"] for sig, info in result.items()}
        if len(set(header_flags.values())) > 1:
            flagged_subjects.append((subj_name, header_flags))

    print("=== Summary across all subjects ===")
    for sig, counts in pattern_counts.items():
        total = counts["header"] + counts["no_header"]
        if total == 0:
            continue
        print(f"  {sig:5s}: header present in {counts['header']}/{total} subjects, "
              f"no header in {counts['no_header']}/{total}")

    print(f"\n=== Subjects with MIXED header presence across their own signals ({len(flagged_subjects)}) ===")
    for name, flags in flagged_subjects:
        print(f"  {name}: {flags}")

    if not flagged_subjects:
        print("  None - every subject is internally consistent (all signals same pattern).")

    print("\nDone. Use this summary to decide the cleaning rule per signal type.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to the 'subjects' folder")
    args = parser.parse_args()
    main(args.root)