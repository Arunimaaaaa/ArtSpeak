"""
Gaze classifier training - matched to Gazedata_final_saliency4asd_features.csv
---------------------------------------------------------------------------------
This file's schema is already fully prepped (filtered to >=3 fixations,
duration features already log-transformed), so unlike earlier drafts this
script does NOT re-filter or re-transform anything - it trains directly.

Anti-overfitting / anti-underfitting measures, all deliberate:
  1. GroupKFold by image_id (NOT random split) - prevents the same
     stimulus image appearing in both train and validation.
  2. Shallow, narrow hyperparameter grid - this dataset (~7000 rows,
     300 images) cannot support deep unconstrained trees without
     memorizing image-specific noise.
  3. RobustScaler - features have different scales and some skew
     remains even after log-transform; robust scaling limits the
     influence of outlier sessions.
  4. Learning curve diagnostic - visually confirms whether train/val
     accuracy are converging (healthy) or diverging (overfitting) or
     both stuck low (underfitting/weak signal).
  5. class_weight='balanced' applied automatically since the classes
     are close but not perfectly even (3306 vs 3655).

Usage:
    python train_gaze_classifier_final.py --input Gazedata_final_saliency4asd_features.csv --outdir model_out
"""

import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

TARGET_COL = "group"
GROUP_COL = "image_id"
FEATURE_COLS = [
    "n_fixations",
    "mean_x_norm",
    "mean_y_norm",
    "spatial_spread",
    "scanpath_length",
    "total_dwell_ms_log",
    "mean_fixation_ms_log",
    "std_fixation_ms_log",
]


def make_learning_curve(pipeline, X, y, groups, outpath):
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y,
        groups=groups,
        cv=GroupKFold(n_splits=5),
        train_sizes=np.linspace(0.2, 1.0, 6),
        scoring="accuracy",
        n_jobs=-1,
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_mean, "o-", label="Training accuracy")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    plt.plot(train_sizes, val_mean, "o-", label="Validation accuracy")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title("Learning curve (GroupKFold by image_id)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

    gap = train_mean[-1] - val_mean[-1]
    print(f"\nLearning curve saved -> {outpath}")
    print(f"Final training accuracy:   {train_mean[-1]:.3f}")
    print(f"Final validation accuracy: {val_mean[-1]:.3f}")
    print(f"Train-val gap: {gap:.3f}", end="  ")
    if gap > 0.15:
        verdict = "OVERFITTING - reduce max_depth / increase min_samples_leaf, or gather more images."
    elif val_mean[-1] < 0.60:
        verdict = "UNDERFITTING or weak signal - features may not separate the classes well at this size."
    else:
        verdict = "reasonable gap for this dataset size."
    print(f"-> {verdict}")
    return gap, val_mean[-1], verdict


def main(input_path: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(input_path)

    missing = [c for c in FEATURE_COLS + [TARGET_COL, GROUP_COL] if c not in df.columns]
    if missing:
        raise SystemExit(f"Input file is missing expected columns: {missing}")

    print(f"Loaded {len(df)} rows, {df[GROUP_COL].nunique()} unique images")
    print("\nClass balance:")
    print(df[TARGET_COL].value_counts())

    X = df[FEATURE_COLS].values
    y = (df[TARGET_COL] == "ASD").astype(int).values  # ASD=1, TD=0
    groups = df[GROUP_COL].values

    class_counts = np.bincount(y)
    imbalance_ratio = min(class_counts) / max(class_counts)
    class_weight = "balanced" if imbalance_ratio < 0.9 else None
    print(f"\nClass imbalance ratio: {imbalance_ratio:.3f} -> "
          f"{'using class_weight=balanced' if class_weight else 'no reweighting needed'}")

    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", RandomForestClassifier(random_state=42, class_weight=class_weight)),
    ])

    # Deliberately shallow - this dataset size cannot support deep trees
    # without overfitting to image-specific noise.
    param_grid = {
        "clf__n_estimators": [150, 300],
        "clf__max_depth": [4, 6, 8],
        "clf__min_samples_leaf": [5, 10, 20],
    }

    gkf = GroupKFold(n_splits=5)
    # IMPORTANT: materialize the splits into a list, not a generator.
    # GridSearchCV with n_jobs=-1 pickles the cv argument to send to worker
    # processes - a generator can't be pickled (fails loudly on newer
    # Python/joblib versions), a list of (train_idx, test_idx) arrays can.
    cv_splits = list(gkf.split(X, y, groups=groups))
    search = GridSearchCV(
        pipeline, param_grid,
        cv=cv_splits,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X, y)

    print(f"\nBest params: {search.best_params_}")
    print(f"Best cross-validated accuracy: {search.best_score_:.3f}")

    best_pipeline = search.best_estimator_

    gap, final_val_acc, verdict = make_learning_curve(
        best_pipeline, X, y, groups, os.path.join(outdir, "learning_curve.png")
    )

    y_pred = cross_val_predict(best_pipeline, X, y, cv=GroupKFold(n_splits=5), groups=groups, n_jobs=-1)
    y_proba = cross_val_predict(best_pipeline, X, y, cv=GroupKFold(n_splits=5), groups=groups,
                                 n_jobs=-1, method="predict_proba")[:, 1]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    report = classification_report(y, y_pred, target_names=["TD", "ASD"])

    print(f"\nOut-of-fold accuracy: {acc:.3f}")
    print(f"Out-of-fold ROC-AUC:  {auc:.3f}")
    print(report)

    cm = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["TD", "ASD"]).plot()
    plt.title("Confusion matrix (out-of-fold, GroupKFold by image)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # feature importances - useful for the report's discussion section
    best_pipeline.fit(X, y)
    importances = best_pipeline.named_steps["clf"].feature_importances_
    importance_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances}) \
        .sort_values("importance", ascending=False)
    print("\nFeature importances:")
    print(importance_df.to_string(index=False))

    joblib.dump(best_pipeline, os.path.join(outdir, "gaze_asd_classifier.joblib"))

    metrics = {
        "best_params": search.best_params_,
        "cv_accuracy": search.best_score_,
        "out_of_fold_accuracy": acc,
        "out_of_fold_roc_auc": auc,
        "train_val_gap": gap,
        "fit_verdict": verdict,
        "n_rows_used": len(df),
        "n_images": df[GROUP_COL].nunique(),
        "feature_columns": FEATURE_COLS,
        "class_counts": {"TD": int(class_counts[0]), "ASD": int(class_counts[1])},
        "feature_importances": importance_df.set_index("feature")["importance"].to_dict(),
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model -> {outdir}/gaze_asd_classifier.joblib")
    print(f"Saved metrics -> {outdir}/metrics.json")
    print(f"Saved plots -> {outdir}/learning_curve.png, {outdir}/confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="model_out")
    args = parser.parse_args()
    main(args.input, args.outdir)