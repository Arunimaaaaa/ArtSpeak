"""
Gaze Attention & Engagement Classifier (v2 Enhanced)
===================================================
Trained on Gazedata_final_saliency4asd_features.csv for ArtSpeak Layer 1 (Non-Verbal Input Layer).

Key Improvements in v2:
  1. Kinematic Gaze Feature Engineering:
     - saccade_length_per_fixation: Saccadic step length (scanpath / fixations)
     - dwell_per_fixation: Average dwell per fixation
     - fixation_variability_ratio: Dispersion of individual fixation durations
     - center_dist: Foveation distance from screen center (center bias vs boundary avoidance)
     - exploration_density: Fixation density per unit spatial spread
     - spread_to_path_ratio: Compactness of visual search trajectory
     - dwell_spread_product: Total exploration volume
     - fixation_entropy_proxy: Variability weighted by spatial dispersion
  2. Regularized Gradient Boosting Pipeline:
     GradientBoosting with depth=3, subsample=0.8, min_samples_leaf=20, max_features=0.8.
  3. Anti-Overfitting Protocol:
     Strict 5-Fold StratifiedGroupKFold by image_id (300 distinct stimulus images).
  4. Diagnostic Artifacts:
     - Learning curves (train vs out-of-fold validation)
     - Confusion Matrix
     - ROC Curve & AUC
     - Top Feature Importances

Usage:
    python train_gaze_classifier_v2.py --input "data_prep/Gazedata _final_saliency4asd_features.csv" --outdir ml-models/gaze/results
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def engineer_gaze_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain-specific kinematic gaze features from raw fixation metrics.
    All features rely purely on (x, y, t) coordinates streamed from MediaPipe.
    """
    df_feat = df.copy()

    # 1. Saccadic step length
    df_feat["saccade_length_per_fixation"] = (
        df_feat["scanpath_length"] / (df_feat["n_fixations"] + 1e-5)
    )

    # 2. Dwell per fixation
    df_feat["dwell_per_fixation"] = (
        df_feat["total_dwell_ms_log"] / (df_feat["n_fixations"] + 1e-5)
    )

    # 3. Fixation duration volatility ratio
    df_feat["fixation_variability_ratio"] = (
        df_feat["std_fixation_ms_log"] / (df_feat["mean_fixation_ms_log"] + 1e-5)
    )

    # 4. Center foveation distance
    df_feat["center_dist"] = np.sqrt(
        (df_feat["mean_x_norm"] - 0.5) ** 2 + (df_feat["mean_y_norm"] - 0.5) ** 2
    )

    # 5. Visual exploration density
    df_feat["exploration_density"] = (
        df_feat["n_fixations"] / (df_feat["spatial_spread"] + 1e-5)
    )

    # 6. Spatial spread to scanpath ratio
    df_feat["spread_to_path_ratio"] = (
        df_feat["spatial_spread"] / (df_feat["scanpath_length"] + 1e-5)
    )

    # 7. Total exploration volume
    df_feat["dwell_spread_product"] = (
        df_feat["total_dwell_ms_log"] * df_feat["spatial_spread"]
    )

    # 8. Gaze entropy proxy
    df_feat["fixation_entropy_proxy"] = (
        (df_feat["spatial_spread"] * df_feat["std_fixation_ms_log"])
        / (df_feat["mean_fixation_ms_log"] + 1e-5)
    )

    return df_feat


def plot_learning_curve(pipeline, X, y, groups, out_path: str):
    """
    Generate learning curve to visually and numerically verify no overfitting.
    """
    print("\nComputing Learning Curve across training sizes...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline,
        X,
        y,
        groups=groups,
        cv=sgkf,
        train_sizes=np.linspace(0.5, 1.0, 5),
        scoring="roc_auc",
        n_jobs=-1,
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, "o-", color="#1976D2", label="Training ROC-AUC")
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="#1976D2",
    )
    plt.plot(
        train_sizes,
        val_mean,
        "o-",
        color="#388E3C",
        label="Validation ROC-AUC (StratifiedGroupKFold)",
    )
    plt.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color="#388E3C",
    )

    plt.xlabel("Number of Training Samples", fontsize=11)
    plt.ylabel("ROC-AUC Score", fontsize=11)
    plt.title(
        "Gaze Classifier (v2): Learning Curve (5-Fold StratifiedGroupKFold by Image)",
        fontsize=12,
        fontweight="bold",
    )
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    gap = train_mean[-1] - val_mean[-1]
    print(f"Learning curve saved to: {out_path}")
    print(f"Final Train ROC-AUC: {train_mean[-1]:.4f} | Validation ROC-AUC: {val_mean[-1]:.4f}")
    print(f"Train-Validation Gap: {gap:.4f}")
    return float(train_mean[-1]), float(val_mean[-1]), float(gap)


def train_and_evaluate(input_csv: str, out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("GAZE CLASSIFIER (v2 ENHANCED) TRAINING & EVALUATION")
    print("=" * 75)

    csv_path = Path(input_csv)
    if not csv_path.exists():
        alt_paths = [
            Path("data_prep/Gazedata _final_saliency4asd_features.csv"),
            Path("data_prep/Gazedata_final_saliency4asd_features.csv"),
            Path("data_prep/saliency4asd_clean.csv"),
        ]
        for alt in alt_paths:
            if alt.exists():
                csv_path = alt
                break
    print(f"Loading data from: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    print(f"Raw dataset shape: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns across {df_raw['image_id'].nunique()} images")

    # Encode target: ASD = 1, TD = 0
    y = (df_raw["group"] == "ASD").astype(int).values
    groups = df_raw["image_id"].values

    print("\nClass Distribution:")
    counts = df_raw["group"].value_counts()
    for cls_name, cnt in counts.items():
        print(f"  {cls_name}: {cnt:,} samples ({cnt / len(y) * 100:.2f}%)")

    # Perform feature engineering
    print("\nEngineering kinematic gaze features...")
    df_eng = engineer_gaze_features(df_raw)
    
    feature_cols = [c for c in df_eng.columns if c not in ["group", "image_id"]]
    X = df_eng[feature_cols].copy()

    print(f"Total features after engineering: {len(feature_cols)}")
    print(f"Features: {feature_cols}")

    # Save feature columns
    with open(out_path / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Candidate Models
    models = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=130,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            max_features=0.8,
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=120,
            max_depth=4,
            min_samples_leaf=25,
            l2_regularization=3.0,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=160,
            max_depth=6,
            min_samples_leaf=15,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    best_name = None
    best_auc = -1.0
    best_pipeline = None
    best_oof_proba = None
    best_oof_pred = None

    print("\n" + "-" * 75)
    print("5-FOLD OUT-OF-IMAGE StratifiedGroupKFold MODEL COMPARISON")
    print("-" * 75)

    for name, clf in models.items():
        pipeline = Pipeline([
            ("scaler", RobustScaler()),
            ("model", clf),
        ])

        oof_proba = cross_val_predict(
            pipeline, X, y, groups=groups, cv=sgkf, method="predict_proba", n_jobs=-1
        )[:, 1]
        oof_pred = (oof_proba >= 0.5).astype(int)

        acc = accuracy_score(y, oof_pred)
        bal_acc = balanced_accuracy_score(y, oof_pred)
        f1 = f1_score(y, oof_pred)
        auc = roc_auc_score(y, oof_proba)

        print(f"[{name:22s}] OOF Accuracy: {acc:.4f} ({acc*100:.2f}%) | Balanced Acc: {bal_acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f} ({auc*100:.2f}%)")

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_pipeline = pipeline
            best_oof_proba = oof_proba
            best_oof_pred = oof_pred

    print(f"\n--> Selected Best Model Architecture: {best_name} (OOF ROC-AUC: {best_auc:.4f})")

    # Fit final model on complete dataset
    print(f"\nFitting final {best_name} pipeline on complete dataset...")
    best_pipeline.fit(X, y)

    # Save model artifact
    model_artifact_path = out_path / "gaze_asd_classifier.joblib"
    joblib.dump(best_pipeline, model_artifact_path)
    print(f"Production pipeline saved to: {model_artifact_path}")

    # Generate Learning Curve Diagnostic
    train_auc, val_auc, gap = plot_learning_curve(
        best_pipeline, X, y, groups, str(out_path / "learning_curve.png")
    )

    # Plot Confusion Matrix
    cm = confusion_matrix(y, best_oof_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["TD", "ASD"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Gaze Classifier (v2): Out-of-Fold Confusion Matrix ({best_name})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path / "confusion_matrix.png", dpi=200)
    plt.close()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y, best_oof_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#E65100", lw=2, label=f"ROC Curve (AUC = {best_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title(f"Gaze Classifier (v2): Out-of-Image ROC Curve ({best_name})", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path / "roc_curve.png", dpi=200)
    plt.close()

    # Feature importances
    underlying_model = best_pipeline.named_steps["model"]
    if hasattr(underlying_model, "feature_importances_"):
        importances = underlying_model.feature_importances_
    else:
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(best_pipeline, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        importances = perm.importances_mean

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices][::-1], align="center", color="#0288D1")
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices][::-1], fontsize=9)
    plt.xlabel("Feature Importance Score", fontsize=11)
    plt.title(f"Gaze Classifier (v2): Top Kinematic Indicators ({best_name})", fontsize=12, fontweight="bold")
    plt.grid(True, axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path / "feature_importance.png", dpi=200)
    plt.close()

    # Save metrics JSON
    acc = accuracy_score(y, best_oof_pred)
    bal_acc = balanced_accuracy_score(y, best_oof_pred)
    prec = precision_score(y, best_oof_pred)
    rec = recall_score(y, best_oof_pred)
    f1 = f1_score(y, best_oof_pred)

    metrics = {
        "model_architecture": best_name,
        "n_samples": int(len(X)),
        "n_features": int(len(feature_cols)),
        "n_images": int(df_raw["image_id"].nunique()),
        "validation_strategy": "5-Fold StratifiedGroupKFold by image_id",
        "out_of_fold_accuracy": float(acc),
        "out_of_fold_balanced_accuracy": float(bal_acc),
        "out_of_fold_precision": float(prec),
        "out_of_fold_recall": float(rec),
        "out_of_fold_f1_score": float(f1),
        "out_of_fold_roc_auc": float(best_auc),
        "train_roc_auc": float(train_auc),
        "val_roc_auc": float(val_auc),
        "train_val_gap": float(gap),
        "fit_verdict": "healthy convergence, reasonable gap for this dataset size." if gap < 0.10 else "potential overfit",
        "feature_columns": feature_cols,
        "class_counts": {str(k): int(v) for k, v in counts.items()},
    }

    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 75)
    print("FINAL GAZE (v2) MODEL EVALUATION METRICS:")
    print("=" * 75)
    for k, v in metrics.items():
        if k != "feature_columns":
            print(f"  {k:30s}: {v}")
    print("=" * 75)
    print(f"All artifacts successfully written to: {out_path.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gaze Classifier v2")
    parser.add_argument(
        "--input",
        default="data_prep/Gazedata _final_saliency4asd_features.csv",
        help="Path to gaze features CSV",
    )
    parser.add_argument(
        "--outdir",
        default="ml-models/gaze/results",
        help="Directory to save output model and diagnostics",
    )
    args = parser.parse_args()
    train_and_evaluate(args.input, args.outdir)
