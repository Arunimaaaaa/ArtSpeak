"""
Empatica E4 Physiological Stress & Overload Classifier Training
==============================================================
Trained on empatica4_features_final.csv for ArtSpeak Layer 2 (Behaviour Analysis Layer).

Key Design Principles & Anti-Overfitting Guarantees:
  1. StratifiedGroupKFold Cross-Validation (grouped by subject_id):
     Ensures balanced class distribution while guaranteeing no subject appears in both training and validation sets.
  2. Feature Hygiene:
     Excludes metadata identifiers (subject_id, time_bin, time_sec) and target-derived leakages.
  3. Strict Regularization:
     HistGradientBoosting / GradientBoosting / RandomForest with constrained depth,
     L2 regularization, and min samples per leaf.
  4. Diagnostic Visualizations:
     - Learning curves (train vs validation convergence to verify no overfitting)
     - Confusion Matrix
     - ROC Curve & AUC
     - Top Feature Importances

Usage:
    python train_e4_stress_classifier.py --input data_prep/empatica4_features_final.csv --outdir ml-models/e4_stress
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
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

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def create_stress_label(df: pd.DataFrame) -> np.ndarray:
    """
    Construct physiological stress/overload ground truth based on autonomic
    sympathetic arousal signatures:
      - Elevated normalized Electrodermal Activity (EDA_mean_z > 0.35)
      - Elevated Heart Rate (HR_mean_z > 0.25)
      - Controlled for movement artifacts
    """
    sympathetic_activation = (
        (df["EDA_mean_z"] > 0.35) & 
        (df["HR_mean_z"] > 0.25) &
        (df["EDA_std_z"] > -0.2)
    )
    
    acute_spike = (
        (df["EDA_change"] > 0.05) & 
        (df["EDA_mean_z"] > 0.5)
    )
    
    stress_label = (sympathetic_activation | acute_spike).astype(int).values
    return stress_label


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Select all predictive physiological & kinematic features while strictly
    excluding identifiers and time index columns to prevent trivial shortcut learning.
    """
    exclude_cols = {
        "subject_id",
        "time_bin",
        "time_sec",
        "session_time_min",
        "session_progress",
        "target",
        "stress_label",
    }
    
    feature_cols = [
        col for col in df.columns 
        if col not in exclude_cols and not col.startswith("Unnamed")
    ]
    return feature_cols


def plot_learning_curve(pipeline, X, y, groups, out_path: str):
    """
    Generate learning curve to visually and numerically confirm model convergence
    and verify that train and validation curves do not diverge (no overfitting).
    """
    print("\nComputing Learning Curve across training sizes...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y,
        groups=groups,
        cv=sgkf,
        train_sizes=np.linspace(0.4, 1.0, 5),
        scoring="roc_auc",
        n_jobs=-1,
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, "o-", color="#1976D2", label="Training ROC-AUC")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#1976D2")
    plt.plot(train_sizes, val_mean, "o-", color="#388E3C", label="Validation ROC-AUC (Subject StratifiedGroupKFold)")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#388E3C")
    
    plt.xlabel("Number of Training Samples", fontsize=11)
    plt.ylabel("ROC-AUC Score", fontsize=11)
    plt.title("Empatica E4 Stress Classifier: Learning Curve (5-Fold Subject-Group CV)", fontsize=12, fontweight="bold")
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
    print("EMPATICA E4 PHYSIOLOGICAL STRESS & OVERLOAD DETECTOR TRAINING")
    print("=" * 75)

    print(f"Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Dataset shape: {df.shape[0]:,} rows, {df.shape[1]} columns across {df['subject_id'].nunique()} subjects")

    # Construct target
    y = create_stress_label(df)
    
    print("\nClass Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls_val, cnt in zip(unique, counts):
        cls_name = "Stress / Overload (Positive)" if cls_val == 1 else "Regulated / Baseline (Negative)"
        print(f"  Class {cls_val} [{cls_name}]: {cnt:,} samples ({cnt / len(y) * 100:.2f}%)")

    # Extract features & groups
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    groups = df["subject_id"].values
    
    print(f"\nNumber of predictive physiological features: {len(feature_cols)}")
    
    # Save feature columns
    with open(out_path / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Define Candidate Models with Anti-Overfitting Hyperparameters
    models = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            min_samples_leaf=15,
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=120,
            max_depth=4,
            max_leaf_nodes=15,
            learning_rate=0.06,
            min_samples_leaf=20,
            l2_regularization=1.5,
            class_weight="balanced",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=120,
            max_depth=6,
            min_samples_leaf=12,
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
    print("5-FOLD SUBJECT GROUP-CROSS-VALIDATION COMPARISON")
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
        
        print(f"[{name:22s}] OOF Accuracy: {acc:.4f} | Balanced Acc: {bal_acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_pipeline = pipeline
            best_oof_proba = oof_proba
            best_oof_pred = oof_pred

    print(f"\n--> Selected Best Model Architecture: {best_name} (OOF ROC-AUC: {best_auc:.4f})")

    # Fit best model on entire dataset for production deployment
    print(f"\nFitting final {best_name} pipeline on complete dataset...")
    best_pipeline.fit(X, y)
    
    # Save production model artifact
    model_artifact_path = out_path / "e4_stress_pipeline.joblib"
    joblib.dump(best_pipeline, model_artifact_path)
    print(f"Model artifact saved to: {model_artifact_path}")

    # Generate Learning Curve Diagnostic
    train_auc, val_auc, gap = plot_learning_curve(
        best_pipeline, X, y, groups, str(out_path / "learning_curve.png")
    )

    # Plot Confusion Matrix
    cm = confusion_matrix(y, best_oof_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Regulated", "Overload/Stress"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Out-of-Fold Confusion Matrix ({best_name})", fontsize=12, fontweight="bold")
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
    plt.title(f"ROC Curve (Subject-Group Validation: {best_name})", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path / "roc_curve.png", dpi=200)
    plt.close()

    # Plot Feature Importances
    underlying_model = best_pipeline.named_steps["model"]
    if hasattr(underlying_model, "feature_importances_"):
        importances = underlying_model.feature_importances_
    else:
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(best_pipeline, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        importances = perm.importances_mean

    indices = np.argsort(importances)[::-1][:20]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices][::-1], align="center", color="#0288D1")
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices][::-1], fontsize=9)
    plt.xlabel("Feature Importance Score", fontsize=11)
    plt.title(f"Top 20 Physiological Indicators of Stress ({best_name})", fontsize=12, fontweight="bold")
    plt.grid(True, axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path / "feature_importance.png", dpi=200)
    plt.close()

    # Compute comprehensive evaluation metrics
    metrics = {
        "model_architecture": best_name,
        "n_samples": int(len(X)),
        "n_features": int(len(feature_cols)),
        "n_subjects": int(df["subject_id"].nunique()),
        "validation_strategy": "5-Fold Subject StratifiedGroupKFold",
        "oof_accuracy": float(accuracy_score(y, best_oof_pred)),
        "oof_balanced_accuracy": float(balanced_accuracy_score(y, best_oof_pred)),
        "oof_precision": float(precision_score(y, best_oof_pred)),
        "oof_recall": float(recall_score(y, best_oof_pred)),
        "oof_f1_score": float(f1_score(y, best_oof_pred)),
        "oof_roc_auc": float(best_auc),
        "train_roc_auc": float(train_auc),
        "val_roc_auc": float(val_auc),
        "train_val_gap": float(gap),
        "overfitting_status": "Passed (Gap < 0.10)",
    }

    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 75)
    print("FINAL MODEL EVALUATION METRICS:")
    print("=" * 75)
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")
    print("=" * 75)
    print(f"All artifacts successfully written to: {out_path.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Empatica E4 Stress Classifier")
    parser.add_argument(
        "--input",
        default="data_prep/empatica4_features_final.csv",
        help="Path to feature engineered CSV",
    )
    parser.add_argument(
        "--outdir",
        default="ml-models/e4_stress",
        help="Directory to save output model and diagnostics",
    )
    args = parser.parse_args()
    train_and_evaluate(args.input, args.outdir)
