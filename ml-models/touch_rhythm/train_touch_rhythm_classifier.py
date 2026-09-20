"""
Touch & Rhythm Emotion & 3D VAD Affect Classifier Training
==========================================================
Trained on synthetic_touch_rhythm_dataset_v2.csv for ArtSpeak:
- Layer 1: Non-Verbal Interaction (Motor Kinematics & Rhythm Dynamics)
- Layer 2: Emotional & Arousal Representation (6 Behavioral States + 3D VAD Vector)

Anti-Overfitting & Anti-Underfitting Protocol:
  1. Stratified 5-Fold Cross-Validation:
     Guarantees balanced representation across all 6 emotion classes.
  2. Domain Feature Engineering:
     Extracts kinematic indices (tap cadence Hz, rhythm stability, motor control, touch intensity).
  3. Strict Hyperparameter Regularization:
     Constrained tree depth, L2 regularization, min samples per leaf.
  4. Diagnostic Artifacts:
     - Learning curves (train vs validation convergence to verify no overfitting)
     - Multi-class Confusion Matrix
     - One-vs-Rest ROC curves for all 6 emotion states
     - Permutation Feature Importances
     - 3D VAD Regression error evaluation (Valence, Arousal, Dominance MAE & R2)

Usage:
    python ml-models/touch_rhythm/train_touch_rhythm_classifier.py --input data_prep/synthetic_touch_rhythm_dataset_v2.csv --outdir ml-models/touch_rhythm
"""

import argparse
import json
import os
import time
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
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, learning_curve
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler


EMOTION_CLASSES = [
    "Anxious_Overstimulated",
    "Distressed_PreMeltdown",
    "Engaged_Focused",
    "Calm_Regulated",
    "Withdrawn_FlatAffect",
    "Excited_Stimming",
]

RAW_FEATURE_COLS = [
    "avg_pressure",
    "avg_speed_px_s",
    "avg_curvature",
    "avg_pressure_variance",
    "avg_tap_interval_ms",
    "avg_tap_regularity",
    "stereotypy_burst_present",
]


def engineer_touch_rhythm_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Extract domain-informed motor kinematic and rhythm interaction features.
    """
    df_feat = df.copy()

    # Cast boolean column to float
    if "stereotypy_burst_present" in df_feat.columns:
        df_feat["stereotypy_burst_present"] = df_feat["stereotypy_burst_present"].astype(float)

    # 1. Tap cadence in Hertz (taps per second)
    df_feat["tap_cadence_hz"] = 1000.0 / (df_feat["avg_tap_interval_ms"] + 1e-5)

    # 2. Touch Intensity / Kinetic Force (pressure * speed)
    df_feat["kinetic_touch_intensity"] = df_feat["avg_pressure"] * (df_feat["avg_speed_px_s"] / 100.0)

    # 3. Rhythm Stability Index (regularity penalized by pressure variance)
    df_feat["rhythm_stability_index"] = df_feat["avg_tap_regularity"] * np.clip(1.0 - df_feat["avg_pressure_variance"], 0.0, 1.0)

    # 4. Motor Fluidity Score (curvature * regularity)
    df_feat["motor_fluidity_score"] = df_feat["avg_curvature"] * df_feat["avg_tap_regularity"]

    # 5. Dysregulation Ratio (pressure variance / tap regularity)
    df_feat["dysregulation_ratio"] = df_feat["avg_pressure_variance"] / (df_feat["avg_tap_regularity"] + 1e-4)

    # 6. Pressure-to-Speed Ratio
    df_feat["pressure_velocity_ratio"] = df_feat["avg_pressure"] / (df_feat["avg_speed_px_s"] / 100.0 + 1e-4)

    feature_cols = [
        col for col in df_feat.columns
        if col not in ["session_id", "valence", "arousal", "dominance", "emotion_label"]
    ]
    return df_feat, feature_cols


def plot_learning_curve_clf(pipeline, X, y, out_path: str):
    """
    Compute and plot learning curve across training set fractions to verify convergence.
    """
    print("\nComputing Learning Curve across dataset increments...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline,
        X,
        y,
        cv=skf,
        train_sizes=np.linspace(0.2, 1.0, 5),
        scoring="accuracy",
        n_jobs=-1,
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, "o-", color="#1976D2", label="Training Accuracy")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#1976D2")
    plt.plot(train_sizes, val_mean, "o-", color="#388E3C", label="5-Fold Cross-Validation Accuracy")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#388E3C")

    plt.xlabel("Number of Training Samples", fontsize=11)
    plt.ylabel("Accuracy Score", fontsize=11)
    plt.title("Touch & Rhythm Emotion Classifier: Learning Curve (5-Fold CV)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    gap = float(train_mean[-1] - val_mean[-1])
    print(f"Learning curve saved to: {out_path}")
    print(f"Final Train Acc: {train_mean[-1]:.4f} | Validation Acc: {val_mean[-1]:.4f} | Gap: {gap:+.4f}")
    return float(train_mean[-1]), float(val_mean[-1]), gap


def train_and_evaluate(input_csv: str, out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ARTSPEAK TOUCH & RHYTHM EMOTION & 3D VAD CLASSIFIER TRAINING")
    print("=" * 80)

    print(f"Loading data from: {input_csv}")
    df_raw = pd.read_csv(input_csv)
    print(f"Dataset shape: {df_raw.shape[0]:,} sessions, {df_raw.shape[1]} columns")

    # Encode categorical target
    label_encoder = LabelEncoder()
    label_encoder.fit(EMOTION_CLASSES)
    y_clf = label_encoder.transform(df_raw["emotion_label"].values)
    
    # 3D VAD Targets
    y_vad = df_raw[["valence", "arousal", "dominance"]].values

    print("\nClass Distribution:")
    for cls_name, count in df_raw["emotion_label"].value_counts().items():
        print(f"  {cls_name:24s}: {count:,} samples ({count / len(df_raw) * 100:.2f}%)")

    # Feature engineering
    print("\nExtracting kinematic and rhythm features...")
    df_feat, feature_cols = engineer_touch_rhythm_features(df_raw)
    X = df_feat[feature_cols].copy()
    print(f"Total features ({len(feature_cols)}): {feature_cols}")

    with open(out_path / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Define Candidate Multi-Class Classifiers with Anti-Overfitting Regularization
    classifiers = {
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=5,
            max_leaf_nodes=20,
            learning_rate=0.06,
            min_samples_leaf=20,
            l2_regularization=2.0,
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.07,
            subsample=0.85,
            min_samples_leaf=15,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=7,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-3,
            max_iter=300,
            early_stopping=True,
            random_state=42,
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_clf_name = None
    best_f1 = -1.0
    best_clf_pipeline = None
    best_oof_proba = None
    best_oof_pred = None

    print("\n" + "-" * 80)
    print("5-FOLD STRATIFIED CROSS-VALIDATION MODEL BENCHMARKING (CLASSIFICATION)")
    print("-" * 80)

    for name, clf in classifiers.items():
        pipeline = Pipeline([
            ("scaler", RobustScaler()),
            ("model", clf),
        ])

        oof_proba = cross_val_predict(
            pipeline, X, y_clf, cv=skf, method="predict_proba", n_jobs=-1
        )
        oof_pred = np.argmax(oof_proba, axis=1)

        acc = accuracy_score(y_clf, oof_pred)
        bal_acc = balanced_accuracy_score(y_clf, oof_pred)
        macro_f1 = f1_score(y_clf, oof_pred, average="macro")
        weighted_f1 = f1_score(y_clf, oof_pred, average="weighted")
        auc = roc_auc_score(y_clf, oof_proba, multi_class="ovr")

        print(
            f"[{name:22s}] Acc: {acc*100:.2f}% | Bal Acc: {bal_acc*100:.2f}% | "
            f"Macro-F1: {macro_f1*100:.2f}% | OVR ROC-AUC: {auc*100:.2f}%"
        )

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_clf_name = name
            best_clf_pipeline = pipeline
            best_oof_proba = oof_proba
            best_oof_pred = oof_pred

    print(f"\n--> Selected Best Classification Architecture: {best_clf_name} (Macro-F1: {best_f1*100:.2f}%)")

    # Fit final classifier on full dataset
    print(f"Fitting final {best_clf_name} classification pipeline on full dataset...")
    best_clf_pipeline.fit(X, y_clf)

    # 3D VAD Regressor Pipeline
    print("\n" + "-" * 80)
    print("5-FOLD CROSS-VALIDATION (3D VAD REGRESSION: VALENCE, AROUSAL, DOMINANCE)")
    print("-" * 80)

    vad_regressor = Pipeline([
        ("scaler", RobustScaler()),
        ("model", MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=150,
                max_depth=5,
                max_leaf_nodes=20,
                learning_rate=0.06,
                min_samples_leaf=20,
                l2_regularization=2.0,
                random_state=42,
            )
        )),
    ])

    oof_vad_preds = cross_val_predict(
        vad_regressor, X, y_vad, cv=5, n_jobs=-1
    )

    overall_mae = mean_absolute_error(y_vad, oof_vad_preds)
    val_mae = mean_absolute_error(y_vad[:, 0], oof_vad_preds[:, 0])
    aro_mae = mean_absolute_error(y_vad[:, 1], oof_vad_preds[:, 1])
    dom_mae = mean_absolute_error(y_vad[:, 2], oof_vad_preds[:, 2])

    val_r2 = r2_score(y_vad[:, 0], oof_vad_preds[:, 0])
    aro_r2 = r2_score(y_vad[:, 1], oof_vad_preds[:, 1])
    dom_r2 = r2_score(y_vad[:, 2], oof_vad_preds[:, 2])

    print(f"Overall VAD Mean Absolute Error: {overall_mae:.4f}")
    print(f"  - Valence:   MAE = {val_mae:.4f}, R2 = {val_r2:.4f}")
    print(f"  - Arousal:   MAE = {aro_mae:.4f}, R2 = {aro_r2:.4f}")
    print(f"  - Dominance: MAE = {dom_mae:.4f}, R2 = {dom_r2:.4f}")

    print("Fitting final 3D VAD regression pipeline on full dataset...")
    vad_regressor.fit(X, y_vad)

    # Save Bundle Artifact (Classifier + Regressor + LabelEncoder)
    touch_bundle = {
        "classifier": best_clf_pipeline,
        "vad_regressor": vad_regressor,
        "label_encoder": label_encoder,
        "classes": list(label_encoder.classes_),
        "feature_cols": feature_cols,
    }
    model_artifact_path = out_path / "touch_rhythm_pipeline.joblib"
    joblib.dump(touch_bundle, model_artifact_path)
    print(f"Production touch & rhythm bundle saved to: {model_artifact_path}")

    # Generate Learning Curve Diagnostic
    train_acc, val_acc, gap = plot_learning_curve_clf(
        best_clf_pipeline, X, y_clf, str(out_path / "learning_curve.png")
    )

    # Confusion Matrix
    cm = confusion_matrix(y_clf, best_oof_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[c.replace("_", "\n") for c in label_encoder.classes_],
    )
    fig, ax = plt.subplots(figsize=(9, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=15)
    ax.set_title(f"Touch & Rhythm 6-Class Out-of-Fold Confusion Matrix ({best_clf_name})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path / "confusion_matrix.png", dpi=200)
    plt.close()

    # Multi-Class ROC Curves
    plt.figure(figsize=(8, 6))
    for i, cls_name in enumerate(label_encoder.classes_):
        binary_y = (y_clf == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_y, best_oof_proba[:, i])
        cls_auc = roc_auc_score(binary_y, best_oof_proba[:, i])
        plt.plot(fpr, tpr, lw=2, label=f"{cls_name} (AUC = {cls_auc:.3f})")

    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title(f"Multi-Class ROC Curves: Out-of-Fold ({best_clf_name})", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path / "roc_curves.png", dpi=200)
    plt.close()

    # Feature Importance Plot
    underlying_model = best_clf_pipeline.named_steps["model"]
    if hasattr(underlying_model, "feature_importances_"):
        importances = underlying_model.feature_importances_
    else:
        perm = permutation_importance(best_clf_pipeline, X, y_clf, n_repeats=5, random_state=42, n_jobs=-1)
        importances = perm.importances_mean

    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices][::-1], align="center", color="#00897B")
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices][::-1], fontsize=9)
    plt.xlabel("Feature Importance Score", fontsize=11)
    plt.title(f"Touch & Rhythm: Top Kinematic Indicators ({best_clf_name})", fontsize=12, fontweight="bold")
    plt.grid(True, axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path / "feature_importance.png", dpi=200)
    plt.close()

    # Metrics JSON
    report = classification_report(
        y_clf,
        best_oof_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model_architecture": best_clf_name,
        "n_samples": int(len(X)),
        "n_features": int(len(feature_cols)),
        "classes": list(label_encoder.classes_),
        "validation_strategy": "5-Fold StratifiedKFold",
        "oof_accuracy": float(accuracy_score(y_clf, best_oof_pred)),
        "oof_balanced_accuracy": float(balanced_accuracy_score(y_clf, best_oof_pred)),
        "oof_macro_f1": float(best_f1),
        "oof_weighted_f1": float(f1_score(y_clf, best_oof_pred, average="weighted")),
        "oof_ovr_roc_auc": float(roc_auc_score(y_clf, best_oof_proba, multi_class="ovr")),
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "train_val_gap": float(gap),
        "fit_verdict": "Strict Anti-Overfitting Passed (Gap < 0.05)" if abs(gap) < 0.05 else "Investigate",
        "vad_overall_mae": float(overall_mae),
        "vad_valence_mae": float(val_mae),
        "vad_arousal_mae": float(aro_mae),
        "vad_dominance_mae": float(dom_mae),
        "vad_valence_r2": float(val_r2),
        "vad_arousal_r2": float(aro_r2),
        "vad_dominance_r2": float(dom_r2),
        "per_class_classification_report": report,
    }

    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL TOUCH & RHYTHM MODEL EVALUATION SUMMARY:")
    print("=" * 80)
    for k in ["model_architecture", "n_samples", "oof_accuracy", "oof_balanced_accuracy", "oof_macro_f1", "oof_ovr_roc_auc", "train_val_gap", "vad_overall_mae"]:
        print(f"  {k:28s}: {metrics[k]}")
    print("=" * 80)
    print(f"All model artifacts successfully written to: {out_path.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Touch & Rhythm Emotion and VAD Model")
    parser.add_argument(
        "--input",
        default="data_prep/synthetic_touch_rhythm_dataset_v2.csv",
        help="Path to touch rhythm dataset CSV",
    )
    parser.add_argument(
        "--outdir",
        default="ml-models/touch_rhythm",
        help="Directory to save output model artifacts and diagnostics",
    )
    args = parser.parse_args()
    train_and_evaluate(args.input, args.outdir)
