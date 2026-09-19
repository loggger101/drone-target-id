#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_model.py — Detailed test evaluation for cnn_fine_coarse

Assumes you have already run train_model.py and that:
  • DATA_ROOT/metadata.csv exists
  • DATA_ROOT/output/best_model.h5 exists
  • metadata.csv has a 'test' split

Outputs:
  • DATA_ROOT/output/test_detailed_metrics.txt
"""

import os, sys, random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

# =========================
# Config (match train_model.py)
# =========================
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 1024
SEED       = 1337

# Override with the DATA_ROOT env var (originally hard-coded to
# C:\Users\Owner\data_kagglehub_unified)
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data_kagglehub_unified")).expanduser().resolve()
os.chdir(DATA_ROOT)
print("Now running from:", Path.cwd())

METADATA_PATH  = Path("metadata.csv")
IMAGE_ROOT_DIR = Path("images")
OUTPUT_DIR     = DATA_ROOT / "output"
MODEL_PATH     = OUTPUT_DIR / "best_model.h5"
METRICS_PATH   = OUTPUT_DIR / "test_detailed_metrics.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# Helpers
# =========================
def resolve_rel_path(p: str) -> str:
    """Mirror path resolution used in training script."""
    p = str(p).strip()
    if not p:
        return p
    p_norm = p.replace("\\", "/")
    if os.path.isabs(p_norm):
        return p_norm
    if p_norm.startswith("./"):
        p_norm = p_norm[2:]
    root_name = str(IMAGE_ROOT_DIR).replace("\\", "/").strip("/")
    if (
        p_norm.startswith(root_name + "/")
        or p_norm.startswith("images/")
        or p_norm == root_name
    ):
        return p_norm.lstrip("/")
    return str((IMAGE_ROOT_DIR / p_norm).as_posix())


def load_and_preprocess_image(path: tf.Tensor) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_test_dataset(paths: np.ndarray) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def per_class_accuracy(y_true, y_pred, label_names):
    lines = []
    for idx, name in enumerate(label_names):
        mask = (y_true == idx)
        n = int(mask.sum())
        if n == 0:
            acc = float("nan")
        else:
            acc = float((y_pred[mask] == y_true[mask]).mean())
        lines.append(f"  {name:>25s}: acc={acc:0.4f} (n={n})")
    return "\n".join(lines)


# =========================
# Load metadata & build test set
# =========================
if not METADATA_PATH.exists():
    print(f"ERROR: {METADATA_PATH} not found.", file=sys.stderr)
    sys.exit(1)

df = pd.read_csv(METADATA_PATH)

required_cols = {"out_relpath", "fine_label", "coarse_label", "split"}
missing = required_cols - set(df.columns)
if missing:
    print(f"ERROR: metadata.csv missing columns: {sorted(missing)}", file=sys.stderr)
    sys.exit(1)

# Normalize coarse labels exactly like train_model.py
df["coarse_label"] = (
    df["coarse_label"].astype(str).str.strip().str.lower()
      .replace({
          "invalid_building": "invalid_background",
          "invalid_backgrounds": "invalid_background",
          "invalid_non_target": "invalid_nontarget",
          "invalid-nontarget": "invalid_nontarget",
          "invalid-non-target": "invalid_nontarget",
      })
)

# Resolve paths & keep only existing files
df["resolved_path"] = df["out_relpath"].astype(str).map(resolve_rel_path)
exists_mask = df["resolved_path"].map(lambda p: Path(p).exists())
missing_count = int((~exists_mask).sum())
if missing_count > 0:
    print(f"[WARN] Skipping {missing_count} rows whose files were not found on disk.", file=sys.stderr)
df = df.loc[exists_mask].copy()

train_df = df[df["split"] == "train"].copy()
test_df  = df[df["split"] == "test"].copy()

if test_df.empty:
    print("ERROR: No test rows found (split == 'test').", file=sys.stderr)
    sys.exit(1)

if train_df.empty:
    print("ERROR: No train rows found (split == 'train'); cannot reconstruct label mapping.", file=sys.stderr)
    sys.exit(1)

# Fine-label mapping (must match training)
fine_labels = sorted(train_df["fine_label"].unique())
fine_to_idx = {c: i for i, c in enumerate(fine_labels)}
num_fine_classes = len(fine_labels)

# Coarse-label mapping (fixed 3-way)
coarse_labels_order = ["invalid_nontarget", "invalid_background", "valid"]
coarse_to_idx = {c: i for i, c in enumerate(coarse_labels_order)}

# Build test arrays
test_paths = test_df["resolved_path"].astype(str).values
y_fine_true = test_df["fine_label"].map(fine_to_idx).values.astype("int32")
y_coarse_true = test_df["coarse_label"].map(coarse_to_idx).values.astype("int32")

print(f"[INFO] Test samples: {len(test_paths)}")

test_ds = make_test_dataset(test_paths)

# =========================
# Load model and predict
# =========================
if not MODEL_PATH.exists():
    print(f"ERROR: model file not found at {MODEL_PATH}", file=sys.stderr)
    sys.exit(1)

print(f"[INFO] Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

print("[INFO] Running predictions on test set...")
preds_fine_prob, preds_coarse_prob = model.predict(test_ds)
y_fine_pred   = np.argmax(preds_fine_prob, axis=1)
y_coarse_pred = np.argmax(preds_coarse_prob, axis=1)

# =========================
# Metrics
# =========================
# --- Fine head ---
fine_acc = accuracy_score(y_fine_true, y_fine_pred)
fine_p_micro, fine_r_micro, fine_f1_micro, _ = precision_recall_fscore_support(
    y_fine_true, y_fine_pred, average="micro", zero_division=0
)
fine_p_macro, fine_r_macro, fine_f1_macro, _ = precision_recall_fscore_support(
    y_fine_true, y_fine_pred, average="macro", zero_division=0
)
fine_p_weight, fine_r_weight, fine_f1_weight, _ = precision_recall_fscore_support(
    y_fine_true, y_fine_pred, average="weighted", zero_division=0
)

fine_report = classification_report(
    y_fine_true, y_fine_pred, target_names=fine_labels, digits=4, zero_division=0
)
fine_cm = confusion_matrix(y_fine_true, y_fine_pred)

# --- Coarse head ---
coarse_acc = accuracy_score(y_coarse_true, y_coarse_pred)
coarse_p_micro, coarse_r_micro, coarse_f1_micro, _ = precision_recall_fscore_support(
    y_coarse_true, y_coarse_pred, average="micro", zero_division=0
)
coarse_p_macro, coarse_r_macro, coarse_f1_macro, _ = precision_recall_fscore_support(
    y_coarse_true, y_coarse_pred, average="macro", zero_division=0
)
coarse_p_weight, coarse_r_weight, coarse_f1_weight, _ = precision_recall_fscore_support(
    y_coarse_true, y_coarse_pred, average="weighted", zero_division=0
)

coarse_report = classification_report(
    y_coarse_true, y_coarse_pred, target_names=coarse_labels_order, digits=4, zero_division=0
)
coarse_cm = confusion_matrix(y_coarse_true, y_coarse_pred)

# Per-class accuracies
fine_per_class_acc   = per_class_accuracy(y_fine_true, y_fine_pred, fine_labels)
coarse_per_class_acc = per_class_accuracy(y_coarse_true, y_coarse_pred, coarse_labels_order)

# =========================
# Save to file
# =========================
with open(METRICS_PATH, "w") as f:
    f.write("DETAILED TEST METRICS\n")
    f.write("=====================\n\n")
    f.write(f"Num test samples: {len(test_paths)}\n")
    f.write(f"Num fine classes: {num_fine_classes}\n\n")

    f.write("== FINE HEAD (fine_output) ==\n")
    f.write(f"Overall accuracy: {fine_acc:0.4f}\n")
    f.write(f"Micro  P/R/F1: {fine_p_micro:0.4f} / {fine_r_micro:0.4f} / {fine_f1_micro:0.4f}\n")
    f.write(f"Macro  P/R/F1: {fine_p_macro:0.4f} / {fine_r_macro:0.4f} / {fine_f1_macro:0.4f}\n")
    f.write(f"Weighted P/R/F1: {fine_p_weight:0.4f} / {fine_r_weight:0.4f} / {fine_f1_weight:0.4f}\n\n")

    f.write("Per-class accuracies (fine):\n")
    f.write(fine_per_class_acc + "\n\n")

    f.write("Classification report (fine):\n")
    f.write(fine_report + "\n")
    f.write("Confusion matrix (fine):\n")
    f.write(str(fine_cm) + "\n\n")

    f.write("== COARSE HEAD (coarse_output) ==\n")
    f.write(f"Overall accuracy: {coarse_acc:0.4f}\n")
    f.write(f"Micro  P/R/F1: {coarse_p_micro:0.4f} / {coarse_r_micro:0.4f} / {coarse_f1_micro:0.4f}\n")
    f.write(f"Macro  P/R/F1: {coarse_p_macro:0.4f} / {coarse_r_macro:0.4f} / {coarse_f1_macro:0.4f}\n")
    f.write(f"Weighted P/R/F1: {coarse_p_weight:0.4f} / {coarse_r_weight:0.4f} / {coarse_f1_weight:0.4f}\n\n")

    f.write("Per-class accuracies (coarse):\n")
    f.write(coarse_per_class_acc + "\n\n")

    f.write("Classification report (coarse):\n")
    f.write(coarse_report + "\n")
    f.write("Confusion matrix (coarse):\n")
    f.write(str(coarse_cm) + "\n")

print("\n[Done] Detailed test metrics written to:")
print(METRICS_PATH.resolve())
print(f"Fine-head overall accuracy:   {fine_acc:0.4f}")
print(f"Coarse-head overall accuracy: {coarse_acc:0.4f}")
