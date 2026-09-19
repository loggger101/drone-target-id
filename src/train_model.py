#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_model.py — Multi-output CNN (Fine class + 3-way Coarse) compatible with your metadata.csv

• Loads images using `metadata.csv` fields: out_relpath, fine_label, coarse_label, split
• Trains a CNN with two heads: fine (softmax) + coarse (softmax with 3 classes)
• Hyperparameters at the top
• Uses sample weights; by default, INVALID images *do contribute* to fine-class loss
  (set IGNORE_INVALID_IN_FINE=True to ignore them if desired)
• Saves: best model, confusion matrices (baseline+final), classification reports, training curves, misclassified images

NEW: Easy weight decay controls
  - Optimizer-level decoupled weight decay (AdamW): set USE_ADAMW=True and WEIGHT_DECAY>0
  - (Optional) Layer L2 regularization: set L2_REG>0 to add kernel regularizers to Conv/Dense
"""

import os, sys, shutil, random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# =========================
# Hyperparameters & Paths
# =========================
IMAGE_SIZE     = (64, 64)     # (H, W)
BATCH_SIZE     = 1024
EPOCHS         = 50
LEARNING_RATE  = 1e-3
VAL_FRACTION   = 0.10
SEED           = 1337

# Weight decay / regularization
USE_ADAMW      = True
WEIGHT_DECAY   = 1e-6
L2_REG         = 1e-5

# Fine/coarse weighting
IGNORE_INVALID_IN_FINE = False  # If True, fine loss is ignored when coarse != 'valid'

# Learning-rate schedule
LR_PATIENCE   = 5      # epochs with no improvement before LR reduced
LR_FACTOR     = 0.5    # LR multiplied by this factor
LR_MIN        = 1e-6   # minimum learning rate

# Dataset root produced by src/build_dataset.py. Override with the DATA_ROOT env var.
# (Originally hard-coded to C:\Users\Owner\data_kagglehub_unified; on Colab it was
#  /content/data_kagglehub_unified)
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data_kagglehub_unified")).expanduser().resolve()
# Change current working directory so relative paths work like before
os.chdir(DATA_ROOT)
print("Now running from:", Path.cwd())

# original constants can stay simple:
METADATA_PATH  = Path("metadata.csv")
IMAGE_ROOT_DIR = Path("images")
OUTPUT_DIR     = DATA_ROOT / "output"

# =========================
# Reproducibility
# =========================
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR / "misclassified", exist_ok=True)

# =========================
# Utilities
# =========================
def resolve_rel_path(p: str) -> str:
    """Return a filesystem path string for an out_relpath that may already contain 'images/...'.
    - Keeps absolute paths as-is.
    - If p starts with 'images/' (or IMAGE_ROOT_DIR name), returns p.
    - Otherwise prefixes with IMAGE_ROOT_DIR.
    """
    p = str(p).strip()
    if not p:
        return p
    p_norm = p.replace("\\", "/")
    if os.path.isabs(p_norm):
        return p_norm
    if p_norm.startswith("./"):
        p_norm = p_norm[2:]
    root_name = str(IMAGE_ROOT_DIR).replace("\\", "/").strip("/")
    if p_norm.startswith(root_name + "/") or p_norm.startswith("images/") or p_norm == root_name:
        return p_norm.lstrip("/")
    return str((IMAGE_ROOT_DIR / p_norm).as_posix())

def load_and_preprocess_image(path: tf.Tensor) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def make_dataset(paths, fine_labels, coarse_labels, fine_weights, coarse_weights, training=False):
    """Yield (image, (fine, coarse), (fine_w, coarse_w)) — tuples match model output order."""
    ds = tf.data.Dataset.from_tensor_slices((paths, fine_labels, coarse_labels, fine_weights, coarse_weights))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
    def _map(p, f, c, fw, cw):
        img = load_and_preprocess_image(p)
        targets = (tf.cast(f, tf.int32), tf.cast(c, tf.int32))
        weights = (tf.cast(fw, tf.float32), tf.cast(cw, tf.float32))
        return img, targets, weights
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def plot_confusion(cm, labels, title, outpath, figsize=(8,6)):
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def get_optimizer():
    if USE_ADAMW:
        if hasattr(tf.keras.optimizers, "AdamW"):
            return tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        else:
            return tf.keras.optimizers.experimental.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        return tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

def reg():
    return (tf.keras.regularizers.l2(L2_REG) if L2_REG and L2_REG > 0.0 else None)

# =========================
# Load metadata
# =========================
if not METADATA_PATH.exists():
    print(f"ERROR: {METADATA_PATH} not found.", file=sys.stderr)
    sys.exit(1)

df = pd.read_csv(METADATA_PATH)

# Expect: out_relpath, fine_label, coarse_label, split
required_cols = {"out_relpath", "fine_label", "coarse_label", "split"}
missing = required_cols - set(df.columns)
if missing:
    print(f"ERROR: metadata.csv missing columns: {sorted(missing)}", file=sys.stderr)
    sys.exit(1)

# Normalize coarse labels to exactly these three: invalid_nontarget, invalid_background, valid
df["coarse_label"] = (
    df["coarse_label"].astype(str).str.strip().str.lower()
      .replace({
          # common typos / aliases → canonical
          "invalid_building": "invalid_background",
          "invalid_backgrounds": "invalid_background",
          "invalid_non_target": "invalid_nontarget",
          "invalid-nontarget": "invalid_nontarget",
          "invalid-non-target": "invalid_nontarget",
      })
)

# Resolve file paths and drop missing files
resolved_paths = []
keep_mask = []
missing_count = 0
for p in df["out_relpath"].astype(str).tolist():
    rp = resolve_rel_path(p)
    resolved_paths.append(rp)
    if Path(rp).exists():
        keep_mask.append(True)
    else:
        keep_mask.append(False)
        missing_count += 1

if missing_count > 0:
    print(f"[WARN] Skipping {missing_count} rows whose files were not found on disk.", file=sys.stderr)

df = df.loc[keep_mask].copy()
df["resolved_path"] = [resolve_rel_path(p) for i, p in enumerate(df["out_relpath"]) if keep_mask[i]]

# Split by split column
train_df = df[df["split"] == "train"].copy()
test_df  = df[df["split"] == "test"].copy()

if train_df.empty:
    print("ERROR: No training rows found (split == 'train') after filtering missing files.", file=sys.stderr); sys.exit(1)
if test_df.empty:
    print("WARN: No test rows found (split == 'test'). Using val split as test later.", file=sys.stderr)

# Build fine label space from training set (will be 3 in your data)
fine_labels = sorted(train_df["fine_label"].unique())
fine_to_idx = {c: i for i, c in enumerate(fine_labels)}
num_fine_classes = len(fine_labels)

# 3-class coarse head mapping (ORDER matters & used throughout)
coarse_labels_order = ["invalid_nontarget", "invalid_background", "valid"]
coarse_to_idx = {c: i for i, c in enumerate(coarse_labels_order)}

# Validation split
val_df = pd.DataFrame(columns=train_df.columns)
if len(train_df) >= 10 and train_df["fine_label"].nunique() > 1:
    train_df, val_df = train_test_split(
        train_df,
        test_size=VAL_FRACTION,
        stratify=train_df["fine_label"],
        random_state=SEED,
    )
elif len(train_df) > 1:
    val_df = train_df.sample(frac=0.2, random_state=SEED)
    train_df = train_df.drop(val_df.index)

def rows_to_arrays(frame: pd.DataFrame):
    paths  = np.array([str(Path(p).as_posix()) for p in frame["resolved_path"]], dtype=str)
    fine   = np.array([fine_to_idx.get(lbl, -1) for lbl in frame["fine_label"]], dtype=np.int32)
    coarse = np.array([coarse_to_idx.get(lbl, 0)  for lbl in frame["coarse_label"]], dtype=np.int32)
    # Sample weights:
    if IGNORE_INVALID_IN_FINE:
        valid_idx = coarse_to_idx["valid"]
        fine_w = (coarse == valid_idx).astype(np.float32)
    else:
        fine_w = np.ones_like(coarse, dtype=np.float32)
    coarse_w = np.ones_like(coarse, dtype=np.float32)
    return paths, fine, coarse, fine_w, coarse_w

train_paths, train_fine, train_coarse, train_fine_w, train_coarse_w = rows_to_arrays(train_df)
val_paths = val_fine = val_coarse = val_fine_w = val_coarse_w = None
if not val_df.empty:
    val_paths, val_fine, val_coarse, val_fine_w, val_coarse_w = rows_to_arrays(val_df)
test_paths = test_fine = test_coarse = test_fine_w = test_coarse_w = None
if not test_df.empty:
    test_paths, test_fine, test_coarse, test_fine_w, test_coarse_w = rows_to_arrays(test_df)

# Datasets
train_ds = make_dataset(train_paths, train_fine, train_coarse, train_fine_w, train_coarse_w, training=True)
val_ds   = None if val_paths is None else make_dataset(val_paths, val_fine, val_coarse, val_fine_w, val_coarse_w)
test_ds  = None if test_paths is None else make_dataset(test_paths, test_fine, test_coarse, test_fine_w, test_coarse_w)

# =========================
# Model (multi-output CNN)
# =========================
inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same", kernel_regularizer=reg())(inputs)
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same", kernel_regularizer=reg())(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same", kernel_regularizer=reg())(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=reg())(x)
x = tf.keras.layers.Dropout(0.5)(x)

fine_output   = tf.keras.layers.Dense(num_fine_classes, activation="softmax", name="fine_output",   kernel_regularizer=reg())(x)
coarse_output = tf.keras.layers.Dense(3,                 activation="softmax", name="coarse_output", kernel_regularizer=reg())(x)

model = tf.keras.Model(inputs=inputs, outputs=[fine_output, coarse_output], name="cnn_fine_coarse")

# Compile with LIST losses/metrics to match LIST outputs
optimizer = get_optimizer()
model.compile(
    optimizer=optimizer,
    loss=[tf.keras.losses.SparseCategoricalCrossentropy(), tf.keras.losses.SparseCategoricalCrossentropy()],
    metrics=[["accuracy"], ["accuracy"]],
)
model.summary()

# =========================
# Training (with checkpoint)
# =========================
checkpoint_path = OUTPUT_DIR / "best_model.h5"
monitor_metric = "val_loss" if val_ds is not None else "loss"
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(checkpoint_path),
    monitor=monitor_metric,
    save_best_only=True,
    verbose=1
)

# Decrease learning rate on plateau after a patience of 5 epochs
lr_scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor=monitor_metric,
    factor=LR_FACTOR,
    patience=LR_PATIENCE,
    min_lr=LR_MIN,
    verbose=1
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[checkpoint_cb, lr_scheduler_cb],
)

# Reload best model
best_model = tf.keras.models.load_model(str(checkpoint_path))

# =========================
# Baseline (majority) metrics + confusion matrices
# =========================
def majority_predictions(train_series, test_len, mapping: dict):
    if len(train_series) == 0 or test_len == 0:
        return None, None
    majority_label = train_series.value_counts().idxmax()
    idx = mapping[majority_label]
    preds = np.full((test_len,), idx, dtype=np.int32)
    return majority_label, preds

# Fine & Coarse baselines (if test exists)
if test_paths is not None and len(test_paths) > 0:
    maj_fine_label,   base_fine_preds   = majority_predictions(train_df["fine_label"],   len(test_paths), fine_to_idx)
    maj_coarse_label, base_coarse_preds = majority_predictions(train_df["coarse_label"], len(test_paths), coarse_to_idx)

    # Baseline confusion matrices
    cm_fine_base = confusion_matrix(test_fine, base_fine_preds)
    plot_confusion(
        cm_fine_base, fine_labels,
        f"BASELINE Confusion (predict all '{maj_fine_label}') — Fine",
        OUTPUT_DIR / "baseline_confusion_fine.png"
    )
    cm_coarse_base = confusion_matrix(test_coarse, base_coarse_preds)
    plot_confusion(
        cm_coarse_base, coarse_labels_order,
        f"BASELINE Confusion (predict all '{maj_coarse_label}') — Coarse",
        OUTPUT_DIR / "baseline_confusion_coarse.png",
        figsize=(5,5)
    )

    # Baseline accuracies
    base_fine_acc   = float(np.mean(base_fine_preds   == test_fine))
    base_coarse_acc = float(np.mean(base_coarse_preds == test_coarse))
    with open(OUTPUT_DIR / "baseline_metrics.txt", "w") as f:
        f.write(f"Baseline fine accuracy (all -> '{maj_fine_label}'): {base_fine_acc:.4f}\n")
        f.write(f"Baseline coarse accuracy (all -> '{maj_coarse_label}'): {base_coarse_acc:.4f}\n")

# =========================
# Evaluation (final) + plots
# =========================
if test_ds is not None:
    preds_fine_prob, preds_coarse_prob = best_model.predict(test_ds)
    preds_fine   = np.argmax(preds_fine_prob, axis=1)
    preds_coarse = np.argmax(preds_coarse_prob, axis=1)  # 3-way argmax

    # Confusion matrices (final)
    cm_fine = confusion_matrix(test_fine, preds_fine)
    plot_confusion(cm_fine, fine_labels, "FINAL Confusion — Fine", OUTPUT_DIR / "final_confusion_fine.png")
    cm_coarse = confusion_matrix(test_coarse, preds_coarse)
    plot_confusion(cm_coarse, coarse_labels_order, "FINAL Confusion — Coarse", OUTPUT_DIR / "final_confusion_coarse.png", figsize=(5,5))

    # Classification reports
    rep_fine   = classification_report(test_fine,   preds_fine,   target_names=fine_labels,          digits=4)
    rep_coarse = classification_report(test_coarse, preds_coarse, target_names=coarse_labels_order,  digits=4)
    with open(OUTPUT_DIR / "classification_report_fine.txt", "w") as f:
        f.write(rep_fine)
    with open(OUTPUT_DIR / "classification_report_coarse.txt", "w") as f:
        f.write(rep_coarse)

    # Misclassified images (only where coarse == valid and fine wrong)
    mis_dir = OUTPUT_DIR / "misclassified"
    valid_idx = coarse_to_idx["valid"]
    for i in range(len(test_paths)):
        if test_coarse[i] == valid_idx and preds_fine[i] != test_fine[i]:
            src = test_paths[i]
            true_lbl = fine_labels[test_fine[i]]
            pred_lbl = fine_labels[preds_fine[i]]
            ext = os.path.splitext(src)[1]
            dst = mis_dir / f"true_{true_lbl}__pred_{pred_lbl}__idx_{i}{ext}"
            try:
                shutil.copy(src, dst)
            except Exception:
                pass

# =========================
# Training curves
# =========================
hist = history.history
epochs = np.arange(1, len(hist.get("loss", [])) + 1)

plt.figure(figsize=(10,4))
# Accuracy
plt.subplot(1,2,1)
for k in ["fine_output_accuracy", "coarse_output_accuracy"]:
    if k in hist:
        style = "o-" if "fine" in k else "s-"
        plt.plot(epochs, hist[k], style, label=k.replace("_", " "))
    vk = "val_" + k
    if vk in hist:
        style = "o--" if "fine" in k else "s--"
        plt.plot(epochs, hist[vk], style, label=vk.replace("_", " "))
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curves"); plt.legend()

# Loss
plt.subplot(1,2,2)
for k in ["fine_output_loss", "coarse_output_loss"]:
    if k in hist:
        style = "o-" if "fine" in k else "s-"
        plt.plot(epochs, hist[k], style, label=k.replace("_", " "))
    vk = "val_" + k
    if vk in hist:
        style = "o--" if "fine" in k else "s--"
        plt.plot(epochs, hist[vk], style, label=k.replace("_", " "))
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves"); plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_curves.png")
plt.close()

print("\n[Done] Outputs saved in:", OUTPUT_DIR.resolve())
print(" - Best model:", (OUTPUT_DIR / 'best_model.h5').resolve())
print(" - Confusion matrices: baseline_* and final_*")
print(" - Reports: classification_report_*.txt, baseline_metrics.txt")
print(" - Training curves: training_curves.png")
print(" - Misclassified images folder:", (OUTPUT_DIR / 'misclassified').resolve())
print(f" - Optimizer: {'AdamW' if USE_ADAMW else 'Adam'} (weight_decay={WEIGHT_DECAY}) | L2_REG={L2_REG} | IGNORE_INVALID_IN_FINE={IGNORE_INVALID_IN_FINE}")
