"""
train_model.py — UPGRADED VERSION (Steps 1–4 integrated)
═══════════════════════════════════════════════════════════════════════════════
Changes from original:
  STEP 1 — Focal Loss replaces sparse_categorical_crossentropy
  STEP 2 — EfficientNetB3 replaces MobileNetV2
  STEP 3 — Merged CSV supported alongside HAM-only CSV
  STEP 4 — Smart per-class oversampling for DF and VASC

═══════════════════════════════════════════════════════════════════════════════
COLAB / PATH FIX (NEW):
  _remap_csv_path() — converts the Windows-absolute paths stored in
  merged_dataset.csv to paths relative to this project's skin_dataset/
  directory, regardless of which OS or machine the script runs on.

  Root cause: merged_dataset.csv was generated on a Windows machine
  and contains absolute paths like:
    C:\\Users\\Anshuman\\...\\skin_dataset\\HAM10000_images_part_1\\ISIC_XXXXXXX.jpg
  These paths do not exist on Google Colab (Linux) or any other machine.

  Fix: extract the sub-path after "skin_dataset/" and rebuild it under
  <_SCRIPT_DIR>/skin_dataset/ using os.path.join, which works on all OSes.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import glob
import time
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import tensorflow as tf

# ── STEP 2: EfficientNetB3 imports (replaces MobileNetV2 block) ───────────────
try:
    from keras.applications import EfficientNetB3
    from keras.applications.efficientnet import preprocess_input
    from keras.layers import (GlobalAveragePooling2D, Dense, Dropout,
                               BatchNormalization)
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                  ReduceLROnPlateau)
    print("[INFO] Using standalone keras — EfficientNetB3")
except ModuleNotFoundError:
    from tensorflow.keras.applications import EfficientNetB3
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense,
                                          Dropout, BatchNormalization)
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                             ReduceLROnPlateau)
    print("[INFO] Using tensorflow.keras — EfficientNetB3")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: FOCAL LOSS
#  Replaces sparse_categorical_crossentropy.
#  gamma=2.0 → easy examples get down-weighted so the model focuses on
#  hard minority classes (BKL, DF, MEL) instead of just coasting on NV.
# ══════════════════════════════════════════════════════════════════════════════

class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for integer (sparse) labels.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    gamma=0 → identical to standard cross-entropy (safe fallback).
    gamma=2 → standard Focal Loss (Lin et al., RetinaNet 2017).
    """
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        probs  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        one_hot      = tf.one_hot(y_true, tf.shape(probs)[-1])
        p_t          = tf.reduce_sum(probs * one_hot, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        loss         = self.alpha * focal_weight * (-tf.math.log(p_t))
        return tf.reduce_mean(loss)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha})
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

HYPERPARAMS = {
    # ── Architecture ──────────────────────────────────────────────────────────
    "dense_units":    256,
    "dropout_rate":   0.4,

    # STEP 2: reduced from 30 (MobileNetV2) to 20 (EfficientNetB3 has more
    # total layers, so 20 is proportionally similar to the old 30)
    "fine_tune_layers": 20,

    # ── Training ──────────────────────────────────────────────────────────────
    "batch_size":     32,
    "lr_phase1":      1e-3,
    "lr_phase2":      1e-5,
    "epochs_phase1":  20,
    "epochs_phase2":  10,
    "patience":       5,
    "lr_reduce_factor":  0.3,
    "lr_reduce_patience": 3,

    # ── Augmentation ──────────────────────────────────────────────────────────
    "aug_rotation":   20,
    "aug_zoom":       (0.85, 1.15),
    "aug_brightness": (0.8, 1.2),
    "aug_hflip":      True,
    "aug_vflip":      True,

    # ── Normal skin threshold ─────────────────────────────────────────────────
    "normal_threshold": 0.55,

    # ── STEP 1: Focal Loss params ─────────────────────────────────────────────
    "focal_gamma":    2.0,
    "focal_alpha":    0.25,

    # ── STEP 4: Per-class oversampling targets ────────────────────────────────
    # These are TARGET total counts in the training split (~70% of dataset).
    # The oversampler will add copies until each class reaches its target.
    # After Step 3 merge: df train samples ≈ 167, vasc ≈ 177.
    # We boost them to make them comparable to akiec (~229 train samples).
    "oversample_targets": {
        6: 600,   # df   → was ~167 train samples → boost to 600
        5: 400,   # vasc → was ~177 train samples → boost to 400
    },

    # ── LR warmup ────────────────────────────────────────────────────────────
    "warmup_epochs":    3,
    "lr_warmup_start":  1e-5,
}


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

IMG_SIZE        = (224, 224)
NUM_CLASSES     = 7
_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(_SCRIPT_DIR, "models", "skin_model.h5")
REPORT_DIR      = os.path.join(_SCRIPT_DIR, "training_report")

LABEL_MAP = {
    "nv": 0, "mel": 1, "bkl": 2, "bcc": 3,
    "akiec": 4, "vasc": 5, "df": 6,
}
CLASS_NAMES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
CLASS_FULL  = [
    "Melanocytic Nevus", "Melanoma", "Benign Keratosis",
    "Basal Cell Carcinoma", "Actinic Keratosis",
    "Vascular Lesion", "Dermatofibroma",
]

# ── Known sub-directories inside skin_dataset/ that hold images ───────────────
# Used as fallback search order inside _remap_csv_path().
_KNOWN_IMAGE_SUBDIRS = [
    "HAM10000_images_part_1",
    "HAM10000_images_part_2",
    "ISIC_2019_Training_Input",
]

DARK_BG  = "#0d1117"; DARK_FIG = "#161b22"
DARK_BD  = "#30363d"; TEXT_COL = "#c9d1d9"
TITLE_COL = "#f0f6fc"


# ══════════════════════════════════════════════════════════════════════════════
#  PATH REMAPPING  ← NEW / CHANGED
#
#  merged_dataset.csv was built on a Windows machine.  Every image_path cell
#  contains an absolute Windows path such as:
#
#    C:\Users\Anshuman\OneDrive\Desktop\skin_project\
#        skin_dataset\HAM10000_images_part_1\ISIC_0027419.jpg
#
#  On Google Colab (or any Linux/macOS machine) those paths do not exist,
#  so the original  df[df["image_path"].apply(os.path.isfile)]  filter
#  drops every single row → zero training images.
#
#  _remap_csv_path() solves this by:
#    1. Normalising backslashes to forward slashes.
#    2. Finding the "skin_dataset/" anchor in the path.
#    3. Extracting the portable sub-path that follows the anchor, e.g.
#         "HAM10000_images_part_1/ISIC_0027419.jpg"
#    4. Re-building the full path using the project's own skin_dataset/ dir:
#         <_SCRIPT_DIR>/skin_dataset/HAM10000_images_part_1/ISIC_0027419.jpg
#
#  If the anchor is not found, it falls back to searching every known
#  sub-directory by filename so partially-correct paths still resolve.
# ══════════════════════════════════════════════════════════════════════════════

def _remap_csv_path(raw_path: str) -> str:
    """
    Convert any OS-absolute path stored in merged_dataset.csv to the correct
    path on the current machine, rooted at <_SCRIPT_DIR>/skin_dataset/.

    Works for:
      • Windows absolute paths  (C:\\Users\\...\\skin_dataset\\sub\\file.jpg)
      • Linux/macOS absolute paths (/home/user/.../skin_dataset/sub/file.jpg)
      • Paths that are already correct (returned unchanged)
      • Paths where only the filename is recognisable (fallback search)

    Returns the remapped path string. Existence is NOT checked here;
    the caller's os.path.isfile() filter handles missing files.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        return raw_path

    # ── Fast path: already correct on this machine ────────────────────────────
    if os.path.isfile(raw_path):
        return raw_path

    # ── Normalise path separators to forward slash ────────────────────────────
    norm = raw_path.replace("\\", "/")

    # ── Primary strategy: find "skin_dataset/" anchor ─────────────────────────
    marker = "skin_dataset/"
    idx = norm.lower().find(marker)
    if idx != -1:
        # sub_path e.g. "HAM10000_images_part_1/ISIC_0027419.jpg"
        sub_path = norm[idx + len(marker):]
        # os.path.join handles the correct separator for the current OS
        remapped = os.path.join(_SCRIPT_DIR, "skin_dataset",
                                *sub_path.split("/"))
        return remapped

    # ── Fallback: search known sub-directories by filename ───────────────────
    basename = os.path.basename(norm)
    skin_dataset_dir = os.path.join(_SCRIPT_DIR, "skin_dataset")
    for subdir in _KNOWN_IMAGE_SUBDIRS:
        candidate = os.path.join(skin_dataset_dir, subdir, basename)
        if os.path.isfile(candidate):
            return candidate

    # ── Last resort: return as-is (will be filtered by isfile check) ─────────
    return raw_path


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: DATASET LOADING (supports both HAM-only and merged CSV)
#  CHANGED: load_data() now calls _remap_csv_path() on every image_path
#           value read from merged_dataset.csv before the isfile() filter.
# ══════════════════════════════════════════════════════════════════════════════

def find_dataset():
    """Original HAM10000-only dataset finder. Used as fallback."""
    csv_cands = [
        os.path.join(_SCRIPT_DIR, "skin_dataset", "HAM10000_metadata.csv"),
        os.path.join(_SCRIPT_DIR, "HAM10000_metadata.csv"),
        os.path.join(os.path.dirname(_SCRIPT_DIR), "skin_dataset", "HAM10000_metadata.csv"),
        os.path.join(os.path.dirname(_SCRIPT_DIR), "HAM10000_metadata.csv"),
    ]
    img_cands = [
        os.path.join(_SCRIPT_DIR, "skin_dataset", "HAM10000_images_part_1"),
        os.path.join(_SCRIPT_DIR, "skin_dataset", "HAM10000_images_part_2"),
        os.path.join(_SCRIPT_DIR, "HAM10000_images_part_1"),
        os.path.join(_SCRIPT_DIR, "HAM10000_images_part_2"),
    ]
    csv_path   = next((c for c in csv_cands if os.path.isfile(c)), None)
    found_dirs = [d for d in img_cands if os.path.isdir(d)]
    if not csv_path:
        print("[ERROR] HAM10000_metadata.csv not found."); sys.exit(1)
    if not found_dirs:
        print("[ERROR] No HAM10000 image dirs found."); sys.exit(1)
    return csv_path, found_dirs


def _img_path(image_id, img_dirs):
    for d in img_dirs:
        p = os.path.join(d, image_id + ".jpg")
        if os.path.isfile(p):
            return p
    return None


def load_data():
    """
    STEP 3 CHANGE: loads merged_dataset.csv if it exists,
    falls back to HAM10000-only otherwise.

    merged_dataset.csv has columns: image_id, dx, image_path, source

    ── PATH FIX (NEW) ─────────────────────────────────────────────────────────
    merged_dataset.csv stores absolute Windows paths that are invalid on any
    other machine.  After reading the CSV we call _remap_csv_path() on every
    row of the image_path column to convert those Windows paths to the correct
    path on the current OS / Colab instance.
    ───────────────────────────────────────────────────────────────────────────
    """
    merged_path = os.path.join(_SCRIPT_DIR, "skin_dataset", "merged_dataset.csv")

    if os.path.isfile(merged_path):
        print(f"[INFO] Loading MERGED dataset from {merged_path}")
        df = pd.read_csv(merged_path)

        # ── PATH FIX: remap OS-absolute Windows paths → current machine paths ─
        # This is the critical fix for Google Colab (and any non-Windows env).
        # Without it, every os.path.isfile() check below returns False and
        # zero training images are loaded.
        print("[INFO] Remapping image paths to current environment ...")
        df["image_path"] = df["image_path"].apply(_remap_csv_path)

        # Report how many paths resolved successfully
        resolved = df["image_path"].apply(os.path.isfile).sum()
        total    = len(df)
        print(f"[INFO] Paths resolved on disk: {resolved}/{total}")
        if resolved == 0:
            print("[WARN] No images found on disk after path remapping.")
            print("       Ensure datasets are downloaded into skin_dataset/:")
            print("         skin_dataset/HAM10000_images_part_1/")
            print("         skin_dataset/HAM10000_images_part_2/")
            print("         skin_dataset/ISIC_2019_Training_Input/  (optional)")
        elif resolved < total:
            missing_count = total - resolved
            print(f"[INFO] {missing_count} images not found on disk and will be skipped.")
            print("       (ISIC_2019 images not downloaded is expected if you only")
            print("        downloaded HAM10000 — training continues with HAM10000.)")

    else:
        print("[INFO] merged_dataset.csv not found — loading HAM10000 only.")
        print("[INFO] Run prepare_dataset.py to generate the merged dataset.")
        csv_path, img_dirs = find_dataset()
        df = pd.read_csv(csv_path)
        df["image_path"] = df["image_id"].apply(lambda i: _img_path(i, img_dirs))

    df["label"] = df["dx"].map(LABEL_MAP)
    df = df.dropna(subset=["label", "image_path"])
    df["label"] = df["label"].astype(int)
    # Drop rows where the image file does not exist on disk
    df = df[df["image_path"].apply(os.path.isfile)]
    df = df.drop_duplicates(subset=["image_id"])

    print(f"\n[INFO] Loaded {len(df)} unique images")
    for lbl_name, lbl_int in LABEL_MAP.items():
        cnt = (df["label"] == lbl_int).sum()
        print(f"  [{lbl_int}] {lbl_name:<8}  {cnt:5d} images")

    if len(df) == 0:
        print("\n[ERROR] No images could be loaded. Training cannot proceed.")
        print("        Please ensure the dataset directories exist inside skin_dataset/:")
        for subdir in _KNOWN_IMAGE_SUBDIRS:
            full = os.path.join(_SCRIPT_DIR, "skin_dataset", subdir)
            status = "✓ found" if os.path.isdir(full) else "✗ missing"
            print(f"          {full}  [{status}]")
        sys.exit(1)

    return df["image_path"].tolist(), df["label"].tolist()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: SMART OVERSAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def oversample_rare_classes(train_paths, train_labels, target_count_map):
    """
    STEP 4: Oversample specific classes to reach target counts.

    Works WITH the existing SkinSequence augmentation — duplicated image paths
    get different random augmentations each epoch because augment=True is set
    on train_seq. So you get genuine variety, not exact copies.

    Parameters
    ----------
    train_paths : list
    train_labels : list
    target_count_map : dict {label_int: target_count}
    """
    paths_arr  = list(train_paths)
    labels_arr = list(train_labels)
    current    = Counter(labels_arr)

    for label_int, target in target_count_map.items():
        have = current.get(label_int, 0)
        if have == 0:
            print(f"[WARN] Label {label_int} has 0 training samples — skipping oversampling.")
            continue
        if have >= target:
            print(f"[INFO] Label {label_int} ({CLASS_NAMES[label_int]}): "
                  f"{have} samples already ≥ target {target}. No oversampling.")
            continue

        needed       = target - have
        class_paths  = [p for p, l in zip(train_paths, train_labels) if l == label_int]
        extra_paths  = (class_paths * ((needed // have) + 2))[:needed]
        extra_labels = [label_int] * len(extra_paths)
        paths_arr.extend(extra_paths)
        labels_arr.extend(extra_labels)
        print(f"[INFO] {CLASS_NAMES[label_int]:<8}: {have} → {have + len(extra_paths)} "
              f"(+{len(extra_paths)} oversampled)")

    combined = list(zip(paths_arr, labels_arr))
    random.shuffle(combined)
    paths_arr, labels_arr = zip(*combined)
    return list(paths_arr), list(labels_arr)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA GENERATOR
#  Unchanged from original except preprocess_input now uses EfficientNet's
# ══════════════════════════════════════════════════════════════════════════════

class SkinSequence(tf.keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size, hp, augment=False, aug_intensity=1.0):
        self.paths         = list(paths)
        self.labels        = list(labels)
        self.batch_size    = batch_size
        self.hp            = hp
        self.augment       = augment
        self.aug_intensity = aug_intensity
        self.indices       = np.arange(len(self.paths))

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.indices)

    def _load(self, path):
        from PIL import Image as PI
        # STEP 2 NOTE: EfficientNet preprocess_input expects [0, 255].
        # We load as float32 in [0, 255] — DO NOT scale to [0, 1] here.
        return np.array(PI.open(path).convert("RGB").resize(IMG_SIZE), dtype=np.float32)

    def _augment(self, img):
        from PIL import Image as PI
        hp = self.hp
        if hp["aug_hflip"] and np.random.rand() > 0.5:
            img = np.fliplr(img)
        if hp["aug_vflip"] and np.random.rand() > 0.5:
            img = np.flipud(img)

        # 90° rotations (dermoscopy is rotation-invariant)
        rot90 = np.random.randint(0, 4)
        if rot90 > 0:
            img = np.rot90(img, rot90)

        angle = np.random.uniform(
            -hp["aug_rotation"] * self.aug_intensity,
             hp["aug_rotation"] * self.aug_intensity,
        )
        pil = PI.fromarray(img.astype(np.uint8)).rotate(angle, resample=PI.BILINEAR)
        img = np.array(pil, dtype=np.float32)

        zoom_min = 1.0 - (1.0 - hp["aug_zoom"][0]) * self.aug_intensity
        zoom_max = 1.0 + (hp["aug_zoom"][1] - 1.0) * self.aug_intensity
        zoom  = np.random.uniform(zoom_min, zoom_max)
        h, w  = img.shape[:2]
        pil   = PI.fromarray(img.astype(np.uint8)).resize(
                    (int(w * zoom), int(h * zoom)), PI.BILINEAR).resize((w, h), PI.BILINEAR)
        img   = np.array(pil, dtype=np.float32)

        try:
            import cv2
            sx = int(np.random.uniform(-0.1, 0.1) * w)
            sy = int(np.random.uniform(-0.1, 0.1) * h)
            img = cv2.warpAffine(img, np.float32([[1, 0, sx], [0, 1, sy]]), (w, h))
        except ImportError:
            pass

        img = np.clip(img * np.random.uniform(*hp["aug_brightness"]), 0, 255)
        return img

    def __getitem__(self, idx):
        bi   = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        imgs = []
        for i in bi:
            try:
                img = self._load(self.paths[i])
                if self.augment:
                    img = self._augment(img)
                # STEP 2: EfficientNet preprocessor (scales [0,255] → normalised)
                imgs.append(preprocess_input(img))
            except Exception as e:
                print(f"[WARN] {self.paths[i]}: {e}")
                imgs.append(np.zeros((*IMG_SIZE, 3), dtype=np.float32))
        return np.array(imgs), np.array([self.labels[i] for i in bi])


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS CALLBACK  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_seq, every_n_epochs=1):
        super().__init__()
        self.val_seq = val_seq
        self.every_n = every_n_epochs
        self.metric_history = {k: [] for k in
                               ["val_precision", "val_recall", "val_f1", "val_f2"]}

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n != 0:
            return
        y_true, y_pred = [], []
        for i in range(len(self.val_seq)):
            X, y  = self.val_seq[i]
            preds = self.model.predict(X, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(y.tolist())
        kw   = dict(average="weighted", zero_division=0)
        prec = precision_score(y_true, y_pred, **kw)
        rec  = recall_score   (y_true, y_pred, **kw)
        f1   = f1_score       (y_true, y_pred, **kw)
        f2   = fbeta_score    (y_true, y_pred, beta=2, **kw)
        for k, v in zip(self.metric_history, [prec, rec, f1, f2]):
            self.metric_history[k].append(v)
        if logs:
            logs.update({"val_precision": prec, "val_recall": rec,
                         "val_f1": f1, "val_f2": f2})
        print(f"\n  Epoch {epoch+1:>3} — Precision: {prec:.4f}  "
              f"Recall: {rec:.4f}  F1: {f1:.4f}  F2: {f2:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: MODEL ARCHITECTURE — EfficientNetB3
# ══════════════════════════════════════════════════════════════════════════════

def build_model(hp, num_classes=NUM_CLASSES, focal_loss=None):
    """
    EfficientNetB3 backbone + custom classification head.

    STEP 2 CHANGES vs original:
      - MobileNetV2 → EfficientNetB3
      - fine_tune_layers default 30 → 20 (EfficientNetB3 has more total layers)
      - Focal Loss passed in and used in compile()

    Two-phase training strategy (unchanged):
      Phase 1: backbone frozen, train Dense head only.
      Phase 2: unfreeze last fine_tune_layers, fine-tune at tiny LR.
    """
    base = EfficientNetB3(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # frozen in Phase 1

    x       = base.output
    x       = GlobalAveragePooling2D()(x)
    x       = Dense(hp["dense_units"], activation="relu")(x)
    x       = BatchNormalization()(x)
    x       = Dropout(hp["dropout_rate"])(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=outputs)

    loss_fn = focal_loss if focal_loss is not None else "sparse_categorical_crossentropy"

    model.compile(
        optimizer=Adam(learning_rate=hp["lr_phase1"]),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    print(f"[INFO] EfficientNetB3 built — {model.count_params():,} total params")
    return model, base


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, val_seq, num_classes):
    print("\n" + "═" * 64)
    print("  FULL EVALUATION REPORT")
    print("═" * 64)

    y_true, y_pred_classes, y_pred_probs = [], [], []
    for i in range(len(val_seq)):
        X, y  = val_seq[i]
        probs = model.predict(X, verbose=0)
        y_true.extend(y.tolist())
        y_pred_classes.extend(np.argmax(probs, axis=1).tolist())
        y_pred_probs.extend(probs.tolist())

    y_true         = np.array(y_true)
    y_pred_classes = np.array(y_pred_classes)
    y_pred_probs   = np.array(y_pred_probs)

    kw   = dict(average="weighted", zero_division=0)
    kw_m = dict(average="macro",    zero_division=0)
    acc  = np.mean(y_true == y_pred_classes)
    print(f"\nOverall Accuracy      : {acc*100:.2f}%")
    print(f"Weighted Precision    : {precision_score(y_true, y_pred_classes, **kw):.4f}")
    print(f"Weighted Recall       : {recall_score   (y_true, y_pred_classes, **kw):.4f}")
    print(f"Weighted F1           : {f1_score       (y_true, y_pred_classes, **kw):.4f}")
    print(f"Weighted F2           : {fbeta_score    (y_true, y_pred_classes, beta=2, **kw):.4f}")
    print(f"Macro F1              : {f1_score       (y_true, y_pred_classes, **kw_m):.4f}")
    print(f"Macro F2              : {fbeta_score    (y_true, y_pred_classes, beta=2, **kw_m):.4f}")

    print("\n" + "─" * 64)
    print(classification_report(y_true, y_pred_classes,
                                 target_names=CLASS_NAMES[:num_classes], zero_division=0))

    f2_per_class = fbeta_score(y_true, y_pred_classes, beta=2,
                               average=None, zero_division=0)
    print("Per-class F2 scores:")
    for i, (name, f2) in enumerate(zip(CLASS_NAMES[:num_classes], f2_per_class)):
        print(f"  [{i}] {name:<8}  F2 = {f2:.4f}")

    roc_aucs = {}
    for i in range(num_classes):
        try:
            auc = roc_auc_score((y_true == i).astype(int), y_pred_probs[:, i])
            roc_aucs[CLASS_NAMES[i]] = auc
            print(f"  [{i}] {CLASS_NAMES[i]:<8}  AUC = {auc:.4f}")
        except Exception:
            pass
    if roc_aucs:
        print(f"\n  Macro AUC = {np.mean(list(roc_aucs.values())):.4f}")

    return y_true, y_pred_classes, y_pred_probs, roc_aucs


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def _ax_style(ax):
    ax.set_facecolor(DARK_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(DARK_BD)
    ax.tick_params(colors=TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TITLE_COL)


def plot_training_history(h1, h2, metrics_cb):
    acc      = h1.history["accuracy"]     + h2.history["accuracy"]
    val_acc  = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss     = h1.history["loss"]         + h2.history["loss"]
    val_loss = h1.history["val_loss"]     + h2.history["val_loss"]
    p1_end   = len(h1.history["accuracy"])
    epochs   = range(1, len(acc) + 1)
    mh       = metrics_cb.metric_history
    m_epochs = range(1, len(mh["val_f1"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(DARK_FIG)
    fig.suptitle("Training History — All Metrics", color=TITLE_COL, fontsize=15)
    axes = axes.flatten()
    for ax in axes:
        _ax_style(ax)

    def _vline(ax):
        ax.axvline(p1_end, color="#6e7681", linestyle="--", linewidth=1, label="Fine-tune →")

    axes[0].plot(epochs, acc,     color="#58a6ff", linewidth=2, label="Train Accuracy")
    axes[0].plot(epochs, val_acc, color="#3fb950", linewidth=2, label="Val Accuracy")
    _vline(axes[0])
    axes[0].set_title("Accuracy"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(facecolor=DARK_FIG, edgecolor=DARK_BD, labelcolor=TEXT_COL)

    axes[1].plot(epochs, loss,     color="#f85149", linewidth=2, label="Train Loss")
    axes[1].plot(epochs, val_loss, color="#d29922", linewidth=2, label="Val Loss")
    _vline(axes[1])
    axes[1].set_title("Loss"); axes[1].set_ylabel("Loss")
    axes[1].legend(facecolor=DARK_FIG, edgecolor=DARK_BD, labelcolor=TEXT_COL)

    if mh["val_f1"]:
        axes[2].plot(m_epochs, mh["val_f1"], color="#58a6ff", linewidth=2, label="Val F1 (weighted)")
        axes[2].plot(m_epochs, mh["val_f2"], color="#bc8cff", linewidth=2, label="Val F2 (weighted)")
        axes[2].set_title("F1 & F2 Score"); axes[2].set_ylabel("Score")
        axes[2].legend(facecolor=DARK_FIG, edgecolor=DARK_BD, labelcolor=TEXT_COL)

    if mh["val_precision"]:
        axes[3].plot(m_epochs, mh["val_precision"], color="#3fb950", linewidth=2, label="Val Precision")
        axes[3].plot(m_epochs, mh["val_recall"],    color="#d29922", linewidth=2, label="Val Recall")
        axes[3].set_title("Precision & Recall"); axes[3].set_ylabel("Score")
        axes[3].legend(facecolor=DARK_FIG, edgecolor=DARK_BD, labelcolor=TEXT_COL)

    for ax in axes:
        ax.set_xlabel("Epoch")
    plt.tight_layout()
    out = os.path.join(REPORT_DIR, "01_training_history.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_FIG)
    plt.close()
    print(f"[SAVED] {out}")


def plot_confusion_matrix(y_true, y_pred, num_classes):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    names   = CLASS_NAMES[:num_classes]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor(DARK_FIG)
    fig.suptitle("Confusion Matrix", color=TITLE_COL, fontsize=14)

    for ax, data, title, fmt in zip(axes, [cm, cm_norm],
                                    ["Absolute Counts", "Normalised (row %)"],
                                    ["d", ".2f"]):
        _ax_style(ax)
        im = ax.imshow(data, cmap="Blues", aspect="auto")
        ax.set_xticks(range(num_classes)); ax.set_xticklabels(names, rotation=45, ha="right", color=TEXT_COL)
        ax.set_yticks(range(num_classes)); ax.set_yticklabels(names, color=TEXT_COL)
        ax.set_title(title, color=TITLE_COL)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.colorbar(im, ax=ax)
        thresh = data.max() / 2
        for i in range(num_classes):
            for j in range(num_classes):
                val   = data[i, j]
                label = (f"{val:.2f}" if fmt == ".2f" else str(val))
                ax.text(j, i, label, ha="center", va="center",
                        color="white" if val > thresh else TEXT_COL, fontsize=8)

    plt.tight_layout()
    out = os.path.join(REPORT_DIR, "02_confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_FIG)
    plt.close()
    print(f"[SAVED] {out}")


def plot_roc_curves(y_true, y_pred_probs, num_classes, roc_aucs):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(DARK_FIG)
    _ax_style(ax)
    colors = ["#58a6ff","#f85149","#3fb950","#d29922","#bc8cff","#79c0ff","#56d364"]
    for i in range(num_classes):
        try:
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_probs[:, i])
            auc         = roc_aucs.get(CLASS_NAMES[i], 0)
            ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                    label=f"[{CLASS_NAMES[i]}] AUC={auc:.3f}")
        except Exception:
            pass
    ax.plot([0,1],[0,1], color="#6e7681", linestyle="--", linewidth=1)
    ax.set_title("ROC Curves", color=TITLE_COL, fontsize=14)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(facecolor=DARK_FIG, edgecolor=DARK_BD, labelcolor=TEXT_COL, fontsize=9)
    plt.tight_layout()
    out = os.path.join(REPORT_DIR, "03_roc_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_FIG)
    plt.close()
    print(f"[SAVED] {out}")


def plot_per_class_metrics(y_true, y_pred, num_classes):
    names = CLASS_NAMES[:num_classes]
    prec  = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec   = recall_score   (y_true, y_pred, average=None, zero_division=0)
    f1    = f1_score       (y_true, y_pred, average=None, zero_division=0)
    f2    = fbeta_score    (y_true, y_pred, beta=2, average=None, zero_division=0)

    x   = np.arange(num_classes)
    w   = 0.2
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(DARK_FIG)
    _ax_style(ax)
    ax.bar(x - 1.5*w, prec, w, label="Precision", color="#58a6ff", alpha=0.9)
    ax.bar(x - 0.5*w, rec,  w, label="Recall",    color="#3fb950", alpha=0.9)
    ax.bar(x + 0.5*w, f1,   w, label="F1",        color="#d29922", alpha=0.9)
    ax.bar(x + 1.5*w, f2,   w, label="F2",        color="#bc8cff", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(names, color=TEXT_COL)
    ax.set_ylim(0, 1.15)
    ax.set_title("Per-Class Metrics", color=TITLE_COL, fontsize=14)
    ax.set_ylabel("Score")
    ax.legend(facecolor=DARK_FIG, edgecolor=DARK_BD, labelcolor=TEXT_COL)
    plt.tight_layout()
    out = os.path.join(REPORT_DIR, "04_per_class_metrics.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_FIG)
    plt.close()
    print(f"[SAVED] {out}")


def save_metrics_txt(y_true, y_pred, y_probs, num_classes, roc_aucs, elapsed):
    kw   = dict(average="weighted", zero_division=0)
    kw_m = dict(average="macro",    zero_division=0)
    lines = [
        "TRAINING RESULTS REPORT (UPGRADED MODEL)",
        "=" * 64,
        f"Training time   : {elapsed/60:.1f} min",
        f"Num classes     : {num_classes}",
        "Backbone        : EfficientNetB3",
        "Loss            : SparseCategoricalFocalLoss(gamma=2.0)",
        "",
        "OVERALL METRICS",
        "-" * 40,
        f"Accuracy        : {np.mean(y_true==y_pred)*100:.2f}%",
        f"Weighted Prec   : {precision_score(y_true,y_pred,**kw):.4f}",
        f"Weighted Recall : {recall_score   (y_true,y_pred,**kw):.4f}",
        f"Weighted F1     : {f1_score       (y_true,y_pred,**kw):.4f}",
        f"Weighted F2     : {fbeta_score    (y_true,y_pred,beta=2,**kw):.4f}",
        f"Macro F1        : {f1_score       (y_true,y_pred,**kw_m):.4f}",
        f"Macro F2        : {fbeta_score    (y_true,y_pred,beta=2,**kw_m):.4f}",
        "",
        "PER-CLASS CLASSIFICATION REPORT",
        "-" * 64,
        classification_report(y_true, y_pred,
                              target_names=CLASS_NAMES[:num_classes], zero_division=0),
        "PER-CLASS F2 SCORES",
        "-" * 40,
    ]
    f2pc = fbeta_score(y_true, y_pred, beta=2, average=None, zero_division=0)
    for i, (n, v) in enumerate(zip(CLASS_NAMES[:num_classes], f2pc)):
        lines.append(f"  [{i}] {n:<8}  {v:.4f}")
    lines += ["", "ROC-AUC (one-vs-rest)", "-" * 40]
    for name, auc in roc_aucs.items():
        lines.append(f"  {name:<8}  {auc:.4f}")
    if roc_aucs:
        lines.append(f"\n  Macro AUC = {np.mean(list(roc_aucs.values())):.4f}")
    lines += ["", "HYPERPARAMETERS USED", "-" * 40]
    for k, v in HYPERPARAMS.items():
        lines.append(f"  {k:<28} = {v}")

    out = os.path.join(REPORT_DIR, "metrics_report.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"[SAVED] {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPERATURE CALIBRATION (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_temperature(model, val_seq, num_classes):
    from scipy.optimize import minimize_scalar
    logits_all, y_true_all = [], []
    for i in range(len(val_seq)):
        X, y = val_seq[i]
        preds = model.predict(X, verbose=0)
        logits_all.extend(preds.tolist())
        y_true_all.extend(y.tolist())
    logits = np.array(logits_all)
    labels = np.array(y_true_all)

    def nll(T):
        scaled = logits / T
        scaled -= scaled.max(axis=1, keepdims=True)
        log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        return -log_probs[np.arange(len(labels)), labels].mean()

    result = minimize_scalar(nll, bounds=(0.1, 5.0), method="bounded")
    T_opt  = float(result.x)
    print(f"[INFO] Optimal temperature T = {T_opt:.4f}")
    return T_opt


# ══════════════════════════════════════════════════════════════════════════════
#  LR WARMUP (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class WarmUpLR(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs, start_lr, target_lr):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.start_lr      = start_lr
        self.target_lr     = target_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.start_lr + (self.target_lr - self.start_lr) * (epoch / self.warmup_epochs)
            opt = self.model.optimizer
            if hasattr(opt, "learning_rate"):
                if hasattr(opt.learning_rate, "assign"):
                    opt.learning_rate.assign(lr)
                else:
                    tf.keras.backend.set_value(opt.learning_rate, lr)
            elif hasattr(opt, "lr"):
                tf.keras.backend.set_value(opt.lr, lr)
            print(f"\n  [WarmUp] epoch {epoch+1}: lr={lr:.2e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train():
    import json
    from datetime import datetime

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    hp = HYPERPARAMS
    t0 = time.time()

    # ── STEP 1: build Focal Loss instance ─────────────────────────────────────
    focal_loss = SparseCategoricalFocalLoss(
        gamma=hp["focal_gamma"],
        alpha=hp["focal_alpha"],
        name="focal_loss",
    )
    print(f"[INFO] Using Focal Loss with gamma={hp['focal_gamma']}, alpha={hp['focal_alpha']}")

    # ── STEP 3: load merged or HAM-only dataset ────────────────────────────────
    # load_data() now includes the path-remapping fix for Colab / non-Windows.
    all_paths, all_labels = load_data()
    num_classes           = len(set(all_labels))

    # Train / val / test split (70/15/15 stratified)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=0.30, random_state=42, stratify=all_labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels
    )
    print(f"\n[INFO] Train: {len(train_paths)}  Val: {len(val_paths)}  Test: {len(test_paths)}")

    # Class weights (keep these even with Focal Loss — they complement each other)
    unique_labels     = np.unique(train_labels)
    cw                = compute_class_weight("balanced", classes=unique_labels, y=train_labels)
    class_weight_dict = dict(zip(unique_labels.tolist(), cw.tolist()))
    print("[INFO] Class weights:")
    for lbl, w in class_weight_dict.items():
        print(f"  [{lbl}] {CLASS_NAMES[lbl]:<8}  {w:.3f}x")

    # ── STEP 4: targeted oversampling for DF and VASC ─────────────────────────
    print("\n[INFO] Applying targeted oversampling for rare classes ...")
    train_paths_aug, train_labels_aug = oversample_rare_classes(
        train_paths, train_labels, hp["oversample_targets"]
    )
    print(f"[INFO] Training set after oversampling: {len(train_paths_aug)} samples")

    # ── Sequences ────────────────────────────────────────────────────────────
    bs         = hp["batch_size"]
    train_seq  = SkinSequence(train_paths_aug, train_labels_aug, bs, hp,
                              augment=True, aug_intensity=1.5)
    val_seq    = SkinSequence(val_paths,  val_labels,  bs, hp, augment=False)
    test_seq   = SkinSequence(test_paths, test_labels, bs, hp, augment=False)
    metrics_cb = MetricsCallback(val_seq, every_n_epochs=1)

    # ── STEP 2: build EfficientNetB3 model ────────────────────────────────────
    model, base = build_model(hp, num_classes, focal_loss=focal_loss)
    model.summary()

    # ── Phase 1: head only ────────────────────────────────────────────────────
    print(f"\n[PHASE 1] Head training — EfficientNetB3 backbone frozen ...")
    cb1 = [
        WarmUpLR(hp["warmup_epochs"], hp["lr_warmup_start"], hp["lr_phase1"]),
        EarlyStopping(patience=hp["patience"], restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, verbose=1),
        ReduceLROnPlateau(factor=hp["lr_reduce_factor"],
                          patience=hp["lr_reduce_patience"], min_lr=1e-8, verbose=1),
        metrics_cb,
    ]
    history1 = model.fit(
        train_seq, validation_data=val_seq,
        epochs=hp["epochs_phase1"], callbacks=cb1,
        class_weight=class_weight_dict, verbose=1,
    )

    # ── Phase 2: fine-tune backbone ───────────────────────────────────────────
    if hp["fine_tune_layers"] > 0:
        print(f"\n[PHASE 2] Fine-tuning last {hp['fine_tune_layers']} EfficientNetB3 layers ...")
        for layer in base.layers[-hp["fine_tune_layers"]:]:
            layer.trainable = True

        # Recompile with smaller LR and same Focal Loss
        model.compile(
            optimizer=Adam(learning_rate=hp["lr_phase2"]),
            loss=focal_loss,
            metrics=["accuracy"],
        )
        cb2 = [
            EarlyStopping(patience=hp["patience"], restore_best_weights=True, verbose=1),
            ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, verbose=1),
            ReduceLROnPlateau(factor=hp["lr_reduce_factor"],
                              patience=hp["lr_reduce_patience"], min_lr=1e-9, verbose=1),
            metrics_cb,
        ]
        history2 = model.fit(
            train_seq, validation_data=val_seq,
            epochs=hp["epochs_phase2"], callbacks=cb2,
            class_weight=class_weight_dict, verbose=1,
        )
    else:
        history2 = history1

    model.save(MODEL_SAVE_PATH)
    elapsed = time.time() - t0
    print(f"\n✅ Model saved → {MODEL_SAVE_PATH}")
    print(f"   Training time : {elapsed/60:.1f} min")

    # Temperature calibration
    T_opt = calibrate_temperature(model, val_seq, num_classes)

    # Save config JSON (STEP 3: note efficientnet preprocessor)
    config = {
        "preprocess":  "efficientnet",   # CHANGED from mobilenet_v2
        "img_size":    list(IMG_SIZE),
        "num_classes": num_classes,
        "class_names": CLASS_NAMES[:num_classes],
        "trained_at":  datetime.now().isoformat(),
        "temperature": T_opt,
        "backbone":    "EfficientNetB3",
        "loss":        f"FocalLoss(gamma={hp['focal_gamma']})",
    }
    config_path = MODEL_SAVE_PATH.replace(".h5", "_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[SAVED] {config_path}")

    # Evaluate on test set
    y_true, y_pred, y_probs, roc_aucs = evaluate_model(model, test_seq, num_classes)

    print(f"\n[INFO] Saving charts to {REPORT_DIR}/ ...")
    plot_training_history(history1, history2, metrics_cb)
    plot_confusion_matrix(y_true, y_pred, num_classes)
    plot_roc_curves(y_true, y_probs, num_classes, roc_aucs)
    plot_per_class_metrics(y_true, y_pred, num_classes)
    save_metrics_txt(y_true, y_pred, y_probs, num_classes, roc_aucs, elapsed)

    print(f"\n✅ All reports saved to {REPORT_DIR}/")
    print("═" * 64)
    print(f"  NORMAL SKIN THRESHOLD = {hp['normal_threshold']}")
    print("  Update models/model.py preprocess_input import:")
    print("  from tensorflow.keras.applications.efficientnet import preprocess_input")
    print("═" * 64)


if __name__ == "__main__":
    train()
