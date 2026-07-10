"""
prepare_dataset.py
══════════════════════════════════════════════════════════════════════════════
Merges HAM10000 + selective ISIC 2019 images into a single merged_dataset.csv
that train_model.py (Step 3) can load directly.

Rules enforced:
  ✅ Only add minority classes from ISIC 2019 — classes already well
     represented in HAM are left unchanged.
  ✅ Skip NV  — adding it makes imbalance worse (6,705 → 12,875)
  ✅ Skip BCC — excluded to avoid inflating an already-represented class
  ✅ Skip AK  — condition overlaps akiec; SCC is used instead (see below)
  ✅ Map SCC → akiec — SCC (Squamous Cell Carcinoma) and akiec (Actinic
     Keratosis / Intraepithelial Carcinoma) are the same keratinocyte
     carcinoma spectrum. The 197 HAM akiec images were re-labelled SCC in
     ISIC 2019. Adding the 431 new ISIC SCC images as akiec raises the
     class from 327 → 758 and cuts its imbalance from 20.5× to 8.8×.
  ✅ Skip UNK — 0 images anyway
  ✅ Deduplicate — all 10,015 HAM images exist in ISIC 2019 (same ISIC IDs)
                   only the 15,316 genuinely new ISIC images are considered
  ✅ Label mapping — ISIC uppercase → model lowercase labels

Expected result
  nv      6,705  (unchanged)
  mel     4,522  (+3,409 from ISIC)
  bkl     2,624  (+1,525 from ISIC)
  bcc       514  (unchanged)
  akiec     758  (+431 from ISIC SCC — FIX: was stuck at 327, 20.5× imbalance)
  vasc      253  (+111 from ISIC)
  df        239  (+124 from ISIC)
  ─────────────
  Total  15,615  images
  Imbalance ratio: 28x → weakest class now df(239) at 28x, akiec improved to 8.8x

Output
  skin_dataset/merged_dataset.csv
  Columns: image_id, dx, image_path, source

Usage
  python prepare_dataset.py
  (run from your project root, i.e. the folder containing train_model.py)
══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS — all relative to project root (where this script lives)
# ══════════════════════════════════════════════════════════════════════════════

_ROOT = os.path.dirname(os.path.abspath(__file__))

# HAM10000
HAM_CSV  = os.path.join(_ROOT, "skin_dataset", "HAM10000_metadata.csv")
HAM_DIRS = [
    os.path.join(_ROOT, "skin_dataset", "HAM10000_images_part_1"),
    os.path.join(_ROOT, "skin_dataset", "HAM10000_images_part_2"),
]

# ISIC 2019
ISIC_CSV    = os.path.join(_ROOT, "skin_dataset", "ISIC_2019_Training_GroundTruth.csv")
ISIC_IMGDIR = os.path.join(_ROOT, "skin_dataset", "ISIC_2019_Training_Input")

# Output
OUT_CSV = os.path.join(_ROOT, "skin_dataset", "merged_dataset.csv")

# ══════════════════════════════════════════════════════════════════════════════
#  LABEL MAPPING
#  ISIC 2019 uses uppercase one-hot columns.
#  We keep 5 minority classes that benefit from more data.
#
#  SCC → akiec  ← FIX: SCC (Squamous Cell Carcinoma) and akiec (Actinic
#    Keratosis / Intraepithelial Carcinoma) are the same keratinocyte
#    carcinoma spectrum. ISIC 2019 re-labelled the 197 HAM akiec images as
#    SCC. Adding the 431 genuinely new ISIC SCC images as akiec raises the
#    class count from 327 → 758 and reduces its imbalance from 20.5× to 8.8×.
#    This is the single highest-impact fix available in this dataset.
#
#  AK  → skipped: condition overlaps akiec; SCC is the correct merge target
#  NV  → skipped: adding 6,170 new NV images worsens imbalance (6,705→12,875)
#  BCC → skipped: excluded to avoid inflating an already-represented class
#  UNK → skipped: 0 images in ISIC 2019
# ══════════════════════════════════════════════════════════════════════════════

ISIC_TO_MODEL = {
    "MEL":  "mel",
    "BKL":  "bkl",
    "DF":   "df",
    "VASC": "vasc",
    "SCC":  "akiec",   # FIX: SCC = same spectrum as akiec; raises 327 → 758
    # NV, BCC, AK, UNK → intentionally excluded
}

ISIC_ALL_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _find_ham_image(image_id: str) -> str | None:
    """Search HAM image directories for image_id.jpg"""
    for d in HAM_DIRS:
        p = os.path.join(d, image_id + ".jpg")
        if os.path.isfile(p):
            return p
    return None


def _check_paths():
    """Validate all required files and folders exist before starting."""
    errors = []
    if not os.path.isfile(HAM_CSV):
        errors.append(f"  ✗ HAM metadata not found : {HAM_CSV}")
    if not os.path.isfile(ISIC_CSV):
        errors.append(f"  ✗ ISIC ground truth not found : {ISIC_CSV}")
    if not os.path.isdir(ISIC_IMGDIR):
        errors.append(f"  ✗ ISIC image folder not found : {ISIC_IMGDIR}")

    ham_dirs_found = [d for d in HAM_DIRS if os.path.isdir(d)]
    if not ham_dirs_found:
        errors.append(f"  ✗ No HAM image folders found in: {_ROOT}/skin_dataset/")

    if errors:
        print("\n[ERROR] Missing required files/folders:")
        for e in errors:
            print(e)
        print("\nExpected folder structure:")
        print("  your_project/")
        print("  ├── prepare_dataset.py  ← this script")
        print("  └── skin_dataset/")
        print("      ├── HAM10000_metadata.csv")
        print("      ├── HAM10000_images_part_1/")
        print("      ├── HAM10000_images_part_2/")
        print("      ├── ISIC_2019_Training_GroundTruth.csv")
        print("      └── ISIC_2019_Training_Input/")
        sys.exit(1)

    return ham_dirs_found


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Load HAM10000
# ══════════════════════════════════════════════════════════════════════════════

def load_ham(ham_dirs_found):
    print("\n[STEP 1] Loading HAM10000 ...")
    df = pd.read_csv(HAM_CSV)

    # Resolve image paths
    df["image_path"] = df["image_id"].apply(_find_ham_image)
    missing = df["image_path"].isna().sum()
    if missing > 0:
        print(f"  [WARN] {missing} HAM images not found on disk — they will be skipped.")

    df = df.dropna(subset=["image_path"])
    df = df.drop_duplicates(subset=["image_id"])
    df["source"] = "HAM10000"

    result = df[["image_id", "dx", "image_path", "source"]].copy()
    print(f"  Loaded {len(result)} HAM10000 images")
    for cls, cnt in result["dx"].value_counts().items():
        print(f"    {cls:<8} {cnt:5d}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Load ISIC 2019, deduplicate, filter to minority classes only
# ══════════════════════════════════════════════════════════════════════════════

def load_isic_selective(ham_df):
    print("\n[STEP 2] Loading ISIC 2019 ...")
    isic = pd.read_csv(ISIC_CSV)

    # Convert one-hot columns → single label string
    isic["dx_isic"] = isic[ISIC_ALL_CLASSES].idxmax(axis=1)
    print(f"  Total ISIC 2019 images: {len(isic)}")

    # ── Deduplication ─────────────────────────────────────────────────────────
    # All HAM10000 images exist in ISIC 2019 with identical ISIC IDs.
    # We only want the genuinely new images.
    ham_ids     = set(ham_df["image_id"].tolist())
    new_only    = isic[~isic["image"].isin(ham_ids)].copy()
    n_dupes     = len(isic) - len(new_only)
    print(f"  Removed {n_dupes} duplicates (HAM images already in ISIC 2019)")
    print(f"  Genuinely new ISIC images: {len(new_only)}")

    # ── Class filter ──────────────────────────────────────────────────────────
    # Keep MEL, BKL, DF, VASC, SCC (mapped to akiec) — minority classes only
    keep_classes = list(ISIC_TO_MODEL.keys())
    filtered     = new_only[new_only["dx_isic"].isin(keep_classes)].copy()
    skipped      = len(new_only) - len(filtered)
    print(f"  Skipped {skipped} images (NV, BCC, AK, UNK — not needed)")
    print(f"  New images to add: {len(filtered)}")
    for cls in keep_classes:
        cnt = (filtered["dx_isic"] == cls).sum()
        print(f"    {cls:<6} → {ISIC_TO_MODEL[cls]:<8} {cnt:5d} new images")

    # ── Resolve image paths ────────────────────────────────────────────────────
    def _isic_path(image_id):
        # ISIC images are typically stored as ISIC_XXXXXXX.jpg
        p = os.path.join(ISIC_IMGDIR, image_id + ".jpg")
        if os.path.isfile(p):
            return p
        # Some datasets use .JPG uppercase
        p2 = os.path.join(ISIC_IMGDIR, image_id + ".JPG")
        if os.path.isfile(p2):
            return p2
        return None

    filtered["image_path"] = filtered["image"].apply(_isic_path)
    missing = filtered["image_path"].isna().sum()
    if missing > 0:
        print(f"  [WARN] {missing} ISIC images listed in CSV but not found on disk — skipping.")
        print( "         Make sure ISIC_2019_Training_Input/ contains all images.")
    filtered = filtered.dropna(subset=["image_path"])

    # ── Map labels to model format ────────────────────────────────────────────
    filtered["dx"]     = filtered["dx_isic"].map(ISIC_TO_MODEL)
    filtered["source"] = "ISIC_2019"
    filtered           = filtered.rename(columns={"image": "image_id"})

    result = filtered[["image_id", "dx", "image_path", "source"]].copy()
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Merge and save
# ══════════════════════════════════════════════════════════════════════════════

def merge_and_save(ham_df, isic_df):
    print("\n[STEP 3] Merging datasets ...")
    merged = pd.concat([ham_df, isic_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["image_id"])

    # Sanity check — no NaN labels or paths
    bad = merged[merged["dx"].isna() | merged["image_path"].isna()]
    if len(bad) > 0:
        print(f"  [WARN] Dropping {len(bad)} rows with missing label or path.")
        merged = merged.dropna(subset=["dx", "image_path"])

    merged.to_csv(OUT_CSV, index=False)
    return merged


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Print final summary
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(merged):
    print("\n" + "═" * 60)
    print("  MERGED DATASET SUMMARY")
    print("═" * 60)

    counts  = merged["dx"].value_counts()
    total   = len(merged)
    max_cnt = counts.max()
    min_cnt = counts.min()

    CLASS_ORDER = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
    print(f"\n  {'Class':<10} {'Count':>6}  {'Bar'}")
    print(f"  {'─'*10} {'─'*6}  {'─'*30}")
    for cls in CLASS_ORDER:
        cnt  = counts.get(cls, 0)
        bar  = "█" * int(cnt / max_cnt * 28)
        print(f"  {cls:<10} {cnt:>6}  {bar}")

    print(f"\n  Total images    : {total:,}")
    print(f"  Imbalance ratio : {max_cnt/min_cnt:.1f}x  (was 58x with HAM only)")

    sources = merged["source"].value_counts()
    print(f"\n  Source breakdown:")
    for src, cnt in sources.items():
        print(f"    {src:<20} {cnt:5d} images")

    print(f"\n  Saved to → {OUT_CSV}")
    print("\n  ✅ Done! You can now run train_model.py")
    print("     It will auto-detect merged_dataset.csv and use it.")
    print("═" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 60)
    print("  DermAI Monitor — Dataset Preparation Script")
    print("  HAM10000 + Selective ISIC 2019 merge")
    print("═" * 60)

    # Validate paths first
    ham_dirs_found = _check_paths()

    # Override HAM_DIRS with only the ones that actually exist
    HAM_DIRS[:] = ham_dirs_found

    # Run pipeline
    ham_df  = load_ham(ham_dirs_found)
    isic_df = load_isic_selective(ham_df)
    merged  = merge_and_save(ham_df, isic_df)
    print_summary(merged)