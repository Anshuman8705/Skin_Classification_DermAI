
import os
import sys
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR  = os.path.join(_SCRIPT_DIR, "dataset_report")

# Full descriptive names for each HAM10000 label
CLASS_INFO = {
    "nv": {
        "full_name"  : "Melanocytic Nevus (Mole)",
        "risk"       : "BENIGN",
        "risk_color" : "#3fb950",
        "description": (
            "The most common skin lesion — a benign cluster of pigmented cells. "
            "Common moles. Rarely become cancerous. Usually brown/black, "
            "round or oval, < 6 mm."
        ),
    },
    "mel": {
        "full_name"  : "Melanoma",
        "risk"       : "MALIGNANT",
        "risk_color" : "#f85149",
        "description": (
            "The most dangerous skin cancer. Arises from melanocytes. "
            "Early detection is critical — 5-year survival drops from ~99% "
            "(local) to ~30% (distant metastasis). ABCDE rule: Asymmetry, "
            "Border irregularity, Colour variation, Diameter > 6 mm, Evolution."
        ),
    },
    "bkl": {
        "full_name"  : "Benign Keratosis",
        "risk"       : "BENIGN",
        "risk_color" : "#3fb950",
        "description": (
            "Umbrella term for seborrhoeic keratosis, solar lentigo, and "
            "lichen-planus-like keratosis. Non-cancerous, common in older adults. "
            "Rough, waxy, 'stuck-on' appearance. No treatment needed unless "
            "cosmetic concern or irritation."
        ),
    },
    "bcc": {
        "full_name"  : "Basal Cell Carcinoma",
        "risk"       : "MALIGNANT",
        "risk_color" : "#f85149",
        "description": (
            "Most common skin cancer overall. Rarely metastasises but can invade "
            "surrounding tissue. Pearly/waxy bump, pink growth, or flat scar-like "
            "lesion. Sun-exposed areas. Highly treatable when caught early."
        ),
    },
    "akiec": {
        "full_name"  : "Actinic Keratosis / Intraepithelial Carcinoma",
        "risk"       : "PRE-MALIGNANT",
        "risk_color" : "#d29922",
        "description": (
            "Rough, scaly patch caused by years of sun exposure. Considered "
            "pre-cancerous — 5–10% progress to squamous cell carcinoma if "
            "untreated. Also called solar keratosis. Treatment: cryotherapy, "
            "topical fluorouracil, photodynamic therapy."
        ),
    },
    "vasc": {
        "full_name"  : "Vascular Lesion",
        "risk"       : "BENIGN",
        "risk_color" : "#3fb950",
        "description": (
            "Includes cherry angiomas, angiokeratomas, and pyogenic granulomas. "
            "Caused by abnormal blood vessel growth. Red/purple colour. "
            "Generally benign. Pyogenic granulomas can bleed heavily. "
            "Treatment if symptomatic: laser, electrocautery."
        ),
    },
    "df": {
        "full_name"  : "Dermatofibroma",
        "risk"       : "BENIGN",
        "risk_color" : "#3fb950",
        "description": (
            "Hard, raised bump — a benign fibrous nodule in the dermis. "
            "Usually on lower legs, common in women. Pink-brown. "
            "Dimples inward when pinched (Fitzpatrick sign). "
            "No treatment needed; rarely becomes malignant."
        ),
    },
}

RISK_ORDER = {"MALIGNANT": 0, "PRE-MALIGNANT": 1, "BENIGN": 2}

DARK_BG = "#0d1117"
DARK_FIG = "#161b22"
DARK_BORDER = "#30363d"
TEXT_COLOR = "#c9d1d9"
TITLE_COLOR = "#f0f6fc"


# ──────────────────────────────────────────────────────────────────────────────
# DATASET FINDER (same logic as train_model.py)
# ──────────────────────────────────────────────────────────────────────────────
def find_dataset():
    csv_candidates = [
        os.path.join(_SCRIPT_DIR, "skin_dataset", "HAM10000_metadata.csv"),
        os.path.join(_SCRIPT_DIR, "HAM10000_metadata.csv"),
        os.path.join(os.path.dirname(_SCRIPT_DIR), "skin_dataset", "HAM10000_metadata.csv"),
        os.path.join(os.path.dirname(_SCRIPT_DIR), "HAM10000_metadata.csv"),
    ]
    img_dir_candidates = [
        os.path.join(_SCRIPT_DIR, "skin_dataset", "HAM10000_images_part_1"),
        os.path.join(_SCRIPT_DIR, "skin_dataset", "HAM10000_images_part_2"),
        os.path.join(_SCRIPT_DIR, "HAM10000_images_part_1"),
        os.path.join(_SCRIPT_DIR, "HAM10000_images_part_2"),
    ]
    csv_path  = next((c for c in csv_candidates if os.path.isfile(c)), None)
    found_dirs = [d for d in img_dir_candidates if os.path.isdir(d)]
    if not csv_path:
        print("[ERROR] HAM10000_metadata.csv not found.")
        sys.exit(1)
    return csv_path, found_dirs


def build_image_path(image_id, img_dirs):
    for d in img_dirs:
        p = os.path.join(d, image_id + ".jpg")
        if os.path.isfile(p):
            return p
    return None


# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATASET
# ──────────────────────────────────────────────────────────────────────────────
def load_dataset():
    csv_path, img_dirs = find_dataset()
    df = pd.read_csv(csv_path)

    # Resolve image paths
    df["image_path"] = df["image_id"].apply(lambda i: build_image_path(i, img_dirs))

    # Deduplicate (HAM10000 has duplicate image_ids for same lesion)
    df_unique = df.drop_duplicates(subset=["image_id"])
    return df, df_unique, csv_path


# ──────────────────────────────────────────────────────────────────────────────
# PRINT CLASS INFO
# ──────────────────────────────────────────────────────────────────────────────
def print_class_info(df_unique):
    print("\n" + "═" * 72)
    print("  HAM10000 — CLASS REFERENCE GUIDE")
    print("═" * 72)

    counts = df_unique["dx"].value_counts()
    total  = counts.sum()

    sorted_labels = sorted(CLASS_INFO.keys(),
                           key=lambda k: RISK_ORDER[CLASS_INFO[k]["risk"]])

    rows = []
    for label in sorted_labels:
        info  = CLASS_INFO[label]
        count = counts.get(label, 0)
        pct   = count / total * 100
        rows.append((label, info["full_name"], info["risk"], count, pct))

    print(f"\n{'Label':<8} {'Full Name':<45} {'Risk':<15} {'Count':>6} {'%':>6}")
    print("-" * 84)
    for label, name, risk, count, pct in rows:
        risk_sym = "🔴" if risk == "MALIGNANT" else ("🟡" if risk == "PRE-MALIGNANT" else "🟢")
        print(f"{label:<8} {name:<45} {risk_sym} {risk:<12} {count:>6} {pct:>5.1f}%")

    print(f"\n{'TOTAL':<54} {total:>6}  100.0%")

    max_count = counts.max()
    min_count = counts.min()
    imbalance = max_count / min_count
    print(f"\nImbalance ratio  : {imbalance:.1f}× (largest class / smallest class)")
    print(f"Largest class    : nv  ({max_count:,} images) — {max_count/total*100:.1f}% of dataset")
    print(f"Smallest class   : df  ({min_count:,} images) — {min_count/total*100:.1f}% of dataset")
    print(f"\n⚠  This is a HEAVILY IMBALANCED dataset.")
    print("   Without class weights, a model that always predicts 'nv'")
    print(f"   would achieve {max_count/total*100:.1f}% accuracy — while being completely useless.")
    print("   train_model.py corrects this using compute_class_weight('balanced').\n")

    print("─" * 72)
    print("CLASS DESCRIPTIONS")
    print("─" * 72)
    for label in sorted_labels:
        info = CLASS_INFO[label]
        print(f"\n[{label.upper()}]  {info['full_name']}  ({info['risk']})")
        wrapped = textwrap.fill(info["description"], width=68, initial_indent="  ",
                                subsequent_indent="  ")
        print(wrapped)

    return rows, counts, total


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 1: CLASS DISTRIBUTION BAR CHART
# ──────────────────────────────────────────────────────────────────────────────
def plot_distribution(counts, total):
    labels   = list(counts.index)
    values   = list(counts.values)
    colors   = [CLASS_INFO[l]["risk_color"] if l in CLASS_INFO else "#58a6ff" for l in labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(DARK_FIG)
    ax.set_facecolor(DARK_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(DARK_BORDER)
    ax.tick_params(colors=TEXT_COLOR)

    bars = ax.bar(labels, values, color=colors, edgecolor=DARK_BORDER, linewidth=0.8)

    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 30,
                f"{val:,}\n({pct:.1f}%)",
                ha="center", va="bottom",
                color=TEXT_COLOR, fontsize=9)

    # Horizontal reference line at mean
    mean_count = total / len(labels)
    ax.axhline(mean_count, color="#8b949e", linestyle="--", linewidth=1,
               label=f"Mean: {mean_count:.0f}")

    ax.set_title("HAM10000 — Image Count per Class", color=TITLE_COLOR, fontsize=15, pad=15)
    ax.set_xlabel("Class Label", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Number of Images", color=TEXT_COLOR, fontsize=11)
    ax.set_xticklabels([f"{l}\n{CLASS_INFO[l]['full_name'].split('(')[0].strip()}"
                        if l in CLASS_INFO else l for l in labels],
                       fontsize=8, color=TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)

    legend = ax.legend(facecolor=DARK_FIG, edgecolor=DARK_BORDER,
                       labelcolor=TEXT_COLOR, fontsize=9)

    # Risk legend
    from matplotlib.patches import Patch
    risk_patches = [
        Patch(color="#f85149", label="Malignant"),
        Patch(color="#d29922", label="Pre-malignant"),
        Patch(color="#3fb950", label="Benign"),
    ]
    ax.legend(handles=risk_patches, loc="upper right",
              facecolor=DARK_FIG, edgecolor=DARK_BORDER, labelcolor=TEXT_COLOR)

    plt.tight_layout()
    out = os.path.join(REPORT_DIR, "01_class_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_FIG)
    plt.close()
    print(f"[SAVED] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 2: SAMPLE IMAGES PER CLASS
# ──────────────────────────────────────────────────────────────────────────────
def plot_sample_images(df_unique):
    n_classes = len(CLASS_INFO)
    n_samples = 3
    fig = plt.figure(figsize=(n_samples * 3.5, n_classes * 3.2))
    fig.patch.set_facecolor(DARK_FIG)
    fig.suptitle("HAM10000 — Sample Images per Class (3 each)",
                 color=TITLE_COLOR, fontsize=14, y=0.995)

    sorted_labels = sorted(CLASS_INFO.keys(),
                           key=lambda k: RISK_ORDER[CLASS_INFO[k]["risk"]])

    for row_idx, label in enumerate(sorted_labels):
        subset = df_unique[
            (df_unique["dx"] == label) & (df_unique["image_path"].notna())
        ].sample(min(n_samples, len(df_unique[df_unique["dx"] == label])),
                 random_state=42)

        for col_idx, (_, record) in enumerate(subset.iterrows()):
            ax = fig.add_subplot(n_classes, n_samples, row_idx * n_samples + col_idx + 1)
            ax.set_facecolor(DARK_BG)

            try:
                img = Image.open(record["image_path"]).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "No image", ha="center", va="center",
                        color=TEXT_COLOR, transform=ax.transAxes)

            if col_idx == 0:
                info = CLASS_INFO[label]
                risk_col = info["risk_color"]
                ax.set_ylabel(
                    f"[{label}]\n{info['full_name'].split('(')[0].strip()}",
                    color=risk_col, fontsize=8, rotation=0,
                    labelpad=90, va="center",
                )

            age  = record.get("age", "?")
            sex  = str(record.get("sex", "?"))[:1].upper()
            loc  = str(record.get("localization", "?"))[:12]
            dx_t = str(record.get("dx_type", "?"))[:4]
            ax.set_title(f"Age:{age} {sex} | {loc}\n[{dx_t}]",
                         color=TEXT_COLOR, fontsize=7, pad=3)
            ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.995])
    out = os.path.join(REPORT_DIR, "03_sample_images.png")
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=DARK_FIG)
    plt.close()
    print(f"[SAVED] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 3: METADATA DISTRIBUTIONS
# ──────────────────────────────────────────────────────────────────────────────
def plot_metadata(df_unique):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(DARK_FIG)
    fig.suptitle("HAM10000 — Patient & Lesion Metadata Distributions",
                 color=TITLE_COLOR, fontsize=14)

    for ax in axes:
        ax.set_facecolor(DARK_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(DARK_BORDER)
        ax.tick_params(colors=TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TITLE_COLOR)

    # ── Age distribution ──────────────────────────────────────────────────────
    ages = df_unique["age"].dropna()
    axes[0].hist(ages, bins=20, color="#58a6ff", edgecolor=DARK_BORDER, linewidth=0.5)
    axes[0].axvline(ages.mean(), color="#f85149", linestyle="--",
                    label=f"Mean: {ages.mean():.1f}")
    axes[0].axvline(ages.median(), color="#d29922", linestyle="--",
                    label=f"Median: {ages.median():.1f}")
    axes[0].set_title("Patient Age Distribution")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")
    axes[0].legend(facecolor=DARK_FIG, edgecolor=DARK_BORDER, labelcolor=TEXT_COLOR)

    # ── Sex distribution ──────────────────────────────────────────────────────
    sex_counts = df_unique["sex"].value_counts()
    colors_sex = ["#58a6ff", "#f778ba", "#8b949e"]
    axes[1].bar(sex_counts.index, sex_counts.values,
                color=colors_sex[:len(sex_counts)], edgecolor=DARK_BORDER)
    for i, (cat, val) in enumerate(sex_counts.items()):
        axes[1].text(i, val + 15, str(val), ha="center", color=TEXT_COLOR, fontsize=10)
    axes[1].set_title("Patient Sex Distribution")
    axes[1].set_xlabel("Sex")
    axes[1].set_ylabel("Count")

    # ── Localization distribution ─────────────────────────────────────────────
    loc_counts = df_unique["localization"].value_counts().head(10)
    y_pos = range(len(loc_counts))
    axes[2].barh(list(y_pos), loc_counts.values,
                 color="#3fb950", edgecolor=DARK_BORDER, linewidth=0.5)
    axes[2].set_yticks(list(y_pos))
    axes[2].set_yticklabels(loc_counts.index, color=TEXT_COLOR, fontsize=9)
    axes[2].set_title("Top 10 Lesion Locations")
    axes[2].set_xlabel("Count")
    for i, val in enumerate(loc_counts.values):
        axes[2].text(val + 10, i, str(val), va="center", color=TEXT_COLOR, fontsize=8)

    plt.tight_layout()
    out = os.path.join(REPORT_DIR, "04_metadata_distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_FIG)
    plt.close()
    print(f"[SAVED] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 4: IMBALANCE HEATMAP (class weight preview)
# ──────────────────────────────────────────────────────────────────────────────
def plot_imbalance(counts, total):
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    labels      = list(counts.index)
    values      = list(counts.values)
    label_ints  = list(range(len(labels)))
    cw          = compute_class_weight("balanced", classes=np.array(label_ints),
                                       y=[i for i, c in zip(label_ints, values) for _ in range(c)])
    weights     = dict(zip(labels, cw))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK_FIG)
    for ax in axes:
        ax.set_facecolor(DARK_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(DARK_BORDER)
        ax.tick_params(colors=TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TITLE_COLOR)

    # Left: image count vs effective count after weighting
    eff_counts  = [v * weights[l] for l, v in zip(labels, values)]
    x           = np.arange(len(labels))
    width       = 0.35
    bars1 = axes[0].bar(x - width/2, values,    width, label="Raw count",      color="#58a6ff", alpha=0.8)
    bars2 = axes[0].bar(x + width/2, eff_counts, width, label="Weighted count", color="#3fb950", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, color=TEXT_COLOR)
    axes[0].set_title("Raw vs Weighted Sample Counts")
    axes[0].set_ylabel("Count")
    axes[0].legend(facecolor=DARK_FIG, edgecolor=DARK_BORDER, labelcolor=TEXT_COLOR)

    # Right: class weights bar chart
    w_vals = [weights[l] for l in labels]
    colors_w = ["#f85149" if w > 5 else ("#d29922" if w > 1.5 else "#3fb950") for w in w_vals]
    axes[1].bar(labels, w_vals, color=colors_w, edgecolor=DARK_BORDER)
    axes[1].axhline(1.0, color="#8b949e", linestyle="--", linewidth=1, label="Weight = 1")
    for i, (l, w) in enumerate(zip(labels, w_vals)):
        axes[1].text(i, w + 0.1, f"{w:.2f}x", ha="center", color=TEXT_COLOR, fontsize=9)
    axes[1].set_title("Computed Class Weights\n(higher = rarer class gets more importance)")
    axes[1].set_ylabel("Weight")
    axes[1].legend(facecolor=DARK_FIG, edgecolor=DARK_BORDER, labelcolor=TEXT_COLOR)

    plt.tight_layout()
    out = os.path.join(REPORT_DIR, "05_class_weights.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_FIG)
    plt.close()
    print(f"[SAVED] {out}")
    return weights


# ──────────────────────────────────────────────────────────────────────────────
# TEXT SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
def save_text_summary(df, df_unique, counts, total, weights, csv_path):
    lines = []
    lines.append("HAM10000 DATASET SUMMARY REPORT")
    lines.append("=" * 70)
    rel_csv = os.path.relpath(csv_path, _SCRIPT_DIR)
    lines.append(f"CSV file          : {rel_csv}  (relative to project root)")
    lines.append(f"Total CSV rows    : {len(df):,}")
    lines.append(f"Unique images     : {len(df_unique):,}  ({len(df)-len(df_unique)} duplicates removed)")
    lines.append(f"Images on disk    : {df_unique['image_path'].notna().sum():,}")
    lines.append(f"Images missing    : {df_unique['image_path'].isna().sum():,}")
    lines.append("")
    lines.append(f"{'Label':<8} {'Full Name':<44} {'Risk':<15} {'Count':>6} {'%':>6}  {'Weight':>7}")
    lines.append("-" * 90)
    sorted_labels = sorted(CLASS_INFO.keys(), key=lambda k: RISK_ORDER[CLASS_INFO[k]["risk"]])
    for label in sorted_labels:
        info  = CLASS_INFO[label]
        count = counts.get(label, 0)
        pct   = count / total * 100
        w     = weights.get(label, 1.0)
        lines.append(f"{label:<8} {info['full_name']:<44} {info['risk']:<15} {count:>6} {pct:>5.1f}%  {w:>7.3f}x")
    lines.append(f"\n{'TOTAL':<68} {total:>6}  100.0%")
    lines.append("")
    lines.append("AGE STATISTICS")
    lines.append("-" * 40)
    ages = df_unique["age"].dropna()
    lines.append(f"Mean age    : {ages.mean():.1f}")
    lines.append(f"Median age  : {ages.median():.1f}")
    lines.append(f"Std dev     : {ages.std():.1f}")
    lines.append(f"Range       : {ages.min():.0f} – {ages.max():.0f}")
    lines.append(f"Missing age : {df_unique['age'].isna().sum()}")
    lines.append("")
    lines.append("SEX DISTRIBUTION")
    lines.append("-" * 40)
    for sex, cnt in df_unique["sex"].value_counts().items():
        lines.append(f"  {sex:<12}: {cnt:,}  ({cnt/len(df_unique)*100:.1f}%)")
    lines.append("")
    lines.append("TOP 10 LESION LOCATIONS")
    lines.append("-" * 40)
    for loc, cnt in df_unique["localization"].value_counts().head(10).items():
        lines.append(f"  {loc:<20}: {cnt:,}  ({cnt/len(df_unique)*100:.1f}%)")
    lines.append("")
    lines.append("DIAGNOSIS TYPE (how was ground truth established?)")
    lines.append("-" * 40)
    dx_type_map = {
        "histo"  : "Histopathology (biopsy) — GOLD STANDARD",
        "follow_up": "Follow-up (lesion monitored over time)",
        "consensus": "Expert consensus",
        "confocal" : "Reflectance confocal microscopy",
    }
    for dt, cnt in df_unique["dx_type"].value_counts().items():
        desc = dx_type_map.get(dt, dt)
        lines.append(f"  {dt:<12}: {cnt:,}  — {desc}")
    lines.append("")
    lines.append("MODEL TRAINING IMPLICATIONS")
    lines.append("-" * 40)
    lines.append(f"  Imbalance ratio : {counts.max()/counts.min():.1f}x")
    lines.append("  → Class weights are REQUIRED (already in train_model.py)")
    lines.append("  → EarlyStopping prevents overfitting on dominant 'nv' class")
    lines.append("  → Augmentation on rare classes (vasc, df) helps generalisation")
    lines.append("")
    lines.append("NORMAL SKIN NOTE")
    lines.append("-" * 40)
    lines.append("  HAM10000 contains no 'normal skin' class.")
    lines.append("  Use a confidence threshold in model.py instead:")
    lines.append("  if max(softmax_probabilities) < 0.55 → predict 'Normal/Healthy Skin'")

    out = os.path.join(REPORT_DIR, "dataset_summary.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[SAVED] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"\n[INFO] Loading dataset…")
    df, df_unique, csv_path = load_dataset()

    print(f"[INFO] CSV rows    : {len(df):,}")
    print(f"[INFO] Unique imgs : {len(df_unique):,}")
    print(f"[INFO] On disk     : {df_unique['image_path'].notna().sum():,}\n")

    rows, counts, total = print_class_info(df_unique)

    print("\n[INFO] Generating charts…")
    plot_distribution(counts, total)
    plot_sample_images(df_unique)
    plot_metadata(df_unique)
    weights = plot_imbalance(counts, total)
    save_text_summary(df, df_unique, counts, total, weights, csv_path)

    print(f"\n✅ All outputs saved to: {REPORT_DIR}")
    print("   Open the PNG files to visualise your dataset.")


if __name__ == "__main__":
    main()