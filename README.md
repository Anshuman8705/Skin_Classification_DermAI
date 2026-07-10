# 🔬 DermAI Monitor

**AI-assisted skin lesion classification, explainability, and longitudinal monitoring — built on EfficientNetB3, trained on a merged HAM10000 + ISIC 2019 dermoscopic dataset.**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13--2.15-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)

> ⚠️ **Not a diagnostic device.** DermAI Monitor is a decision-support and triage-aid prototype built for an academic AI/ML internship. It is **not** validated for clinical use and must not be used as a substitute for evaluation by a qualified dermatologist.

---

## Overview

DermAI Monitor is an end-to-end system for automatic classification of dermoscopic skin lesion images into seven diagnostic categories, paired with tools that make the prediction *usable* rather than just a label:

- **Grad-CAM** heatmaps showing which region of the image drove the prediction
- An independent **ABCD dermoscopic scoring rule** (Asymmetry, Border, Colour, Diameter) as a classical cross-check alongside the CNN output
- A **patient history database** for tracking a lesion across repeated visits
- **Automated PDF report generation** for each scan

The classifier is an **EfficientNetB3** backbone fine-tuned with a two-phase transfer-learning strategy, focal loss, class weighting, and targeted oversampling to handle a ~58× class imbalance between the most and least common lesion types.

Full methodology and results are written up in the accompanying IEEE-format paper (`docs/DermAI_Monitor_Paper.pdf`).

---

## Key Features

- 🧠 7-class dermoscopic classification: `nv`, `mel`, `bkl`, `bcc`, `akiec`, `vasc`, `df`
- 🩹 Confidence-threshold fallback to a "Normal / Healthy Skin" label when the model is unsure (see [Limitations](#limitations))
- 🔥 Grad-CAM visual explainability overlay
- 📏 ABCD Total Dermoscopy Score (TDS) as a rule-based second opinion
- 📁 SQLite-backed patient records with longitudinal progress tracking
- 📄 One-click PDF report export (ReportLab)
- 🖥️ Multi-page Streamlit interface — Home, Scan, Progress, History, About

---

## Results

Final model, evaluated on a held-out test set of 2,343 images from the merged 15,615-image HAM10000 + ISIC 2019 dataset:

| Metric | Score |
|---|---|
| Accuracy | **68.97%** |
| Weighted F1 | 0.6926 |
| Macro F1 | 0.5426 |
| Macro AUC (one-vs-rest) | **0.9143** |

**Per-class breakdown**

| Class | Diagnosis | Type | Precision | Recall | F1 | ROC-AUC | Support |
|---|---|---|---|---|---|---|---|
| `nv` | Melanocytic Nevus (mole) | Benign | 0.87 | 0.82 | 0.84 | 0.944 | 1006 |
| `mel` | Melanoma | Malignant | 0.67 | 0.69 | 0.68 | 0.877 | 679 |
| `bkl` | Benign Keratosis | Benign | 0.49 | 0.51 | 0.50 | 0.835 | 394 |
| `bcc` | Basal Cell Carcinoma | Malignant | 0.41 | 0.57 | 0.48 | 0.950 | 77 |
| `akiec` | Actinic Keratosis | Pre-malignant | 0.49 | 0.40 | 0.44 | 0.927 | 113 |
| `vasc` | Vascular Lesion | Benign | 0.52 | 0.42 | 0.46 | 0.920 | 38 |
| `df` | Dermatofibroma | Benign | 0.35 | 0.47 | 0.40 | 0.947 | 36 |

Per-class ROC-AUC stays high (0.83–0.95) even where raw accuracy is lower, meaning the model ranks classes reliably — most errors are between visually similar categories (e.g. `bkl` vs. `mel`/`nv`) rather than random. See `training_report/` (regenerate locally — not tracked in git) for the confusion matrix and ROC curves.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Model | TensorFlow / Keras, EfficientNetB3 (ImageNet-pretrained) |
| Training | Google Colab (GPU), Focal Loss, class weighting, oversampling |
| App | Streamlit |
| Explainability | Grad-CAM |
| Storage | SQLite |
| Reporting | ReportLab |
| Data/CV | pandas, NumPy, OpenCV, scikit-learn, Pillow, SciPy |

---

## Project Structure

```
dermai-monitor/
├── app.py                     # Streamlit entry point
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── pages/
│   ├── home.py
│   ├── scan.py                 # upload + classify + Grad-CAM + ABCD
│   ├── progress.py             # longitudinal tracking
│   ├── history.py              # patient records
│   └── about.py
├── models/
│   ├── model.py                # inference wrapper + Grad-CAM + ABCD scoring
│   └── skin_model.h5           # ⚠️ NOT in repo — see "Model Weights" below
├── database/
│   └── db.py                   # SQLite layer (WAL mode)
├── utils/
│   ├── pdf_report.py           # PDF report generation
│   └── seed_demo.py            # optional synthetic demo data
├── train_model.py              # two-phase EfficientNetB3 training script
├── prepare_dataset.py          # HAM10000 + ISIC 2019 merge/dedup logic
├── dataset_explorer.py         # generates dataset_report/ (EDA)
├── skin_dataset/
│   └── merged_dataset.csv      # image_id, dx label, path, source
└── notebooks/
    └── DermAI_Training_FIXED.ipynb   # Colab pipeline: fetch → train → checkpoint → download
```

---

## Getting Started

```bash
git clone https://github.com/<your-username>/dermai-monitor.git
cd dermai-monitor

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

### Model weights

`models/skin_model.h5` is **not tracked in this repository** (trained-model binaries don't belong in git history). Without it, the app still runs — `models/model.py` falls back to a heuristic mode automatically. To use the real trained model:

1. Download `skin_model.h5` (and its companion `skin_model_config.json`) from **[add your hosting link here — e.g. a GitHub Release, Hugging Face Hub, or Drive link]**
2. Place both files in `models/`
3. Restart the app

### Reproducing training

The full pipeline — downloading HAM10000 + ISIC 2019 via the Kaggle API, merging/deduplicating, and running the two-phase fine-tune — is documented step-by-step in `notebooks/DermAI_Training_FIXED.ipynb`, designed to run on Google Colab with checkpointing to Drive (useful given Colab's session time limits). `train_model.py` and `prepare_dataset.py` are the underlying scripts it calls.

---

## Dataset & Training

- **HAM10000**: 10,015 dermoscopic images, 7 classes ([Tschandl et al., 2018](https://doi.org/10.1038/sdata.2018.161)) — released for non-commercial research use with attribution; confirm current terms on [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) before any redistribution.
- **ISIC 2019**: additional images added selectively to under-represented classes only ([Codella et al., 2019](https://arxiv.org/abs/1902.03368)).
- Merged, deduplicated dataset: **15,615 images**, imbalance reduced from ~58× to ~28×.
- Split: 10,930 train / 2,342 val / 2,343 test.
- **Model**: EfficientNetB3 backbone → GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.4) → Dense(7, softmax).
- **Phase 1**: backbone frozen, head trained 20 epochs, lr = 1e-3.
- **Phase 2**: last 30 backbone layers unfrozen, fine-tuned 10 epochs, lr = 1e-5, 3-epoch warm-up, LR reduced ×0.3 every 3 epochs on plateau, early stopping (patience 5).
- **Loss**: sparse categorical focal loss (γ = 2.0, α = 0.25) to focus learning on hard/minority-class examples.

---

## Limitations

Carried over honestly from the accompanying paper, rather than hidden:

- **68.97% accuracy / 0.5426 macro F1 is not sufficient for clinical decision-making.** This is a triage/decision-support aid, not a diagnostic tool.
- The confidence-threshold fallback (predictions below a threshold get relabeled "Normal/Healthy Skin") can currently relabel a low-confidence melanoma or basal-cell-carcinoma prediction as normal — this is flagged in the paper as unsafe for clinical support and is an open item, not a solved problem.
- Training/test data comes from curated dermoscopic archives captured under controlled conditions. Real-world phone photos (variable lighting, focus, magnification) may perform worse.
- `df` and `vasc` (the rarest classes) remain the hardest to classify despite oversampling and focal loss.

## Roadmap

- Replace the confidence-cutoff fallback with an uncertainty-aware / dedicated "skin" class approach
- Expand training data with harder examples and additional sources
- Validate against dermatologist-reviewed cases
- Model compression for on-device / low-connectivity deployment

---

## Team

Built under the **IEEE EMBS Pune Chapter AI/ML Internship Programme 2026**, Dept. of Information Technology, Shri Sant Gajanan Maharaj College of Engineering, Shegaon.

- Anshuman Agrawal — model training, system architecture, application development
- Tanushri Khawale — dataset engineering, model validation

Thanks to Dr. Ankita Tidake and the IEEE EMBS Pune Chapter for guidance and support.

*(Edit names/roles above if your team differs from the paper byline.)*

## Citation

If you build on this work, please cite the paper:

```bibtex
@inproceedings{dermaimonitor2026,
  title     = {DermAI Monitor: A Deep Learning Framework for Multi-Class Skin Lesion Classification with Severity Scoring and Longitudinal Monitoring},
  author    = {Agrawal, Anshuman and Khawale, Tanushri},
  year      = {2026},
  note      = {IEEE EMBS Pune Chapter AI/ML Internship Programme}
}
```

## License

Code in this repository is available under the [MIT License](LICENSE) (add a `LICENSE` file — MIT is a common permissive default, but you're free to choose another). This license covers the **code only**. HAM10000 and ISIC 2019 imagery/metadata carry their own separate non-commercial research licenses — check each source before any redistribution or commercial use.

## Disclaimer

DermAI Monitor is a research prototype developed for educational purposes. It has not been evaluated or approved by any regulatory body and must not be used for actual medical diagnosis or to delay seeking care from a licensed healthcare professional.
