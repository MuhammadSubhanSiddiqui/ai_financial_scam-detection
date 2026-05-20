# AI Scam Detection (project README)

This folder contains the primary project README for the AI Scam Detection prototype.

## Quick Summary

Prototype to detect possible financial influencer scams using a fine-tuned DistilBERT model. Includes data, helpers, and a small demo app for inference.

## Quick start

- Create a virtualenv and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r Scam_Detector_Project\requirements.txt
```

- Run the demo app (example):

```powershell
python Scam_Detector_Project\app.py
```

## Where things live

- `datasets/` — source and cleaned datasets.
- `Scam_Detector_Project/best_distilbert_model/` — trained model artifacts.
- `Scam_Detector_Project/app.py` — demo inference app.
- `helpers.py` and `Scam_Detector_Project/helpers.py` — utility functions used by notebooks and scripts.

## Screenshots

Place images in `assets/screenshots/` named `overview.png`, `training_metrics.png`, `app_demo.png`.

## Notes

- GitHub does not apply custom fonts or external CSS in README files; use images for styled content.
- If you want a shorter or more detailed version, tell me which sections to keep or remove.

## Data & Model (numeric summary)

- Master cleaned dataset: `datasets/master_cleaned_dataset.csv` — **160,852** samples
- Number of dataset files detected under `datasets/`: **11**
- Model artifact: `Scam_Detector_Project/best_distilbert_model/model.safetensors` — **~255 MB** (267,832,560 bytes)
- Model architecture: `DistilBertForSequenceClassification` (DistilBERT)
  - hidden dimension: **768**
  - transformer layers: **6**
  - attention heads: **12**
  - vocab size: **30522**

These numbers were collected from the repository files; if you update datasets or replace the model file, I can refresh the stats.

## Contact

Open an issue or message the maintainer for reproduction help.
