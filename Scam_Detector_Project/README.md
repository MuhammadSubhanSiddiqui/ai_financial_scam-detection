# Scam_Detector_Project

Project scaffold for deploying the AI Scam Detection system.

Structure:
- `app.py` — Streamlit app (paste your code here).
- `requirements.txt` — Python dependencies for deployment.
- `best_distilbert_model/` — Place the trained DistilBERT model files here.

Secrets:
- Copy `.env.example` to `.env` for local runs.
- Or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` for Streamlit secrets.
- Set `GEMINI_API_KEY` in either place.

Quick start:
1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your model files to `best_distilbert_model/`.
4. Paste your Streamlit code into `app.py` and run:

```bash
streamlit run app.py
```

For GitHub deployment, upload only the code files and the model folder if you want local inference. Do not upload `.env` or `.streamlit/secrets.toml`.
