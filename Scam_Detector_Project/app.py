from __future__ import annotations

import json
import os
import re
from pathlib import Path

import google.generativeai as genai
import spacy
import streamlit as st
import torch
from spacy.matcher import Matcher
from transformers import AutoModelForSequenceClassification, AutoTokenizer


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "best_distilbert_model"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def load_local_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


load_local_env_file(APP_DIR / ".env")
load_local_env_file(APP_DIR / ".env.example")


st.set_page_config(page_title="AI Scam Detector", page_icon="Shield", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(13, 110, 253, 0.16), transparent 32%),
                radial-gradient(circle at top right, rgba(25, 135, 84, 0.12), transparent 28%),
                linear-gradient(180deg, #07111f 0%, #0c1728 45%, #111827 100%);
            color: #eef2ff;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 24px;
            padding: 1.5rem 1.5rem 1.1rem 1.5rem;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.18);
        }
        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
            line-height: 1.1;
            color: #f8fafc;
        }
        .hero p {
            margin: 0.45rem 0 0;
            color: #cbd5e1;
            max-width: 72ch;
        }
        .card {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 18px;
            padding: 1rem 1rem 0.85rem 1rem;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.12);
        }
        .muted {
            color: #94a3b8;
            font-size: 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_resources(model_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

    model_path = Path(model_dir)
    if model_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device)
        model.eval()
    else:
        tokenizer = None
        model = None
    return nlp, tokenizer, model, device


def get_distilbert_score(text: str, tokenizer, model, device: str) -> float:
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    if len(probabilities) > 1:
        return float(probabilities[1] * 100)
    return float(probabilities[0] * 100)


def get_fallback_score(text: str, claims: list[str]) -> float:
    lower_text = text.lower()
    score = 10.0

    score += min(len(claims) * 14.0, 42.0)

    if re.search(r"\b\d{1,4}%\b|\b\d{1,4}\s?percent\b", lower_text):
        score += 18.0
    if re.search(r"\$\d{2,}", lower_text):
        score += 10.0

    trigger_terms = ["guaranteed", "urgent", "dm me", "click now", "risk-free", "no risk", "passive income", "get rich"]
    score += sum(6.0 for term in trigger_terms if term in lower_text)

    return float(max(0.0, min(100.0, score)))


def extract_suspicious_claims(text: str, nlp) -> list[str]:
    doc = nlp(text.lower())
    matcher = Matcher(nlp.vocab)
    claims: list[str] = []

    pattern_promise = [
        {"LOWER": {"IN": ["guarantee", "guaranteed", "risk-free", "absolute", "assured", "passive"]}},
        {"OP": "*"},
        {"LOWER": {"IN": ["return", "profit", "gain", "money", "income", "payout", "yield", "cash"]}},
    ]
    pattern_urgency = [
        {"LOWER": {"IN": ["dm", "text", "click", "join", "buy", "claim", "invest", "deposit"]}},
        {"LOWER": {"IN": ["me", "now", "here", "fast", "today", "urgently"]}},
    ]

    matcher.add("PROMISE", [pattern_promise])
    matcher.add("URGENCY", [pattern_urgency])

    for match_id, start, end in matcher(doc):
        label = nlp.vocab.strings[match_id]
        if label == "PROMISE":
            claims.append(f"Unrealistic return promise: '{doc[start:end].text}'")
        elif label == "URGENCY":
            claims.append(f"High-pressure action: '{doc[start:end].text}'")

    percent_match = re.search(r"\b\d{1,4}%\s*|\b\d{1,4}\s?percent\b", text, re.IGNORECASE)
    if percent_match:
        claims.append(f"Specific yield target: '{percent_match.group().strip()}'")

    cash_match = re.search(r"\$\d{1,7}", text)
    if cash_match:
        claims.append(f"Cash bait target: '{cash_match.group()}'")

    for phrase in ["free money", "crypto secrets", "get rich", "passive income", "no risk"]:
        if phrase in text.lower():
            claims.append(f"High-risk buzzword: '{phrase}'")

    return list(dict.fromkeys(claims))


def build_gemini_client(api_key: str | None):
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(DEFAULT_GEMINI_MODEL)
    except Exception:
        return None


def resolve_gemini_api_key(sidebar_value: str) -> str | None:
    if sidebar_value.strip():
        return sidebar_value.strip()

    try:
        secrets_key = st.secrets.get("GEMINI_API_KEY", "")
        if isinstance(secrets_key, str) and secrets_key.strip():
            return secrets_key.strip()
    except Exception:
        pass

    env_key = os.environ.get("GEMINI_API_KEY", "")
    if env_key.strip():
        return env_key.strip()

    return None


def verify_with_gemini(text: str, claims: list[str], gemini_model):
    if not gemini_model:
        return {"gemini_risk_score": 50 if claims else 0, "explanation": "Gemini verification is disabled until an API key is provided."}

    prompt = (
        "You are a financial fraud investigator. Analyze this text and its extracted claims.\n"
        f"Text: {text!r}\n"
        f"Claims: {claims}\n"
        "Respond STRICTLY in JSON format without markdown:\n"
        '{"gemini_risk_score": <integer 0-100>, "explanation": "<One clear sentence explaining why it is safe or fraudulent>"}'
    )

    try:
        response = gemini_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        parsed = json.loads(response.text.strip())
        score = int(parsed.get("gemini_risk_score", 50))
        explanation = str(parsed.get("explanation", "No explanation provided."))
        return {"gemini_risk_score": max(0, min(100, score)), "explanation": explanation}
    except Exception:
        return {"gemini_risk_score": 50 if claims else 0, "explanation": "Gemini verification failed for this input."}


def score_to_label(score: float) -> str:
    if score <= 30:
        return "Low risk"
    if score <= 70:
        return "Medium risk"
    return "High risk"


st.markdown(
    """
    <div class="hero">
        <h1>AI Scam Detector</h1>
        <p>Analyze suspicious financial captions with a local DistilBERT model, rule-based evidence extraction, and Gemini verification.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input("Gemini API key", value="", type="password", help="Leave blank to use the environment or Streamlit secrets.")
    model_path_input = st.text_input("Model folder", value=str(MODEL_DIR))
    st.caption("The local model is loaded from the folder above.")

    st.markdown("### Examples")
    example_one = "Guaranteed 20% monthly returns on crypto trading. DM me now!"
    example_two = "Learn budgeting tips and long-term investing basics with transparent risk."
    if st.button("Load high-risk example"):
        st.session_state.user_input = example_one
    if st.button("Load safe example"):
        st.session_state.user_input = example_two


try:
    nlp, tokenizer, model, device = load_resources(model_path_input)
    if tokenizer and model:
        model_status = f"Loaded local model on {device}."
    else:
        model_status = "Local model folder not found. Using fallback scoring so the app can still run."
except Exception as exc:
    nlp = tokenizer = model = None
    device = "cpu"
    model_status = f"Model load failed: {exc}"

gemini_key = resolve_gemini_api_key(api_key_input)
gemini_model = build_gemini_client(gemini_key)

status_col1, status_col2 = st.columns(2)
with status_col1:
    st.markdown('<div class="card"><div class="muted">Model status</div><strong>{}</strong></div>'.format(model_status), unsafe_allow_html=True)
with status_col2:
    gemini_state = "Enabled" if gemini_model else "Disabled"
    st.markdown('<div class="card"><div class="muted">Gemini status</div><strong>{}</strong></div>'.format(gemini_state), unsafe_allow_html=True)

st.write("")

st.markdown("### Input")
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

user_input = st.text_area(
    "Social media text",
    height=150,
    value=st.session_state.user_input,
    placeholder="Example: Guaranteed 20% monthly returns on crypto trading. DM me now!",
)

analyze_clicked = st.button("Analyze text", type="primary")

if analyze_clicked:
    if not user_input.strip():
        st.error("Please enter some text to analyze.")
    elif not all([nlp, tokenizer, model]):
        st.warning(model_status)
        extracted_claims = extract_suspicious_claims(user_input, nlp)
        nlp_score = get_fallback_score(user_input, extracted_claims)
        gemini_data = verify_with_gemini(user_input, extracted_claims, gemini_model)
        gemini_score = float(gemini_data.get("gemini_risk_score", 0))
        final_score = float((0.70 * nlp_score) + (0.30 * gemini_score))

        st.markdown("### Results")
        metric_cols = st.columns(4)
        metric_cols[0].metric("NLP intent", f"{nlp_score:.1f}%")
        metric_cols[1].metric("Claims extracted", len(extracted_claims))
        metric_cols[2].metric("Gemini risk", f"{gemini_score:.0f}%")
        metric_cols[3].metric("Final score", f"{final_score:.1f}/100")

        st.progress(int(max(0, min(100, final_score))))
        st.subheader(score_to_label(final_score))
        st.info(gemini_data.get("explanation", "No reasoning provided."))

        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("#### Extracted evidence")
            if extracted_claims:
                for claim in extracted_claims:
                    st.warning(claim)
            else:
                st.success("No obvious scam patterns were extracted.")

        with right:
            st.markdown("#### Input preview")
            st.write(user_input)

        if final_score <= 30:
            st.success(f"Low risk: {final_score:.1f}/100")
        elif final_score <= 70:
            st.warning(f"Medium risk: {final_score:.1f}/100")
        else:
            st.error(f"High risk: {final_score:.1f}/100")
    else:
        with st.spinner("Analyzing linguistic patterns and verifying claims..."):
            nlp_score = get_distilbert_score(user_input, tokenizer, model, device)
            extracted_claims = extract_suspicious_claims(user_input, nlp)
            gemini_data = verify_with_gemini(user_input, extracted_claims, gemini_model)
            gemini_score = float(gemini_data.get("gemini_risk_score", 0))
            final_score = float((0.70 * nlp_score) + (0.30 * gemini_score))

        st.markdown("### Results")
        metric_cols = st.columns(4)
        metric_cols[0].metric("NLP intent", f"{nlp_score:.1f}%")
        metric_cols[1].metric("Claims extracted", len(extracted_claims))
        metric_cols[2].metric("Gemini risk", f"{gemini_score:.0f}%")
        metric_cols[3].metric("Final score", f"{final_score:.1f}/100")

        st.progress(int(max(0, min(100, final_score))))
        st.subheader(score_to_label(final_score))
        st.info(gemini_data.get("explanation", "No reasoning provided."))

        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("#### Extracted evidence")
            if extracted_claims:
                for claim in extracted_claims:
                    st.warning(claim)
            else:
                st.success("No obvious scam patterns were extracted.")

        with right:
            st.markdown("#### Input preview")
            st.write(user_input)

        if final_score <= 30:
            st.success(f"Low risk: {final_score:.1f}/100")
        elif final_score <= 70:
            st.warning(f"Medium risk: {final_score:.1f}/100")
        else:
            st.error(f"High risk: {final_score:.1f}/100")
