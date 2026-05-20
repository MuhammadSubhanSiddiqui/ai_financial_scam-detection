"""
helpers.py
Consolidated helper functions for the AI Scam Detection notebook.
This file collects utility functions so the main notebook can stay focused
on the core phases. It intentionally does NOT modify the notebook's main
phase logic — import and use these helpers from the notebook as desired.
"""

import re
import json
from spacy.matcher import Matcher


def get_distilbert_score(text, tokenizer, model, device, max_length=128):
    """Return the probability (%) for class 1 (manipulative) from a DistilBERT classifier.

    Parameters:
    - text: str
    - tokenizer: transformers tokenizer
    - model: transformers model
    - device: 'cpu' or 'cuda'
    - max_length: int
    """
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with __import__("torch").no_grad():
        outputs = model(**inputs)
        probabilities = __import__("torch").softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    return float(probabilities[1] * 100)


def extract_suspicious_claims(text, nlp):
    """Extract simple suspicious claim patterns from `text` using a spaCy `nlp` pipeline.

    Returns a list of unique descriptive claim strings.
    """
    doc = nlp(text.lower())
    matcher = Matcher(nlp.vocab)
    claims = []

    pattern_guarantee = [
        {"LEMMA": {"IN": ["guarantee", "guaranteed", "risk-free", "absolute", "100%", "assured", "passive"]}},
        {"OP": "*"},
        {"LEMMA": {"IN": ["return", "profit", "gain", "money", "income", "payout", "yield", "cash"]}}
    ]
    pattern_urgency = [
        {"LEMMA": {"IN": ["dm", "text", "click", "join", "buy", "claim", "invest", "deposit"]}},
        {"LOWER": {"IN": ["me", "now", "here", "fast", "today", "urgently"]}}
    ]

    matcher.add("PROMISE", [pattern_guarantee])
    matcher.add("URGENCY", [pattern_urgency])

    for match_id, start, end in matcher(doc):
        string_id = nlp.vocab.strings[match_id]
        if string_id == "PROMISE":
            claims.append(f"Unrealistic Return Promise: '{doc[start:end].text}'")
        elif string_id == "URGENCY":
            claims.append(f"High-Pressure Action: '{doc[start:end].text}'")

    match_percent = re.search(r'\b\d{1,4}%\s*|\b\d{1,4}\s?percent\b', text, re.IGNORECASE)
    if match_percent:
        claims.append(f"Specific Yield Target: '{match_percent.group().strip()}'")

    match_cash = re.search(r'\$\d{1,7}', text)
    if match_cash:
        claims.append(f"Cash Bait Target: '{match_cash.group()}'")

    for word in ["free money", "crypto secrets", "get rich", "passive income", "no risk"]:
        if word in text.lower():
            claims.append(f"High-Risk Buzzword: '{word.title()}'")

    # Preserve order but remove duplicates
    return list(dict.fromkeys(claims))


def verify_with_gemini(text, claims, gemini_model):
    """Call the Gemini model wrapper to verify claims.

    Returns a dict with keys `gemini_risk_score` and `explanation`.
    This function keeps the same fallback behavior as the notebook.
    """
    prompt = f"""
    You are a financial fraud investigator. Analyze this text and its extracted claims.
    Text: "{text}"
    Claims: {claims}
    Respond STRICTLY in JSON format without markdown:
    {{"gemini_risk_score": <integer 0-100>, "explanation": "<One clear sentence explaining why it is safe or fraudulent>"}}
    """
    try:
        response = gemini_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text.strip())
    except Exception:
        return {"gemini_risk_score": 50 if claims else 0, "explanation": "API verification failed."}


__all__ = [
    "get_distilbert_score",
    "extract_suspicious_claims",
    "verify_with_gemini",
]
