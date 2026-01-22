from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from flask_cors import CORS

# ─── Imports for XLM-RoBERTa ────────────────────────────────────
import os
import torch
import traceback
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
CORS(app)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ─── OLD MODEL LOADING ──────────────────────────────────────────
tfidf = pickle.load(open("models/vec.pkl", "rb"))
model = pickle.load(open("models/mod.pkl", "rb"))

stop = set(stopwords.words('english'))
stemmer = nltk.stem.SnowballStemmer('english')

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
THRESHOLD = 50.0

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_stopwords(text):
    words = [w for w in text.split() if w not in stop]
    return ' '.join(words)

def stem_text(text):
    words = text.split()
    stem_words = [stemmer.stem(word) for word in words]
    return ' '.join(stem_words)

def preprocess_old(comment):
    comment = str(comment).lower()
    return stem_text(remove_stopwords(remove_punctuation(comment))).strip()

def risk_label(overall_pct):
    if overall_pct < 20:
        return "Low"
    if overall_pct < 50:
        return "Medium"
    if overall_pct < 80:
        return "High"
    return "Critical"

# ─── NEW: Load XLM-RoBERTa from local folder ────────────────────
# ─── Load XLM-RoBERTa from Hugging Face Hub ───────────────────────────────
HF_REPO = "durgeshptl/toxic-xlmr-v3"  # your repo

model_new = None
tokenizer_new = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    print("\n" + "="*70)
    print("Loading XLM-RoBERTa from Hugging Face:", HF_REPO)

    tokenizer_new = AutoTokenizer.from_pretrained(
        HF_REPO,
        use_fast=True
    )

    model_new = AutoModelForSequenceClassification.from_pretrained(
        HF_REPO,
        problem_type="multi_label_classification"
    )

    model_new.to(device)
    model_new.eval()

    print("SUCCESS: Loaded from Hugging Face on", device.type)
    print("="*70 + "\n")

except Exception as e:
    print("\n" + "="*70)
    print("WARNING: Failed to load XLM-RoBERTa from Hugging Face")
    print("Error:", str(e))
    traceback.print_exc()
    print("Old TF-IDF model still works")
    print("="*70 + "\n")
# ─── PREPROCESSING ──────────────────────────────────────────────
# Old
def preprocess_old(comment):
    comment = str(comment).lower()
    return stem_text(remove_stopwords(remove_punctuation(comment))).strip()

# New (light)
def preprocess_new(comment):
    comment = str(comment)
    comment = re.sub(r"http\S+|www\.\S+", " ", comment)
    comment = re.sub(r"\s+", " ", comment).strip()
    return comment.lower()

# ─── ROUTES ─────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    comment = (data.get("comment") or "").strip()
    if not comment:
        return jsonify({"error": "Empty comment"}), 400

    clean = preprocess_old(comment)
    vec = tfidf.transform([clean])

    probs = model.predict_proba(vec)[0]
    percent = {label: round(float(p) * 100.0, 2) for label, p in zip(LABELS, probs)}

    overall = round(max(percent.values()) if percent else 0.0, 2)
    risk = risk_label(overall)

    return jsonify({
        "percent": percent,
        "overall": overall,
        "risk": risk,
        "threshold": THRESHOLD
    })

@app.route("/predict_new", methods=["POST"])
def predict_new():
    data = request.get_json(silent=True) or {}
    comment = (data.get("comment") or "").strip()
    if not comment:
        return jsonify({"error": "Empty comment"}), 400

    clean_text = preprocess_new(comment)

    enc = tokenizer_new(clean_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model_new(**enc).logits[0].cpu().numpy()

    probs = 1 / (1 + np.exp(-logits))
    percent = {label: round(float(p) * 100.0, 2) for label, p in zip(LABELS, probs)}

    overall = round(max(percent.values()) if percent else 0.0, 2)
    risk = risk_label(overall)

    return jsonify({
        "percent": percent,
        "overall": overall,
        "risk": risk,
        "threshold": THRESHOLD
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)