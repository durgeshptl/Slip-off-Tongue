from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords

from flask_cors import CORS



app = Flask(__name__)
CORS(app)  # ← this line allows requests from chrome-extension:// and other origins

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load models from the models/ folder
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

def preprocess(comment):
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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    comment = (data.get("comment") or "").strip()
    if not comment:
        return jsonify({"error": "Empty comment"}), 400

    clean = preprocess(comment)
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)