from flask import Flask, jsonify, request
import io
import re
import statistics
from docx import Document
import pdfplumber

from predict import get_detailed_prediction, predict_text

app = Flask(__name__, static_folder="frontend", static_url_path="")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()

    if len(text) < 10:
        return jsonify(error="Please provide at least 10 characters."), 400

    try:
        result = get_detailed_prediction(text)
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 500
    except Exception as exc:
        return jsonify(error=f"Prediction failed: {exc}"), 500

    safe_result = {
        "prediction": result.get("prediction"),
        "confidence": float(result.get("confidence", 0)),
        "human_probability": float(result.get("human_probability", 0)),
        "ai_probability": float(result.get("ai_probability", 0)),
        "word_count": int(result.get("word_count", 0)),
        "warning": result.get("warning"),
        "needs_review": bool(result.get("needs_review")),
    }

    return jsonify(safe_result)


@app.post("/api/extract")
def api_extract():
    if "file" not in request.files:
        return jsonify(error="No file uploaded."), 400

    file = request.files["file"]
    filename = (file.filename or "").lower()

    if filename.endswith(".pdf"):
        with pdfplumber.open(file.stream) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n".join(pages).strip()
    elif filename.endswith(".docx"):
        content = io.BytesIO(file.read())
        doc = Document(content)
        text = "\n".join([p.text for p in doc.paragraphs]).strip()
    elif filename.endswith((".txt", ".md", ".csv", ".json")):
        text = (file.read() or b"").decode("utf-8", errors="ignore").strip()
    else:
        return jsonify(error="Unsupported file type. Use .pdf, .docx, .txt, .md, .csv, or .json."), 400

    if not text:
        return jsonify(error="Unable to extract text from file."), 400

    return jsonify(text=text)


def _split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _compute_reasons(text):
    reasons = []
    sentences = _split_sentences(text)
    words = re.findall(r"\b\w+\b", text.lower())

    if sentences:
        lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
        if len(lengths) > 1:
            variance = statistics.pvariance(lengths)
            if variance < 8:
                reasons.append("Low sentence length variation")

    if words:
        bigrams = list(zip(words, words[1:]))
        if bigrams:
            counts = {}
            for bg in bigrams:
                counts[bg] = counts.get(bg, 0) + 1
            top = max(counts.values())
            if top / max(len(bigrams), 1) > 0.08:
                reasons.append("Repetitive phrasing patterns detected")

        function_words = {
            "the",
            "and",
            "to",
            "of",
            "in",
            "that",
            "for",
            "on",
            "with",
            "as",
            "is",
            "it",
            "this",
            "by",
            "from",
        }
        ratio = sum(1 for w in words if w in function_words) / max(len(words), 1)
        if ratio > 0.55:
            reasons.append("High function-word density")

    if not reasons:
        reasons.append("Balanced structure and phrasing")

    return reasons


@app.post("/api/analyze")
def api_analyze():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()

    if len(text) < 10:
        return jsonify(error="Please provide at least 10 characters."), 400

    try:
        overall = get_detailed_prediction(text)
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 500
    except Exception as exc:
        return jsonify(error=f"Analysis failed: {exc}"), 500

    sentences = _split_sentences(text)
    sentence_results = []
    for sentence in sentences:
        label, confidence, probs, warning, word_count = predict_text(sentence)
        sentence_results.append(
            {
                "text": sentence,
                "label": label,
                "confidence": float(confidence),
                "human_probability": float(probs[0] * 100),
                "ai_probability": float(probs[1] * 100),
                "word_count": int(word_count),
                "warning": warning,
            }
        )

    safe_overall = {
        "prediction": overall.get("prediction"),
        "confidence": float(overall.get("confidence", 0)),
        "human_probability": float(overall.get("human_probability", 0)),
        "ai_probability": float(overall.get("ai_probability", 0)),
        "word_count": int(overall.get("word_count", 0)),
        "warning": overall.get("warning"),
        "needs_review": bool(overall.get("needs_review")),
    }

    return jsonify(
        {
            "overall": safe_overall,
            "sentences": sentence_results,
            "reasons": _compute_reasons(text),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
