from flask import Flask, jsonify, request
import io
from docx import Document
import pdfplumber

from predict import get_detailed_prediction

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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
