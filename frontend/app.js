
const form = document.getElementById("detector-form");
const input = document.getElementById("detector-input");
const wordCount = document.getElementById("word-count");
const charCount = document.getElementById("char-count");
const resultLabel = document.getElementById("result-label");
const resultConfidence = document.getElementById("result-confidence");
const humanProb = document.getElementById("human-prob");
const aiProb = document.getElementById("ai-prob");
const humanBar = document.getElementById("human-bar");
const aiBar = document.getElementById("ai-bar");
const resultNotes = document.getElementById("result-notes");
const pasteBtn = document.getElementById("paste-btn");
const clearBtn = document.getElementById("clear-btn");
const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const driveBtn = document.getElementById("drive-btn");
const startScanBtn = document.getElementById("start-scan-btn");
const exampleBtn = document.getElementById("example-btn");
const resultTagPrimary = document.getElementById("result-tag-primary");
const resultWords = document.getElementById("result-words");
const resultReview = document.getElementById("result-review");
const tabButtons = Array.from(document.querySelectorAll(".tab-btn"));
const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));

const updateCounts = () => {
  const text = input.value.trim();
  const words = text ? text.split(/\s+/).length : 0;
  wordCount.textContent = `${words} words`;
  charCount.textContent = `${input.value.length} characters`;
};

const setText = (text) => {
  input.value = text;
  updateCounts();
};

const setResult = (label, confidence, human, ai, note, words, review) => {
  resultLabel.textContent = label;
  resultConfidence.textContent = confidence;
  humanProb.textContent = human;
  aiProb.textContent = ai;
  humanBar.style.width = human;
  aiBar.style.width = ai;
  resultNotes.textContent = note;
  if (resultTagPrimary) {
    resultTagPrimary.textContent = label;
  }
  if (resultWords) {
    resultWords.textContent = words ?? "--";
  }
  if (resultReview) {
    resultReview.textContent = review ?? "--";
  }
};

const runPrediction = async (text) => {
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const errorMessage = payload.error || "Prediction failed.";
    throw new Error(errorMessage);
  }

  return response.json();
};

input.addEventListener("input", updateCounts);
updateCounts();

startScanBtn.addEventListener("click", () => {
  input.focus();
});

const setActiveTab = (tab) => {
  tabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
  tabPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.panel === tab);
  });
};

tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    setActiveTab(button.dataset.tab);
  });
});

pasteBtn.addEventListener("click", async () => {
  if (!navigator.clipboard?.readText) {
    setResult(
      "Clipboard blocked",
      "--%",
      "--%",
      "--%",
      "Clipboard access isn't available in this browser.",
      "--",
      "--"
    );
    return;
  }

  try {
    const text = await navigator.clipboard.readText();
    setText(text);
    setActiveTab("paste");
  } catch (error) {
    setResult(
      "Clipboard blocked",
      "--%",
      "--%",
      "--%",
      "Clipboard access was denied.",
      "--",
      "--"
    );
  }
});

clearBtn.addEventListener("click", () => {
  setText("");
  setResult(
    "Awaiting input",
    "--%",
    "--%",
    "--%",
    "Paste or upload text to start.",
    "--",
    "--"
  );
});

exampleBtn.addEventListener("click", () => {
  const sample =
    "Artificial intelligence systems have become more capable in recent years, " +
    "but the most effective assessments still blend automation with human review. " +
    "This example text includes varied sentence length, natural phrasing, and " +
    "consistent narrative flow to simulate a human-written paragraph.";
  setText(sample);
  setActiveTab("paste");
});

const extractFileText = async (file) => {
  if (!file) return "";
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/api/extract", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || "Failed to extract text.");
  }

  const payload = await response.json();
  return payload.text || "";
};

const readFileToInput = async (file) => {
  if (!file) return;
  setResult("Extracting", "--%", "--%", "--%", "Reading file...", "--", "--");

  try {
    const text = await extractFileText(file);
    setText(text);
  } catch (error) {
    setResult(
      "File error",
      "--%",
      "--%",
      "--%",
      error.message || "Unable to read that file.",
      "--",
      "--"
    );
  }
};

fileInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  readFileToInput(file);
  setActiveTab("upload");
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragover");
  const file = event.dataTransfer.files?.[0];
  readFileToInput(file);
  setActiveTab("upload");
});

driveBtn.addEventListener("click", () => {
  setResult(
    "Google Drive",
    "--%",
    "--%",
    "--%",
    "Google Drive picker requires OAuth setup. Add your API keys to enable it.",
    "--",
    "--"
  );
  setActiveTab("drive");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();

  if (text.length < 10) {
    setResult(
      "Needs more text",
      "--%",
      "--%",
      "--%",
      "Please enter at least 10 characters to run a scan.",
      "--",
      "--"
    );
    return;
  }

  setResult("Scanning", "--%", "--%", "--%", "Running analysis...", "--", "--");

  try {
    const result = await runPrediction(text);
    const label = result.prediction || "Unknown";
    const confidence = `${result.confidence.toFixed(1)}%`;
    const human = `${result.human_probability.toFixed(1)}%`;
    const ai = `${result.ai_probability.toFixed(1)}%`;
    const note = result.warning
      ? result.warning
      : result.needs_review
      ? "Moderate confidence. Consider manual review."
      : "Result looks confident. No additional review needed.";
    const review = result.needs_review ? "Review" : "Clear";

    setResult(label, confidence, human, ai, note, result.word_count, review);
  } catch (error) {
    setResult(
      "Offline",
      "--%",
      "--%",
      "--%",
      `Backend unavailable. Start "python web_app.py" to run scans. ${error.message}`,
      "--",
      "--"
    );
  }
});
