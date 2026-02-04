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
const scoreRing = document.getElementById("score-ring");
const resultHighlights = document.getElementById("result-highlights");
const pasteBtn = document.getElementById("paste-btn");
const clearBtn = document.getElementById("clear-btn");
const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const driveBtn = document.getElementById("drive-btn");
const startScanBtn = document.getElementById("start-scan-btn");
const exampleBtn = document.getElementById("example-btn");
const newScanBtn = document.getElementById("new-scan-btn");
const themeToggle = document.getElementById("theme-toggle");
const resultTagAi = document.getElementById("result-tag-ai");
const resultTagHuman = document.getElementById("result-tag-human");
const resultTagMixed = document.getElementById("result-tag-mixed");
const resultWords = document.getElementById("result-words");
const tabButtons = Array.from(document.querySelectorAll(".tab-btn"));
const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));

let isExtracting = false;
let isDriveAuthenticated = false;
let gapi = null;
let tokenClient = null;

// Google API Configuration - You'll need to add your own API Key and Client ID
const GOOGLE_API_KEY = 'YOUR_GOOGLE_API_KEY';  // Get from Google Cloud Console
const GOOGLE_CLIENT_ID = 'YOUR_CLIENT_ID.apps.googleusercontent.com';  // Get from Google Cloud Console
const DISCOVERY_DOC = 'https://www.googleapis.com/discovery/v1/apis/drive/v3/rest';
const SCOPES = 'https://www.googleapis.com/auth/drive.readonly';

// Check Drive authentication status on load
async function checkDriveStatus() {
  try {
    const response = await fetch('/api/drive/status');
    const data = await response.json();
    isDriveAuthenticated = data.authenticated;
    updateDriveButton();
  } catch (error) {
    console.error('Failed to check Drive status:', error);
  }
}

function updateDriveButton() {
  if (isDriveAuthenticated) {
    driveBtn.textContent = '✓ Connected - Pick a file';
    driveBtn.classList.add('connected');
  } else {
    driveBtn.textContent = 'Connect Google Drive';
    driveBtn.classList.remove('connected');
  }
}

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
  if (scoreRing) {
    scoreRing.style.setProperty("--score", confidence);
  }
  if (resultNotes) {
    if (note) {
      resultNotes.textContent = note;
      resultNotes.classList.remove("is-hidden");
    } else {
      resultNotes.textContent = "";
      resultNotes.classList.add("is-hidden");
    }
  }
  if (resultTagAi && resultTagHuman && resultTagMixed) {
    resultTagAi.classList.toggle("active", label === "AI");
    resultTagHuman.classList.toggle("active", label === "Human");
    resultTagMixed.classList.toggle("active", label === "UNCERTAIN");
  }
  if (resultWords) {
    resultWords.textContent = words ?? "--";
  }
};

const resetScanState = () => {
  setText("");
  setResult(
    "Awaiting input",
    "--%",
    "--%",
    "--%",
    "",
    "--",
    "--"
  );
  if (resultHighlights) {
    resultHighlights.innerHTML = "";
  }
  if (dropZone) {
    dropZone.classList.remove("is-hidden");
  }
  setActiveTab("paste");
};

const escapeHtml = (value) =>
  value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

const buildHighlights = (text, sentences) => {
  const chunks = text.match(/[^.!?]+[.!?]?\s*/g) || [text];
  let idx = 0;

  return chunks
    .map((chunk) => {
      if (!/\w/.test(chunk)) {
        return escapeHtml(chunk);
      }

      const sentence = sentences[idx];
      if (sentence) {
        idx += 1;
        const aiProb = Number(sentence.ai_probability || 0);
        const cls =
          aiProb >= 60
            ? "highlight-ai"
            : aiProb <= 40
            ? "highlight-human"
            : "highlight-mixed";
        return `<span class="${cls}">${escapeHtml(chunk)}</span>`;
      }

      return `<span class="highlight-mixed">${escapeHtml(chunk)}</span>`;
    })
    .join("");
};

const runAnalysis = async (text) => {
  const response = await fetch("/api/analyze", {
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

document.addEventListener("dragover", (event) => {
  event.preventDefault();
});

document.addEventListener("drop", (event) => {
  event.preventDefault();
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
  if (resultHighlights) {
    resultHighlights.textContent = "Highlights will appear after scanning.";
  }
  if (dropZone) {
    dropZone.classList.remove("is-hidden");
  }
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

newScanBtn.addEventListener("click", () => {
  resetScanState();
});

themeToggle.addEventListener("click", () => {
  document.body.classList.toggle("dark");
  themeToggle.textContent = document.body.classList.contains("dark")
    ? "Dark mode"
    : "Light mode";
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
  isExtracting = true;
  if (startScanBtn) {
    startScanBtn.disabled = true;
    startScanBtn.textContent = "Extracting...";
  }

  try {
    const text = await extractFileText(file);
    setText(text);
    setActiveTab("upload");
    if (resultHighlights) {
      resultHighlights.innerHTML = "";
    }
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
  } finally {
    isExtracting = false;
    if (startScanBtn) {
      startScanBtn.disabled = false;
      startScanBtn.textContent = "Scan";
    }
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
  if (!file) {
    setResult(
      "File error",
      "--%",
      "--%",
      "--%",
      "No file detected. Try dropping a file again.",
      "--",
      "--"
    );
    return;
  }
  readFileToInput(file);
  setActiveTab("upload");
});

// Google Drive Authentication and File Picker
async function authenticateDrive() {
  try {
    setResult(
      "Connecting",
      "--%",
      "--%",
      "--%",
      "Opening Google authentication...",
      "--",
      "--"
    );

    const response = await fetch('/api/drive/auth');
    const data = await response.json();

    if (data.error) {
      setResult(
        "Auth Error",
        "--%",
        "--%",
        "--%",
        data.error,
        "--",
        "--"
      );
      return false;
    }

    // Open OAuth popup
    const authWindow = window.open(
      data.auth_url,
      'Google Drive Authentication',
      'width=600,height=700'
    );

    // Listen for authentication success
    return new Promise((resolve) => {
      const messageHandler = (event) => {
        if (event.data.type === 'drive-auth-success') {
          window.removeEventListener('message', messageHandler);
          isDriveAuthenticated = true;
          updateDriveButton();
          setResult(
            "Connected",
            "--%",
            "--%",
            "--%",
            "Successfully connected to Google Drive! Now pick a file.",
            "--",
            "--"
          );
          resolve(true);
        }
      };
      window.addEventListener('message', messageHandler);

      // Check if popup was closed without completing auth
      const checkClosed = setInterval(() => {
        if (authWindow && authWindow.closed) {
          clearInterval(checkClosed);
          window.removeEventListener('message', messageHandler);
          if (!isDriveAuthenticated) {
            setResult(
              "Auth Cancelled",
              "--%",
              "--%",
              "--%",
              "Authentication was cancelled.",
              "--",
              "--"
            );
          }
          resolve(false);
        }
      }, 1000);
    });
  } catch (error) {
    setResult(
      "Connection Error",
      "--%",
      "--%",
      "--%",
      `Failed to connect: ${error.message}`,
      "--",
      "--"
    );
    return false;
  }
}

// Load Google Picker API
function loadGooglePicker() {
  if (window.google && window.google.picker) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'https://apis.google.com/js/api.js';
    script.onload = () => {
      gapi = window.gapi;
      gapi.load('picker', () => {
        resolve();
      });
    };
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

async function showDrivePicker() {
  try {
    await loadGooglePicker();
    
    // Note: For the picker to work, you need to use the Google Picker API
    // which requires an API key. For simplicity, we'll use a file selection approach
    
    // Create a simple file selector using Google Drive API
    setResult(
      "Loading Drive",
      "--%",
      "--%",
      "--%",
      "Loading your Google Drive files...",
      "--",
      "--"
    );

    // Show a modal with file selection
    showDriveFileModal();
    
  } catch (error) {
    setResult(
      "Picker Error",
      "--%",
      "--%",
      "--%",
      `Failed to load Drive picker: ${error.message}`,
      "--",
      "--"
    );
  }
}

function showDriveFileModal() {
  // Create a simple modal for file selection
  const modal = document.createElement('div');
  modal.className = 'drive-modal';
  modal.innerHTML = `
    <div class="drive-modal-content">
      <div class="drive-modal-header">
        <h3>Select a file from Google Drive</h3>
        <button class="drive-modal-close">&times;</button>
      </div>
      <div class="drive-modal-body">
        <p>Enter the Google Drive File ID:</p>
        <input type="text" id="drive-file-id" placeholder="e.g., 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms" />
        <p class="drive-help">To get the file ID, open the file in Google Drive and copy the ID from the URL.</p>
        <p class="drive-help">Example: https://docs.google.com/document/d/<strong>FILE_ID_HERE</strong>/edit</p>
      </div>
      <div class="drive-modal-footer">
        <button class="ghost-btn drive-modal-cancel">Cancel</button>
        <button class="primary-btn drive-modal-select">Load File</button>
      </div>
    </div>
  `;
  
  document.body.appendChild(modal);
  
  const fileIdInput = modal.querySelector('#drive-file-id');
  const closeBtn = modal.querySelector('.drive-modal-close');
  const cancelBtn = modal.querySelector('.drive-modal-cancel');
  const selectBtn = modal.querySelector('.drive-modal-select');
  
  const closeModal = () => {
    document.body.removeChild(modal);
  };
  
  closeBtn.addEventListener('click', closeModal);
  cancelBtn.addEventListener('click', closeModal);
  
  selectBtn.addEventListener('click', async () => {
    const fileId = fileIdInput.value.trim();
    if (!fileId) {
      alert('Please enter a file ID');
      return;
    }
    
    closeModal();
    await downloadDriveFile(fileId);
  });
  
  fileIdInput.focus();
}

async function downloadDriveFile(fileId) {
  try {
    setResult(
      "Downloading",
      "--%",
      "--%",
      "--%",
      "Downloading file from Google Drive...",
      "--",
      "--"
    );
    
    isExtracting = true;
    if (startScanBtn) {
      startScanBtn.disabled = true;
      startScanBtn.textContent = "Downloading...";
    }

    const response = await fetch('/api/drive/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_id: fileId })
    });

    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || 'Failed to download file');
    }

    const data = await response.json();
    setText(data.text);
    
    setResult(
      "Ready",
      "--%",
      "--%",
      "--%",
      `File "${data.filename}" loaded successfully! Click Scan to analyze.`,
      "--",
      "--"
    );
    
    setActiveTab("paste");
    
  } catch (error) {
    setResult(
      "Download Error",
      "--%",
      "--%",
      "--%",
      error.message || "Failed to download file from Drive.",
      "--",
      "--"
    );
  } finally {
    isExtracting = false;
    if (startScanBtn) {
      startScanBtn.disabled = false;
      startScanBtn.textContent = "Scan";
    }
  }
}

driveBtn.addEventListener("click", async () => {
  if (!isDriveAuthenticated) {
    const authenticated = await authenticateDrive();
    if (authenticated) {
      await showDrivePicker();
    }
  } else {
    await showDrivePicker();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();

  if (isExtracting) {
    setResult(
      "Extracting",
      "--%",
      "--%",
      "--%",
      "Please wait for the file to finish extracting.",
      "--",
      "--"
    );
    return;
  }

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
    const analysis = await runAnalysis(text);
    const overall = analysis.overall || {};
    const label = overall.prediction || "Unknown";
    const confidence = `${Number(overall.confidence || 0).toFixed(1)}%`;
    const human = `${Number(overall.human_probability || 0).toFixed(1)}%`;
    const ai = `${Number(overall.ai_probability || 0).toFixed(1)}%`;
    const note = overall.warning
      ? overall.warning
      : overall.needs_review
      ? "Moderate confidence. Consider manual review."
      : "Result looks confident. No additional review needed.";
    const review = overall.needs_review ? "Review" : "Clear";

    setResult(label, confidence, human, ai, note, overall.word_count, review);

    if (resultHighlights) {
      const sentences = analysis.sentences || [];
      const html = buildHighlights(text, sentences);
      resultHighlights.innerHTML = html || "No highlights available.";
    }

    if (dropZone) {
      dropZone.classList.add("is-hidden");
    }

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

// Check Drive authentication status on page load
checkDriveStatus();