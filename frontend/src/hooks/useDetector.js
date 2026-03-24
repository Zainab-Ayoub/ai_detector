import { useState, useCallback } from 'react';

const INITIAL_RESULT = {
  label: 'Awaiting input',
  confidence: '--%',
  humanProb: '--%',
  aiProb: '--%',
  note: '',
  words: '--',
};

const escapeHtml = (value) =>
  value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

const buildHighlights = (rawText, sentences) => {
  const chunks = rawText.match(/[^.!?]+[.!?]?\s*/g) || [rawText];
  let idx = 0;
  return chunks
    .map((chunk) => {
      if (!/\w/.test(chunk)) return escapeHtml(chunk);
      const sentence = sentences[idx];
      if (sentence) {
        idx += 1;
        const aiProb = Number(sentence.ai_probability || 0);
        const cls =
          aiProb >= 60 ? 'highlight-ai'
          : aiProb <= 40 ? 'highlight-human'
          : 'highlight-mixed';
        return `<span class="${cls}">${escapeHtml(chunk)}</span>`;
      }
      return `<span class="highlight-mixed">${escapeHtml(chunk)}</span>`;
    })
    .join('');
};

export function useDetector() {
  const [text, setText] = useState('');
  const [activeTab, setActiveTab] = useState('paste');
  const [isExtracting, setIsExtracting] = useState(false);
  const [result, setResult] = useState(INITIAL_RESULT);
  const [highlights, setHighlights] = useState('');
  const [isDark, setIsDark] = useState(true);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const charCount = text.length;

  const reset = useCallback(() => {
    setText('');
    setResult(INITIAL_RESULT);
    setHighlights('');
    setActiveTab('paste');
  }, []);

  const loadExample = useCallback(() => {
    setText(
      'Artificial intelligence systems have become more capable in recent years, ' +
      'but the most effective assessments still blend automation with human review. ' +
      'This example text includes varied sentence length, natural phrasing, and ' +
      'consistent narrative flow to simulate a human-written paragraph.'
    );
    setActiveTab('paste');
  }, []);

  const pasteFromClipboard = useCallback(async () => {
    if (!navigator.clipboard?.readText) {
      setResult((prev) => ({ ...prev, label: 'Clipboard blocked', note: "Clipboard access isn't available in this browser." }));
      return;
    }
    try {
      const clipText = await navigator.clipboard.readText();
      setText(clipText);
      setActiveTab('paste');
    } catch {
      setResult((prev) => ({ ...prev, label: 'Clipboard blocked', note: 'Clipboard access was denied.' }));
    }
  }, []);

  const handleFileRead = useCallback(async (file) => {
    if (!file) return;
    setResult({ label: 'Extracting', confidence: '--%', humanProb: '--%', aiProb: '--%', note: 'Reading file...', words: '--' });
    setIsExtracting(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await fetch('/api/extract', { method: 'POST', body: formData });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || 'Failed to extract text.');
      }
      const payload = await response.json();
      setText(payload.text || '');
      setActiveTab('upload');
      setHighlights('');
    } catch (error) {
      setResult({ label: 'File error', confidence: '--%', humanProb: '--%', aiProb: '--%', note: error.message || 'Unable to read that file.', words: '--' });
    } finally {
      setIsExtracting(false);
    }
  }, []);

  const scan = useCallback(async () => {
    const trimmed = text.trim();
    if (isExtracting) {
      setResult((prev) => ({ ...prev, label: 'Extracting', note: 'Please wait for the file to finish extracting.' }));
      return;
    }
    if (trimmed.length < 10) {
      setResult((prev) => ({ ...prev, label: 'Needs more text', note: 'Please enter at least 10 characters to run a scan.' }));
      return;
    }
    setResult({ label: 'Scanning', confidence: '--%', humanProb: '--%', aiProb: '--%', note: 'Running analysis...', words: '--' });
    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: trimmed }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || 'Prediction failed.');
      }
      const analysis = await response.json();
      const overall = analysis.overall || {};
      setResult({
        label: overall.prediction || 'Unknown',
        confidence: `${Number(overall.confidence || 0).toFixed(1)}%`,
        humanProb: `${Number(overall.human_probability || 0).toFixed(1)}%`,
        aiProb: `${Number(overall.ai_probability || 0).toFixed(1)}%`,
        note: overall.warning ? overall.warning
          : overall.needs_review ? 'Moderate confidence. Consider manual review.'
          : 'Result looks confident. No additional review needed.',
        words: overall.word_count,
      });
      const sentences = analysis.sentences || [];
      setHighlights(buildHighlights(trimmed, sentences) || 'No highlights available.');
    } catch (error) {
      setResult({ label: 'Offline', confidence: '--%', humanProb: '--%', aiProb: '--%', note: `Backend unavailable. Start "python web_app.py" to run scans. ${error.message}`, words: '--' });
    }
  }, [text, isExtracting]);

  return {
    text, setText, activeTab, setActiveTab, isExtracting,
    result, highlights, isDark, setIsDark,
    wordCount, charCount,
    reset, loadExample, pasteFromClipboard, handleFileRead, scan,
  };
}