export default function TextEditor({ text, setText, wordCount, charCount, onPaste, onClear }) {
    return (
      <div className="tab-panel active" data-panel="paste">
        <label htmlFor="detector-input">Text to scan</label>
        <textarea
          id="detector-input"
          placeholder="Paste your text here..."
          rows={10}
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <div className="form-meta">
          <span>{wordCount} words</span>
          <span>{charCount} characters</span>
        </div>
        <div className="input-actions">
          <button className="ghost-btn" type="button" onClick={onPaste}>Paste from clipboard</button>
          <button className="ghost-btn" type="button" onClick={onClear}>Clear text</button>
        </div>
      </div>
    );
  }