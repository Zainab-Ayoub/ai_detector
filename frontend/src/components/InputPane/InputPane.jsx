import TextEditor from './TextEditor';
import UploadPanel from './UploadPanel';
import HighlightPanel from './HighlightPanel';

export default function InputPane({
  text, setText, activeTab, setActiveTab, isExtracting,
  wordCount, charCount, highlights,
  pasteFromClipboard, handleFileRead, reset, loadExample, scan,
}) {
  return (
    <section className="pane pane-left">
      <div className="pane-header">
        <h1>Scan for AI writing.</h1>
        <p className="card-subtitle">
          Paste text or upload a document to evaluate authenticity with confidence and review guidance.
        </p>
      </div>

      <form className="detector-form" onSubmit={(e) => { e.preventDefault(); scan(); }}>
        <div className="toolbar">
          <div className="tab-row">
            {['paste', 'upload'].map((tab) => (
              <button
                key={tab}
                className={`tab-btn${activeTab === tab ? ' active' : ''}`}
                type="button"
                onClick={() => setActiveTab(tab)}
              >
                {tab === 'paste' ? 'Editor' : 'Upload'}
              </button>
            ))}
          </div>
          <div className="toolbar-actions">
            <button className="ghost-btn" type="button" onClick={reset}>+ New scan</button>
            <button className="ghost-btn" type="button" onClick={loadExample}>Try an example</button>
            <button className="primary-btn" type="submit" disabled={isExtracting}>
              {isExtracting ? 'Extracting…' : 'Scan'}
            </button>
          </div>
        </div>

        {activeTab === 'paste' ? (
          <TextEditor
            text={text} setText={setText}
            wordCount={wordCount} charCount={charCount}
            onPaste={pasteFromClipboard}
            onClear={() => setText('')}
          />
        ) : (
          <UploadPanel onFileRead={handleFileRead} />
        )}

        <p className="form-hint">Tip: provide 50+ words for higher confidence.</p>
      </form>

      <HighlightPanel highlights={highlights} />
    </section>
  );
}