import { useRef, useState } from 'react';

export default function UploadPanel({ onFileRead }) {
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [fileName, setFileName] = useState('');

  const handleFile = (file) => {
    if (!file) return;
    setFileName(file.name);
    onFileRead(file);
  };

  return (
    <div className="tab-panel active" data-panel="upload">
      <button
        type="button"
        className="ghost-btn"
        onClick={() => fileInputRef.current?.click()}
      >
        Upload file
      </button>

      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.docx,.txt,.md,.csv,.json"
        style={{ display: 'none' }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          handleFile(f);
          e.target.value = '';
        }}
      />

      {fileName ? (
        <div className="drop-zone" style={{ borderStyle: 'solid', borderColor: 'var(--accent)' }}>
          <p style={{ color: 'var(--accent)', fontWeight: 600 }}>📄 {fileName}</p>
          <span>File selected</span>
        </div>
      ) : (
        <div
          className={`drop-zone${isDragOver ? ' dragover' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragOver(false);
            handleFile(e.dataTransfer.files?.[0]);
          }}
        >
          <p>Drag & drop a file here</p>
          <span>.pdf, .docx, .txt, .md, .csv, .json supported</span>
        </div>
      )}
    </div>
  );
}