const TAGS = [
    { key: 'AI', label: 'AI' },
    { key: 'Human', label: 'Human' },
    { key: 'UNCERTAIN', label: 'Mixed' },
  ];
  
  export default function ResultTags({ label }) {
    return (
      <div className="result-tags">
        {TAGS.map((tag) => (
          <span key={tag.key} className={`result-tag${label === tag.key ? ' active' : ''}`}>
            {tag.label}
          </span>
        ))}
      </div>
    );
  }