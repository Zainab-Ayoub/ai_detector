function ProbBar({ label, value }) {
    const width = value && value !== '--%' ? value : '0%';
    return (
      <div>
        <div className="bar-label">
          <span>{label}</span>
          <span>{value}</span>
        </div>
        <div className="bar">
          <span style={{ width }} />
        </div>
      </div>
    );
  }
  
  export default function ResultBars({ humanProb, aiProb }) {
    return (
      <div className="result-bars">
        <ProbBar label="Human probability" value={humanProb} />
        <ProbBar label="AI probability" value={aiProb} />
      </div>
    );
  }