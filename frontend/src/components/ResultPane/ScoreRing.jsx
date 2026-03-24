export default function ScoreRing({ confidence }) {
    const scoreValue = confidence && confidence !== '--%' ? confidence : '0%';
    return (
      <div className="result-score">
        <div className="score-ring" style={{ '--score': scoreValue }}>
          <span className="score-value">{confidence}</span>
        </div>
        <p className="score-label">Confidence</p>
      </div>
    );
  }