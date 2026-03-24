import ScoreRing from './ScoreRing';
import ResultTags from './ResultTags';
import ResultBars from './ResultBars';

export default function ResultPane({ result }) {
  const { label, confidence, humanProb, aiProb, note, words } = result;
  return (
    <section className="pane pane-right">
      <div className="panel-header">
        <h2>Result summary</h2>
        <span className="pill">{label}</span>
      </div>
      <div className="result-panel">
        <ScoreRing confidence={confidence} />
        <ResultTags label={label} />
        <ResultBars humanProb={humanProb} aiProb={aiProb} />
        <div className="result-insights">
          <div>
            <p className="insight-label">Words</p>
            <p className="insight-value">{words}</p>
          </div>
        </div>
        {note && <div className="result-notes">{note}</div>}
      </div>
    </section>
  );
}