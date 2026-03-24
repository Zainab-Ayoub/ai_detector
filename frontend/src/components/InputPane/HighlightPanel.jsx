export default function HighlightPanel({ highlights }) {
    return (
      <>
        <div className="result-highlights" dangerouslySetInnerHTML={{ __html: highlights || '' }} />
        <div className="highlight-legend">
          <span className="legend-item"><span className="legend-swatch legend-ai" /> Likely AI</span>
          <span className="legend-item"><span className="legend-swatch legend-mixed" /> Mixed/uncertain</span>
          <span className="legend-item"><span className="legend-swatch legend-human" /> Likely human</span>
        </div>
      </>
    );
  }