export default function Header({ isDark, onToggleTheme }) {
    return (
      <header className="app-header">
        <div className="logo">AI Detector</div>
        <div className="header-actions">
          <button className="ghost-btn" onClick={onToggleTheme}>
            {isDark ? '☀️ Light mode' : '🌙 Dark mode'}
          </button>
        </div>
      </header>
    );
  }