import { useEffect } from 'react';
import { useDetector } from './hooks/useDetector';
import Header from './components/Header';
import InputPane from './components/InputPane/InputPane';
import ResultPane from './components/ResultPane/ResultPane';
import './index.css';

export default function App() {
  const detector = useDetector();

  useEffect(() => {
    document.body.classList.toggle('dark', detector.isDark);
  }, [detector.isDark]);

  return (
    <>
      <div className="page-bg" />
      <Header
        isDark={detector.isDark}
        onToggleTheme={() => detector.setIsDark((d) => !d)}
      />
      <main className="workspace">
        <InputPane
          text={detector.text}
          setText={detector.setText}
          activeTab={detector.activeTab}
          setActiveTab={detector.setActiveTab}
          isExtracting={detector.isExtracting}
          wordCount={detector.wordCount}
          charCount={detector.charCount}
          highlights={detector.highlights}
          pasteFromClipboard={detector.pasteFromClipboard}
          handleFileRead={detector.handleFileRead}
          reset={detector.reset}
          loadExample={detector.loadExample}
          scan={detector.scan}
        />
        <ResultPane result={detector.result} />
      </main>
    </>
  );
}