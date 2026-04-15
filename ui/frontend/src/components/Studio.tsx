import { useAudioStream } from '../hooks/useAudioStream';
import SpectrogramCanvas from './SpectrogramCanvas';
import type { Prediction } from '../types';

export default function Studio() {
  const { isRecording, analyser, predictions, wsConnected, sendProgress, start, stop } = useAudioStream();

  const handleToggle = async () => {
    if (isRecording) {
      stop();
    } else {
      try {
        await start();
      } catch (e) {
        console.error('Failed to start recording:', e);
        alert('Could not access microphone. Please allow microphone access.');
      }
    }
  };

  return (
    <div className="studio">
      {/* Live Spectrogram */}
      <div className="card" style={{ padding: '16px' }}>
        <div className="card__title">Spectrogram_Feed</div>
        <SpectrogramCanvas analyser={analyser} isRecording={isRecording} />
      </div>

      {/* Inference countdown bar */}
      {isRecording && (
        <div className="send-progress">
          <div
            className="send-progress__fill"
            style={{ width: `${sendProgress * 100}%` }}
          />
          <span className="send-progress__label">
            INFERENCE_CYCLE: {Math.max(0, (3 - sendProgress * 3)).toFixed(1)}s
          </span>
        </div>
      )}

      {/* Controls */}
      <div className="studio__controls">
        <button
          className={`record-btn ${isRecording ? 'record-btn--recording' : ''}`}
          onClick={handleToggle}
          aria-label={isRecording ? 'Stop recording' : 'Start recording'}
          id="record-button"
        >
          {isRecording ? 'STOP_RECORDING' : 'START_RECORD'}
        </button>

        <div className="studio__status">
          <div
            className={`studio__status-dot ${isRecording ? 'studio__status-dot--active' : ''}`}
          />
          {isRecording ? (
            <span>
              STATUS: REC_ACTIVE{' // '}
              {wsConnected ? (
                <span style={{ color: 'var(--green)' }}>LINK_ESTABLISHED</span>
              ) : (
                <span style={{ color: 'var(--yellow)' }}>CONNECTING...</span>
              )}
            </span>
          ) : (
            <span>STATUS: STANDBY</span>
          )}
        </div>
      </div>

      {/* Results */}
      <div className="card">
        <div className="card__title">Inference_Results</div>
        {predictions.length > 0 ? (
          <ResultsPanel predictions={predictions} />
        ) : (
          <div className="empty-state">
            <div className="empty-state__icon">
              <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2v20M17 5v14M7 5v14M22 8v8M2 8v8" stroke="currentColor" strokeWidth="2" strokeLinecap="square"/>
              </svg>
            </div>
            <div className="empty-state__text">
              {isRecording
                ? 'AWAITING_TELEMETRY...'
                : 'NO_DATA_FEED'}
            </div>
            <div className="empty-state__sub">
              {isRecording
                ? 'INFERENCE BATCH TRANSMITTED EVERY 3000MS'
                : 'ENGAGE RECORDING TO INITIALIZE CLASSIFICATION'}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Results Panel ────────────────────────────────────────────────────────────

function ResultsPanel({ predictions }: { predictions: Prediction[] }) {
  return (
    <div className="results-panel">
      {predictions.map((pred, i) => {
        const pct = pred.confidence * 100;
        return (
          <div className="result-row" key={pred.class_name} style={{ animationDelay: `${i * 60}ms` }}>
            <div className="result-row__rank">0{i + 1}</div>
            <div className="result-row__name">{pred.class_name.replace(/_/g, ' ')}</div>
            <div className="result-row__bar-bg">
              <div
                className="result-row__bar-fill"
                style={{ width: `${pct}%` }}
              />
            </div>
            <div className="result-row__conf">{pct.toFixed(1)}%</div>
          </div>
        );
      })}
    </div>
  );
}
