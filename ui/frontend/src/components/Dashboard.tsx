import { useEffect, useState, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts';

import { fetchModelInfo, fetchEvaluation, getConfusionMatrixUrl, getAudioSampleUrl } from '../api';
import type { ModelInfo, EvaluationResult, ClassAccuracy } from '../types';

// ── Dashboard Tab ────────────────────────────────────────────────────────────

export default function Dashboard() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [evalData, setEvalData] = useState<EvaluationResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const [infoRes, evalRes] = await Promise.all([
          fetchModelInfo(),
          fetchEvaluation(),
        ]);
        setInfo(infoRes);
        setEvalData(evalRes);
      } catch (e) {
        console.error('Dashboard fetch failed:', e);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) {
    return (
      <div className="loading">
        <div className="loading__spinner" />
        <span>SYS.LOADING...</span>
      </div>
    );
  }

  if (!info) {
    return (
      <div className="empty-state">
        <div className="empty-state__icon">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 9V14M12 17.5V18M12 3L2 21H22L12 3Z" stroke="currentColor" strokeWidth="2" strokeLinecap="square"/>
          </svg>
        </div>
        <div className="empty-state__text">SYSTEM OFFLINE</div>
        <div className="empty-state__sub">
          Verify backend connectivity: uv run uvicorn ui.backend.main:app --port 8000
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      {/* Model Info */}
      <div className="card">
        <div className="card__title">Model Spec</div>
        <div className="model-info">
          <StatItem label="Model_ID" value={info.model_name} accent />
          <StatItem label="Params" value={formatNumber(info.total_params)} />
          <StatItem label="Classes" value={String(info.num_classes)} />
          <StatItem label="Epochs" value={String(info.total_epochs)} />
          <StatItem label="Best_Ep" value={String(info.best_epoch)} accent />
          <StatItem
            label="Peak_Val_Acc"
            value={`${(info.best_val_acc * 100).toFixed(2)}%`}
            green
          />
        </div>
      </div>

      {/* Charts Row */}
      {info.history.length > 0 && (
        <div className="dashboard__row">
          <div className="card">
            <div className="card__title">Loss_Metrics</div>
            <TrainingChart
              data={info.history}
              lines={[
                { key: 'tr_loss', name: 'TR_LOSS', color: '#FF5500' },
                { key: 'val_loss', name: 'VAL_LOSS', color: '#A3A3A3' },
              ]}
            />
          </div>
          <div className="card">
            <div className="card__title">Accuracy_Progression</div>
            <TrainingChart
              data={info.history.map((h) => ({
                ...h,
                tr_acc: h.tr_acc * 100,
                val_acc: h.val_acc * 100,
              }))}
              lines={[
                { key: 'tr_acc', name: 'TR_ACC', color: '#525252' },
                { key: 'val_acc', name: 'VAL_ACC', color: '#FF5500' },
              ]}
              yDomain={[0, 100]}
              ySuffix="%"
            />
          </div>
        </div>
      )}

      {/* Per-class accuracy */}
      {evalData && (
        <div className="card">
          <div className="card__title">
            Class_Accuracy_Distribution (Avg: {(evalData.overall_accuracy * 100).toFixed(2)}%)
          </div>
          <AccuracyBars classes={evalData.per_class} />
        </div>
      )}

      {/* Class Grid */}
      {evalData && (
        <div className="card">
          <div className="card__title">Class_Matrix</div>
          <ClassGrid classes={evalData.per_class} />
        </div>
      )}

      {/* Confusion Matrix */}
      <div className="card">
        <div className="card__title">Confusion_Matrix</div>
        <div className="confusion-matrix">
          <img
            src={getConfusionMatrixUrl()}
            alt="Confusion matrix heatmap"
            loading="lazy"
          />
        </div>
      </div>
    </div>
  );
}

// ── Sub-components ───────────────────────────────────────────────────────────

function StatItem({
  label,
  value,
  accent,
  green,
}: {
  label: string;
  value: string;
  accent?: boolean;
  green?: boolean;
}) {
  const cls = green
    ? 'stat-item__value stat-item__value--green'
    : accent
      ? 'stat-item__value stat-item__value--accent'
      : 'stat-item__value';
  return (
    <div className="stat-item">
      <div className={cls}>{value}</div>
      <div className="stat-item__label">{label}</div>
    </div>
  );
}

function TrainingChart({
  data,
  lines,
  yDomain,
  ySuffix,
}: {
  data: Record<string, number>[];
  lines: { key: string; name: string; color: string }[];
  yDomain?: [number, number];
  ySuffix?: string;
}) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
        <CartesianGrid strokeDasharray="2 2" stroke="#333333" />
        <XAxis dataKey="epoch" tick={{ fontSize: 10, fontFamily: 'monospace' }} />
        <YAxis domain={yDomain} tick={{ fontSize: 10, fontFamily: 'monospace' }} />
        <Tooltip
          contentStyle={{
            background: '#000000',
            border: '1px solid #FF5500',
            borderRadius: 0,
            fontSize: 12,
            fontFamily: 'monospace'
          }}
          formatter={(value: number) =>
            `${value.toFixed(3)}${ySuffix || ''}`
          }
        />
        <Legend
          wrapperStyle={{ fontSize: 10, fontFamily: 'monospace', paddingTop: 8 }}
        />
        {lines.map((l) => (
          <Line
            key={l.key}
            type="monotone"
            dataKey={l.key}
            name={l.name}
            stroke={l.color}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: '#000', stroke: l.color, strokeWidth: 2 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

function AccuracyBars({ classes }: { classes: ClassAccuracy[] }) {
  const sorted = [...classes].sort((a, b) => b.accuracy - a.accuracy);
  return (
    <div style={{ maxHeight: 500, overflowY: 'auto', paddingRight: 8 }}>
      {sorted.map((c) => {
        const pct = c.accuracy * 100;
        const color =
          pct >= 80
            ? 'var(--green)'
            : pct >= 50
              ? 'var(--yellow)'
              : 'var(--red)';
        return (
          <div className="acc-bar-row" key={c.idx}>
            <div className="acc-bar-row__name">{c.name.replace(/_/g, ' ')}</div>
            <div className="acc-bar-row__bar">
              <div
                className="acc-bar-row__fill"
                style={{ width: `${pct}%`, background: color }}
              />
            </div>
            <div className="acc-bar-row__value" style={{ color }}>
              {pct.toFixed(1)}%
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ClassGrid({ classes }: { classes: ClassAccuracy[] }) {
  return (
    <div className="class-grid">
      {classes.map((c) => {
        return (
          <div className="class-card" key={c.idx}>
            <div className="class-card__name">{c.name.replace(/_/g, ' ')}</div>
            <AudioButton url={getAudioSampleUrl(c.idx)} />
            <div className="class-card__detail">
              VAL: {c.correct}/{c.total}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Utilities ────────────────────────────────────────────────────────────────

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return String(n);
}

function AudioButton({ url }: { url: string }) {
  const [playing, setPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const toggle = () => {
    if (!audioRef.current) {
      const a = new Audio(url);
      a.onended = () => setPlaying(false);
      audioRef.current = a;
    }
    const a = audioRef.current;
    if (playing) {
      a.pause();
      a.currentTime = 0;
      setPlaying(false);
    } else {
      a.play();
      setPlaying(true);
    }
  };

  return (
    <button 
      onClick={toggle}
      className={`audio-btn ${playing ? 'audio-btn--playing' : ''}`}
      style={playing ? { 
        color: '#000', 
        background: 'var(--accent)',
        borderColor: 'var(--accent)'
      } : {
        color: 'var(--text-secondary)'
      }}
    >
      <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" style={{ flexShrink: 0 }}>
        {playing ? (
          <rect x="6" y="6" width="12" height="12" />
        ) : (
          <polygon points="8,5 19,12 8,19" />
        )}
      </svg>
      {playing ? 'STOP' : 'PLAY_SAMPLE'}
    </button>
  );
}
