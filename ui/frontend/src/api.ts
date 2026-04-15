import type { EvaluationResult, ModelInfo } from './types';

const BASE = '';  // Vite proxy handles /api → localhost:8000

export async function fetchModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${BASE}/api/dashboard/model-info`);
  if (!res.ok) throw new Error(`Failed to fetch model info: ${res.statusText}`);
  return res.json();
}

export async function fetchEvaluation(): Promise<EvaluationResult> {
  const res = await fetch(`${BASE}/api/dashboard/evaluation`);
  if (!res.ok) throw new Error(`Failed to fetch evaluation: ${res.statusText}`);
  return res.json();
}

export function getConfusionMatrixUrl(): string {
  return `${BASE}/api/dashboard/confusion-matrix`;
}

export function getTrainingHistoryImageUrl(): string {
  return `${BASE}/api/dashboard/training-history-image`;
}

export function getWebSocketUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  // In dev, connect directly to the FastAPI server
  return `${proto}//localhost:8000/api/infer/stream`;
}

export function getAudioSampleUrl(classIdx: number): string {
  return `${BASE}/api/dashboard/audio-sample/${classIdx}`;
}
