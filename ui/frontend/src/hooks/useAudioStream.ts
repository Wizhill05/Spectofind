/**
 * useAudioStream — Microphone capture, real-time FFT analysis, and WebSocket streaming.
 *
 * Manages:
 *  1. getUserMedia → AudioContext → AnalyserNode (for spectrogram canvas)
 *  2. ScriptProcessorNode → accumulates PCM → sends over WebSocket every ~3s
 *  3. WebSocket connection to FastAPI for live classification
 */

import { useCallback, useRef, useState } from 'react';
import type { Prediction } from '../types';
import { getWebSocketUrl } from '../api';

const SEND_INTERVAL_MS = 3000;   // send audio to server every 3 seconds
const MAX_WINDOW_SEC = 5;        // max seconds of audio to send at once
const KEEP_OVERLAP_SEC = 2;      // seconds to keep for next window overlap

function mergeFloat32Arrays(arrays: Float32Array[]): Float32Array {
  let totalLength = 0;
  for (const a of arrays) totalLength += a.length;
  const result = new Float32Array(totalLength);
  let offset = 0;
  for (const a of arrays) {
    result.set(a, offset);
    offset += a.length;
  }
  return result;
}

export interface AudioStreamState {
  isRecording: boolean;
  analyser: AnalyserNode | null;
  predictions: Prediction[];
  wsConnected: boolean;
  sendProgress: number; // 0..1 — progress toward next inference send
  start: () => Promise<void>;
  stop: () => void;
}

export function useAudioStream(): AudioStreamState {
  const [isRecording, setIsRecording] = useState(false);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [sendProgress, setSendProgress] = useState(0);

  const audioCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const intervalRef = useRef<number | null>(null);
  const progressIntervalRef = useRef<number | null>(null);
  const lastSendTimeRef = useRef<number>(0);

  const start = useCallback(async () => {
    // Get microphone
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    const audioCtx = new AudioContext();
    const source = audioCtx.createMediaStreamSource(stream);

    // AnalyserNode for real-time FFT (used by SpectrogramCanvas)
    const analyserNode = audioCtx.createAnalyser();
    analyserNode.fftSize = 2048;
    analyserNode.smoothingTimeConstant = 0.3;
    analyserNode.minDecibels = -100;
    analyserNode.maxDecibels = -10;

    // ScriptProcessorNode to capture raw PCM data
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);

    // Mute playback (prevent feedback) with zero-gain node
    const gainNode = audioCtx.createGain();
    gainNode.gain.value = 0;

    source.connect(analyserNode);
    analyserNode.connect(processor);
    processor.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    // WebSocket connection
    const ws = new WebSocket(getWebSocketUrl());
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      setWsConnected(true);
      // Send config message with browser's sample rate
      ws.send(JSON.stringify({ sampleRate: audioCtx.sampleRate }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.predictions) {
          setPredictions(data.predictions);
        }
      } catch (e) {
        console.error('Failed to parse WS message:', e);
      }
    };

    ws.onclose = () => setWsConnected(false);
    ws.onerror = () => setWsConnected(false);

    // Capture PCM audio chunks
    chunksRef.current = [];
    processor.onaudioprocess = (e) => {
      const data = e.inputBuffer.getChannelData(0);
      chunksRef.current.push(new Float32Array(data));
    };

    // Periodically send accumulated audio over WebSocket
    lastSendTimeRef.current = Date.now();
    const interval = window.setInterval(() => {
      if (chunksRef.current.length === 0 || !ws || ws.readyState !== WebSocket.OPEN) {
        return;
      }

      const merged = mergeFloat32Arrays(chunksRef.current);
      const maxSamples = Math.floor(audioCtx.sampleRate * MAX_WINDOW_SEC);
      const startIdx = Math.max(0, merged.length - maxSamples);
      const audioWindow = merged.slice(startIdx);

      // Send as raw Float32 binary
      ws.send(audioWindow.buffer);
      lastSendTimeRef.current = Date.now();

      // Keep overlap for next window
      const keepSamples = Math.floor(audioCtx.sampleRate * KEEP_OVERLAP_SEC);
      const keepFrom = Math.max(0, merged.length - keepSamples);
      chunksRef.current = [merged.slice(keepFrom)];
    }, SEND_INTERVAL_MS);

    // Progress timer (updates ~20x/sec for smooth bar)
    const progressInterval = window.setInterval(() => {
      const elapsed = Date.now() - lastSendTimeRef.current;
      setSendProgress(Math.min(elapsed / SEND_INTERVAL_MS, 1));
    }, 50);

    // Store refs
    audioCtxRef.current = audioCtx;
    streamRef.current = stream;
    wsRef.current = ws;
    processorRef.current = processor;
    gainRef.current = gainNode;
    intervalRef.current = interval;
    progressIntervalRef.current = progressInterval;

    setAnalyser(analyserNode);
    setIsRecording(true);
  }, []);

  const stop = useCallback(() => {
    // Clear intervals
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (progressIntervalRef.current !== null) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }

    // Disconnect audio nodes
    processorRef.current?.disconnect();
    gainRef.current?.disconnect();

    // Stop mic stream tracks
    streamRef.current?.getTracks().forEach((track) => track.stop());

    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    // Close AudioContext
    audioCtxRef.current?.close();

    // Reset state
    setIsRecording(false);
    setAnalyser(null);
    setWsConnected(false);
    setSendProgress(0);
    chunksRef.current = [];
  }, []);

  return { isRecording, analyser, predictions, wsConnected, sendProgress, start, stop };
}
