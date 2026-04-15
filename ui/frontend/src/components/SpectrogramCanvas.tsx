/**
 * SpectrogramCanvas — real-time scrolling spectrogram drawn from an AnalyserNode.
 *
 * Uses requestAnimationFrame to draw one column per frame (60fps).
 * The canvas scrolls left, painting new FFT data on the right edge.
 * Uses a viridis-like colormap for visual consistency with the training spectrograms.
 */

import { useEffect, useRef } from 'react';

// ── Viridis colormap (20 key stops) ─────────────────────────────────────────
const VIRIDIS: [number, number, number][] = [
  [68, 1, 84],
  [72, 20, 103],
  [72, 38, 119],
  [69, 55, 129],
  [63, 71, 136],
  [56, 87, 140],
  [49, 104, 142],
  [42, 118, 142],
  [35, 132, 141],
  [31, 148, 139],
  [30, 163, 136],
  [34, 177, 132],
  [53, 191, 122],
  [82, 204, 107],
  [119, 215, 86],
  [161, 224, 59],
  [195, 229, 39],
  [224, 228, 24],
  [243, 229, 30],
  [253, 231, 37],
];

function viridisColor(t: number): string {
  // t in [0, 1]
  const clamped = Math.max(0, Math.min(1, t));
  const idx = clamped * (VIRIDIS.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, VIRIDIS.length - 1);
  const frac = idx - lo;
  const r = Math.round(VIRIDIS[lo][0] + (VIRIDIS[hi][0] - VIRIDIS[lo][0]) * frac);
  const g = Math.round(VIRIDIS[lo][1] + (VIRIDIS[hi][1] - VIRIDIS[lo][1]) * frac);
  const b = Math.round(VIRIDIS[lo][2] + (VIRIDIS[hi][2] - VIRIDIS[lo][2]) * frac);
  return `rgb(${r},${g},${b})`;
}

interface SpectrogramCanvasProps {
  analyser: AnalyserNode | null;
  isRecording: boolean;
}

export default function SpectrogramCanvas({ analyser, isRecording }: SpectrogramCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !analyser || !isRecording) return;

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    // Set canvas resolution to actual pixel size
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;

    const W = canvas.width;
    const H = canvas.height;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    // Fill with black initially
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, W, H);

    function draw() {
      rafRef.current = requestAnimationFrame(draw);
      analyser!.getByteFrequencyData(dataArray);

      // Shift existing image left by 2 pixels (faster scrolling)
      const imageData = ctx!.getImageData(2, 0, W - 2, H);
      ctx!.putImageData(imageData, 0, 0);

      // Draw new column on the right edge (2px wide for visibility)
      // Only use the lower ~60% of frequency bins (most environmental sounds are < 8kHz)
      const useBins = Math.floor(bufferLength * 0.6);
      for (let y = 0; y < H; y++) {
        // Map canvas y (top=high freq, bottom=low freq) to frequency bin
        const freqIdx = Math.floor((1 - y / H) * useBins);
        const value = dataArray[Math.min(freqIdx, bufferLength - 1)] / 255;
        const color = viridisColor(value);
        ctx!.fillStyle = color;
        ctx!.fillRect(W - 2, y, 2, 1);
      }
    }

    draw();

    return () => {
      cancelAnimationFrame(rafRef.current);
    };
  }, [analyser, isRecording]);

  // When not recording, show dark idle state
  useEffect(() => {
    if (!isRecording && canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        const rect = canvasRef.current.getBoundingClientRect();
        canvasRef.current.width = rect.width * window.devicePixelRatio;
        canvasRef.current.height = rect.height * window.devicePixelRatio;
        ctx.fillStyle = '#0a0a0f';
        ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);

        // Draw placeholder text
        ctx.fillStyle = 'rgba(255,255,255,0.15)';
        ctx.font = `${14 * window.devicePixelRatio}px Inter, sans-serif`;
        ctx.textAlign = 'center';
        ctx.fillText(
          'Spectrogram will appear here when recording',
          canvasRef.current.width / 2,
          canvasRef.current.height / 2,
        );
      }
    }
  }, [isRecording]);

  return (
    <div className="spectrogram-container">
      <canvas ref={canvasRef} />
      {isRecording && (
        <>
          <span className="spectrogram-label">Frequency ↑</span>
          <span className="spectrogram-label spectrogram-label--right">Time →</span>
        </>
      )}
    </div>
  );
}
