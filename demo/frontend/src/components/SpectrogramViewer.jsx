import { useRef, useEffect, useCallback } from "react";

// Custom colormap: dark navy → teal → amber → bright yellow
function spectrogramColor(value, min, max) {
  const t = Math.max(0, Math.min(1, (value - min) / (max - min + 1e-8)));
  // 4-stop gradient
  let r, g, b;
  if (t < 0.25) {
    const s = t / 0.25;
    r = 11 + s * (15 - 11);
    g = 16 + s * (43 - 16);
    b = 38 + s * (76 - 38);
  } else if (t < 0.5) {
    const s = (t - 0.25) / 0.25;
    r = 15 + s * (26 - 15);
    g = 43 + s * (107 - 43);
    b = 76 + s * (106 - 76);
  } else if (t < 0.75) {
    const s = (t - 0.5) / 0.25;
    r = 26 + s * (212 - 26);
    g = 107 + s * (160 - 107);
    b = 106 + s * (23 - 106);
  } else {
    const s = (t - 0.75) / 0.25;
    r = 212 + s * (245 - 212);
    g = 160 + s * (230 - 160);
    b = 23 + s * (66 - 23);
  }
  return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
}

function attentionColor(value, min, max) {
  const t = Math.max(0, Math.min(1, (value - min) / (max - min + 1e-8)));
  // Transparent to bright orange
  return `rgba(232, 82, 10, ${t * 0.7})`;
}

function drawSpectrogram(ctx, data, width, height) {
  if (!data || !data.length) return;
  const freqBins = data.length;
  const timeBins = data[0].length;
  const cellW = width / timeBins;
  const cellH = height / freqBins;

  // Find global min/max
  let min = Infinity, max = -Infinity;
  for (let f = 0; f < freqBins; f++) {
    for (let t = 0; t < timeBins; t++) {
      const v = data[f][t];
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }

  for (let f = 0; f < freqBins; f++) {
    for (let t = 0; t < timeBins; t++) {
      ctx.fillStyle = spectrogramColor(data[f][t], min, max);
      // Flip frequency axis (low freq at bottom)
      ctx.fillRect(t * cellW, (freqBins - 1 - f) * cellH, cellW + 0.5, cellH + 0.5);
    }
  }
}

function drawAttentionOverlay(ctx, norms, width, height, freqBins, timeBins) {
  if (!norms || !norms.length) return;
  const nFreq = norms.length;
  const nTime = norms[0].length;
  const cellW = width / nTime;
  const cellH = height / nFreq;

  let min = Infinity, max = -Infinity;
  for (let f = 0; f < nFreq; f++) {
    for (let t = 0; t < nTime; t++) {
      const v = norms[f][t];
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }

  for (let f = 0; f < nFreq; f++) {
    for (let t = 0; t < nTime; t++) {
      ctx.fillStyle = attentionColor(norms[f][t], min, max);
      ctx.fillRect(t * cellW, (nFreq - 1 - f) * cellH, cellW + 0.5, cellH + 0.5);
    }
  }
}

export default function SpectrogramViewer({
  magnitude,      // [16][256][51] — ground truth spectrograms
  patchNorms,     // [16][n_freq][n_time] — attention overlay
  currentStep,
  onStepChange,
  playing,
  onPlayPause,
}) {
  const canvasLeftRef = useRef(null);
  const canvasRightRef = useRef(null);
  const step = currentStep ?? 0;

  const CANVAS_W = 400;
  const CANVAS_H = 280;

  // Draw ground truth spectrogram
  useEffect(() => {
    const canvas = canvasLeftRef.current;
    if (!canvas || !magnitude || !magnitude[step]) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    drawSpectrogram(ctx, magnitude[step], CANVAS_W, CANVAS_H);
  }, [magnitude, step]);

  // Draw spectrogram + attention overlay
  useEffect(() => {
    const canvas = canvasRightRef.current;
    if (!canvas || !magnitude || !magnitude[step]) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    // Base spectrogram (dimmed)
    ctx.globalAlpha = 0.4;
    drawSpectrogram(ctx, magnitude[step], CANVAS_W, CANVAS_H);
    ctx.globalAlpha = 1.0;
    // Attention overlay
    if (patchNorms && patchNorms[step]) {
      drawAttentionOverlay(ctx, patchNorms[step], CANVAS_W, CANVAS_H);
    }
  }, [magnitude, patchNorms, step]);

  return (
    <div className="panel" style={{ height: "100%" }}>
      <div className="panel-header">
        <h3>RF Scene Playback</h3>
        <div style={{ display: "flex", gap: "var(--gap-md)", alignItems: "center" }}>
          <span className="metric">
            <span className="metric-label">Step: </span>
            <span className="metric-value">{step} / 15</span>
          </span>
          <button className="secondary" onClick={onPlayPause}>
            {playing ? "⏸ Pause" : "▶ Play"}
          </button>
        </div>
      </div>

      <div className="panel-body" style={{
        display: "flex", gap: "var(--gap-md)", justifyContent: "center",
      }}>
        {/* Ground truth */}
        <div>
          <div style={{
            fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text-muted)",
            marginBottom: "var(--gap-xs)", textAlign: "center",
          }}>
            Ground Truth
          </div>
          <canvas
            ref={canvasLeftRef}
            width={CANVAS_W}
            height={CANVAS_H}
            style={{ borderRadius: "var(--radius-sm)", border: "1px solid var(--border)" }}
          />
        </div>

        {/* Attention overlay */}
        <div>
          <div style={{
            fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--accent)",
            marginBottom: "var(--gap-xs)", textAlign: "center",
          }}>
            Encoder Attention
          </div>
          <canvas
            ref={canvasRightRef}
            width={CANVAS_W}
            height={CANVAS_H}
            style={{
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--accent-dim)",
            }}
          />
        </div>
      </div>

      {/* Timestep scrubber */}
      <div style={{ padding: "0 var(--gap-md) var(--gap-md)" }}>
        <input
          type="range"
          min={0}
          max={15}
          value={step}
          onChange={e => onStepChange(Number(e.target.value))}
          style={{ width: "100%", accentColor: "var(--accent)" }}
        />
      </div>
    </div>
  );
}
