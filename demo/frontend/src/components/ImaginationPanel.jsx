import { useState } from "react";

function FrameImage({ src, label, border }) {
  if (!src) {
    return (
      <div style={{
        width: 400, height: 280,
        background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)",
        border: `1px dashed ${border || "var(--border)"}`,
        display: "flex", alignItems: "center", justifyContent: "center",
        color: "var(--text-muted)", fontFamily: "var(--font-mono)", fontSize: 12,
      }}>
        {label || "No data"}
      </div>
    );
  }
  return (
    <img
      src={`data:image/png;base64,${src}`}
      alt={label}
      style={{
        width: 400, height: 280, objectFit: "fill",
        borderRadius: "var(--radius-sm)",
        border: `1px solid ${border || "var(--border)"}`,
        imageRendering: "pixelated",
      }}
    />
  );
}

function ThumbnailStrip({ frames, currentIdx, offset, onSelect, accent }) {
  if (!frames || !frames.length) return null;
  return (
    <div style={{
      display: "flex", gap: 2, overflowX: "auto", padding: "2px 0",
    }}>
      {frames.map((f, i) => (
        <img
          key={i}
          src={`data:image/png;base64,${f}`}
          alt={`Step ${offset + i}`}
          onClick={() => onSelect(offset + i)}
          style={{
            width: 40, height: 28, objectFit: "fill",
            borderRadius: 2, cursor: "pointer",
            imageRendering: "pixelated",
            border: (offset + i) === currentIdx
              ? `2px solid ${accent || "var(--accent)"}`
              : "1px solid var(--border)",
            opacity: (offset + i) === currentIdx ? 1 : 0.6,
          }}
        />
      ))}
    </div>
  );
}

export default function ImaginationPanel({
  imagineResult,
  currentStep,
  onStepChange,
  loading,
  error,
}) {
  if (error) {
    return (
      <div className="panel" style={{ height: "100%" }}>
        <div className="panel-header">
          <h3>World Model Imagination</h3>
        </div>
        <div className="panel-body" style={{
          display: "flex", alignItems: "center", justifyContent: "center",
          color: "var(--text-muted)", fontFamily: "var(--font-mono)", fontSize: 12,
          minHeight: 300,
        }}>
          Decoder not available — train MAE + bridge first
        </div>
      </div>
    );
  }

  if (loading || !imagineResult) {
    return (
      <div className="panel" style={{ height: "100%" }}>
        <div className="panel-header">
          <h3>World Model Imagination</h3>
        </div>
        <div className="panel-body" style={{
          display: "flex", alignItems: "center", justifyContent: "center",
          color: "var(--text-muted)", fontFamily: "var(--font-mono)", fontSize: 12,
          minHeight: 300,
        }}>
          {loading ? "Imagining..." : "Select a trajectory"}
        </div>
      </div>
    );
  }

  const { ground_truth_frames, imagined_frames, surprise_scores, context_len } = imagineResult;
  const step = currentStep ?? 0;
  const isImagined = step >= context_len;
  const imaginedIdx = step - context_len;

  // Current surprise
  const currentSurprise = isImagined && surprise_scores[imaginedIdx] != null
    ? surprise_scores[imaginedIdx] : null;
  const meanSurprise = surprise_scores.length > 0
    ? surprise_scores.reduce((a, b) => a + b, 0) / surprise_scores.length : 0;

  return (
    <div className="panel" style={{ height: "100%" }}>
      <div className="panel-header">
        <h3>World Model Imagination</h3>
        <div style={{ display: "flex", gap: "var(--gap-md)", alignItems: "center" }}>
          {!isImagined && (
            <span className="badge badge-regime">Context Window</span>
          )}
          {currentSurprise != null && (
            <span className="metric">
              <span className="metric-label">Surprise: </span>
              <span className="metric-value" style={{
                color: currentSurprise > meanSurprise * 1.5 ? "var(--alert)" : "var(--text-primary)"
              }}>
                {currentSurprise.toFixed(3)}
              </span>
            </span>
          )}
        </div>
      </div>

      <div className="panel-body">
        {/* Side-by-side spectrograms */}
        <div style={{
          display: "flex", gap: "var(--gap-md)", justifyContent: "center",
        }}>
          {/* Ground truth */}
          <div>
            <div style={{
              fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text-muted)",
              marginBottom: "var(--gap-xs)", textAlign: "center",
            }}>
              Ground Truth (t={step})
            </div>
            <FrameImage
              src={ground_truth_frames[step]}
              label="Ground Truth"
            />
          </div>

          {/* Imagined */}
          <div>
            <div style={{
              fontSize: 11, fontFamily: "var(--font-mono)",
              color: isImagined ? "var(--accent)" : "var(--text-muted)",
              marginBottom: "var(--gap-xs)", textAlign: "center",
            }}>
              {isImagined ? `Imagined (t=${step})` : "Context — no prediction"}
            </div>
            <FrameImage
              src={isImagined ? imagined_frames[imaginedIdx] : null}
              label={isImagined ? "Imagined" : `Context window (t < ${context_len})`}
              border={isImagined ? "var(--accent-dim)" : "var(--border)"}
            />
          </div>
        </div>

        {/* Thumbnail strips */}
        <div style={{ marginTop: "var(--gap-md)" }}>
          <div style={{
            fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--text-muted)",
            marginBottom: 2,
          }}>
            Ground Truth
          </div>
          <ThumbnailStrip
            frames={ground_truth_frames}
            currentIdx={step}
            offset={0}
            onSelect={onStepChange}
          />
          <div style={{
            fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--accent)",
            marginBottom: 2, marginTop: "var(--gap-xs)",
          }}>
            Imagined (t={context_len}+)
          </div>
          <ThumbnailStrip
            frames={imagined_frames}
            currentIdx={step}
            offset={context_len}
            onSelect={onStepChange}
            accent="var(--accent)"
          />
        </div>

        {/* Mini surprise sparkline */}
        {surprise_scores.length > 0 && (
          <div style={{ marginTop: "var(--gap-sm)" }}>
            <div style={{
              fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--text-muted)",
              marginBottom: 2,
            }}>
              Surprise per step
            </div>
            <div style={{
              display: "flex", gap: 1, alignItems: "flex-end", height: 30,
            }}>
              {surprise_scores.map((s, i) => {
                const maxS = Math.max(...surprise_scores);
                const h = maxS > 0 ? (s / maxS) * 28 + 2 : 2;
                const active = i === imaginedIdx;
                return (
                  <div
                    key={i}
                    onClick={() => onStepChange(context_len + i)}
                    style={{
                      flex: 1, height: h, cursor: "pointer",
                      background: active ? "var(--accent)" : "var(--border-focus)",
                      borderRadius: 1, minWidth: 3,
                      transition: "height 0.15s",
                    }}
                  />
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
