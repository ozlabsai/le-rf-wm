import { useState, useEffect, useRef, useCallback } from "react";
import { api } from "./api/client";
import SpectrogramViewer from "./components/SpectrogramViewer";
import LatentSpace from "./components/LatentSpace";
import SurpriseTimeline from "./components/SurpriseTimeline";
import "./theme.css";

const CURATOR_PICKS = [
  { id: 0, label: "Leader-follower dynamics" },
  { id: 50, label: "Bursty transitions" },
  { id: 100, label: "Quiet scene (baseline)" },
  { id: 200, label: "Interference event" },
  { id: 400, label: "Dense multi-signal" },
];

export default function App() {
  const [trajectoryId, setTrajectoryId] = useState(null);
  const [trajectoryData, setTrajectoryData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [injection, setInjection] = useState(null);
  const [pcaBackground, setPcaBackground] = useState(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [injecting, setInjecting] = useState(false);
  const [error, setError] = useState(null);
  const [regimes, setRegimes] = useState(null);
  const [inputId, setInputId] = useState("");

  const playRef = useRef(null);

  // Load PCA background + regimes on mount
  useEffect(() => {
    api.pcaBackground().then(setPcaBackground).catch(console.error);
    api.stats().then(setRegimes).catch(console.error);
  }, []);

  // Load trajectory
  const loadTrajectory = useCallback(async (id) => {
    setLoading(true);
    setError(null);
    setInjection(null);
    try {
      const [traj, pred] = await Promise.all([
        api.trajectory(id),
        api.predict(id),
      ]);
      setTrajectoryId(id);
      setTrajectoryData(traj);
      setPrediction(pred);
      setCurrentStep(0);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  // Inject anomaly
  const handleInject = useCallback(async (tid, pertType, injectStep) => {
    setInjecting(true);
    try {
      const result = await api.inject(tid, pertType, injectStep);
      setInjection(result);
    } catch (e) {
      setError(e.message);
    } finally {
      setInjecting(false);
    }
  }, []);

  // Playback timer
  useEffect(() => {
    if (playing) {
      playRef.current = setInterval(() => {
        setCurrentStep(s => {
          if (s >= 15) { setPlaying(false); return 15; }
          return s + 1;
        });
      }, 400);
    } else if (playRef.current) {
      clearInterval(playRef.current);
    }
    return () => clearInterval(playRef.current);
  }, [playing]);

  return (
    <div style={{
      display: "flex", flexDirection: "column", height: "100vh",
      maxWidth: 1400, margin: "0 auto", padding: "var(--gap-md)",
    }}>
      {/* Header */}
      <header style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        marginBottom: "var(--gap-md)", padding: "var(--gap-sm) 0",
        borderBottom: "1px solid var(--border)",
      }}>
        <div>
          <h1 style={{
            fontFamily: "var(--font-mono)", fontSize: 18, fontWeight: 600,
            letterSpacing: "0.02em",
          }}>
            <span style={{ color: "var(--accent)" }}>RF-LeWM</span>
            <span style={{ color: "var(--text-muted)", fontWeight: 400 }}> — RF World Model Demo</span>
          </h1>
          <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>
            JEPA-based latent dynamics prediction for RF spectral environments
          </p>
        </div>
        <div style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
          Oz Labs Research
        </div>
      </header>

      {/* Scene selector */}
      <div style={{
        display: "flex", gap: "var(--gap-sm)", marginBottom: "var(--gap-md)",
        alignItems: "center", flexWrap: "wrap",
      }}>
        {CURATOR_PICKS.map(pick => (
          <button
            key={pick.id}
            className={trajectoryId === pick.id ? "primary" : "secondary"}
            onClick={() => loadTrajectory(pick.id)}
            disabled={loading}
          >
            {pick.label}
          </button>
        ))}

        <div style={{ display: "flex", gap: "var(--gap-xs)", alignItems: "center", marginLeft: "auto" }}>
          <input
            type="number"
            min={0}
            max={2998}
            placeholder="ID"
            value={inputId}
            onChange={e => setInputId(e.target.value)}
            onKeyDown={e => e.key === "Enter" && loadTrajectory(Number(inputId))}
            style={{
              width: 80, padding: "4px 8px", fontFamily: "var(--font-mono)", fontSize: 12,
              background: "var(--bg-elevated)", border: "1px solid var(--border)",
              borderRadius: "var(--radius-sm)", color: "var(--text-primary)",
            }}
          />
          <button className="secondary" onClick={() => loadTrajectory(Number(inputId))} disabled={loading || !inputId}>
            Load
          </button>
        </div>

        {trajectoryData && (
          <div style={{ display: "flex", gap: "var(--gap-xs)" }}>
            <span className="badge badge-regime">{trajectoryData.regime}</span>
            <span className="badge badge-regime">{trajectoryData.snr_db} dB</span>
            <span className="badge badge-regime">{trajectoryData.scene_id}</span>
          </div>
        )}
      </div>

      {error && (
        <div style={{
          padding: "var(--gap-sm) var(--gap-md)", marginBottom: "var(--gap-md)",
          background: "var(--alert-dim)", border: "1px solid var(--alert)",
          borderRadius: "var(--radius-sm)", color: "var(--alert)", fontSize: 13,
        }}>
          {error}
        </div>
      )}

      {loading && (
        <div style={{
          textAlign: "center", padding: "var(--gap-xl)",
          color: "var(--text-muted)", fontFamily: "var(--font-mono)",
        }}>
          Loading trajectory...
        </div>
      )}

      {/* Three-panel layout */}
      {prediction && !loading && (
        <div style={{ display: "flex", flexDirection: "column", gap: "var(--gap-md)", flex: 1, minHeight: 0 }}>
          {/* Top row: Spectrogram + Latent Space */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--gap-md)" }}>
            <SpectrogramViewer
              magnitude={trajectoryData?.magnitude}
              patchNorms={prediction?.patch_norms}
              currentStep={currentStep}
              onStepChange={setCurrentStep}
              playing={playing}
              onPlayPause={() => setPlaying(p => !p)}
            />
            <LatentSpace
              pcaActual={prediction?.pca_actual}
              pcaPredicted={prediction?.pca_predicted}
              pcaBackground={pcaBackground?.trajectories}
              currentStep={currentStep}
              deltaCosines={prediction?.delta_cosines}
            />
          </div>

          {/* Bottom: Surprise Timeline */}
          <SurpriseTimeline
            surpriseScores={prediction?.surprise_scores}
            injectionResult={injection}
            onInject={handleInject}
            loading={injecting}
            trajectoryId={trajectoryId}
          />
        </div>
      )}

      {/* Empty state */}
      {!prediction && !loading && (
        <div style={{
          flex: 1, display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center",
          color: "var(--text-muted)",
        }}>
          <p style={{ fontSize: 16, marginBottom: "var(--gap-sm)" }}>
            Select a trajectory to begin
          </p>
          <p style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>
            {regimes ? `${regimes.total} trajectories across ${Object.keys(regimes.regimes).length} RF regimes` : "Connecting to server..."}
          </p>
        </div>
      )}

      {/* Footer */}
      <footer style={{
        marginTop: "auto", padding: "var(--gap-sm) 0",
        borderTop: "1px solid var(--border)",
        display: "flex", justifyContent: "space-between",
        fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)",
      }}>
        <span>RF-LeWM v0 — 16.4M params — Residual JEPA</span>
        <span>
          <a href="https://huggingface.co/OzLabs/RF-LeWM-v0" target="_blank" rel="noreferrer"
             style={{ color: "var(--accent)", textDecoration: "none" }}>Model</a>
          {" · "}
          <a href="https://huggingface.co/datasets/OzLabs/rf-spectral-trajectories" target="_blank" rel="noreferrer"
             style={{ color: "var(--accent)", textDecoration: "none" }}>Dataset</a>
          {" · "}
          <a href="https://github.com/ozlabsai/le-rf-wm" target="_blank" rel="noreferrer"
             style={{ color: "var(--accent)", textDecoration: "none" }}>Code</a>
        </span>
      </footer>
    </div>
  );
}
