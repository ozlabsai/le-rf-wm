import { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Area, ComposedChart,
} from "recharts";

const PERTURBATION_TYPES = [
  { value: "noise_burst", label: "Noise Burst", status: "detected" },
  { value: "signal_injection", label: "Signal Injection", status: "blind" },
  { value: "signal_dropout", label: "Signal Dropout", status: "blind" },
  { value: "frequency_shift", label: "Frequency Shift", status: "unknown" },
  { value: "temporal_reversal", label: "Temporal Reversal", status: "blind" },
];

function StatusBadge({ status }) {
  const cls = status === "detected" ? "badge-detected"
    : status === "blind" ? "badge-not-detected"
    : "badge-regime";
  const label = status === "detected" ? "Detectable"
    : status === "blind" ? "Known blind spot"
    : "Untested";
  return <span className={`badge ${cls}`}>{label}</span>;
}

export default function SurpriseTimeline({
  surpriseScores,
  injectionResult,
  onInject,
  loading,
  trajectoryId,
}) {
  const [pertType, setPertType] = useState("noise_burst");
  const [injectStep, setInjectStep] = useState(8);

  // Build chart data
  const chartData = [];
  const normal = surpriseScores || [];
  const perturbed = injectionResult?.surprise_perturbed || [];

  const maxLen = Math.max(normal.length, perturbed.length);
  for (let i = 0; i < maxLen; i++) {
    const step = i + 3; // steps start after history_size=3
    const d = { step, timestep: `t${step}` };
    if (i < normal.length) d.normal = normal[i];
    if (i < perturbed.length) d.perturbed = perturbed[i];
    chartData.push(d);
  }

  // Mean surprise for reference line
  const meanSurprise = normal.length > 0
    ? normal.reduce((a, b) => a + b, 0) / normal.length
    : 0;

  const selectedPert = PERTURBATION_TYPES.find(p => p.value === pertType);

  return (
    <div className="panel" style={{ height: "100%" }}>
      <div className="panel-header">
        <h3>Surprise Score Timeline</h3>
        {injectionResult && (
          <div style={{ display: "flex", gap: "var(--gap-md)", alignItems: "center" }}>
            <span className="metric">
              <span className="metric-label">Ratio: </span>
              <span className="metric-value" style={{
                color: injectionResult.surprise_ratio > 1.2 ? "var(--alert)" : "var(--text-muted)"
              }}>
                {injectionResult.surprise_ratio.toFixed(2)}×
              </span>
            </span>
            <span className={`badge ${injectionResult.detected ? "badge-detected" : "badge-not-detected"}`}>
              {injectionResult.detected ? "DETECTED" : "NOT DETECTED"}
            </span>
          </div>
        )}
      </div>

      <div className="panel-body">
        {/* Chart */}
        <div style={{ width: "100%", height: 220 }}>
          <ResponsiveContainer>
            <ComposedChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis
                dataKey="timestep"
                tick={{ fill: "var(--text-muted)", fontSize: 11, fontFamily: "var(--font-mono)" }}
                stroke="var(--border)"
              />
              <YAxis
                tick={{ fill: "var(--text-muted)", fontSize: 11, fontFamily: "var(--font-mono)" }}
                stroke="var(--border)"
                label={{ value: "Surprise", angle: -90, position: "insideLeft",
                         fill: "var(--text-muted)", fontSize: 11 }}
              />
              <Tooltip
                contentStyle={{
                  background: "var(--bg-elevated)", border: "1px solid var(--border)",
                  borderRadius: "var(--radius-sm)", fontFamily: "var(--font-mono)", fontSize: 12,
                }}
                labelStyle={{ color: "var(--text-secondary)" }}
              />
              {meanSurprise > 0 && (
                <ReferenceLine
                  y={meanSurprise}
                  stroke="var(--text-muted)"
                  strokeDasharray="4 4"
                  label={{ value: "mean", fill: "var(--text-muted)", fontSize: 10 }}
                />
              )}
              <Line
                type="monotone"
                dataKey="normal"
                stroke="var(--success)"
                strokeWidth={2}
                dot={{ r: 3, fill: "var(--success)" }}
                name="Normal"
              />
              {perturbed.length > 0 && (
                <Line
                  type="monotone"
                  dataKey="perturbed"
                  stroke="var(--alert)"
                  strokeWidth={2}
                  dot={{ r: 3, fill: "var(--alert)" }}
                  name="Perturbed"
                  strokeDasharray="5 3"
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* Injection controls */}
        <div style={{
          marginTop: "var(--gap-md)",
          display: "flex", gap: "var(--gap-md)", alignItems: "center",
          flexWrap: "wrap",
        }}>
          <div style={{ display: "flex", gap: "var(--gap-sm)", alignItems: "center" }}>
            <label className="metric-label">Perturbation:</label>
            <select value={pertType} onChange={e => setPertType(e.target.value)}>
              {PERTURBATION_TYPES.map(p => (
                <option key={p.value} value={p.value}>{p.label}</option>
              ))}
            </select>
            <StatusBadge status={selectedPert?.status} />
          </div>

          <div style={{ display: "flex", gap: "var(--gap-sm)", alignItems: "center" }}>
            <label className="metric-label">Inject at:</label>
            <select value={injectStep} onChange={e => setInjectStep(Number(e.target.value))}>
              {[4, 5, 6, 7, 8, 9, 10, 11, 12].map(s => (
                <option key={s} value={s}>t{s}</option>
              ))}
            </select>
          </div>

          <button
            className="primary"
            onClick={() => onInject(trajectoryId, pertType, injectStep)}
            disabled={loading || trajectoryId == null}
          >
            {loading ? "Running..." : "Inject Anomaly"}
          </button>
        </div>

        {/* Honest framing */}
        {selectedPert?.status === "blind" && (
          <div style={{
            marginTop: "var(--gap-sm)",
            padding: "var(--gap-sm) var(--gap-md)",
            background: "var(--bg-elevated)",
            borderRadius: "var(--radius-sm)",
            borderLeft: "3px solid var(--warning)",
            fontSize: 12, color: "var(--text-secondary)",
          }}>
            Current model blind spot — structural perturbations are not yet detected.
            Under active development in v2 (token-level prediction).
          </div>
        )}
      </div>
    </div>
  );
}
