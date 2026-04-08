import { useRef, useEffect } from "react";
import * as d3 from "d3";

const REGIME_COLORS = {
  quiet: "#4A6B8A",
  dense: "#8B6B4A",
  bursty: "#7A4A6B",
  ramp_up: "#4A8B6B",
  interference_event: "#8B4A4A",
  correlated_alternating: "#6B8B4A",
  correlated_leader_follower: "#4A6B8B",
  random: "#6B4A8B",
};

export default function LatentSpace({
  pcaActual,        // [[x,y], ...] — 16 points
  pcaPredicted,     // [[x,y], ...] — 16 points
  pcaBackground,    // [{scene_id, regime, points: [[x,y],...]}]
  currentStep,
  deltaCosines,     // [cos_val, ...] for steps 3-15
}) {
  const svgRef = useRef(null);
  const W = 440;
  const H = 340;
  const margin = { top: 20, right: 20, bottom: 30, left: 40 };

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const iw = W - margin.left - margin.right;
    const ih = H - margin.top - margin.bottom;
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    // Collect all points for scale
    const allPoints = [];
    if (pcaBackground) {
      pcaBackground.forEach(traj => traj.points.forEach(p => allPoints.push(p)));
    }
    if (pcaActual) pcaActual.forEach(p => allPoints.push(p));
    if (pcaPredicted) pcaPredicted.forEach(p => allPoints.push(p));

    if (allPoints.length === 0) return;

    const xExt = d3.extent(allPoints, d => d[0]);
    const yExt = d3.extent(allPoints, d => d[1]);
    const xPad = (xExt[1] - xExt[0]) * 0.1 || 1;
    const yPad = (yExt[1] - yExt[0]) * 0.1 || 1;

    const x = d3.scaleLinear()
      .domain([xExt[0] - xPad, xExt[1] + xPad])
      .range([0, iw]);
    const y = d3.scaleLinear()
      .domain([yExt[0] - yPad, yExt[1] + yPad])
      .range([ih, 0]);

    // Axes
    g.append("g")
      .attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(5))
      .selectAll("text").style("fill", "#55576A").style("font-size", "10px");
    g.append("g")
      .call(d3.axisLeft(y).ticks(5))
      .selectAll("text").style("fill", "#55576A").style("font-size", "10px");
    g.selectAll(".domain, .tick line").style("stroke", "#1E2028");

    // Background scatter
    if (pcaBackground) {
      pcaBackground.forEach(traj => {
        const color = REGIME_COLORS[traj.regime] || "#555";
        traj.points.forEach(p => {
          g.append("circle")
            .attr("cx", x(p[0]))
            .attr("cy", y(p[1]))
            .attr("r", 1.5)
            .attr("fill", color)
            .attr("opacity", 0.15);
        });
      });
    }

    // Actual trajectory path
    if (pcaActual && pcaActual.length > 1) {
      const line = d3.line().x(d => x(d[0])).y(d => y(d[1])).curve(d3.curveCardinal);
      g.append("path")
        .datum(pcaActual)
        .attr("d", line)
        .attr("fill", "none")
        .attr("stroke", "#2A7A4B")
        .attr("stroke-width", 1.5)
        .attr("opacity", 0.6);

      pcaActual.forEach((p, i) => {
        const isActive = i === currentStep;
        g.append("circle")
          .attr("cx", x(p[0]))
          .attr("cy", y(p[1]))
          .attr("r", isActive ? 5 : 3)
          .attr("fill", isActive ? "#2A7A4B" : d3.interpolateGreens(0.3 + 0.7 * i / 15))
          .attr("stroke", isActive ? "#E8E9ED" : "none")
          .attr("stroke-width", isActive ? 2 : 0);
      });
    }

    // Predicted trajectory path
    if (pcaPredicted && pcaPredicted.length > 3) {
      const predPoints = pcaPredicted.slice(3); // predictions start after context
      const line = d3.line().x(d => x(d[0])).y(d => y(d[1])).curve(d3.curveCardinal);
      g.append("path")
        .datum(predPoints)
        .attr("d", line)
        .attr("fill", "none")
        .attr("stroke", "var(--accent)")
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "4,3")
        .attr("opacity", 0.6);

      predPoints.forEach((p, i) => {
        const stepIdx = i + 3;
        const isActive = stepIdx === currentStep;
        g.append("circle")
          .attr("cx", x(p[0]))
          .attr("cy", y(p[1]))
          .attr("r", isActive ? 5 : 3)
          .attr("fill", "none")
          .attr("stroke", isActive ? "#E8520A" : d3.interpolateOranges(0.3 + 0.7 * i / 12))
          .attr("stroke-width", isActive ? 2 : 1.5);
      });
    }
  }, [pcaActual, pcaPredicted, pcaBackground, currentStep]);

  // Current delta cosine
  const stepIdx = currentStep != null ? currentStep - 3 : -1;
  const currentDeltaCos = deltaCosines && stepIdx >= 0 && stepIdx < deltaCosines.length
    ? deltaCosines[stepIdx] : null;

  return (
    <div className="panel" style={{ height: "100%" }}>
      <div className="panel-header">
        <h3>Latent Space Dynamics</h3>
        {currentDeltaCos != null && (
          <span className="metric">
            <span className="metric-label">Δ CosSim: </span>
            <span className="metric-value" style={{
              color: currentDeltaCos > 0.3 ? "var(--success)" : "var(--text-secondary)"
            }}>
              {currentDeltaCos.toFixed(3)}
            </span>
          </span>
        )}
      </div>
      <div className="panel-body" style={{ display: "flex", justifyContent: "center" }}>
        <svg ref={svgRef} width={W} height={H} />
      </div>
      <div style={{
        padding: "0 var(--gap-md) var(--gap-sm)",
        display: "flex", gap: "var(--gap-lg)", justifyContent: "center",
      }}>
        <span style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "var(--text-muted)" }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#2A7A4B", display: "inline-block" }} />
          Actual
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "var(--text-muted)" }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", border: "2px solid #E8520A", display: "inline-block" }} />
          Predicted
        </span>
      </div>
    </div>
  );
}
