const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error ${res.status}`);
  }
  return res.json();
}

export const api = {
  health: () => request("/health"),
  stats: () => request("/regimes"),
  trajectory: (id) => request(`/trajectory/${id}`),
  predict: (trajectoryId) =>
    request("/predict", {
      method: "POST",
      body: JSON.stringify({ trajectory_id: trajectoryId }),
    }),
  inject: (trajectoryId, perturbationType, injectStep = 8, strength = 3.0) =>
    request("/inject", {
      method: "POST",
      body: JSON.stringify({
        trajectory_id: trajectoryId,
        perturbation_type: perturbationType,
        inject_step: injectStep,
        strength,
      }),
    }),
  pcaBackground: () => request("/pca_background"),
  imagine: (trajectoryId, contextLen = 4) =>
    request(`/imagine/${trajectoryId}?context_len=${contextLen}`),
  imaginePerturbed: (params) =>
    request("/imagine_perturbed", {
      method: "POST",
      body: JSON.stringify(params),
    }),
};
