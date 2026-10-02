// Sin VITE_API_URL usa mismo origen (/api -> nginx proxy). VITE_API_URL solo para `npm run dev`.
const BASE = import.meta.env.VITE_API_URL || "/api";

async function json(res) {
  if (!res.ok) {
    const err = new Error(`HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

export function predict(deporte, sueno) {
  return fetch(`${BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ deporte, sueno }),
  }).then(json);
}

export function health() {
  return fetch(`${BASE}/health`).then(json);
}

export function metrics() {
  return fetch(`${BASE}/metrics`).then(json);
}
