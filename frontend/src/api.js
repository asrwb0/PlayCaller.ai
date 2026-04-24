const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function post(path, body) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.detail ?? `HTTP ${res.status}`);
  }

  return data;
}

async function get(path) {
  const res = await fetch(`${BASE_URL}${path}`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
  return data;
}

/**
 * Predict fantasy points for a player.
 * @param {string} playerName
 * @param {{ season?: number, week?: number }} opts
 * @returns {{ player, position, team, predicted_points, low, high, range, recent_avg }}
 */
export async function predictPlayer(playerName, opts = {}) {
  return post("/predict", { player_name: playerName, ...opts });
}

/**
 * Get injury risk for a player.
 * @param {string} playerName
 * @param {{ season?: number, week?: number }} opts
 * @returns {{ risk_tier, risk_pct, risk_color, recommendation, key_flags,
 *             recent_avg_snap, season_carries }}
 */
export async function getInjuryRisk(playerName, opts = {}) {
  return post("/injury", { player_name: playerName, ...opts });
}

/**
 * Evaluate a fantasy trade.
 * @param {string[]} giving
 * @param {string[]} receiving
 * @returns {{ verdict, net_value_swing, summary,
 *             giving_total, receiving_total,
 *             giving_players, receiving_players, warnings }}
 */
export async function evaluateTrade(giving, receiving) {
  return post("/trade/evaluate", { giving, receiving });
}

/**
 * Autocomplete player name search.
 * @param {string} query
 * @param {number} limit
 * @returns {string[]}
 */
export async function searchPlayers(query, limit = 10) {
  const data = await get(`/players/search/${encodeURIComponent(query)}?limit=${limit}`);
  return data.results ?? [];
}

/**
 * Get both projection + injury for one player in a single call.
 * @param {string} playerName
 * @returns {{ projection: {...}, injury: {...} | null }}
 */
export async function getPlayerFull(playerName) {
  return post("/player/full", { player_name: playerName });
}

/**
 * Ping the backend health endpoint.
 * @returns {{ status: "ok", models: "loaded" }}
 */
export async function checkHealth() {
  return get("/health");
}