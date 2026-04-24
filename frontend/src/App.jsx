import { useState, useRef, useEffect } from "react";
import {
  evaluateTrade,
  getPlayerFull,
} from "./api";

const OPENAI_API_KEY = import.meta.env.VITE_OPENAI_KEY;

const MODEL_MAP = {
  "GPT-4o":       "gpt-4o",
  "GPT-4o-Mini":  "gpt-4o-mini",
};

const MODELS   = Object.keys(MODEL_MAP);
const PERSONAS = ["Analyst", "Coach", "Stathead", "Skeptic", "Insider", "Casual"];
const STARTERS = [
  "Should I start Josh Allen or Lamar Jackson?",
  "Best waiver wire pickups this week?",
  "Analyze my running back depth",
  "Trade advice: Tyreek for two flex players?",
  "Who are the best sleepers in PPR?",
];

const PERSONA_DESCRIPTIONS = {
  Analyst:  "Analytical, data-driven, cite trends.",
  Coach:    "Strategic coaching mindset, game-plan focused.",
  Stathead: "Deep stat dives, advanced metrics.",
  Skeptic:  "Devil's advocate, challenge hype.",
  Insider:  "Insider tone, locker room vibes.",
  Casual:   "Friendly, conversational, jargon-light.",
};

function buildSystemPrompt(persona, focus, depth, creativity) {
  const depthDesc =
    depth < 30 ? "brief and punchy" :
    depth < 60 ? "moderately detailed" :
    "deeply comprehensive";

  const creativityDesc = {
    "Fact-Based": "strictly factual, cite real statistics where possible",
    "Neutral":    "balanced between facts and insight",
    "Creative":   "creative, opinionated, and entertaining",
  }[creativity] ?? "balanced";

  return `You are a fantasy football assistant with the persona of a "${persona}". ${PERSONA_DESCRIPTIONS[persona]}

Your response style:
- Focus: ${focus} — prioritize ${focus === "Statistics" ? "stats and numbers" : focus === "Narrative" ? "storytelling and reasoning" : "a mix of stats and narrative"}
- Depth: Be ${depthDesc} (depth level ${depth}/100)
- Creativity: Be ${creativityDesc}

You have access to LIVE ML model predictions injected into this conversation as tool results.
When ML data is provided, prioritize it and reference specific numbers (predicted_points, risk_tier, trade verdict).
Give helpful, confident fantasy football advice. Stay in character as the "${persona}".`;
}

function formatTime(d) {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function extractPlayerNames(text) {
  const matches = text.match(/[A-Z][a-z]+ [A-Z][a-z]+/g) ?? [];
  return [...new Set(matches)];
}

function detectIntent(text) {
  const t = text.toLowerCase();
  if (/trade|swap|deal|give|receiv/.test(t))  return "trade";
  if (/injur|risk|hurt|miss|health/.test(t))  return "injury";
  if (/start|predict|points|project|score/.test(t)) return "predict";
  return "general";
}

const RISK_COLOR_MAP = {
  green:  "#4af083",
  yellow: "#f0c45a",
  orange: "#f97316",
  red:    "#f87171",
};

const STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=Geist+Mono:wght@300;400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #080a0e;
    --sidebar-bg: #0d0f14;
    --surface: #111318;
    --surface2: #181b22;
    --border: #232630;
    --border2: #2e3240;
    --accent: #4af083;
    --accent2: #2dba60;
    --accent-dim: #1a3d2a;
    --gold: #f0c45a;
    --gold-dim: #3a2e0e;
    --text: #e8eaf0;
    --text-muted: #606470;
    --text-mid: #9399a8;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'Geist Mono', monospace;
    --radius: 12px;
    --sidebar-w: 260px;
  }

  html, body, #root { height: 100%; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-mono);
    height: 100vh;
    overflow: hidden;
  }

  .app { display: flex; height: 100vh; overflow: hidden; }

  .sidebar {
    width: var(--sidebar-w);
    min-width: var(--sidebar-w);
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    overflow-x: hidden;
  }
  .sidebar::-webkit-scrollbar { width: 3px; }
  .sidebar::-webkit-scrollbar-thumb { background: var(--border2); }

  .sidebar-logo {
    padding: 20px 18px 14px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .logo-mark {
    width: 30px; height: 30px;
    background: linear-gradient(135deg, var(--accent2), var(--accent));
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
  }
  .logo-text {
    font-family: var(--font-head);
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.02em;
    line-height: 1.2;
    color: var(--text);
  }
  .logo-text span {
    display: block;
    font-size: 9px;
    font-weight: 400;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 1px;
  }


  .sidebar-section { padding: 14px 16px 6px; }
  .sidebar-label {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
    display: block;
  }

  .ml-card {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
    font-size: 11.5px;
  }
  .ml-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }
  .ml-card-title { font-family: var(--font-head); font-size: 13px; font-weight: 600; }
  .ml-card-badge {
    font-size: 9px;
    padding: 3px 8px;
    border-radius: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border: 1px solid;
  }
  .ml-card-row { display: flex; justify-content: space-between; margin-bottom: 4px; }
  .ml-card-label { color: var(--text-muted); }
  .ml-card-value { color: var(--text); font-weight: 500; }
  .ml-card-value.accent { color: var(--accent); }
  .ml-card-value.gold   { color: var(--gold); }
  .ml-card .flags { margin-top: 8px; font-size: 10.5px; color: var(--text-muted); line-height: 1.7; }

  .trade-verdict {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 10px;
    border-radius: 8px;
    margin-top: 10px;
    font-family: var(--font-head);
    font-size: 13px;
    font-weight: 700;
    border: 1px solid;
  }

  .persona-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; }
  .persona-btn {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text-mid);
    font-family: var(--font-mono);
    font-size: 10.5px;
    padding: 7px 6px;
    border-radius: 8px;
    cursor: pointer;
    text-align: center;
    transition: all 0.14s;
    letter-spacing: 0.02em;
  }
  .persona-btn:hover { border-color: var(--accent2); color: var(--text); }
  .persona-btn.active {
    background: var(--accent-dim);
    border-color: var(--accent2);
    color: var(--accent);
    font-weight: 500;
  }

  .radio-group { display: flex; flex-direction: column; gap: 4px; }
  .radio-option {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 10px;
    border-radius: 8px;
    cursor: pointer;
    border: 1px solid transparent;
    transition: all 0.14s;
  }
  .radio-option:hover { background: var(--surface2); }
  .radio-option.active { background: var(--surface2); border-color: var(--border2); }
  .radio-dot {
    width: 14px; height: 14px;
    border-radius: 50%;
    border: 1.5px solid var(--border2);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    transition: all 0.14s;
  }
  .radio-option.active .radio-dot { border-color: var(--accent2); }
  .radio-dot-inner {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    opacity: 0;
    transition: opacity 0.14s;
  }
  .radio-option.active .radio-dot-inner { opacity: 1; }
  .radio-label { font-size: 11px; color: var(--text-mid); transition: color 0.14s; }
  .radio-option.active .radio-label { color: var(--text); }

  .slider-wrap { padding: 0 2px; }
  .slider-value {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 8px;
  }
  .slider-val-num { font-size: 18px; font-family: var(--font-head); font-weight: 700; color: var(--accent); }
  .slider-val-label { font-size: 9px; color: var(--text-muted); letter-spacing: 0.08em; text-transform: uppercase; }
  input[type=range] {
    -webkit-appearance: none;
    width: 100%; height: 3px;
    background: var(--border2);
    border-radius: 2px;
    outline: none;
    cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent2);
    cursor: pointer;
  }

  .select-slider { display: flex; gap: 4px; }
  .cs-btn {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 9.5px;
    padding: 6px 4px;
    border-radius: 7px;
    cursor: pointer;
    text-align: center;
    transition: all 0.14s;
    letter-spacing: 0.02em;
  }
  .cs-btn:hover { border-color: var(--gold); color: var(--text); }
  .cs-btn.active { background: var(--gold-dim); border-color: var(--gold); color: var(--gold); }

  .checkbox-row {
    display: flex; align-items: center; gap: 9px;
    cursor: pointer; padding: 8px 2px;
  }
  .checkbox-box {
    width: 16px; height: 16px;
    border-radius: 4px;
    border: 1.5px solid var(--border2);
    display: flex; align-items: center; justify-content: center;
    transition: all 0.14s;
    flex-shrink: 0;
  }
  .checkbox-box.checked { background: var(--accent-dim); border-color: var(--accent2); }
  .checkbox-check { font-size: 10px; color: var(--accent); opacity: 0; transition: opacity 0.14s; }
  .checkbox-box.checked .checkbox-check { opacity: 1; }
  .checkbox-label { font-size: 11px; color: var(--text-mid); }

  .model-select {
    width: 100%;
    background: var(--surface2);
    border: 1px solid var(--border2);
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 11px;
    padding: 7px 10px;
    border-radius: 8px;
    outline: none;
    cursor: pointer;
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath fill='%23606470' d='M0 0l5 6 5-6z'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 10px center;
    padding-right: 28px;
    margin-top: 4px;
  }
  .model-select:focus { border-color: var(--accent2); }
  .model-select option { background: #12141a; }

  .sidebar-divider { height: 1px; background: var(--border); margin: 6px 16px; }

  .sidebar-footer {
    margin-top: auto;
    padding: 14px 16px;
    border-top: 1px solid var(--border);
  }
  .active-config { font-size: 10px; color: var(--text-muted); line-height: 1.7; }
  .active-config strong { color: var(--accent); font-weight: 500; }

  .chat-main { flex: 1; display: flex; flex-direction: column; overflow: hidden; background: var(--bg); }

  .chat-header {
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
    background: var(--surface);
  }
  .chat-title { font-family: var(--font-head); font-size: 16px; font-weight: 600; letter-spacing: 0.01em; }
  .chat-subtitle { font-size: 10px; color: var(--text-muted); letter-spacing: 0.08em; text-transform: uppercase; margin-top: 2px; }
  .persona-badge {
    font-size: 10px;
    font-family: var(--font-mono);
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid var(--accent2);
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  .messages { flex: 1; overflow-y: auto; padding: 20px 24px; display: flex; flex-direction: column; gap: 6px; }
  .messages::-webkit-scrollbar { width: 4px; }
  .messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .welcome {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    flex: 1; text-align: center; gap: 14px; padding: 40px 20px;
  }
  .welcome-icon { font-size: 44px; line-height: 1; filter: drop-shadow(0 0 20px #4af08344); }
  .welcome h2 { font-family: var(--font-head); font-size: 26px; font-weight: 700; letter-spacing: -0.02em; }
  .welcome p { font-size: 12px; color: var(--text-muted); max-width: 360px; line-height: 1.75; }
  .starter-chips { display: flex; flex-wrap: wrap; gap: 7px; justify-content: center; margin-top: 6px; }
  .chip {
    background: var(--surface2);
    border: 1px solid var(--border2);
    color: var(--text-mid);
    font-family: var(--font-mono);
    font-size: 11px;
    padding: 7px 14px;
    border-radius: 20px;
    cursor: pointer;
    transition: all 0.14s;
  }
  .chip:hover { border-color: var(--accent2); color: var(--accent); background: var(--accent-dim); }

  .msg { display: flex; gap: 11px; animation: fadeUp 0.2s ease-out both; padding: 4px 0; }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(5px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .msg.user { flex-direction: row-reverse; }
  .msg-avatar {
    width: 28px; height: 28px;
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .msg.user .msg-avatar { background: #1a1e2a; border: 1px solid var(--border2); }
  .msg.ai   .msg-avatar { background: var(--accent-dim); border: 1px solid var(--accent2); color: var(--accent); }

  .msg-body { max-width: 80%; }
  .msg.user .msg-body { display: flex; flex-direction: column; align-items: flex-end; }
  .msg-name { font-size: 9.5px; color: var(--text-muted); letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 4px; }
  .msg.ai .msg-name { color: var(--accent2); }
  .msg-bubble { padding: 10px 14px; border-radius: 12px; font-size: 13px; line-height: 1.65; white-space: pre-wrap; word-break: break-word; }
  .msg.user .msg-bubble { background: var(--surface2); border: 1px solid var(--border2); border-bottom-right-radius: 3px; color: var(--text); }
  .msg.ai   .msg-bubble { background: var(--surface);  border: 1px solid var(--border);  border-bottom-left-radius: 3px; color: #d8dce8; }
  .msg-time { font-size: 9.5px; color: var(--text-muted); margin-top: 4px; letter-spacing: 0.04em; }

  .msg-ml-card {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 10px;
    padding: 12px 14px;
    margin-top: 10px;
    font-size: 11.5px;
    max-width: 340px;
  }
  .msg-ml-card .card-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 8px;
  }
  .msg-ml-card .card-title { font-family: var(--font-head); font-size: 13px; font-weight: 600; }
  .msg-ml-card .badge {
    font-size: 9px; padding: 3px 8px; border-radius: 12px;
    letter-spacing: 0.08em; text-transform: uppercase; border: 1px solid;
  }
  .msg-ml-card .row { display: flex; justify-content: space-between; margin-bottom: 4px; }
  .msg-ml-card .lbl { color: var(--text-muted); }
  .msg-ml-card .val { color: var(--text); font-weight: 500; }
  .msg-ml-card .val.g { color: var(--accent); }
  .msg-ml-card .val.y { color: var(--gold); }
  .msg-ml-card .val.r { color: #f87171; }
  .msg-ml-card .flags { margin-top: 8px; font-size: 10px; color: var(--text-muted); line-height: 1.7; }

  .typing-dots {
    display: flex; gap: 4px; align-items: center;
    padding: 10px 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    border-bottom-left-radius: 3px;
    width: fit-content;
  }
  .typing-dots span {
    width: 5px; height: 5px;
    background: var(--accent2);
    border-radius: 50%;
    animation: tdot 1.2s infinite;
    opacity: 0.4;
  }
  .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
  .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes tdot {
    0%,80%,100% { transform: scale(0.7); opacity: 0.3; }
    40%          { transform: scale(1);   opacity: 1; }
  }

  .input-area { padding: 14px 24px 18px; border-top: 1px solid var(--border); flex-shrink: 0; background: var(--surface); }
  .error-msg {
    font-size: 11.5px; color: #f87171;
    background: #1a0808; border: 1px solid #3d1515;
    border-radius: 8px; padding: 7px 12px; margin-bottom: 10px;
  }
  .input-row {
    display: flex; gap: 9px; align-items: flex-end;
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    padding: 9px 12px;
    transition: border-color 0.15s;
  }
  .input-row:focus-within { border-color: var(--accent2); }
  .input-row textarea {
    flex: 1;
    background: none; border: none; outline: none;
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 13px; line-height: 1.55;
    resize: none; max-height: 130px; min-height: 22px;
  }
  .input-row textarea::placeholder { color: var(--text-muted); }
  .send-btn {
    background: var(--accent2);
    border: none; border-radius: 8px;
    width: 32px; height: 32px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; transition: all 0.14s;
    flex-shrink: 0; color: #080a0e; font-size: 15px; font-weight: bold;
  }
  .send-btn:hover:not(:disabled) { background: var(--accent); transform: scale(1.06); }
  .send-btn:disabled { opacity: 0.3; cursor: not-allowed; }
  .input-hint { font-size: 10px; color: var(--text-muted); margin-top: 7px; text-align: center; letter-spacing: 0.04em; }
`;

function SourceBadge({ source }) {
  const isML = source === "ml_model";
  return (
    <span style={{
      fontSize: 9,
      padding: "2px 7px",
      borderRadius: 10,
      border: `1px solid ${isML ? "var(--accent2)" : "#f0c45a55"}`,
      color: isML ? "var(--accent)" : "var(--gold)",
      letterSpacing: "0.08em",
      textTransform: "uppercase",
      marginLeft: 6,
      verticalAlign: "middle",
    }}>
      {isML ? "⚙ ML Model" : "✦ GPT"}
    </span>
  );
}

function ProjectionCard({ data }) {
  if (!data) return null;
  return (
    <div className="msg-ml-card">
      <div className="card-header">
        <span className="card-title">
          {data.player}
          <SourceBadge source={data._source} />
        </span>
        <span className="badge" style={{ borderColor: "#4af08344", color: "var(--accent)" }}>
          {data.position} · {data.team}
        </span>
      </div>
      <div className="row"><span className="lbl">Projected pts</span><span className="val g">{data.predicted_points}</span></div>
      <div className="row"><span className="lbl">Range (80% CI)</span><span className="val">{data.range}</span></div>
      <div className="row"><span className="lbl">Recent avg</span><span className="val">{data.recent_avg}</span></div>
    </div>
  );
}

function InjuryCard({ data }) {
  if (!data) return null;
  const color = RISK_COLOR_MAP[data.risk_color] ?? "var(--text-mid)";
  return (
    <div className="msg-ml-card">
      <div className="card-header">
        <span className="card-title">
          Injury Risk
          <SourceBadge source={data._source} />
        </span>
        <span className="badge" style={{ borderColor: color, color }}>{data.risk_tier}</span>
      </div>
      <div className="row"><span className="lbl">Risk %</span><span className="val" style={{ color }}>{data.risk_pct}</span></div>
      <div className="row"><span className="lbl">Snap share</span><span className="val">{data.recent_avg_snap != null ? `${Math.round(data.recent_avg_snap * 100)}%` : "—"}</span></div>
      <div className="row"><span className="lbl">Season carries</span><span className="val">{data.season_carries ?? "—"}</span></div>
      {data.key_flags?.length > 0 && (
        <div className="flags">⚑ {data.key_flags.join(" · ")}</div>
      )}
      <div style={{ marginTop: 8, fontSize: 11, color: "var(--text-mid)" }}>{data.recommendation}</div>
    </div>
  );
}

function TradeCard({ data }) {
  if (!data) return null;
  const swing  = data.net_value_swing;
  const isGood = swing >= 0;
  const color  = isGood ? "var(--accent)" : "#f87171";
  return (
    <div className="msg-ml-card" style={{ maxWidth: 400 }}>
      <div className="card-header">
        <span className="card-title">
          Trade Analysis
          <SourceBadge source={data._source} />
        </span>
        <span className="badge" style={{ borderColor: color, color }}>{data.verdict}</span>
      </div>
      <div className="row">
        <span className="lbl">You give</span>
        <span className="val">{data.giving_players?.map(p => `${p.player} (${p.value})`).join(", ") ?? "—"}</span>
      </div>
      <div className="row">
        <span className="lbl">You receive</span>
        <span className="val">{data.receiving_players?.map(p => `${p.player} (${p.value})`).join(", ") ?? "—"}</span>
      </div>
      <div className="row">
        <span className="lbl">Net swing</span>
        <span className="val" style={{ color }}>{isGood ? "+" : ""}{swing?.toFixed(1)}</span>
      </div>
      {data.warnings?.length > 0 && (
        <div className="flags">⚠ {data.warnings.join(" · ")}</div>
      )}
      <div style={{ marginTop: 8, fontSize: 11, color: "var(--text-mid)" }}>{data.summary}</div>
    </div>
  );
}

export default function App() {
  const [persona,      setPersona]      = useState("Analyst");
  const [focus,        setFocus]        = useState("Mixed");
  const [depth,        setDepth]        = useState(50);
  const [creativity,   setCreativity]   = useState("Neutral");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [modelApi,     setModelApi]     = useState("GPT-4o");

  const [messages, setMessages] = useState([]);
  const [input,    setInput]    = useState("");
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState("");

  const bottomRef   = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);


  function autoResize() {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 130) + "px";
  }

  function buildMLContext(intent, playerNames, mlData, tradeData) {
    const parts = [];

    if (mlData?.projection) {
      const p = mlData.projection;
      parts.push(
        `[ML PROJECTION] ${p.player} (${p.position}, ${p.team}): ` +
        `Predicted ${p.predicted_points} pts (range ${p.range}), recent avg ${p.recent_avg}.`
      );
    }
    if (mlData?.injury) {
      const r = mlData.injury;
      parts.push(
        `[ML INJURY RISK] ${r.risk_tier} (${r.risk_pct}). ` +
        `${r.key_flags?.join(", ") ?? "No flags"}. ${r.recommendation}`
      );
    }
    if (tradeData) {
      const t = tradeData;
      parts.push(
        `[ML TRADE EVAL] Verdict: ${t.verdict}. Net swing: ${t.net_value_swing?.toFixed(1)}. ` +
        `${t.summary}`
      );
    }

    return parts.length > 0
      ? `\n\n--- LIVE ML DATA ---\n${parts.join("\n")}\n---\n\nUser question: `
      : "";
  }

  async function sendMessage(textOverride) {
    const userText = (textOverride || input).trim();
    if (!userText) return;
    setError("");

    const intent      = detectIntent(userText);
    const playerNames = extractPlayerNames(userText);

    let autoML    = null;
    let autoTrade = null;

    try {
        if (intent === "trade") {
          const forIdx  = userText.toLowerCase().indexOf(" for ");
          if (forIdx !== -1) {
            const givingRaw    = userText.slice(0, forIdx);
            const receivingRaw = userText.slice(forIdx + 5);
            const gNames = extractPlayerNames(givingRaw);
            const rNames = extractPlayerNames(receivingRaw);
            if (gNames.length && rNames.length) {
              autoTrade = await evaluateTrade(gNames, rNames).catch(() => null);
            }
          }
        } else if (playerNames.length > 0) {
          autoML = await getPlayerFull(playerNames[0]).catch(() => null);
        }
    } catch { /* silent */ }

    const mlContext  = buildMLContext(intent, playerNames, autoML, autoTrade);
    const promptText = mlContext ? mlContext + userText : userText;

    const userMsg = { role: "user", content: userText, time: new Date(), mlData: autoML, tradeData: autoTrade };
    const history = [...messages, userMsg];
    setMessages(history);
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setLoading(true);

    const resolvedModel = MODEL_MAP[modelApi] ?? "gpt-4o-mini";
    const systemPrompt  = buildSystemPrompt(persona, focus, depth, creativity);

    try {
      const response = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type":  "application/json",
          "Authorization": `Bearer ${OPENAI_API_KEY}`,
        },
        body: JSON.stringify({
          model:       resolvedModel,
          temperature: 0.7,
          messages: [
            { role: "system", content: systemPrompt },
            ...history.slice(0, -1).map(({ role, content }) => ({ role, content })),
            { role: "user", content: promptText },
          ],
        }),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error?.message ?? `HTTP ${response.status}`);

      const aiText = data.choices?.[0]?.message?.content ?? "Sorry, I couldn't generate a response.";
      setMessages(prev => [...prev, { role: "assistant", content: aiText, time: new Date() }]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <>
      <style>{STYLES}</style>
      <div className="app">

        {/* App sidebar */}
        <div className="sidebar">
          <div className="sidebar-logo">
            <div className="logo-mark">📋</div>
            <div className="logo-text">
              PlayCaller
            </div>
          </div>

          {/* Model persona */}
          <div className="sidebar-section">
            <span className="sidebar-label">Model Persona</span>
            <div className="persona-grid">
              {PERSONAS.map(p => (
                <button
                  key={p}
                  className={`persona-btn${persona === p ? " active" : ""}`}
                  onClick={() => setPersona(p)}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>

          <div className="sidebar-divider" />

          {/* Model's response type */}
          <div className="sidebar-section">
            <span className="sidebar-label">Response Focus</span>
            <div className="radio-group">
              {["Statistics", "Narrative", "Mixed"].map(opt => (
                <div key={opt} className={`radio-option${focus === opt ? " active" : ""}`} onClick={() => setFocus(opt)}>
                  <div className="radio-dot"><div className="radio-dot-inner" /></div>
                  <span className="radio-label">{opt}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="sidebar-divider" />

          {/* Depth */}
          <div className="sidebar-section">
            <span className="sidebar-label">Response Depth</span>
            <div className="slider-wrap">
              <div className="slider-value">
                <span className="slider-val-num">{depth}</span>
                <span className="slider-val-label">/ 100</span>
              </div>
              <input type="range" min={5} max={100} step={5} value={depth} onChange={e => setDepth(Number(e.target.value))} />
            </div>
          </div>

          <div className="sidebar-divider" />

          {/* Creativity */}
          <div className="sidebar-section">
            <span className="sidebar-label">Response Creativity</span>
            <div className="select-slider">
              {["Fact-Based", "Neutral", "Creative"].map(opt => (
                <button key={opt} className={`cs-btn${creativity === opt ? " active" : ""}`} onClick={() => setCreativity(opt)}>
                  {opt}
                </button>
              ))}
            </div>
          </div>

          <div className="sidebar-divider" />

          {/* Advanced */}
          <div className="sidebar-section">
            <div className="checkbox-row" onClick={() => setShowAdvanced(v => !v)}>
              <div className={`checkbox-box${showAdvanced ? " checked" : ""}`}>
                <span className="checkbox-check">✓</span>
              </div>
              <span className="checkbox-label">Advanced Options</span>
            </div>
            {showAdvanced && (
              <div>
                <span className="sidebar-label" style={{ marginTop: 6, display: "block" }}>Model API</span>
                <select className="model-select" value={modelApi} onChange={e => setModelApi(e.target.value)}>
                  {MODELS.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
            )}
          </div>

          <div className="sidebar-footer">
            <div className="active-config">
              <strong>{persona}</strong> · {focus} · Depth {depth}
              <br />{creativity} · {modelApi}
            </div>
          </div>
        </div>

        {/* Chat */}
        <div className="chat-main">
          <div className="chat-header">
            <div>
              <div className="chat-title">PlayCaller</div>
              <div className="chat-subtitle">{focus} Focus</div>
            </div>
            <div className="persona-badge">{persona}</div>
          </div>

          <div className="messages">
            {messages.length === 0 && !loading ? (
              <div className="welcome">
                <h2>Your All-in-One Fantasy Assistant</h2>
                <p>
                  PlayCaller is powered
                  by GPT-4 and five years of <br></br> high-quality data from the Sleeper API.
                </p>
                <div className="starter-chips">
                  {STARTERS.map(s => (
                    <button key={s} className="chip" onClick={() => sendMessage(s)}>{s}</button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((msg, i) => (
                <div key={i} className={`msg ${msg.role === "user" ? "user" : "ai"}`}>
                  <div className="msg-avatar">
                    {msg.role === "user" ? "✧" : "⚡"}
                  </div>
                  <div className="msg-body">
                    <div className="msg-name">
                      {msg.role === "user" ? "You" : (
                        <>
                          {persona} · PlayCaller
                          {(messages[i - 1]?.mlData || messages[i - 1]?.tradeData) && (
                            <span style={{
                              fontSize: 9,
                              marginLeft: 8,
                              padding: "1px 6px",
                              borderRadius: 8,
                              border: "1px solid var(--accent2)",
                              color: "var(--accent)",
                              letterSpacing: "0.06em",
                            }}>
                              + ML context
                            </span>
                          )}
                        </>
                      )}
                    </div>
                    <div className="msg-bubble">{msg.content}</div>
                    <div className="msg-time">{formatTime(msg.time)}</div>
                  </div>
                </div>
              ))
            )}

            {loading && (
              <div className="msg ai">
                <div className="msg-avatar">⚡</div>
                <div className="msg-body">
                  <div className="msg-name">{persona} · PlayCaller</div>
                  <div className="typing-dots"><span /><span /><span /></div>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="input-area">
            {error && <div className="error-msg">⚠ {error}</div>}
            <div className="input-row">
              <textarea
                ref={textareaRef}
                placeholder="Ask me anything"
                value={input}
                onChange={e => { setInput(e.target.value); autoResize(); }}
                onKeyDown={onKeyDown}
              />
              <button className="send-btn" onClick={() => sendMessage()} disabled={loading || !input.trim()} title="Send">
                ↑
              </button>
            </div>
            <div className="input-hint">
              PlayCaller is AI and can make mistakes. Double-check responses.
            </div>
          </div>
        </div>
      </div>
    </>
  );
}