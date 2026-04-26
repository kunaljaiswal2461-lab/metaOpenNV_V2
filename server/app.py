"""
metaOpenNV_V2 — SPY RL OpenEnv Space.

This module owns:

1. The FastAPI surface (``/reset``, ``/step``, ``/state``, ``/health``) used by
   remote agents / judges. The contract is **untouched** — it matches
   ``openenv.yaml`` and the README Phase 2 table.
2. A Gradio UI mounted at ``/`` shaped like a real quant research terminal:
   live regime/agent controls, all 7 technical indicators visible at every
   step, full per-step reward + portfolio readout, equity / price charts
   with action markers, and the same "Episode Performance Metrics" strip
   judges expect (return, Sharpe, Calmar, max drawdown, total trades,
   episode score).
"""

from __future__ import annotations

import math
import os
import random
import statistics
import sys
import time
from typing import Any, Iterator

import gradio as gr
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import TradingAction, TradingObservation, TradingState
from server.trading_environment import TradingEnvironment

# ---------------------------------------------------------------------------
# FastAPI (OpenEnv contract — unchanged)
# ---------------------------------------------------------------------------

app = FastAPI(title="SPY RL Environment API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: TradingEnvironment | None = None


def get_env() -> TradingEnvironment:
    global _env
    if _env is None:
        _env = TradingEnvironment()
    return _env


@app.get("/web")
def web_redirect():
    return RedirectResponse(url="/")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=TradingObservation)
async def reset(request: Request):
    task_name = None
    try:
        body = await request.json()
        if isinstance(body, dict) and body.get("task_name"):
            task_name = str(body["task_name"])
    except Exception:
        pass
    return get_env().reset(task_name=task_name)


@app.post("/step", response_model=TradingObservation)
def step(action: TradingAction):
    return get_env().step(action)


@app.get("/state", response_model=TradingState)
def state() -> TradingState:
    return get_env().state()


# ---------------------------------------------------------------------------
# UI helpers — pure functions of (env, episode trace)
# ---------------------------------------------------------------------------

# 7-feature schema as emitted by ``data/preprocess.py`` and consumed by all
# teachers / prompts. Keep the order in sync with TradingEnvironment.feat.
_FEATURE_NAMES: tuple[str, ...] = (
    "log_return",
    "sma5_dist",
    "sma20_dist",
    "rsi",
    "norm_volume",
    "volatility",
    "vwap_dist",
)
_FEATURE_LABELS: dict[str, str] = {
    "log_return": "Log Return",
    "sma5_dist": "SMA-5 dist",
    "sma20_dist": "SMA-20 dist",
    "rsi": "RSI (0-1)",
    "norm_volume": "Volume / 20-MA",
    "volatility": "Volatility",
    "vwap_dist": "VWAP dist",
}
_FEATURE_HELP: dict[str, str] = {
    "log_return": "ln(P_t / P_t-1) — last bar return",
    "sma5_dist": "(P - SMA5) / SMA5 — short-term mean-reversion",
    "sma20_dist": "(P - SMA20) / SMA20 — medium-term trend",
    "rsi": "Normalized RSI(14); >0.7 = overbought",
    "norm_volume": "Vol / 20-bar avg vol",
    "volatility": "StdDev of log return over 10 bars",
    "vwap_dist": "(P - VWAP) / VWAP — cumulative fair value gap",
}
_FEATURE_COUNT = len(_FEATURE_NAMES)

# Human-readable labels for the active OHLCV dataset shown in the live UI.
_DATASET_LABELS: dict[str, str] = {
    "spy": "SPY (S&P 500 ETF) · Primary",
    "nifty50": "Nifty 50 / RELIANCE.NS · Secondary",
}


def _dataset_label(name: str | None) -> str:
    if not name:
        return "—"
    return _DATASET_LABELS.get(str(name).lower(), str(name))

_TASK_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "spy_trading": {
        "title": "SPY TRADING — short window (10 bars)",
        "body": (
            "Tight 10-minute lookback. The agent must commit early — there is barely enough "
            "context to confirm a reversal, so disciplined trend-following beats over-thinking. "
            "Best for stress-testing reactive policies."
        ),
        "rules": [
            "Window = 10 bars (73-dim observation: 70 market + 3 portfolio).",
            "Per-step features: log return, SMA-5/20 distances, RSI, volume z, volatility, VWAP distance.",
            "Reward = composite (alpha + downside + dSharpe + Treynor − tx friction).",
        ],
    },
    "risk_aware_trading": {
        "title": "RISK-AWARE TRADING — standard benchmark (20 bars)",
        "body": (
            "The default judge benchmark. 20-minute lookback gives the agent enough context "
            "to read the SMA20 trend and balance composite risk components. Phase 3 adapter "
            "(Kj2461/metaOpenNV-sft-qwen15) was trained on this regime."
        ),
        "rules": [
            "Window = 20 bars (143-dim observation).",
            "Reward dominated by alpha + dSharpe; downside variance keeps drawdowns honest.",
            "Liquidation triggers automatically at PV < 40% of initial capital.",
        ],
    },
    "multi_horizon_trading": {
        "title": "MULTI-HORIZON TRADING — deep planning (50 bars)",
        "body": (
            "Long 50-bar context for sequence-aware models. Track-H training (full prompt + "
            "composite teacher + 5-step action history) targets this regime, so the LLM can "
            "reason across sub-trends."
        ),
        "rules": [
            "Window = 50 bars (353-dim observation).",
            "Best regime for long-context LLMs / Track-H 3B + LoRA runs.",
            "Highest reward variance; expect bigger swings in equity curve.",
        ],
    },
}

_AGENTS: tuple[str, ...] = (
    "Manual",
    "SMA-20 Trend (heuristic)",
    "Composite (heuristic)",
    "Buy & Hold",
    "Random",
)

_INITIAL_CASH = 10000.0


def _last_bar_features(obs: TradingObservation, window: int) -> list[float]:
    """Return the latest bar's feature row from the flattened observation."""
    mf = list(obs.market_features or [])
    if len(mf) < _FEATURE_COUNT:
        return [0.0] * _FEATURE_COUNT
    return [float(x) for x in mf[-_FEATURE_COUNT:]]


def _series_for_feature(obs: TradingObservation, window: int, feat_idx: int) -> list[float]:
    """Pull one feature's time series across the obs window for sparkline charts."""
    mf = list(obs.market_features or [])
    if len(mf) % _FEATURE_COUNT != 0 or len(mf) == 0:
        return []
    n = len(mf) // _FEATURE_COUNT
    return [float(mf[i * _FEATURE_COUNT + feat_idx]) for i in range(n)]


def _action_label(a: int) -> str:
    return {0: "HOLD", 1: "BUY", 2: "SELL"}.get(int(a), "?")


def _action_color(a: int) -> str:
    return {0: "#94a3b8", 1: "#22c55e", 2: "#ef4444"}.get(int(a), "#cbd5e1")


def _max_drawdown(pv_curve: list[float]) -> float:
    if not pv_curve:
        return 0.0
    peak = pv_curve[0]
    dd = 0.0
    for v in pv_curve:
        peak = max(peak, v)
        if peak > 0:
            dd = min(dd, (v - peak) / peak)
    return dd  # negative number, e.g. -0.15 means -15%


def _sharpe(pv_curve: list[float]) -> float:
    if len(pv_curve) < 2:
        return 0.0
    rets = [
        math.log(pv_curve[i] / pv_curve[i - 1])
        for i in range(1, len(pv_curve))
        if pv_curve[i - 1] > 0
    ]
    if len(rets) < 2:
        return 0.0
    mu = statistics.mean(rets)
    sd = statistics.pstdev(rets)
    if sd <= 1e-9:
        return 0.0
    return (mu / sd) * math.sqrt(252 * 390)


def _calmar(pv_curve: list[float], steps: int) -> float:
    if not pv_curve or steps < 1:
        return 0.0
    total_return = pv_curve[-1] / pv_curve[0] - 1.0 if pv_curve[0] > 0 else 0.0
    annual_return = total_return * (252 * 390 / max(steps, 1))
    mdd = abs(_max_drawdown(pv_curve))
    if mdd <= 1e-6:
        return float("inf") if total_return > 0 else 0.0
    return annual_return / mdd


def _format_pct(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x*100:.2f}%"


def _build_indicator_html(values: list[float], series: list[list[float]]) -> str:
    cards: list[str] = []
    for i, name in enumerate(_FEATURE_NAMES):
        v = values[i] if i < len(values) else 0.0
        ser = series[i] if i < len(series) else []
        # Color-code per-feature: green-ish when "bullish-ish", red when "bearish-ish".
        if name in ("log_return", "sma5_dist", "sma20_dist", "vwap_dist"):
            color = "#22c55e" if v > 0 else ("#ef4444" if v < 0 else "#94a3b8")
        elif name == "rsi":
            color = "#ef4444" if v > 0.7 else ("#f59e0b" if v < 0.3 else "#22c55e")
        else:
            color = "#7dd3fc"

        # Tiny ASCII sparkline using block characters.
        spark = ""
        if ser:
            lo, hi = min(ser), max(ser)
            span = max(hi - lo, 1e-9)
            blocks = "▁▂▃▄▅▆▇█"
            spark = "".join(
                blocks[min(len(blocks) - 1, int((x - lo) / span * (len(blocks) - 1)))]
                for x in ser[-24:]
            )

        if name == "rsi":
            display_val = f"{v:.3f}"
        elif name in ("norm_volume",):
            display_val = f"{v:.2f}×"
        else:
            display_val = f"{v:+.4f}"

        cards.append(
            f"""
            <div class="ind-card" title="{_FEATURE_HELP[name]}">
              <div class="ind-name">{_FEATURE_LABELS[name]}</div>
              <div class="ind-val" style="color:{color};">{display_val}</div>
              <div class="ind-spark">{spark or '—'}</div>
            </div>
            """
        )
    return f'<div class="ind-grid">{"".join(cards)}</div>'


def _empty_pv_df() -> pd.DataFrame:
    return pd.DataFrame({"Step": [0], "Portfolio Value": [_INITIAL_CASH]})


def _empty_price_df() -> pd.DataFrame:
    return pd.DataFrame({"Step": [0], "SPY Price": [0.0]})


def _empty_log_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["Step", "Action", "Reward", "Cum. Reward", "Cash", "Holdings", "PV"]
    )


def _kpi_html(label: str, value: str, sub: str = "", color: str = "#22c55e") -> str:
    return f"""
    <div class="kpi-box">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{color};">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """


def _initial_kpi_strip() -> str:
    boxes = [
        ("Final Return", "—", "vs $10,000", "#94a3b8"),
        ("Sharpe Ratio", "—", "annualised", "#94a3b8"),
        ("Calmar Ratio", "—", "ann. return / |MDD|", "#94a3b8"),
        ("Max Drawdown", "—", "trough vs peak", "#94a3b8"),
        ("Total Trades", "—", "BUY + SELL", "#94a3b8"),
        ("Episode Score", "—", "Σ composite reward", "#94a3b8"),
    ]
    return '<div class="kpi-strip">' + "".join(_kpi_html(*b) for b in boxes) + "</div>"


def _build_kpi_strip(
    pv_curve: list[float],
    total_reward: float,
    trades: int,
) -> str:
    if not pv_curve:
        return _initial_kpi_strip()

    total_return = (pv_curve[-1] / pv_curve[0] - 1.0) if pv_curve[0] > 0 else 0.0
    sharpe = _sharpe(pv_curve)
    calmar = _calmar(pv_curve, len(pv_curve))
    mdd = _max_drawdown(pv_curve)
    score = total_reward

    def col(x: float, neutral: bool = False) -> str:
        if neutral:
            return "#7dd3fc"
        return "#22c55e" if x > 0 else ("#ef4444" if x < 0 else "#94a3b8")

    boxes = [
        ("Final Return", _format_pct(total_return), f"PV ${pv_curve[-1]:,.2f}", col(total_return)),
        ("Sharpe Ratio", f"{sharpe:+.2f}", "annualised", col(sharpe)),
        ("Calmar Ratio", "∞" if math.isinf(calmar) else f"{calmar:+.2f}", "ann. return / |MDD|", col(calmar if not math.isinf(calmar) else 1)),
        ("Max Drawdown", _format_pct(mdd), "trough vs peak", col(mdd)),
        ("Total Trades", str(trades), "BUY + SELL", "#7dd3fc"),
        ("Episode Score", f"{score:+.4f}", "Σ composite reward", col(score)),
    ]
    return '<div class="kpi-strip">' + "".join(_kpi_html(*b) for b in boxes) + "</div>"


def _regime_panel_html(regime: str) -> str:
    info = _TASK_DESCRIPTIONS.get(regime, _TASK_DESCRIPTIONS["risk_aware_trading"])
    rules = "".join(f"<li>{r}</li>" for r in info["rules"])
    return f"""
    <div class="regime-card">
      <div class="regime-head">⚡ {info['title']}</div>
      <div class="regime-body">{info['body']}</div>
      <ul class="regime-rules">{rules}</ul>
    </div>
    """


def _action_for(strategy: str, obs: TradingObservation, step_idx: int) -> int:
    """Map a strategy choice to a discrete action."""
    from trl_data.teacher import get_teacher

    if strategy == "Manual":
        return 0
    if strategy == "SMA-20 Trend (heuristic)":
        return get_teacher("sma20")(obs)
    if strategy == "Composite (heuristic)":
        return get_teacher("composite")(obs)
    if strategy == "Buy & Hold":
        return 1 if step_idx == 0 else 0
    if strategy == "Random":
        return random.randint(0, 2)
    return 0


# ---------------------------------------------------------------------------
# Episode runner — Gradio generator
# ---------------------------------------------------------------------------


def run_episode(
    regime: str,
    strategy: str,
    episode_length: int,
    seed: int,
    speed_ms: int,
) -> Iterator[tuple]:
    """Roll one episode and yield UI updates every step."""
    speed = max(0.0, min(0.5, speed_ms / 1000.0))
    env = TradingEnvironment(
        random_episode_start=False,
        seed=int(seed),
        episode_length=int(episode_length),
    )
    obs = env.reset(task_name=regime)
    window = int(env.window)

    pv_curve = [float(obs.port_val)]
    pv_rows = [(0, float(obs.port_val))]
    price_rows = [(0, float(obs.close_price))]
    log_rows: list[list] = []
    total_reward = 0.0
    trades = 0
    last_action = 0

    # First render: dashboard at t=0 before any step is taken.
    last_feats = _last_bar_features(obs, window)
    feat_series = [_series_for_feature(obs, window, i) for i in range(_FEATURE_COUNT)]
    yield (
        _build_indicator_html(last_feats, feat_series),
        pd.DataFrame(pv_rows, columns=["Step", "Portfolio Value"]),
        pd.DataFrame(price_rows, columns=["Step", "SPY Price"]),
        _empty_log_df(),
        _build_kpi_strip(pv_curve, total_reward, trades),
        f"Step 0/{episode_length} • {strategy} • {regime}",
        _action_label(0),
        f"{0:.6f}",
        f"{0:.6f}",
        f"${obs.port_val:,.2f}",
        f"{obs.holdings:.4f} SPY",
        f"${obs.port_cash:,.2f}",
        f"${obs.close_price:,.2f}",
        f"{0:.2%}",
        gr.update(value=False),
        _dataset_label(getattr(obs, "dataset", "spy")),
    )

    for step_idx in range(int(episode_length)):
        if obs.done:
            break
        a = _action_for(strategy, obs, step_idx)
        obs = env.step(TradingAction(action=int(a), amount=1.0))
        last_action = int(a)
        if a in (1, 2):
            trades += 1
        total_reward += float(obs.reward)
        pv_curve.append(float(obs.port_val))
        pv_rows.append((step_idx + 1, float(obs.port_val)))
        price_rows.append((step_idx + 1, float(obs.close_price)))
        log_rows.append(
            [
                step_idx + 1,
                _action_label(a),
                round(float(obs.reward), 6),
                round(total_reward, 6),
                round(float(obs.port_cash), 2),
                round(float(obs.holdings), 4),
                round(float(obs.port_val), 2),
            ]
        )

        last_feats = _last_bar_features(obs, window)
        feat_series = [_series_for_feature(obs, window, i) for i in range(_FEATURE_COUNT)]

        ret_pct = (obs.port_val / _INITIAL_CASH) - 1.0
        log_df = pd.DataFrame(
            log_rows[-200:],
            columns=["Step", "Action", "Reward", "Cum. Reward", "Cash", "Holdings", "PV"],
        )

        yield (
            _build_indicator_html(last_feats, feat_series),
            pd.DataFrame(pv_rows, columns=["Step", "Portfolio Value"]),
            pd.DataFrame(price_rows, columns=["Step", "SPY Price"]),
            log_df,
            _build_kpi_strip(pv_curve, total_reward, trades),
            f"Step {step_idx+1}/{episode_length} • {strategy} • {regime} • done={obs.done}",
            _action_label(last_action),
            f"{float(obs.reward):+.6f}",
            f"{total_reward:+.6f}",
            f"${obs.port_val:,.2f}",
            f"{obs.holdings:.4f} SPY",
            f"${obs.port_cash:,.2f}",
            f"${obs.close_price:,.2f}",
            _format_pct(ret_pct),
            gr.update(value=obs.done),
            _dataset_label(getattr(obs, "dataset", "spy")),
        )

        if speed > 0:
            time.sleep(speed)


def _coerce_df(value, default_factory):
    """Normalize Gradio component values back to a pandas DataFrame.

    Gradio passes LinePlot/Dataframe values back to Python as either a
    ``pd.DataFrame`` or a ``{"headers": [...], "data": [[...]]}`` dict
    depending on version. This helper hides that asymmetry so callers can
    always operate on a DataFrame.
    """
    if value is None:
        return default_factory()
    if isinstance(value, pd.DataFrame):
        return default_factory() if value.empty else value
    if isinstance(value, dict):
        headers = value.get("headers") or value.get("columns")
        data = value.get("data") or value.get("value")
        if headers and data is not None:
            try:
                df = pd.DataFrame(data, columns=headers)
                return default_factory() if df.empty else df
            except Exception:
                return default_factory()
        return default_factory()
    if isinstance(value, list):
        try:
            df = pd.DataFrame(value)
            return default_factory() if df.empty else df
        except Exception:
            return default_factory()
    return default_factory()


def _coerce_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip().replace(",", "").replace("$", "").replace("+", "")
        try:
            return float(s) if s else default
        except ValueError:
            return default
    return default


def manual_step_action(
    action_idx: int,
    amount_pct: float,
    pv_state,
    price_state,
    log_state,
    cum_reward_str,
    trade_count,
    regime: str,
):
    """Manual control pad: step the singleton env once and return a full UI refresh."""
    env = get_env()
    obs = env.step(TradingAction(action=int(action_idx), amount=float(amount_pct) / 100.0))
    window = int(env.window)
    step = int(obs.current_step) - window

    pv_state = _coerce_df(pv_state, _empty_pv_df)
    price_state = _coerce_df(price_state, _empty_price_df)
    log_state = _coerce_df(log_state, _empty_log_df)
    cum_reward = _coerce_float(cum_reward_str, 0.0)
    trade_count = int(trade_count) if trade_count is not None else 0

    cum_reward += float(obs.reward)
    if int(action_idx) in (1, 2):
        trade_count += 1

    pv_state = pd.concat(
        [pv_state, pd.DataFrame([{"Step": step, "Portfolio Value": float(obs.port_val)}])],
        ignore_index=True,
    )
    price_state = pd.concat(
        [price_state, pd.DataFrame([{"Step": step, "SPY Price": float(obs.close_price)}])],
        ignore_index=True,
    )
    new_row = pd.DataFrame(
        [
            {
                "Step": step,
                "Action": _action_label(int(action_idx)),
                "Reward": round(float(obs.reward), 6),
                "Cum. Reward": round(cum_reward, 6),
                "Cash": round(float(obs.port_cash), 2),
                "Holdings": round(float(obs.holdings), 4),
                "PV": round(float(obs.port_val), 2),
            }
        ]
    )
    log_state = pd.concat([log_state, new_row], ignore_index=True).tail(200)

    last_feats = _last_bar_features(obs, window)
    feat_series = [_series_for_feature(obs, window, i) for i in range(_FEATURE_COUNT)]
    pv_curve = pv_state["Portfolio Value"].astype(float).tolist()
    ret_pct = (obs.port_val / _INITIAL_CASH) - 1.0

    return (
        _build_indicator_html(last_feats, feat_series),
        pv_state,
        price_state,
        log_state,
        _build_kpi_strip(pv_curve, cum_reward, trade_count),
        f"Manual step {step} • action={_action_label(int(action_idx))} • amount={amount_pct:.0f}%",
        _action_label(int(action_idx)),
        f"{float(obs.reward):+.6f}",
        f"{cum_reward:+.6f}",
        f"${obs.port_val:,.2f}",
        f"{obs.holdings:.4f} SPY",
        f"${obs.port_cash:,.2f}",
        f"${obs.close_price:,.2f}",
        _format_pct(ret_pct),
        gr.update(value=obs.done),
        _dataset_label(getattr(obs, "dataset", "spy")),
        cum_reward,
        trade_count,
    )


def manual_reset(regime: str):
    env = get_env()
    obs = env.reset(task_name=regime)
    window = int(env.window)
    pv_state = pd.DataFrame({"Step": [0], "Portfolio Value": [float(obs.port_val)]})
    price_state = pd.DataFrame({"Step": [0], "SPY Price": [float(obs.close_price)]})
    log_state = _empty_log_df()
    last_feats = _last_bar_features(obs, window)
    feat_series = [_series_for_feature(obs, window, i) for i in range(_FEATURE_COUNT)]
    return (
        _build_indicator_html(last_feats, feat_series),
        pv_state,
        price_state,
        log_state,
        _initial_kpi_strip(),
        f"Reset • regime={regime} • window={window}",
        "—",
        "0.000000",
        "0.000000",
        f"${obs.port_val:,.2f}",
        f"{obs.holdings:.4f} SPY",
        f"${obs.port_cash:,.2f}",
        f"${obs.close_price:,.2f}",
        "+0.00%",
        gr.update(value=False),
        _dataset_label(getattr(obs, "dataset", "spy")),
        0.0,
        0,
    )


# ---------------------------------------------------------------------------
# Gradio Blocks — the actual terminal UI
# ---------------------------------------------------------------------------

_CSS = """
:root, body, .gradio-container, .gradio-container > div {
    background: #06080d !important;
    color: #d8e2f0 !important;
    font-family: 'JetBrains Mono', 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important;
}
.gradio-container { max-width: 1400px !important; margin: 0 auto; padding: 0 !important; }
.gradio-container h1, .gradio-container h2, .gradio-container h3, .gradio-container h4 {
    color: #d8ffe9 !important; letter-spacing: 0.02em;
}
.gradio-container button.primary, .gradio-container .lg.primary, .gradio-container button[variant="primary"] {
    background: linear-gradient(180deg, #22c55e, #16a34a) !important;
    color: #06080d !important;
    border: 1px solid #22c55e !important;
    font-weight: 700 !important;
}
.gradio-container button { border-radius: 4px !important; font-family: inherit !important; }
.gradio-container .gr-form, .gradio-container .form, .gradio-container .block {
    background: transparent !important; border: none !important;
}
.gradio-container input, .gradio-container select, .gradio-container textarea,
.gradio-container .gr-input, .gradio-container .gr-textbox, .gradio-container .gr-dropdown {
    background: #0a1018 !important;
    color: #d8e2f0 !important;
    border: 1px solid #1f2937 !important;
}
.gradio-container .gr-box, .gradio-container .panel, .gradio-container .block.padded {
    background: #0a0e15 !important; border: 1px solid #1f2937 !important; border-radius: 6px !important;
}
.gradio-container .tabs > .tab-nav { background: #06080d !important; border-bottom: 1px solid #1f2937 !important; }
.gradio-container .tabs > .tab-nav button {
    background: transparent !important; color: #94a3b8 !important;
    border: none !important; border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    font-family: inherit !important; letter-spacing: 0.04em;
}
.gradio-container .tabs > .tab-nav button.selected {
    color: #22c55e !important; border-bottom-color: #22c55e !important;
}

/* === Header === */
.term-header {
    background: linear-gradient(180deg, #0c1420 0%, #06080d 100%);
    border: 1px solid #1f2937;
    border-radius: 6px;
    padding: 18px 22px;
    margin: 14px 0 18px 0;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 12px;
}
.term-title { font-size: 22px; font-weight: 700; color: #e6fff0; letter-spacing: 0.04em; }
.term-title .accent { color: #22c55e; }
.term-sub { font-size: 11px; color: #64748b; letter-spacing: 0.18em; margin-top: 2px; text-transform: uppercase; }
.term-pills { display: flex; gap: 10px; flex-wrap: wrap; }
.pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(34, 197, 94, 0.08); color: #86efac;
    border: 1px solid rgba(34, 197, 94, 0.4);
    padding: 4px 10px; border-radius: 999px;
    font-size: 11px; letter-spacing: 0.06em;
}
.pill .dot { width: 6px; height: 6px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 6px #22c55e; }
.pill.amber { background: rgba(245, 158, 11, 0.08); color: #fcd34d; border-color: rgba(245, 158, 11, 0.4); }
.pill.amber .dot { background: #f59e0b; box-shadow: 0 0 6px #f59e0b; }
.pill.cyan { background: rgba(56, 189, 248, 0.08); color: #7dd3fc; border-color: rgba(56, 189, 248, 0.4); }
.pill.cyan .dot { background: #38bdf8; box-shadow: 0 0 6px #38bdf8; }

/* === Section blocks === */
.section-card {
    background: #0a0e15; border: 1px solid #1f2937;
    border-left: 3px solid #22c55e;
    border-radius: 4px; padding: 14px 18px; margin-bottom: 14px;
}
.section-card h3 { color: #22c55e !important; font-size: 12px !important; letter-spacing: 0.18em; margin: 0 0 8px 0 !important; text-transform: uppercase; }

/* === Regime card === */
.regime-card {
    background: linear-gradient(180deg, rgba(34,197,94,0.06), rgba(34,197,94,0.02));
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-left: 4px solid #22c55e;
    border-radius: 4px; padding: 14px 18px;
}
.regime-head { color: #4ade80; font-weight: 700; font-size: 13px; letter-spacing: 0.1em; }
.regime-body { color: #cbd5e1; margin: 8px 0 8px 0; font-size: 13px; line-height: 1.55; }
.regime-rules { color: #94a3b8; font-size: 12px; margin: 0; padding-left: 22px; }
.regime-rules li { margin: 4px 0; }

/* === Indicator strip (single horizontal row) === */
.ind-grid {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 8px !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    padding-bottom: 4px;
    width: 100% !important;
}
.ind-grid::-webkit-scrollbar { height: 6px; }
.ind-grid::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 3px; }
.ind-card {
    flex: 1 1 0 !important;
    min-width: 108px !important;
    max-width: 1fr;
    background: #0a0e15;
    border: 1px solid #1f2937;
    border-radius: 4px;
    padding: 8px 10px;
    min-height: 78px;
    display: flex;
    flex-direction: column;
    gap: 3px;
    transition: border-color 0.15s;
}
.ind-card:hover { border-color: #22c55e; }
.ind-name { color: #64748b; font-size: 9px; letter-spacing: 0.08em; text-transform: uppercase; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ind-val { font-size: 15px; font-weight: 700; font-family: inherit; white-space: nowrap; }
.ind-spark { color: #475569; font-size: 11px; letter-spacing: 0.04em; line-height: 1; opacity: 0.85; white-space: nowrap; overflow: hidden; }

/* === KPI strip === */
.kpi-strip {
    display: grid; grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 10px; margin-top: 6px;
}
.kpi-box {
    background: #0a0e15; border: 1px solid #1f2937; border-radius: 4px;
    padding: 12px 14px; text-align: center;
    border-top: 2px solid #1f2937;
}
.kpi-box:nth-child(1) { border-top-color: #22c55e; }
.kpi-box:nth-child(2) { border-top-color: #38bdf8; }
.kpi-box:nth-child(3) { border-top-color: #facc15; }
.kpi-box:nth-child(4) { border-top-color: #ef4444; }
.kpi-box:nth-child(5) { border-top-color: #a855f7; }
.kpi-box:nth-child(6) { border-top-color: #22c55e; }
.kpi-label { color: #64748b; font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase; }
.kpi-value { font-size: 22px; font-weight: 700; margin: 6px 0 4px 0; }
.kpi-sub { color: #475569; font-size: 10px; letter-spacing: 0.06em; }

/* === Live readout === */
.readout-row { display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 8px; }
.readout-row .gr-form { background: #0a0e15 !important; border: 1px solid #1f2937 !important; border-radius: 4px !important; }
.readout-row label > span { color: #64748b !important; font-size: 10px !important; letter-spacing: 0.1em; text-transform: uppercase; }
.readout-row input { color: #d8ffe9 !important; font-weight: 700 !important; font-size: 14px !important; }
"""


def _header_html() -> str:
    return """
    <div class="term-header">
      <div>
        <div class="term-title">metaOpenNV<span class="accent"> · SPY RL Quantitative Terminal</span></div>
        <div class="term-sub">META × OpenEnv Hackathon 2026 · OpenEnv-compliant trading MDP</div>
      </div>
      <div class="term-pills">
        <span class="pill"><span class="dot"></span>OpenEnv Active</span>
        <span class="pill cyan"><span class="dot"></span>FastAPI Online</span>
        <span class="pill"><span class="dot"></span>TRL SFT Trained</span>
        <span class="pill amber"><span class="dot"></span>Qwen 2.5 Adapter Ready</span>
      </div>
    </div>
    """


with gr.Blocks(theme=gr.themes.Base(), css=_CSS, title="metaOpenNV — SPY RL Terminal") as demo:
    gr.HTML(_header_html())

    cum_reward_state = gr.State(0.0)
    trade_count_state = gr.State(0)

    with gr.Tabs():
        # =========================================================
        # Tab 1: Live Trading Terminal
        # =========================================================
        with gr.Tab("● Live Trading Terminal"):
            with gr.Group(elem_classes="section-card"):
                gr.HTML("<h3>Episode Controls</h3>")
                with gr.Row():
                    regime_dd = gr.Dropdown(
                        choices=list(_TASK_DESCRIPTIONS.keys()),
                        value="risk_aware_trading",
                        label="Market Regime / Task",
                    )
                    strategy_dd = gr.Dropdown(
                        choices=list(_AGENTS),
                        value="Composite (heuristic)",
                        label="Agent Strategy",
                    )
                    length_slider = gr.Slider(
                        minimum=50, maximum=390, step=10, value=200, label="Episode Length (steps)"
                    )
                    seed_num = gr.Number(value=42, precision=0, label="Seed", minimum=0)
                    speed_slider = gr.Slider(
                        minimum=0, maximum=200, step=10, value=20, label="Step delay (ms)"
                    )
                with gr.Row():
                    run_btn = gr.Button("▶  Run Episode", variant="primary", scale=2)
                    reset_btn = gr.Button("↺  Reset Singleton Env", scale=1)

            regime_panel = gr.HTML(_regime_panel_html("risk_aware_trading"))

            with gr.Group(elem_classes="section-card"):
                gr.HTML("<h3>Technical Indicators · current bar (last 24-bar sparkline)</h3>")
                indicator_panel = gr.HTML(
                    _build_indicator_html(
                        [0.0] * _FEATURE_COUNT,
                        [[] for _ in range(_FEATURE_COUNT)],
                    )
                )

            with gr.Group(elem_classes="section-card"):
                gr.HTML("<h3>Live Readout · per-step state</h3>")
                with gr.Row(elem_classes="readout-row"):
                    step_label = gr.Textbox(label="Step / Mode", interactive=False, show_label=True, value="awaiting episode")
                    action_label_box = gr.Textbox(label="Last Action", interactive=False, value="—")
                    reward_box = gr.Textbox(label="Reward (this step)", interactive=False, value="0.000000")
                    cum_reward_box = gr.Textbox(label="Σ Reward", interactive=False, value="0.000000")
                    pv_box = gr.Textbox(label="Portfolio Value", interactive=False, value=f"${_INITIAL_CASH:,.2f}")
                    cash_box = gr.Textbox(label="Cash", interactive=False, value=f"${_INITIAL_CASH:,.2f}")
                    holdings_box = gr.Textbox(label="Holdings", interactive=False, value="0.0000 SPY")
                with gr.Row(elem_classes="readout-row"):
                    price_box = gr.Textbox(label="Last SPY Price", interactive=False, value="$0.00")
                    return_box = gr.Textbox(label="Episode Return", interactive=False, value="+0.00%")
                    done_chk = gr.Checkbox(label="done", value=False, interactive=False)
                    dataset_box = gr.Textbox(
                        label="Active Dataset",
                        interactive=False,
                        value=_dataset_label("spy"),
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    pv_chart = gr.LinePlot(
                        value=_empty_pv_df(),
                        x="Step",
                        y="Portfolio Value",
                        title="Equity curve",
                        height=320,
                    )
                with gr.Column(scale=1):
                    price_chart = gr.LinePlot(
                        value=_empty_price_df(),
                        x="Step",
                        y="SPY Price",
                        title="SPY price",
                        height=320,
                    )

            with gr.Group(elem_classes="section-card"):
                gr.HTML("<h3>Episode Performance Metrics</h3>")
                kpi_panel = gr.HTML(_initial_kpi_strip())

            with gr.Group(elem_classes="section-card"):
                gr.HTML("<h3>Step-by-step trace</h3>")
                step_log = gr.Dataframe(
                    value=_empty_log_df(),
                    headers=["Step", "Action", "Reward", "Cum. Reward", "Cash", "Holdings", "PV"],
                    datatype=["number", "str", "number", "number", "number", "number", "number"],
                    interactive=False,
                    wrap=True,
                    max_height=300,
                )

            with gr.Group(elem_classes="section-card"):
                gr.HTML("<h3>Manual Control Pad · drives the singleton env (used by /step)</h3>")
                with gr.Row():
                    amount_slider = gr.Slider(1, 100, step=1, value=100, label="Trade Order Size (% of available)")
                with gr.Row():
                    hold_btn = gr.Button("⏸  HOLD", variant="secondary")
                    buy_btn = gr.Button("▲  BUY", variant="primary")
                    sell_btn = gr.Button("▼  SELL", variant="stop")
                    full_reset_btn = gr.Button("↺  FULL RESET")

            run_outputs = [
                indicator_panel,
                pv_chart,
                price_chart,
                step_log,
                kpi_panel,
                step_label,
                action_label_box,
                reward_box,
                cum_reward_box,
                pv_box,
                holdings_box,
                cash_box,
                price_box,
                return_box,
                done_chk,
                dataset_box,
            ]

            run_btn.click(
                run_episode,
                inputs=[regime_dd, strategy_dd, length_slider, seed_num, speed_slider],
                outputs=run_outputs,
            )

            regime_dd.change(_regime_panel_html, inputs=[regime_dd], outputs=[regime_panel])

            manual_outputs = [
                indicator_panel,
                pv_chart,
                price_chart,
                step_log,
                kpi_panel,
                step_label,
                action_label_box,
                reward_box,
                cum_reward_box,
                pv_box,
                holdings_box,
                cash_box,
                price_box,
                return_box,
                done_chk,
                dataset_box,
                cum_reward_state,
                trade_count_state,
            ]

            for btn, idx in ((hold_btn, 0), (buy_btn, 1), (sell_btn, 2)):
                btn.click(
                    manual_step_action,
                    inputs=[
                        gr.Number(idx, visible=False),
                        amount_slider,
                        pv_chart,
                        price_chart,
                        step_log,
                        cum_reward_box,
                        trade_count_state,
                        regime_dd,
                    ],
                    outputs=manual_outputs,
                )

            reset_outputs = manual_outputs  # same shape
            full_reset_btn.click(manual_reset, inputs=[regime_dd], outputs=reset_outputs)
            reset_btn.click(manual_reset, inputs=[regime_dd], outputs=reset_outputs)

        # =========================================================
        # Tab 2: Project Documentation (kept rich-HTML page)
        # =========================================================
        with gr.Tab("◇ Project Documentation"):
            gr.HTML(
                """
                <style>
                .doc-wrapper { padding: 1.5rem 0.5rem; color: #d8e2f0; line-height: 1.65; }
                .doc-wrapper h1 { color: #22c55e; }
                .doc-wrapper h2 { color: #4ade80; border-left: 3px solid #22c55e; padding-left: 0.75rem; margin-top: 2rem; }
                .doc-wrapper h3 { color: #7dd3fc; }
                .doc-wrapper code { background: #0a1018; color: #4ade80; padding: 1px 6px; border-radius: 3px; }
                .doc-wrapper table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
                .doc-wrapper th, .doc-wrapper td { border: 1px solid #1f2937; padding: 8px 12px; text-align: left; }
                .doc-wrapper th { background: #0a1018; color: #22c55e; }
                .doc-wrapper .formula-box { background: #000; color: #4ade80; padding: 14px 18px; border-left: 3px solid #22c55e; border-radius: 4px; margin: 12px 0; font-family: inherit; }
                .doc-wrapper .reward-component { background: #0a0e15; border: 1px solid #1f2937; border-radius: 4px; padding: 12px 16px; margin: 10px 0; }
                .doc-wrapper .reward-component h4 { color: #facc15; margin: 0 0 4px 0; }
                .doc-wrapper a { color: #38bdf8; }
                </style>
                <div class="doc-wrapper">
                  <h1>metaOpenNV — engine specifications</h1>
                  <p>High-fidelity Markov Decision Process for SPY trading. The Gradio UI on the left is a live view of what the agent sees + does; the FastAPI surface (<code>/reset</code>, <code>/step</code>, <code>/state</code>, <code>/health</code>) is the OpenEnv contract judges connect to.</p>

                  <h2>1 · Observation (window × 7 + 3)</h2>
                  <p>Flattened temporal hierarchy: last <code>WINDOW_SIZE</code> bars of 7 engineered features, plus 3 portfolio scalars. <code>WINDOW_SIZE</code> per task: <code>spy_trading=10</code>, <code>risk_aware_trading=20</code> (default), <code>multi_horizon_trading=50</code>.</p>
                  <table>
                    <thead><tr><th>Indicator</th><th>Formula</th><th>Intent</th></tr></thead>
                    <tbody>
                      <tr><td>Log Return</td><td><code>ln(P_t / P_t-1)</code></td><td>Stationary bar return</td></tr>
                      <tr><td>SMA-5 dist</td><td><code>(P − SMA5)/SMA5</code></td><td>Short-term mean-reversion</td></tr>
                      <tr><td>SMA-20 dist</td><td><code>(P − SMA20)/SMA20</code></td><td>Medium-trend alignment</td></tr>
                      <tr><td>RSI(14) (0..1)</td><td><code>100 − 100/(1+RS)</code></td><td>Overbought/oversold</td></tr>
                      <tr><td>Norm volume</td><td><code>V / SMA20(V)</code></td><td>Liquidity conviction</td></tr>
                      <tr><td>Volatility</td><td><code>σ(LogRet, 10)</code></td><td>Risk regime</td></tr>
                      <tr><td>VWAP dist</td><td><code>(P − VWAP)/VWAP</code></td><td>Cumulative fair-value gap</td></tr>
                    </tbody>
                  </table>

                  <h2>2 · Reward (composite, Moody &amp; Saffell-style)</h2>
                  <div class="formula-box">R_t = 0.40·C1 + 0.25·C2 + 0.15·C3 + 0.05·C4 + 0.15·C5</div>
                  <div class="reward-component"><h4>C1 · Alpha capture (40%)</h4>Log return of portfolio value.</div>
                  <div class="reward-component"><h4>C2 · Downside variance (25%)</h4>−max(0, −r)² penalises drawdowns asymmetrically.</div>
                  <div class="reward-component"><h4>C3 · Differential Sharpe (15%)</h4>Online recursive estimator (Moody &amp; Saffell 2001).</div>
                  <div class="reward-component"><h4>C4 · Transaction friction (5%)</h4>−0.1% on each executed trade.</div>
                  <div class="reward-component"><h4>C5 · Treynor (15%)</h4>(r − rf)/β over a 20-bar rolling window vs SPY return.</div>

                  <h2>3 · Action space</h2>
                  <ul>
                    <li><b>0 HOLD</b> — no transaction.</li>
                    <li><b>1 BUY</b> — invest <code>amount × cash</code> at next bar close, 0.1% friction.</li>
                    <li><b>2 SELL</b> — liquidate <code>amount × holdings</code> at next bar close, 0.1% friction.</li>
                  </ul>

                  <h2>4 · OpenEnv contract</h2>
                  <p>POST <code>/reset</code> (optional <code>{"task_name": ...}</code>) → <code>TradingObservation</code>. POST <code>/step</code> with <code>{"action": 0|1|2, "amount": 0..1}</code> → <code>TradingObservation</code>. GET <code>/state</code> → <code>TradingState</code>. GET <code>/health</code> → status. The Gradio terminal above drives the same env singleton via these endpoints.</p>

                  <h2>5 · Trained adapters</h2>
                  <ul>
                    <li><b>Phase 3</b>: <a href="https://huggingface.co/Kj2461/metaOpenNV-sft-qwen15">Kj2461/metaOpenNV-sft-qwen15</a> — Qwen 2.5-1.5B + LoRA, T4-medium. Lifts mean episode reward from −0.044 to +1.844 on <code>risk_aware_trading</code>.</li>
                    <li><b>Track H</b> (in progress): Qwen 2.5-3B + 4-bit LoRA on <code>multi_horizon_trading</code> with full prompt + composite teacher + 5-step history.</li>
                  </ul>
                </div>
                """
            )


app = gr.mount_gradio_app(app, demo, path="/")


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))


if __name__ == "__main__":
    main()
