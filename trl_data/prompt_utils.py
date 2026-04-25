"""
Build compact text prompts and a simple teacher policy from TradingObservation.

Feature order must match server.trading_environment.TradingEnvironment.feat.
"""

from __future__ import annotations

from typing import List, Tuple

from models import TradingObservation

MARKET_FEAT_NAMES: Tuple[str, ...] = (
    "log_return",
    "sma5_dist",
    "sma20_dist",
    "rsi",
    "norm_volume",
    "volatility",
    "vwap_dist",
    "ema12_dist",
    "macd_signal_gap",
    "atr_pct",
)
_N_FEAT = len(MARKET_FEAT_NAMES)
_SMA20_IDX = MARKET_FEAT_NAMES.index("sma20_dist")


def window_from_obs(obs: TradingObservation) -> int:
    mf = obs.market_features
    if not mf or len(mf) % _N_FEAT != 0:
        return 0
    return len(mf) // _N_FEAT


def last_sma20_dist(obs: TradingObservation) -> float:
    w = window_from_obs(obs)
    if w < 1:
        return 0.0
    mf = obs.market_features
    last_row_start = (w - 1) * _N_FEAT
    return float(mf[last_row_start + _SMA20_IDX])


def teacher_sma20_action(obs: TradingObservation) -> int:
    """Same rule as eval.phase4_benchmark sma20_trend."""
    d = last_sma20_dist(obs)
    if d > 0.0:
        return 1
    if d < -0.02:
        return 2
    return 0


def build_user_prompt(obs: TradingObservation, task_name: str | None = None) -> str:
    """Compact, human-readable state for the LLM (no raw 203-float dump)."""
    w = window_from_obs(obs)
    d20 = last_sma20_dist(obs)
    task = task_name or "risk_aware_trading"
    lines = [
        "You are a trading agent for SPY in a simulated environment.",
        f"task={task} lookback_bars={w}",
        f"portfolio_usd={obs.port_val:.2f} cash_usd={obs.port_cash:.2f} holdings_shares={obs.holdings:.6f}",
        f"close_usd={obs.close_price:.4f} step={obs.current_step} done={obs.done}",
        f"last_bar_sma20_dist={d20:.6f} (positive suggests price above SMA20)",
        "Respond with a single digit only: 0=HOLD, 1=BUY, 2=SELL.",
    ]
    return "\n".join(lines)


def build_sft_messages(
    obs: TradingObservation, teacher_action: int, task_name: str | None = None
) -> dict:
    """One training example in TRL conversational SFT format."""
    user = build_user_prompt(obs, task_name=task_name)
    assistant = str(int(teacher_action))
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def build_sft_text(
    obs: TradingObservation, teacher_action: int, task_name: str | None = None
) -> str:
    """Single-string prompt+completion (works with any causal LM via dataset_text_field=text)."""
    user = build_user_prompt(obs, task_name=task_name)
    assistant = str(int(teacher_action))
    return (
        "### Instruction\n"
        f"{user}\n"
        "### Response\n"
        f"{assistant}"
    )


def build_sft_row(
    obs: TradingObservation, teacher_action: int, task_name: str | None = None, *, use_messages: bool = False
) -> dict:
    if use_messages:
        return build_sft_messages(obs, teacher_action, task_name=task_name)
    return {"text": build_sft_text(obs, teacher_action, task_name=task_name)}
