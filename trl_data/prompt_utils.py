"""
Build text prompts and SFT rows from a ``TradingObservation``.

Two prompt styles are supported:

- ``compact`` (default, used by Phase 3 small-model SFT). One numeric summary
  line plus the action map. Cheap on tokens; ideal for a 0.5B model.

- ``full``. Dumps the full last-bar feature row by name, plus an optional
  ``(action, reward)`` history tail (last K steps the agent took before the
  current observation). Designed for the larger-context Track H model
  (Qwen2.5-3B-Instruct or similar) where the extra structure pays off.

Both styles end with the **same single-digit response contract**, so a model
fine-tuned in one style still generates "0", "1", or "2" greedy-decoded.

Teachers live in ``trl_data.teacher``; the legacy import path
``from trl_data.prompt_utils import teacher_sma20_action`` keeps working.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from models import TradingObservation

from trl_data.teacher import (
    MARKET_FEAT_NAMES,
    _N_FEAT,
    _SMA20_IDX,
    teacher_composite_action,
    teacher_sma20_action,
)

PromptStyle = str  # "compact" | "full"
HistoryTail = Iterable[Tuple[int, float]]


def window_from_obs(obs: TradingObservation) -> int:
    """Number of bars currently in ``market_features``."""
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


def _last_bar_named(obs: TradingObservation) -> List[Tuple[str, float]]:
    """Return ``[(feature_name, value), ...]`` for the most recent bar."""
    w = window_from_obs(obs)
    if w < 1:
        return []
    mf = obs.market_features
    last_row_start = (w - 1) * _N_FEAT
    row = mf[last_row_start : last_row_start + _N_FEAT]
    return [(name, float(v)) for name, v in zip(MARKET_FEAT_NAMES, row)]


def _format_history(history: HistoryTail | None) -> str:
    """Render the trajectory tail. Empty / ``None`` is fine; we just say so."""
    if not history:
        return "Recent steps (action, reward):\n  (none yet)"
    items = list(history)
    lines = ["Recent steps (action, reward):"]
    n = len(items)
    for offset, (action, reward) in enumerate(items, start=1):
        rel = n - offset + 1
        lines.append(f"  t-{rel}: action={int(action)} reward={float(reward):.6f}")
    return "\n".join(lines)


def build_user_prompt(
    obs: TradingObservation,
    task_name: str | None = None,
    *,
    style: PromptStyle = "compact",
    history: HistoryTail | None = None,
) -> str:
    """
    Build the user-side prompt for a single observation.

    Defaults preserve the original compact wording so existing callers and
    already-collected JSONL datasets do not need to change.
    """
    if style not in ("compact", "full"):
        raise ValueError(f"Unknown prompt style: {style!r}")

    w = window_from_obs(obs)
    task = task_name or "risk_aware_trading"

    if style == "compact":
        d20 = last_sma20_dist(obs)
        return "\n".join(
            [
                "You are a trading agent for SPY in a simulated environment.",
                f"task={task} lookback_bars={w}",
                f"portfolio_usd={obs.port_val:.2f} cash_usd={obs.port_cash:.2f} "
                f"holdings_shares={obs.holdings:.6f}",
                f"close_usd={obs.close_price:.4f} step={obs.current_step} done={obs.done}",
                f"last_bar_sma20_dist={d20:.6f} (positive suggests price above SMA20)",
                "Respond with a single digit only: 0=HOLD, 1=BUY, 2=SELL.",
            ]
        )

    feats = _last_bar_named(obs)
    feat_lines = ["Latest indicators (last bar):"]
    # Pack 4 per line so the prompt stays under ~25 lines even with K=5 history.
    chunk = []
    for i, (name, value) in enumerate(feats, start=1):
        chunk.append(f"{name}={value:.6f}")
        if i % 4 == 0:
            feat_lines.append("  " + "  ".join(chunk))
            chunk = []
    if chunk:
        feat_lines.append("  " + "  ".join(chunk))

    return "\n".join(
        [
            "You are a trading agent for SPY in a simulated environment.",
            f"task={task} lookback_bars={w} step={obs.current_step} done={obs.done}",
            f"portfolio_usd={obs.port_val:.2f} cash_usd={obs.port_cash:.2f} "
            f"holdings_shares={obs.holdings:.6f} close_usd={obs.close_price:.4f}",
            "",
            *feat_lines,
            "",
            _format_history(history),
            "",
            "Respond with a single digit only: 0=HOLD, 1=BUY, 2=SELL.",
        ]
    )


def build_sft_messages(
    obs: TradingObservation,
    teacher_action: int,
    task_name: str | None = None,
    *,
    style: PromptStyle = "compact",
    history: HistoryTail | None = None,
) -> dict:
    user = build_user_prompt(obs, task_name=task_name, style=style, history=history)
    assistant = str(int(teacher_action))
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def build_sft_text(
    obs: TradingObservation,
    teacher_action: int,
    task_name: str | None = None,
    *,
    style: PromptStyle = "compact",
    history: HistoryTail | None = None,
) -> str:
    user = build_user_prompt(obs, task_name=task_name, style=style, history=history)
    assistant = str(int(teacher_action))
    return "### Instruction\n" f"{user}\n" "### Response\n" f"{assistant}"


def build_sft_row(
    obs: TradingObservation,
    teacher_action: int,
    task_name: str | None = None,
    *,
    use_messages: bool = False,
    style: PromptStyle = "compact",
    history: HistoryTail | None = None,
) -> dict:
    if use_messages:
        return build_sft_messages(
            obs, teacher_action, task_name=task_name, style=style, history=history
        )
    return {
        "text": build_sft_text(
            obs, teacher_action, task_name=task_name, style=style, history=history
        )
    }


__all__ = [
    "MARKET_FEAT_NAMES",
    "PromptStyle",
    "build_sft_messages",
    "build_sft_row",
    "build_sft_text",
    "build_user_prompt",
    "last_sma20_dist",
    "teacher_composite_action",
    "teacher_sma20_action",
    "window_from_obs",
]
