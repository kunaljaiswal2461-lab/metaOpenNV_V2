"""
Teacher policies that turn a `TradingObservation` into a discrete action label
for supervised fine-tuning.

Why teachers exist
------------------
Phase 3 is **supervised** fine-tuning — we need a label per observation. Two
heuristic teachers are shipped:

- ``sma20``: the same single-feature rule used in ``eval.phase4_benchmark``
  (``sma20_trend``). Cheap, easy to reason about, and what the existing
  collected dataset already uses.

- ``composite``: a multi-feature rule that mixes trend (``sma20_dist``) with
  momentum exhaustion (``rsi``). It is intentionally close to a discretionary
  rule a human swing trader would write, so the LLM has something more
  interesting than a 1-feature mapping to imitate. The thresholds match the
  spec in the project plan:

      sma20_dist >  0.0       AND  rsi < 0.7   -> BUY  (1)
      sma20_dist < -0.02      OR   rsi > 0.7   -> SELL (2)
      otherwise                                 -> HOLD (0)

  ``rsi`` is the normalized 0..1 value emitted by ``data/preprocess.py``
  (i.e. divide-by-100). 0.7 therefore corresponds to RSI 70.

The teachers are pure functions of the latest bar in ``obs.market_features``;
no state, no env access, safe to call from any process.
"""

from __future__ import annotations

from typing import Callable, Tuple

from models import TradingObservation

MARKET_FEAT_NAMES: Tuple[str, ...] = (
    "log_return",
    "sma5_dist",
    "sma20_dist",
    "rsi",
    "norm_volume",
    "volatility",
    "vwap_dist",
)
_N_FEAT: int = len(MARKET_FEAT_NAMES)
_SMA20_IDX: int = MARKET_FEAT_NAMES.index("sma20_dist")
_RSI_IDX: int = MARKET_FEAT_NAMES.index("rsi")

# Action codes (kept identical to ``models.TradingAction``).
HOLD: int = 0
BUY: int = 1
SELL: int = 2

# Composite-teacher thresholds; named so they can be tuned in one place.
SMA20_BUY_THRESHOLD: float = 0.0
SMA20_SELL_THRESHOLD: float = -0.02
RSI_OVERBOUGHT_THRESHOLD: float = 0.7


def _last_bar_features(obs: TradingObservation) -> Tuple[float, ...] | None:
    """Return the most recent bar's 7-feature row, or ``None`` if the window is empty."""
    mf = obs.market_features
    if not mf or len(mf) % _N_FEAT != 0:
        return None
    last_row_start = len(mf) - _N_FEAT
    return tuple(float(x) for x in mf[last_row_start : last_row_start + _N_FEAT])


def teacher_sma20_action(obs: TradingObservation) -> int:
    """Single-feature SMA20 trend teacher. Matches ``eval.phase4_benchmark.sma20_trend``."""
    row = _last_bar_features(obs)
    if row is None:
        return HOLD
    sma20_dist = row[_SMA20_IDX]
    if sma20_dist > SMA20_BUY_THRESHOLD:
        return BUY
    if sma20_dist < SMA20_SELL_THRESHOLD:
        return SELL
    return HOLD


def teacher_composite_action(obs: TradingObservation) -> int:
    """Composite teacher: SMA20 trend gated by RSI exhaustion (RSI is 0..1 here)."""
    row = _last_bar_features(obs)
    if row is None:
        return HOLD
    sma20_dist = row[_SMA20_IDX]
    rsi = row[_RSI_IDX]

    if sma20_dist > SMA20_BUY_THRESHOLD and rsi < RSI_OVERBOUGHT_THRESHOLD:
        return BUY
    if sma20_dist < SMA20_SELL_THRESHOLD or rsi > RSI_OVERBOUGHT_THRESHOLD:
        return SELL
    return HOLD


TEACHERS: dict[str, Callable[[TradingObservation], int]] = {
    "sma20": teacher_sma20_action,
    "composite": teacher_composite_action,
}


def get_teacher(name: str) -> Callable[[TradingObservation], int]:
    """Resolve a teacher by name; raises ``KeyError`` with a helpful message."""
    if name not in TEACHERS:
        raise KeyError(
            f"Unknown teacher {name!r}. Available: {sorted(TEACHERS)}"
        )
    return TEACHERS[name]
