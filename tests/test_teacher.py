"""Unit tests for trl_data.teacher (composite + sma20 rules)."""

from __future__ import annotations

from models import TradingObservation
from trl_data.teacher import (
    BUY,
    HOLD,
    MARKET_FEAT_NAMES,
    SELL,
    get_teacher,
    teacher_composite_action,
    teacher_sma20_action,
)

_N = len(MARKET_FEAT_NAMES)
_SMA20 = MARKET_FEAT_NAMES.index("sma20_dist")
_RSI = MARKET_FEAT_NAMES.index("rsi")


def _make_obs(*, sma20_dist: float, rsi: float, window: int = 1) -> TradingObservation:
    """Build a minimal observation whose last bar carries the requested features."""
    bars = []
    for _ in range(window):
        row = [0.0] * _N
        row[_SMA20] = sma20_dist
        row[_RSI] = rsi
        bars.extend(row)
    return TradingObservation(
        market_features=bars,
        port_cash=10_000.0,
        holdings=0.0,
        port_val=10_000.0,
        portfolio_value=10_000.0,
        current_step=window,
    )


def test_sma20_teacher_buy_sell_hold_thresholds():
    assert teacher_sma20_action(_make_obs(sma20_dist=0.001, rsi=0.5)) == BUY
    assert teacher_sma20_action(_make_obs(sma20_dist=-0.05, rsi=0.5)) == SELL
    assert teacher_sma20_action(_make_obs(sma20_dist=-0.01, rsi=0.5)) == HOLD


def test_composite_teacher_blocks_buy_when_overbought():
    assert teacher_composite_action(_make_obs(sma20_dist=0.01, rsi=0.5)) == BUY
    assert teacher_composite_action(_make_obs(sma20_dist=0.01, rsi=0.85)) == SELL


def test_composite_teacher_sells_on_strong_downtrend():
    assert teacher_composite_action(_make_obs(sma20_dist=-0.05, rsi=0.5)) == SELL


def test_composite_teacher_holds_in_neutral_zone():
    assert teacher_composite_action(_make_obs(sma20_dist=-0.01, rsi=0.5)) == HOLD


def test_composite_teacher_handles_empty_window_safely():
    obs = TradingObservation(
        market_features=[],
        port_cash=10_000.0,
        holdings=0.0,
        port_val=10_000.0,
        portfolio_value=10_000.0,
        current_step=0,
    )
    assert teacher_composite_action(obs) == HOLD
    assert teacher_sma20_action(obs) == HOLD


def test_get_teacher_registry():
    assert get_teacher("sma20") is teacher_sma20_action
    assert get_teacher("composite") is teacher_composite_action
