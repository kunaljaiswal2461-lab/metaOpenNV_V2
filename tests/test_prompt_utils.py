"""Tests for trl_data.prompt_utils (compact + full styles, history tail)."""

from __future__ import annotations

import pytest

from models import TradingObservation
from trl_data.prompt_utils import (
    MARKET_FEAT_NAMES,
    build_sft_row,
    build_user_prompt,
    last_sma20_dist,
)

_N = len(MARKET_FEAT_NAMES)
_SMA20 = MARKET_FEAT_NAMES.index("sma20_dist")


def _obs(sma20_dist: float = 0.01) -> TradingObservation:
    row = [0.0] * _N
    row[_SMA20] = sma20_dist
    return TradingObservation(
        market_features=row,
        port_cash=4_000.0,
        holdings=2.0,
        port_val=10_500.0,
        portfolio_value=10_500.0,
        current_step=42,
        close_price=336.78,
    )


def test_compact_prompt_contains_action_map_and_sma20():
    text = build_user_prompt(_obs(0.005), task_name="risk_aware_trading", style="compact")
    assert "0=HOLD" in text and "1=BUY" in text and "2=SELL" in text
    assert "last_bar_sma20_dist=0.005000" in text
    assert "task=risk_aware_trading" in text
    assert "Latest indicators" not in text


def test_full_prompt_dumps_named_features_and_history():
    history = [(1, 0.0001), (0, -0.0005), (2, 0.0008)]
    text = build_user_prompt(
        _obs(-0.03),
        task_name="multi_horizon_trading",
        style="full",
        history=history,
    )
    for name in MARKET_FEAT_NAMES:
        assert f"{name}=" in text
    assert "Recent steps (action, reward):" in text
    assert "t-1: action=2 reward=0.000800" in text
    assert "Respond with a single digit only" in text


def test_full_prompt_no_history_section_message():
    text = build_user_prompt(_obs(), style="full", history=None)
    assert "(none yet)" in text


def test_unknown_style_raises():
    with pytest.raises(ValueError):
        build_user_prompt(_obs(), style="exotic")


def test_build_sft_row_text_default_is_compact():
    row = build_sft_row(_obs(), teacher_action=1)
    assert "text" in row and "messages" not in row
    assert "### Response\n1" in row["text"]
    assert "Latest indicators" not in row["text"]


def test_build_sft_row_full_messages_format():
    row = build_sft_row(
        _obs(),
        teacher_action=2,
        use_messages=True,
        style="full",
        history=[(0, 0.0)],
    )
    assert "messages" in row and "text" not in row
    msgs = row["messages"]
    assert msgs[0]["role"] == "user"
    assert msgs[1] == {"role": "assistant", "content": "2"}
    assert "Latest indicators" in msgs[0]["content"]


def test_last_sma20_dist_returns_zero_on_empty_window():
    obs = TradingObservation(
        market_features=[],
        port_cash=0,
        holdings=0,
        port_val=0,
        portfolio_value=0,
        current_step=0,
    )
    assert last_sma20_dist(obs) == 0.0
