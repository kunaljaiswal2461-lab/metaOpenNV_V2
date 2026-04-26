"""Unit tests for trl_data.eval_utils (parsing + Markdown rendering)."""

from __future__ import annotations

import pytest

from trl_data.eval_utils import format_metrics_md, parse_action, parse_models


def test_parse_action_picks_first_valid_digit():
    assert parse_action("1") == 1
    assert parse_action("\n2\n") == 2
    assert parse_action("0.") == 0
    assert parse_action("The action is 2.") == 2
    # "9 then 1" -> 9 is invalid, 1 is the first valid digit
    assert parse_action("9 then 1") == 1


def test_parse_action_falls_back_to_zero():
    assert parse_action("") == 0
    assert parse_action("BUY") == 0
    # "3" is not a valid env action
    assert parse_action("3") == 0


def test_parse_models_named_and_bare():
    out = parse_models(["base=Qwen/Qwen2.5-0.5B-Instruct", "results/phase3_lora"])
    assert out["base"] == "Qwen/Qwen2.5-0.5B-Instruct"
    # bare ident gets a filesystem-friendly auto-name
    assert out["results_phase3_lora"] == "results/phase3_lora"


def test_parse_models_rejects_duplicates():
    with pytest.raises(ValueError):
        parse_models(["a=x", "a=y"])


def test_parse_models_rejects_empty_input():
    with pytest.raises(ValueError):
        parse_models([])


def test_format_metrics_md_table_shape():
    rows = [
        {
            "name": "base",
            "mean_r": -0.001234,
            "std_r": 0.000456,
            "final_pv": 9876.54,
            "h_pct": 0.5,
            "b_pct": 0.3,
            "s_pct": 0.2,
            "teacher_agreement": 0.42,
        },
        {
            "name": "sft",
            "mean_r": 0.002000,
            "std_r": 0.000999,
            "final_pv": 10142.30,
            "h_pct": 0.1,
            "b_pct": 0.6,
            "s_pct": 0.3,
            "teacher_agreement": 0.81,
        },
    ]
    md = format_metrics_md(
        rows,
        task_name="risk_aware_trading",
        prompt_style="compact",
        teacher="sma20",
        episodes=10,
    )
    assert md.startswith("# Phase 3 - LLM on env evaluation")
    assert "task=risk_aware_trading prompt=compact teacher=sma20 episodes=10" in md
    assert "| base | -0.001234 | 0.000456 | 9876.54 | 50.0% | 30.0% | 20.0% | 0.420 |" in md
    assert "| sft | 0.002000 | 0.000999 | 10142.30 | 10.0% | 60.0% | 30.0% | 0.810 |" in md
