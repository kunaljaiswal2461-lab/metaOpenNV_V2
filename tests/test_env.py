"""Phase 2: remote client must not import server implementation.

Also covers Phase-N dataset rotation: the env alternates between SPY (primary)
and Nifty 50 / RELIANCE.NS (secondary) on each reset() call.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_client_py_has_no_server_imports():
    text = (ROOT / "client.py").read_text(encoding="utf-8")
    assert "from server" not in text
    assert "import server" not in text


def test_inference_entrypoint_uses_http_client_only():
    text = (ROOT / "inference.py").read_text(encoding="utf-8")
    assert "from server" not in text
    assert "import server" not in text


# ---------------------------------------------------------------------------
# Dataset rotation tests
# ---------------------------------------------------------------------------

from server.trading_environment import TradingEnvironment  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_class_state():
    TradingEnvironment._global_episode_idx = 0
    TradingEnvironment._dataset_cache.clear()
    yield
    TradingEnvironment._global_episode_idx = 0
    TradingEnvironment._dataset_cache.clear()


def _make_env():
    return TradingEnvironment(
        episode_length=20,
        random_episode_start=False,
    )


def test_first_reset_uses_primary_spy_dataset():
    env = _make_env()
    obs = env.reset()
    assert obs.dataset == "spy"
    assert env.active_dataset == "spy"


def test_second_reset_switches_to_nifty50():
    nifty_csv = ROOT / "data" / "nifty50_prices.csv"
    if not nifty_csv.exists():
        pytest.skip("nifty50_prices.csv not present in data/ folder")
    env = _make_env()
    env.reset()
    obs = env.reset()
    assert obs.dataset == "nifty50"
    assert env.active_dataset == "nifty50"


def test_third_reset_returns_to_spy():
    nifty_csv = ROOT / "data" / "nifty50_prices.csv"
    if not nifty_csv.exists():
        pytest.skip("nifty50_prices.csv not present in data/ folder")
    env = _make_env()
    env.reset()
    env.reset()
    obs = env.reset()
    assert obs.dataset == "spy"


def test_state_endpoint_carries_active_dataset():
    env = _make_env()
    env.reset()
    assert env.state().dataset == "spy"
    nifty_csv = ROOT / "data" / "nifty50_prices.csv"
    if not nifty_csv.exists():
        return
    env.reset()
    assert env.state().dataset == "nifty50"


def test_observations_remain_finite_across_dataset_switch():
    nifty_csv = ROOT / "data" / "nifty50_prices.csv"
    if not nifty_csv.exists():
        pytest.skip("nifty50_prices.csv not present in data/ folder")
    env = _make_env()
    for _ in range(2):
        obs = env.reset()
        assert all(
            isinstance(v, float) and (v == v) and v not in (float("inf"), float("-inf"))
            for v in obs.market_features
        ), f"Non-finite feature on dataset {obs.dataset}"
