import numpy as np
import pandas as pd
import sys
import os
import json

try:
    from openenv import Environment
except ImportError:
    from openenv_core import Environment

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models import TradingAction, TradingObservation, TradingState
from reward import RewardCalculator
from data.preprocess import load_and_preprocess

# Maps openenv.yaml task names to lookback window length (minutes of history).
_TASK_WINDOWS = {
    "spy_trading": 10,
    "risk_aware_trading": 20,
    "multi_horizon_trading": 50,
}


def sanitize_value(val):
    """Deep cleaning of values to ensure JSON compliance for OpenAI."""
    try:
        if isinstance(val, (list, np.ndarray)):
            return [sanitize_value(v) for v in val]
        if pd.isna(val) or np.isnan(val) or np.isinf(val):
            return 0.0
        return float(val)
    except:
        return 0.0

class TradingEnvironment(Environment):
    # Datasets are visited in this order, alternating across reset() calls.
    # The very first reset uses the primary (SPY) set, the next reset
    # switches to the secondary (Nifty 50 / RELIANCE.NS) set, and after
    # that episode finishes the cycle wraps back to SPY. Counter is a
    # class attribute so a single FastAPI process alternates consistently.
    _DATASETS = [
        ("spy", os.path.join("data", "spy_prices.csv")),
        ("nifty50", os.path.join("data", "nifty50_prices.csv")),
    ]
    _dataset_cache: dict = {}
    _global_episode_idx: int = 0

    def __init__(
        self,
        window=None,
        initial_capital=10000.0,
        cost=0.001,
        random_episode_start=True,
        episode_length=390,
        seed=None,
    ):
        if window is None:
            window = int(os.environ.get("WINDOW_SIZE", "20"))
        self._default_window = int(window)
        self.window = self._default_window
        self.initial_cash = initial_capital
        self.cost = cost
        self.random_episode_start = random_episode_start
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)
        # 7-feature observation schema (multicollinearity-audited Apr 2026).
        # EMA-12 dist, MACD-signal gap, and ATR% were dropped as redundant.
        self.feat = [
            'log_return', 'sma5_dist', 'sma20_dist', 'rsi',
            'norm_volume', 'volatility', 'vwap_dist'
        ]
        # Active dataset for this episode. Boot up on the primary set so
        # the very first request that lands before any reset() still has
        # a usable observation window.
        self.active_dataset = "spy"
        self._load_dataset(self.active_dataset)
        self._reset_state()

    @classmethod
    def _resolve_csv_path(cls, name: str) -> str:
        for n, rel_path in cls._DATASETS:
            if n == name:
                root_path = os.path.join(ROOT_DIR, rel_path)
                if os.path.exists(root_path):
                    return root_path
                return rel_path
        raise ValueError(f"Unknown dataset name: {name}")

    def _load_dataset(self, name: str) -> None:
        """Load (and cache) the preprocessed dataframe for a given dataset.

        Falls back to the primary 'spy' dataset if the requested file is
        missing on disk so the environment never breaks for users who
        only have the primary CSV checked out.
        """
        cache = TradingEnvironment._dataset_cache
        if name not in cache:
            csv_path = self._resolve_csv_path(name)
            if not os.path.exists(csv_path):
                if name != "spy":
                    self._load_dataset("spy")
                    self.active_dataset = "spy"
                    return
                raise FileNotFoundError(
                    f"Required primary dataset not found at {csv_path}"
                )
            train_df, _ = load_and_preprocess(csv_path)
            cache[name] = train_df.reset_index(drop=True)
        self.df = cache[name].reset_index(drop=True)
        self.active_dataset = name

    def _reset_state(self):
        self.cash = self.initial_cash
        self.hold = 0.0
        self.pv = self.initial_cash
        self.peak = self.initial_cash
        self.current_step = self._sample_episode_start()
        self.end_step = self._sample_episode_end(self.current_step)
        self.reward_calc = RewardCalculator(tc_rate=self.cost)
        self.done = False
        self.last_reward = 0.0

    def _sample_episode_start(self):
        min_start = self.window
        max_start = max(min_start, len(self.df) - self.episode_length - 2)
        if not self.random_episode_start or max_start <= min_start:
            return min_start
        return int(self.rng.integers(min_start, max_start + 1))

    def _sample_episode_end(self, start_step):
        return min(len(self.df) - 1, start_step + self.episode_length)

    def reset(self, task_name=None) -> TradingObservation:
        # Rotate datasets across reset() calls so each new episode gets
        # the next dataset in TradingEnvironment._DATASETS. The counter
        # is process-global (class attribute), guaranteeing a consistent
        # alternation regardless of which client kicked off the reset.
        cls = TradingEnvironment
        idx = cls._global_episode_idx % len(cls._DATASETS)
        next_dataset = cls._DATASETS[idx][0]
        cls._global_episode_idx += 1
        self._load_dataset(next_dataset)

        if task_name and task_name in _TASK_WINDOWS:
            self.window = _TASK_WINDOWS[task_name]
        else:
            self.window = self._default_window
        self._reset_state()
        return self._get_obs(reward=0.0, done=False)

    def step(self, action: TradingAction) -> TradingObservation:
        if self.current_step >= self.end_step:
            self.done = True
            return self._get_obs(reward=0, done=True)

        price = float(self.df.loc[self.current_step, 'close'])
        prev_pv = self.pv
        
        act = action.action  
        amount = getattr(action, 'amount', 1.0)
        
        trade_executed = False
        if act == 1 and self.cash > 0:
            invest = self.cash * amount
            self.hold += invest / price * (1 - self.cost)
            self.cash -= invest
            trade_executed = True
        elif act == 2 and self.hold > 0:
            selling = self.hold * amount
            self.cash += selling * price * (1 - self.cost)
            self.hold -= selling
            trade_executed = True
            
        self.pv = self.cash + (self.hold * price)
        self.peak = max(self.peak, self.pv)
        
        market_return = float(self.df.iloc[self.current_step]['log_return'])
        
        rew = self.reward_calc.compute(
            pv=self.pv,
            prev_pv=prev_pv,
            peak=self.peak,
            trade_executed=trade_executed,
            market_return=market_return
        )
        self.last_reward = rew 
        
        self.current_step += 1
        if self.pv < self.initial_cash * 0.4 or self.current_step >= self.end_step:
            self.done = True
            
        return self._get_obs(reward=self.last_reward, done=self.done)

    def _get_obs(self, reward, done) -> TradingObservation:
        s = self.current_step
        w = self.window
        
        # 🧪 CRITICAL SANITIZATION STEP
        raw_feats = self.df.loc[s-w:s-1, self.feat].values.flatten().tolist()
        clean_feats = sanitize_value(raw_feats)
        
        price = float(self.df.loc[s, 'close'])
        
        return TradingObservation(
            market_features=clean_feats,
            port_cash=sanitize_value(self.cash),
            holdings=sanitize_value(self.hold),
            port_val=sanitize_value(self.pv),
            portfolio_value=sanitize_value(self.pv),
            current_step=int(s),
            close_price=sanitize_value(price),
            reward=sanitize_value(reward if reward != 0 else self.last_reward),
            done=bool(done),
            dataset=str(self.active_dataset),
        )

    def state(self) -> TradingState:
        return TradingState(
            portfolio_value=sanitize_value(self.pv),
            cash=sanitize_value(self.cash),
            holdings=sanitize_value(self.hold),
            current_step=int(self.current_step),
            history=[],
            INITIAL_CASH=int(self.initial_cash),
            TRANSACTION_COST=self.cost,
            dataset=str(self.active_dataset),
        )
