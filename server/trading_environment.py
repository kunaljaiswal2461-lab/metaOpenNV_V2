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
    def __init__(
        self,
        window=20,
        initial_capital=10000.0,
        cost=0.001,
        random_episode_start=True,
        episode_length=390,
        seed=None,
    ):
        csv_path = os.path.join(ROOT_DIR, 'data', 'spy_prices.csv')
        if not os.path.exists(csv_path):
            csv_path = 'data/spy_prices.csv'
        
        train_df, _ = load_and_preprocess(csv_path)
        self.df = train_df.reset_index(drop=True)
        self.window = window
        self.initial_cash = initial_capital
        self.cost = cost
        self.random_episode_start = random_episode_start
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)
        self.feat = [
            'log_return', 'sma5_dist', 'sma20_dist', 'rsi',
            'norm_volume', 'volatility', 'vwap_dist', 'ema12_dist',
            'macd_signal_gap', 'atr_pct'
        ]
        self._reset_state()

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
            done=bool(done)
        )

    def state(self) -> TradingState:
        return TradingState(
            portfolio_value=sanitize_value(self.pv),
            cash=sanitize_value(self.cash),
            holdings=sanitize_value(self.hold),
            current_step=int(self.current_step),
            history=[],
            INITIAL_CASH=int(self.initial_cash),
            TRANSACTION_COST=self.cost
        )
