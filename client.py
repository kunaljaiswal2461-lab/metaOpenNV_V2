import os
import numpy as np
import requests
from models import TradingAction, TradingObservation, TradingState

class TradingEnv:
    def __init__(self):
        # Use SPACE_URL env var when running against live HF Space,
        # falls back to localhost for local development
        self.base_url = os.environ.get("SPACE_URL", "http://localhost:7860").rstrip("/")
        self.obs = None
        self.session = requests.Session()

    def reset(self, task_name=None, **kwargs):
        payload = {}
        if task_name is not None:
            payload["task_name"] = task_name
        res = self.session.post(
            f"{self.base_url}/reset",
            json=payload if payload else {},
        )
        self.obs = TradingObservation(**res.json())
        return self.obs

    def step(self, action: int):
        act = TradingAction(action=action)
        res = self.session.post(f"{self.base_url}/step", json=act.model_dump())
        self.obs = TradingObservation(**res.json())
        return self.obs

    def obs_to_array(self, obs: TradingObservation) -> np.ndarray:
        return np.array(
            obs.market_features + [obs.port_cash, obs.holdings, obs.port_val],
            dtype=np.float32
        )
