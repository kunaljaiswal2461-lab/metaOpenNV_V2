import os
import time
import numpy as np
import requests
from models import TradingAction, TradingObservation, TradingState

# How many transient HTTP/JSON failures we will absorb before giving up.
# Tuned for HF Space proxy hiccups during long collection runs (thousands
# of sequential POSTs through the gateway).
_DEFAULT_RETRIES = int(os.environ.get("TRADING_API_RETRIES", "6"))
_DEFAULT_TIMEOUT = float(os.environ.get("TRADING_API_TIMEOUT", "30"))


def _request_json(session: requests.Session, method: str, url: str, *, json_body=None,
                  retries: int = _DEFAULT_RETRIES, timeout: float = _DEFAULT_TIMEOUT):
    """POST/GET that retries with exponential backoff on transient errors.

    Treats any non-2xx response, connection error, or JSON-decode failure as
    transient (HF Space gateway sometimes returns HTML 5xx pages mid-stream).
    """
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if method == "POST":
                resp = session.post(url, json=json_body, timeout=timeout)
            else:
                resp = session.get(url, timeout=timeout)
            if resp.status_code >= 500 or resp.status_code == 429:
                raise requests.HTTPError(f"HTTP {resp.status_code} body={resp.text[:200]!r}")
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError, ValueError) as e:
            last_err = e
            if attempt == retries:
                break
            sleep_for = min(8.0, 0.5 * (2 ** attempt))
            time.sleep(sleep_for)
    raise RuntimeError(f"{method} {url} failed after {retries+1} attempts: {last_err!r}")


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
        data = _request_json(self.session, "POST", f"{self.base_url}/reset", json_body=payload)
        self.obs = TradingObservation(**data)
        return self.obs

    def step(self, action: int, amount: float = 1.0):
        act = TradingAction(action=action, amount=amount)
        data = _request_json(self.session, "POST", f"{self.base_url}/step", json_body=act.model_dump())
        self.obs = TradingObservation(**data)
        return self.obs

    def state(self) -> TradingState:
        """Read-only MDP state (portfolio, step, costs). Does not advance the episode."""
        data = _request_json(self.session, "GET", f"{self.base_url}/state")
        return TradingState(**data)

    def obs_to_array(self, obs: TradingObservation) -> np.ndarray:
        return np.array(
            obs.market_features + [obs.port_cash, obs.holdings, obs.port_val],
            dtype=np.float32
        )
