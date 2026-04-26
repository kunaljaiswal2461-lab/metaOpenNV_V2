from pydantic import BaseModel
from typing import List, Optional

class TradingAction(BaseModel):
    action: int          # 0=HOLD, 1=BUY, 2=SELL
    amount: float = 1.0  # 0.0 to 1.0 (Percentage of Cash/Holdings)

class TradingObservation(BaseModel):
    market_features: List[float]  # flattened window_size * feature_count values
    port_cash: float              # cash in bank in USD
    holdings: float               # stocks held
    port_val: float               # (portfolio + holdings) in USD
    portfolio_value: float
    current_step: int
    close_price: float = 0.0      # Current Stock Price for human display
    reward: float = 0.0
    done: bool = False
    dataset: str = "spy"          # active OHLCV dataset for this episode

class TradingState(BaseModel):
    portfolio_value: float
    cash: float
    holdings: float
    current_step: int
    history: list
    INITIAL_CASH: int
    TRANSACTION_COST: float = 0.001
    dataset: str = "spy"          # active OHLCV dataset for this episode
