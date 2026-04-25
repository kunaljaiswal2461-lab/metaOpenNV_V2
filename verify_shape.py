import sys
from server.trading_environment import TradingEnvironment
from client import TradingEnv
from models import TradingObservation

env = TradingEnvironment()
obs = env.reset()

import numpy as np
arr = np.array(obs.market_features + [obs.port_cash, obs.holdings, obs.port_val], dtype=np.float32)
expected_size = env.window * len(env.feat) + 3
print('Array shape:', arr.shape)
print(f'Expected:   ({expected_size},)')
print('Match:', arr.shape == (expected_size,))
