import numpy as np
from reward import RewardCalculator
from server.trading_environment import TradingEnvironment
from fastapi.testclient import TestClient
from server.app import app

def run_tests():
    print("Testing env/reward.py importable with no errors...")
    calc = RewardCalculator()
    rew = calc.compute(10100.0, 10000.0, 10100.0, True)
    if isinstance(rew, float) and np.isfinite(rew):
        print(f"Reward OK: {rew}")
    else:
        print(f"Reward Error: {rew}")

    print("\nTesting trading_environment.py and /reset endpoint...")
    client = TestClient(app)
    response = client.post("/reset")
    if response.status_code == 200:
        data = response.json()
        print(f"curl -X POST /reset still returns valid JSON: {True}")
        # print specific keys to verify
        print(f"Keys: {list(data.keys())}")
    else:
        print(f"Endpoint failed with status {response.status_code}")

if __name__ == '__main__':
    run_tests()
