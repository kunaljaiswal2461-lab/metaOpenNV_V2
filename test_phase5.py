import numpy as np
from reward import RewardCalculator
from fastapi.testclient import TestClient
from server.app import app


def run_tests():
    print("Testing reward.py importable with no errors...")
    calc = RewardCalculator()
    rew = calc.compute(10100.0, 10000.0, 10100.0, True)
    if isinstance(rew, float) and np.isfinite(rew):
        print(f"Reward OK: {rew}")
    else:
        print(f"Reward Error: {rew}")

    print("\nTesting HTTP API (reset / step / state) via TestClient...")
    client = TestClient(app)
    r = client.post("/reset", json={})
    assert r.status_code == 200, r.text
    data = r.json()
    print(f"POST /reset keys: {list(data.keys())}")

    r2 = client.get("/state")
    assert r2.status_code == 200, r2.text
    st = r2.json()
    assert "portfolio_value" in st and "current_step" in st
    print(f"GET /state keys: {list(st.keys())}")

    r3 = client.post("/step", json={"action": 0, "amount": 1.0})
    assert r3.status_code == 200, r3.text
    print("POST /step (HOLD) OK")

if __name__ == '__main__':
    run_tests()
