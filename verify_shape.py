"""
Verify flattened observation shape via the HTTP client only (no server imports).

Requires a running API: `python server/app.py` or set SPACE_URL to your Space.
"""
import sys

import numpy as np

from client import TradingEnv

# Expected (window x 7 market features) + 3 portfolio scalars.
# 7-feature schema, multicollinearity-audited Apr 2026.
_EXPECTED = {
    "spy_trading": 10 * 7 + 3,
    "risk_aware_trading": 20 * 7 + 3,
    "multi_horizon_trading": 50 * 7 + 3,
}


def main() -> None:
    env = TradingEnv()
    for task, expected in _EXPECTED.items():
        try:
            obs = env.reset(task_name=task)
        except Exception as exc:
            print(
                "Connection failed - start the server "
                "(e.g. `python server/app.py`) or set SPACE_URL.\n",
                exc,
                file=sys.stderr,
            )
            sys.exit(1)
        arr = env.obs_to_array(obs)
        ok = arr.shape == (expected,) and len(obs.market_features) + 3 == expected
        print(f"task={task!r} shape={arr.shape} expected=({expected},) match={ok}")
        if not ok:
            sys.exit(1)
    st = env.state()
    print("state():", st.model_dump())
    print("OK: client-only shape checks passed")


if __name__ == "__main__":
    main()
