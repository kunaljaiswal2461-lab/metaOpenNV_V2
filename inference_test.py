import os

import numpy as np

from agent.dqn_agent import DQNAgent
from client import TradingEnv

# Must match training / default Space window (20 -> 203-dim vector).
TASK = "risk_aware_trading"
STATE_SIZE = 203


def main() -> None:
    env = TradingEnv()
    agent = DQNAgent(state_size=STATE_SIZE, action_size=3)
    model_path = os.path.join("checkpoints", "best_model.pt")
    if not os.path.isfile(model_path):
        print(f"Skip: no checkpoint at {model_path} (train first with python -m training.train)")
        return

    print(f"Loading trained model from {model_path}...")
    agent.load(model_path)
    agent.epsilon = 0.0

    print(f"Running test on {TASK} for 5 steps...")
    obs = env.reset(task_name=TASK)
    for i in range(5):
        arr = env.obs_to_array(obs)
        assert arr.shape[0] == STATE_SIZE, (arr.shape, STATE_SIZE)
        action = agent.select_action(arr)
        obs = env.step(action)
        print(f"  Step {i+1}: Action {action}, Reward {obs.reward:.6f}")

    print("\nLOCAL INFERENCE TEST PASSED")


if __name__ == "__main__":
    main()
