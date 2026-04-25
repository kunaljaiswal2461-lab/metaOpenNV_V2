import os, json, time
import numpy as np
from client import TradingEnv
from agent.dqn_agent import DQNAgent

# Verify local FastAPI
env = TradingEnv()
agent = DQNAgent(state_size=123, action_size=3)
model_path = os.path.join("checkpoints", "best_model.pt")

print(f"Loading trained model from {model_path}...")
agent.load(model_path)

task = "spy_trading"
print(f"Running test on {task} for 5 steps...")
obs = env.reset(task_name=task)
for i in range(5):
    arr = env.obs_to_array(obs)
    action = agent.select_action(arr)
    obs = env.step(action)
    print(f"  Step {i+1}: Action {action}, Reward {obs.reward:.6f}")

print("\nLOCAL INFERENCE TEST PASSED")
