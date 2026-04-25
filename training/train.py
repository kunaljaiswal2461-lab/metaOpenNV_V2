"""
training/train.py — DQN training loop for the SPY trading agent.

**OpenEnv client/server note:** Remote agents and judge-facing demos must use
`client.TradingEnv` (HTTP to `/reset`, `/step`, `/state` only). This file imports
`TradingEnvironment` directly for fast local RL; that is intentional for the
training job, not a pattern for production clients.

Usage:
    python -m training.train                     # 300 episodes (default)
    python -m training.train --episodes 10       # quick smoke test
    python -m training.train --resume            # resume from last checkpoint

Saves:
    checkpoints/best_model.pt   — best episode score seen during training
    checkpoints/last_model.pt   — model after every checkpoint interval
"""

import argparse
import os
import sys
import time
import numpy as np

import torch

# ── project root on path ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from server.trading_environment import TradingEnvironment
from models import TradingAction
from agent.dqn_agent import DQNAgent

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def obs_to_array(obs) -> np.ndarray:
    """Flatten TradingObservation into a 1-D float32 numpy vector."""
    return np.array(
        obs.market_features + [obs.port_cash, obs.holdings, obs.port_val],
        dtype=np.float32,
    )


def train(
    episodes: int = 300,
    resume: bool = False,
    ckpt_dir: str = "checkpoints",
    ckpt_interval: int = 25,
    log_interval: int = 10,
) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    last_path = os.path.join(ckpt_dir, "last_model.pt")

    env = TradingEnvironment(window=20, random_episode_start=True)
    state_size = env.window * len(env.feat) + 3
    agent = DQNAgent(
        state_size=state_size,
        action_size=3,
        lr=3e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        tau=0.005,
        batch_size=64,
        buffer_size=50_000,
    )

    start_episode = 0
    best_score = -float("inf")

    if resume and os.path.exists(last_path):
        agent.load(last_path)
        print(f"[RESUME] Loaded checkpoint from {last_path}")

    print(
        f"[TRAIN] Starting {episodes} episodes | "
        f"state={state_size} ({env.window}x{len(env.feat)}+3) | "
        "actions=3 (HOLD/BUY/SELL)"
    )
    print(f"        checkpoint dir: {ckpt_dir}/")
    print()

    episode_scores = []

    for ep in range(start_episode, episodes):
        t0 = time.time()
        obs = env.reset()
        state = obs_to_array(obs)
        total_reward = 0.0
        total_loss = 0.0
        loss_count = 0
        done = False
        step = 0

        while not done:
            action = agent.select_action(state)
            obs_next = env.step(TradingAction(action=action))
            next_state = obs_to_array(obs_next)
            reward = obs_next.reward
            done = obs_next.done

            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                total_loss += loss
                loss_count += 1

            state = next_state
            total_reward += reward
            step += 1

        agent.decay_epsilon()

        # Normalize score to a rough 0-1 range for display
        score = (total_reward + 1.0) / 2.0
        episode_scores.append(score)
        elapsed = time.time() - t0

        # Save best model
        if score > best_score:
            best_score = score
            agent.save(best_path)
            best_tag = " ← BEST"
        else:
            best_tag = ""

        # Periodic checkpoint
        if (ep + 1) % ckpt_interval == 0:
            agent.save(last_path)

        # Console log
        if (ep + 1) % log_interval == 0:
            avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
            recent_avg = float(np.mean(episode_scores[-log_interval:]))
            print(
                f"Ep {ep+1:4d}/{episodes} | "
                f"Score: {score:.4f} | "
                f"Avg({log_interval}): {recent_avg:.4f} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Loss: {avg_loss:.5f} | "
                f"Steps: {step:4d} | "
                f"{elapsed:.1f}s"
                f"{best_tag}"
            )

    # Final save
    agent.save(last_path)
    print()
    print(f"[DONE] Training complete after {episodes} episodes.")
    print(f"       Best score : {best_score:.4f}")
    print(f"       Best model : {best_path}")
    print(f"       Last model : {last_path}")


# ─────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the DQN trading agent")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes")
    parser.add_argument("--resume",   action="store_true",    help="Resume from last checkpoint")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--ckpt-interval", type=int, default=25, help="Save checkpoint every N episodes")
    parser.add_argument("--log-interval",  type=int, default=10, help="Print progress every N episodes")
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        resume=args.resume,
        ckpt_dir=args.ckpt_dir,
        ckpt_interval=args.ckpt_interval,
        log_interval=args.log_interval,
    )
