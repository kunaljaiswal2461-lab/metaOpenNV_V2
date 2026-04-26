"""
Phase 4 - Evidence of learning: baselines vs DQN on the same TradingEnvironment.

Produces:
  - results/phase4_episode_return.png   (episode total reward per policy)
  - results/phase4_metrics.md           (table: mean return, std, max drawdown, trades)

Usage (from repo root):
    python -m eval.phase4_benchmark
    python -m eval.phase4_benchmark --eval-episodes 25 --train-episodes 40

Requires: torch, matplotlib (see requirements.txt).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent.dqn_agent import DQNAgent
from models import TradingAction, TradingObservation
from server.trading_environment import TradingEnvironment

# Feature order must match trading_environment.self.feat (7-feature schema,
# multicollinearity-audited Apr 2026).
_FEAT = [
    "log_return",
    "sma5_dist",
    "sma20_dist",
    "rsi",
    "norm_volume",
    "volatility",
    "vwap_dist",
]
_SMA20_IDX = _FEAT.index("sma20_dist")


def obs_to_array(obs: TradingObservation, window: int) -> np.ndarray:
    return np.array(
        obs.market_features + [obs.port_cash, obs.holdings, obs.port_val],
        dtype=np.float32,
    )


def _last_bar_sma20_dist(obs: TradingObservation, window: int) -> float:
    mf = obs.market_features
    n_feat = len(_FEAT)
    if len(mf) < n_feat * window:
        return 0.0
    last_row_start = (window - 1) * n_feat
    return float(mf[last_row_start + _SMA20_IDX])


def max_drawdown(portfolio_values: List[float]) -> float:
    peak = portfolio_values[0]
    worst = 0.0
    for v in portfolio_values:
        peak = max(peak, v)
        if peak > 0:
            worst = max(worst, (peak - v) / peak)
    return float(worst)


@dataclass
class EpisodeStats:
    total_reward: float
    final_pv: float
    max_drawdown: float
    n_trades: int
    steps: int


def run_episode(
    env: TradingEnvironment,
    act_fn: Callable[[TradingObservation, int], int],
) -> EpisodeStats:
    obs = env.reset()
    total_r = 0.0
    pv_path = [float(obs.port_val)]
    trades = 0
    step_i = 0
    while not obs.done:
        a = act_fn(obs, step_i)
        prev_cash, prev_hold = obs.port_cash, obs.holdings
        obs = env.step(TradingAction(action=a, amount=1.0))
        total_r += float(obs.reward)
        pv_path.append(float(obs.port_val))
        if a in (1, 2):
            if abs(obs.port_cash - prev_cash) > 1e-6 or abs(obs.holdings - prev_hold) > 1e-6:
                trades += 1
        step_i += 1
    return EpisodeStats(
        total_reward=total_r,
        final_pv=float(obs.port_val),
        max_drawdown=max_drawdown(pv_path),
        n_trades=trades,
        steps=step_i,
)


def train_dqn(
    env: TradingEnvironment,
    episodes: int,
    state_size: int,
    ckpt_out: str,
) -> DQNAgent:
    agent = DQNAgent(
        state_size=state_size,
        action_size=3,
        lr=3e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.97,
        tau=0.005,
        batch_size=64,
        buffer_size=20_000,
    )
    for ep in range(episodes):
        obs = env.reset()
        state = obs_to_array(obs, env.window)
        done = False
        while not done:
            action = agent.select_action(state)
            nxt = env.step(TradingAction(action=action, amount=1.0))
            next_state = obs_to_array(nxt, env.window)
            agent.remember(state, action, float(nxt.reward), next_state, bool(nxt.done))
            loss = agent.learn()
            _ = loss
            state = next_state
            done = nxt.done
            obs = nxt
        agent.decay_epsilon()
    agent.save(ckpt_out)
    return agent


def evaluate_policy(
    env: TradingEnvironment,
    act_fn: Callable[[TradingObservation, int], int],
    episodes: int,
) -> List[EpisodeStats]:
    rows: List[EpisodeStats] = []
    for _ in range(episodes):
        rows.append(run_episode(env, act_fn))
    return rows


def summarize(name: str, rows: List[EpisodeStats]) -> Dict[str, float]:
    rewards = [r.total_reward for r in rows]
    dds = [r.max_drawdown for r in rows]
    trades = [r.n_trades for r in rows]
    return {
        "policy": name,
        "mean_return": float(np.mean(rewards)),
        "std_return": float(np.std(rewards)),
        "mean_max_dd": float(np.mean(dds)),
        "mean_trades": float(np.mean(trades)),
        "mean_final_pv": float(np.mean([r.final_pv for r in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 benchmark: baselines vs DQN")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Episodes per policy for curves")
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=70,
        help="DQN training episodes before eval (raise to 120+ for a clearer greedy lift)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)
    ckpt_path = os.path.join(args.results_dir, "phase4_dqn.pt")

    env_train = TradingEnvironment(
        window=20,
        random_episode_start=True,
        seed=args.seed,
    )
    env_eval = TradingEnvironment(
        window=20,
        random_episode_start=False,
        seed=args.seed + 1,
    )
    w = env_eval.window
    state_size = w * len(env_eval.feat) + 3

    print(f"[Phase4] state_size={state_size} eval_episodes={args.eval_episodes}")

    # --- Train DQN (same physics as Space; diverse episodes during training) ---
    print(f"[Phase4] Training DQN for {args.train_episodes} episodes -> {ckpt_path}")
    agent = train_dqn(env_train, args.train_episodes, state_size, ckpt_path)
    agent.epsilon = 0.0  # greedy eval

    policies: Dict[str, Callable[[TradingObservation, int], int]] = {
        "random": lambda obs, t: random.randrange(3),
        "always_hold": lambda obs, t: 0,
        "buy_once_then_hold": lambda obs, t: 1 if t == 0 else 0,
        "sma20_trend": lambda obs, t: (
            1
            if _last_bar_sma20_dist(obs, w) > 0.0
            else (2 if _last_bar_sma20_dist(obs, w) < -0.02 else 0)
        ),
        "dqn_greedy": lambda obs, t: agent.select_action(obs_to_array(obs, w)),
    }

    series: Dict[str, List[float]] = {}
    table_rows: List[Dict[str, float]] = []

    for pname, fn in policies.items():
        stats = evaluate_policy(env_eval, fn, args.eval_episodes)
        series[pname] = [s.total_reward for s in stats]
        table_rows.append(summarize(pname, stats))
        print(f"[Phase4] {pname}: mean_return={table_rows[-1]['mean_return']:.4f}")

    # --- Plot: episode index vs total reward (lines per policy) ---
    plt.figure(figsize=(10, 5))
    ep_x = np.arange(1, args.eval_episodes + 1)
    for pname, ys in series.items():
        plt.plot(ep_x, ys, linewidth=1.5, label=pname, alpha=0.85)
    plt.xlabel("Episode index")
    plt.ylabel("Episode total reward (sum of step rewards)")
    plt.title("Phase 4 - Baselines vs DQN (same env, fixed episode starts)")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, "phase4_episode_return.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[Phase4] Wrote {plot_path}")

    # --- Bar: mean episode return ---
    plt.figure(figsize=(8, 4))
    names = [r["policy"] for r in table_rows]
    means = [r["mean_return"] for r in table_rows]
    stds = [r["std_return"] for r in table_rows]
    x = np.arange(len(names))
    plt.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.85)
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylabel("Mean episode total reward ± std")
    plt.title("Phase 4 - Average performance by policy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    bar_path = os.path.join(args.results_dir, "phase4_mean_return_bar.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"[Phase4] Wrote {bar_path}")

    # --- Markdown table ---
    md_path = os.path.join(args.results_dir, "phase4_metrics.md")
    lines = [
        "# Phase 4 metrics (auto-generated)",
        "",
        f"Eval episodes per policy: **{args.eval_episodes}**. DQN train episodes: **{args.train_episodes}**. `random_episode_start=False` so policies are comparable on aligned rollouts.",
        "",
        "| Policy | Mean episode reward | Std | Mean max drawdown | Mean # trades | Mean final PV |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in table_rows:
        lines.append(
            f"| {r['policy']} | {r['mean_return']:.4f} | {r['std_return']:.4f} | "
            f"{r['mean_max_dd']:.4f} | {r['mean_trades']:.2f} | {r['mean_final_pv']:.2f} |"
        )
    lines.append("")
    ckpt_rel = ckpt_path.replace(os.sep, "/")
    lines.append(f"Checkpoint (local, gitignored): `{ckpt_rel}`")
    lines.append("")
    lines.append(
        "Interpretation: compare **mean episode reward** and **mean final PV** vs `random` and `always_hold`. "
        "Increase `--train-episodes` (e.g. 120+) for a stronger DQN signal on this env."
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Phase4] Wrote {md_path}")


if __name__ == "__main__":
    main()
