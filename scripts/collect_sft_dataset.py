"""
Collect SFT JSONL from the live Trading API (HTTP only).

Uses SMA20-distance teacher labels (same heuristic as Phase 4 benchmark).

Usage (repo root, Space or local server running):
    set SPACE_URL=https://huggingface.co/spaces/...
    python scripts/collect_sft_dataset.py --episodes 8 --max-steps 400

Output:
    data/trl_sft_train.jsonl   (one JSON object per line, TRL \"messages\" format)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from client import TradingEnv
from trl_data.prompt_utils import build_sft_row, teacher_sma20_action


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10, help="Number of reset/step rollouts")
    p.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    p.add_argument(
        "--task-name",
        type=str,
        default="risk_aware_trading",
        help="OpenEnv task name for /reset body",
    )
    p.add_argument(
        "--out",
        type=str,
        default=os.path.join("data", "trl_sft_train.jsonl"),
        help="Output JSONL path",
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="Use in-process TradingEnvironment (dev only; judges should use HTTP + SPACE_URL)",
    )
    p.add_argument(
        "--messages-format",
        action="store_true",
        help="Write {\"messages\": [...]} instead of {\"text\": ...} (instruct-tuned models)",
    )
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    n = 0

    if args.local:
        from models import TradingAction
        from server.trading_environment import TradingEnvironment

        env = TradingEnvironment(window=20, random_episode_start=True, seed=42)
        print("Collecting LOCAL (not HTTP) task=", args.task_name)

        def reset_task():
            return env.reset(task_name=args.task_name)

        def step_act(a: int):
            return env.step(TradingAction(action=a, amount=1.0))

    else:
        env = TradingEnv()
        base = os.environ.get("SPACE_URL", "http://localhost:7860")
        print(f"Collecting from SPACE_URL={base} task={args.task_name}")

        def reset_task():
            return env.reset(task_name=args.task_name)

        def step_act(a: int):
            return env.step(a)

    with open(args.out, "w", encoding="utf-8") as f:
        for ep in range(args.episodes):
            obs = reset_task()
            steps = 0
            while not obs.done and steps < args.max_steps:
                a = teacher_sma20_action(obs)
                row = build_sft_row(
                    obs,
                    a,
                    task_name=args.task_name,
                    use_messages=args.messages_format,
                )
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                obs = step_act(a)
                steps += 1
            print(f"  episode {ep+1}/{args.episodes} steps={steps} total_rows={n}")

    print(f"Wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
