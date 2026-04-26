"""
Collect SFT JSONL from the live Trading API (HTTP) or the in-process env.

Defaults reproduce the original Phase-3 dataset (compact prompt, SMA20 teacher,
no history tail), so previously trained adapters remain valid. Opt in to the
richer setup with ``--prompt full`` / ``--teacher composite`` / ``--history-len``.

Usage (Space or local server running):

    set SPACE_URL=https://huggingface.co/spaces/<user>/<repo>
    python scripts/collect_sft_dataset.py --episodes 8 --max-steps 400

Track-H ("big-context") run:

    python scripts/collect_sft_dataset.py \
        --episodes 30 --max-steps 400 --task-name multi_horizon_trading \
        --prompt full --teacher composite --history-len 5 \
        --out data/trl_sft_train_full.jsonl

Output:
    data/trl_sft_train.jsonl   one JSON object per line ({"text": ...} by default,
                               or {"messages": [...]} with --messages-format).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from client import TradingEnv
from trl_data.prompt_utils import build_sft_row
from trl_data.teacher import TEACHERS, get_teacher


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        help='Write {"messages": [...]} instead of {"text": ...} (instruct-tuned models)',
    )
    p.add_argument(
        "--prompt",
        choices=("compact", "full"),
        default="compact",
        help="Prompt style; full exposes all 10 last-bar features (Track H).",
    )
    p.add_argument(
        "--teacher",
        choices=tuple(TEACHERS),
        default="sma20",
        help="Teacher policy used to label each observation.",
    )
    p.add_argument(
        "--history-len",
        type=int,
        default=0,
        help="Last K (action, reward) pairs to attach to the prompt (0 disables).",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    teacher = get_teacher(args.teacher)
    history_len = max(0, int(args.history_len))

    if args.local:
        from models import TradingAction
        from server.trading_environment import TradingEnvironment

        env = TradingEnvironment(window=20, random_episode_start=True, seed=42)
        print(f"Collecting LOCAL (not HTTP) task={args.task_name}")

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

    n_rows = 0
    print(
        f"prompt={args.prompt} teacher={args.teacher} history_len={history_len} "
        f"messages_format={args.messages_format}"
    )

    dataset_counts: dict[str, int] = {}
    with open(args.out, "w", encoding="utf-8") as f:
        for ep in range(args.episodes):
            obs = reset_task()
            ds_name = str(getattr(obs, "dataset", "spy") or "spy")
            dataset_counts[ds_name] = dataset_counts.get(ds_name, 0) + 1
            history: deque[tuple[int, float]] = deque(maxlen=history_len) if history_len else deque(maxlen=0)
            steps = 0
            while not obs.done and steps < args.max_steps:
                a = teacher(obs)
                row = build_sft_row(
                    obs,
                    a,
                    task_name=args.task_name,
                    use_messages=args.messages_format,
                    style=args.prompt,
                    history=tuple(history) if history_len else None,
                )
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
                obs = step_act(a)
                if history_len:
                    history.append((int(a), float(obs.reward)))
                steps += 1
            print(
                f"  episode {ep+1}/{args.episodes} dataset={ds_name} "
                f"steps={steps} total_rows={n_rows}"
            )

    summary = ", ".join(f"{k}={v}" for k, v in sorted(dataset_counts.items())) or "n/a"
    print(f"Episode dataset mix: {summary}")
    print(f"Wrote {n_rows} rows to {args.out}")


if __name__ == "__main__":
    main()
