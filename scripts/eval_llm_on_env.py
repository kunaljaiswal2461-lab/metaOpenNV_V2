"""
Run one or more LLMs through the live Trading API and write a metrics table +
bar chart so judges can see whether fine-tuning moved the needle on the env.

Usage:

    # 1) Compare base vs the smoke-fine-tuned distilgpt2 (CPU, slow but works)
    python scripts/eval_llm_on_env.py \
        --models base=Qwen/Qwen2.5-0.5B-Instruct sft=results/phase3_smoke/checkpoint-90 \
        --episodes 10 --max-steps 300

    # 2) Single-model snapshot (any HF id or local adapter dir)
    python scripts/eval_llm_on_env.py --models base=Qwen/Qwen2.5-0.5B-Instruct

    # 3) Track-H eval against the rich prompt (must match what was trained)
    python scripts/eval_llm_on_env.py \
        --models sft_3b=results/phase3_lora_3b \
        --prompt full --history-len 5 --task-name multi_horizon_trading

Outputs:
    results/phase3_eval_metrics.md
    results/phase3_eval_bar.png

Notes:
    - Reuses ``build_user_prompt`` so eval prompts match the training prompts.
    - Loads models sequentially (``del`` + ``torch.cuda.empty_cache()`` between)
      so several models fit on a single Colab T4.
    - HTTP client by default; honour ``SPACE_URL`` for the live Space.
    - ``--local`` uses the in-process ``TradingEnvironment`` (same physics as
      ``collect_sft_dataset --local``; no server required).
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import Counter, deque
from typing import Any, Protocol

# Windows: TRL/transformers read bundled Jinja templates as UTF-8.
os.environ.setdefault("PYTHONUTF8", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from client import TradingEnv
from trl_data.eval_utils import format_metrics_md, parse_action, parse_models
from trl_data.prompt_utils import build_user_prompt
from trl_data.teacher import TEACHERS, get_teacher


class _EnvLike(Protocol):
    def reset(self, task_name: str | None = None, **kwargs: Any) -> Any: ...
    def step(self, action: int, amount: float = 1.0) -> Any: ...


def _resolve_hf_model_path(ident: str) -> str:
    """
    If ``ident`` is an output directory without ``config.json`` at the root
    (e.g. ``results/phase3_lora`` after TRL saves only ``checkpoint-*``),
    pick the lexically greatest ``checkpoint-*`` subfolder so users can pass
    the parent ``--output-dir`` from training.
    """
    if not ident or not os.path.isdir(ident):
        return ident
    root_cfg = os.path.join(ident, "config.json")
    if os.path.isfile(root_cfg):
        return ident
    candidates: list[tuple[int, str]] = []
    try:
        for name in os.listdir(ident):
            if name.startswith("checkpoint-") and name[len("checkpoint-") :].isdigit():
                step = int(name.split("-", 1)[1])
                sub = os.path.join(ident, name)
                if os.path.isfile(os.path.join(sub, "config.json")):
                    candidates.append((step, sub))
    except OSError:
        return ident
    if not candidates:
        return ident
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


class _LocalTradingEnvGym:
    """Same reset/step contract as ``client.TradingEnv`` but in-process."""

    def __init__(self) -> None:
        from models import TradingAction
        from server.trading_environment import TradingEnvironment

        self._TradingAction = TradingAction
        self._env = TradingEnvironment(window=20, random_episode_start=True, seed=42)

    def reset(self, task_name: str | None = None, **kwargs: Any):
        return self._env.reset(task_name=task_name)

    def step(self, action: int, amount: float = 1.0):
        return self._env.step(self._TradingAction(action=action, amount=amount))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        help='One or more "name=identifier" pairs (or bare HF ids).',
    )
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--task-name", type=str, default="risk_aware_trading")
    p.add_argument(
        "--prompt",
        choices=("compact", "full"),
        default="compact",
        help="Must match the prompt style used during SFT.",
    )
    p.add_argument(
        "--teacher",
        choices=tuple(TEACHERS),
        default="sma20",
        help="Teacher used for the agreement metric only (does not change actions).",
    )
    p.add_argument(
        "--history-len",
        type=int,
        default=0,
        help="Last K (action, reward) pairs to attach to the prompt (full style).",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=4,
        help="Greedy decode budget; only the first valid digit is consumed.",
    )
    p.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=1024,
        help="Tokenizer truncation cap for the user prompt.",
    )
    p.add_argument("--metrics-out", type=str, default=os.path.join("results", "phase3_eval_metrics.md"))
    p.add_argument("--bar-out", type=str, default=os.path.join("results", "phase3_eval_bar.png"))
    p.add_argument(
        "--local",
        action="store_true",
        help="Use in-process TradingEnvironment (dev); omit for HTTP + SPACE_URL.",
    )
    return p


def _run_episodes(
    model,
    tokenizer,
    env: _EnvLike,
    *,
    episodes: int,
    max_steps: int,
    task_name: str,
    prompt_style: str,
    history_len: int,
    teacher_fn,
    device: str,
    max_new_tokens: int,
    max_prompt_tokens: int,
):
    """Roll a single model through ``episodes`` runs and return aggregate stats."""
    import torch  # local import: only this helper depends on torch

    rewards_per_ep: list[float] = []
    final_pvs: list[float] = []
    action_counts: Counter[int] = Counter({0: 0, 1: 0, 2: 0})
    teacher_match = 0
    total_steps = 0

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for ep in range(episodes):
        history: deque[tuple[int, float]] = (
            deque(maxlen=history_len) if history_len else deque(maxlen=0)
        )
        obs = env.reset(task_name=task_name)
        ep_reward = 0.0
        steps = 0

        while not obs.done and steps < max_steps:
            prompt = build_user_prompt(
                obs,
                task_name=task_name,
                style=prompt_style,
                history=tuple(history) if history_len else None,
            )
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_tokens,
            ).to(device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
            generated = output[0][inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            action = parse_action(text)
            teacher_action = teacher_fn(obs)

            if action == teacher_action:
                teacher_match += 1
            action_counts[action] += 1
            total_steps += 1

            obs = env.step(action)
            ep_reward += float(obs.reward)
            if history_len:
                history.append((int(action), float(obs.reward)))
            steps += 1

        rewards_per_ep.append(ep_reward)
        final_pvs.append(float(obs.port_val))
        print(
            f"  ep {ep+1}/{episodes} steps={steps} ep_reward={ep_reward:.6f} "
            f"final_pv={obs.port_val:.2f}"
        )

    return rewards_per_ep, final_pvs, action_counts, teacher_match, total_steps


def _bar_plot(rows, *, out_path: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"] for r in rows]
    means = [r["mean_r"] for r in rows]
    stds = [r["std_r"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, means, yerr=stds, capsize=4, color="#4f8ef7")
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_ylabel("Mean episode reward")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = _build_arg_parser().parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install -r requirements-trl.txt")
        sys.exit(1)

    models = parse_models(args.models)
    teacher_fn = get_teacher(args.teacher)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    if args.local:
        env: _EnvLike = _LocalTradingEnvGym()
        env_desc = "local TradingEnvironment(seed=42)"
    else:
        env = TradingEnv()
        env_desc = f"HTTP base_url={env.base_url}"
    print(
        f"env {env_desc} device={device} models={list(models)} "
        f"prompt={args.prompt} task={args.task_name}"
    )

    rows = []
    for name, ident in models.items():
        ident = _resolve_hf_model_path(ident)
        print(f"\n=== {name} ({ident}) ===")
        tokenizer = AutoTokenizer.from_pretrained(ident, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            ident, torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval()

        rewards, final_pvs, action_counts, teacher_match, total_steps = _run_episodes(
            model,
            tokenizer,
            env,
            episodes=args.episodes,
            max_steps=args.max_steps,
            task_name=args.task_name,
            prompt_style=args.prompt,
            history_len=max(0, args.history_len),
            teacher_fn=teacher_fn,
            device=device,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
        )

        mean_r = statistics.mean(rewards) if rewards else 0.0
        std_r = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        last_pv = final_pvs[-1] if final_pvs else 0.0
        total = sum(action_counts.values()) or 1

        rows.append(
            {
                "name": name,
                "mean_r": mean_r,
                "std_r": std_r,
                "final_pv": last_pv,
                "h_pct": action_counts[0] / total,
                "b_pct": action_counts[1] / total,
                "s_pct": action_counts[2] / total,
                "teacher_agreement": teacher_match / total_steps if total_steps else 0.0,
                "episodes": len(rewards),
            }
        )

        del model, tokenizer
        try:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    md = format_metrics_md(
        rows,
        task_name=args.task_name,
        prompt_style=args.prompt,
        teacher=args.teacher,
        episodes=args.episodes,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)) or ".", exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nWrote {args.metrics_out}")

    _bar_plot(rows, out_path=args.bar_out, title=f"Phase 3 - LLM on env ({args.task_name})")
    print(f"Wrote {args.bar_out}")


if __name__ == "__main__":
    main()
