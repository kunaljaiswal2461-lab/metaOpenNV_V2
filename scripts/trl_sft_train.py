"""
Fine-tune a small instruct model with TRL SFT on data/trl_sft_train.jsonl (\"text\" field).

Usage (GPU recommended, e.g. Colab):
    pip install -r requirements-trl.txt
    python scripts/collect_sft_dataset.py --episodes 15 --max-steps 300
    python scripts/trl_sft_train.py --epochs 1

Writes:
    results/trl_sft_loss.png
    results/phase3_lora/   (adapter weights)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Windows: TRL reads bundled Jinja templates as UTF-8; avoid cp1252 decode errors.
os.environ.setdefault("PYTHONUTF8", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _plot_loss(log_history: list, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, losses = [], []
    for row in log_history:
        if "loss" in row and row["loss"] is not None:
            steps.append(row.get("step", len(steps)))
            losses.append(row["loss"])
    if not losses:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(steps, losses, marker="o", markersize=3)
    plt.xlabel("Logging step")
    plt.ylabel("Training loss")
    plt.title("Phase 3 - TRL SFT training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join("data", "trl_sft_train.jsonl"),
        help="JSONL from scripts/collect_sft_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("results", "phase3_lora"),
        help="Where to save the adapter + checkpoints",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Small instruct model on the Hub",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (faster on small models / CPU smoke tests)",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Train LoRA adapters only (recommended for >=1B models on T4-class GPUs).",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        print("Missing dependency:", e)
        print("Install: pip install -r requirements-trl.txt")
        sys.exit(1)

    # Auto-enable LoRA on GPUs with limited VRAM (e.g. T4 16GB) for >=1B models.
    if not args.use_lora and torch.cuda.is_available():
        try:
            free_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            free_gb = 0.0
        looks_big = any(s in args.model_id for s in ("1.5B", "3B", "7B", "8B", "13B"))
        if looks_big and free_gb < 24.0:
            print(f"[trl_sft_train] auto-enabling LoRA: model={args.model_id}, total VRAM={free_gb:.1f} GB")
            args.use_lora = True

    if not os.path.isfile(args.data):
        print(f"Dataset not found: {args.data}")
        print("Run: python scripts/collect_sft_dataset.py")
        sys.exit(1)

    rows: list = []
    with open(args.data, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if len(rows) < 8:
        print(f"Warning: only {len(rows)} examples; consider collecting more data.")

    train_ds = Dataset.from_list(rows)

    # TRL v1.x: SFTConfig carries training + SFT-specific fields.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    sft_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=max(1, min(10, len(rows) // 3)),
        max_length=args.max_length,
        dataset_text_field="text",
        report_to="none",
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        save_strategy="epoch",
        save_total_limit=1,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )

    peft_config = None
    if args.use_lora:
        try:
            from peft import LoraConfig
        except ImportError:
            print("peft is required for --use-lora; install: pip install peft")
            sys.exit(1)
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        print(f"[trl_sft_train] using LoRA r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")

    trainer_kwargs = dict(
        model=args.model_id,
        args=sft_args,
        train_dataset=train_ds,
    )
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()

    loss_png = os.path.join("results", "trl_sft_loss.png")
    _plot_loss(trainer.state.log_history, loss_png)
    print(f"Saved loss plot: {loss_png}")
    print(f"Adapter / model files under: {args.output_dir}")


if __name__ == "__main__":
    main()
