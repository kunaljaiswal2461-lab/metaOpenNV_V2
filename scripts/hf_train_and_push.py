"""
Train Phase 3 TRL SFT on Hugging Face GPU infra and push weights to the Hub.

Intended layout
---------------
1. Create a **new** Docker Space (do not replace the submission API Space) from
   this repo, e.g. ``<user>/metaOpenNV_V2-train``.
2. Space **Settings -> Hardware**: pick **Nvidia T4** (or better). This uses
   your Hugging Face paid GPU / credits, not Google Colab.
3. Space **Settings -> Dockerfile path**: ``Dockerfile.train``
4. **Secrets**: add ``HF_TOKEN`` (write access) and ``HF_HUB_MODEL_ID`` (target
   model repo id, e.g. ``Kj2461/metaOpenNV-sft-qwen05``). Optional: ``SPACE_URL``
   if your trading API lives somewhere other than the default public Space.

On boot the container runs: collect JSONL from ``SPACE_URL`` -> ``trl_sft_train``
-> ``upload_folder`` of the latest checkpoint (training-only shards omitted).

After a successful push, a minimal FastAPI app stays bound to ``PORT`` (7860)
so the Space remains healthy for log inspection.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

os.environ.setdefault("PYTHONUTF8", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trl_data.eval_utils import resolve_hf_checkpoint_dir


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def _push_checkpoint(folder: str, repo_id: str, *, private: bool, token: str | None) -> None:
    from huggingface_hub import HfApi, upload_folder

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
    upload_folder(
        repo_id=repo_id,
        folder_path=folder,
        repo_type="model",
        token=token,
        ignore_patterns=[
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "training_args.bin",
            "trainer_state.json",
        ],
    )
    print(f"Pushed model files from {folder} -> https://huggingface.co/{repo_id}", flush=True)


def _serve_status(*, hub_model_id: str | None, error: str | None) -> None:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    port = int(os.environ.get("PORT", "7860"))
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        if error:
            return f"<h1>Training failed</h1><pre>{error}</pre>"
        body = (
            "<h1>Phase 3 training job finished</h1>"
            f"<p>Hub model: <code>{hub_model_id or 'n/a'}</code></p>"
            "<p>See Space <strong>Build logs</strong> for full stdout.</p>"
        )
        return body

    print(f"Serving status on 0.0.0.0:{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--hub-model-id",
        default=None,
        help="Target Hub model repo (else env HF_HUB_MODEL_ID / HUB_MODEL_ID).",
    )
    p.add_argument("--space-url", default=os.environ.get("SPACE_URL"), help="Trading API base URL for collection.")
    p.add_argument("--data-out", default=os.path.join("data", "trl_sft_train.jsonl"))
    p.add_argument("--collect-episodes", type=int, default=15)
    p.add_argument("--collect-max-steps", type=int, default=400)
    p.add_argument("--task-name", default="risk_aware_trading")
    p.add_argument(
        "--model-id",
        default=os.environ.get("PHASE3_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct"),
        help="Base model id (env PHASE3_MODEL_ID overrides the CLI default).",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--output-dir", default=os.path.join("results", "phase3_lora"))
    p.add_argument("--skip-collect", action="store_true")
    p.add_argument("--skip-push", action="store_true")
    p.add_argument("--private-repo", action="store_true", help="Create the Hub model repo as private.")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    hub_id = (
        (args.hub_model_id or os.environ.get("HF_HUB_MODEL_ID") or os.environ.get("HUB_MODEL_ID") or "")
        .strip()
    )
    if not args.skip_push and not hub_id:
        print("Set --hub-model-id or HF_HUB_MODEL_ID (target model repo id).", file=sys.stderr)
        sys.exit(1)
    if not args.skip_push and not token:
        print("Set HF_TOKEN (Hub write token) for push.", file=sys.stderr)
        sys.exit(1)

    space_url = (args.space_url or "").strip() or "https://huggingface.co/spaces/Kj2461/metaOpenNV_V2"
    os.environ["SPACE_URL"] = space_url.rstrip("/")

    err: str | None = None
    try:
        if not args.skip_collect:
            _run(
                [
                    sys.executable,
                    os.path.join("scripts", "collect_sft_dataset.py"),
                    "--episodes",
                    str(args.collect_episodes),
                    "--max-steps",
                    str(args.collect_max_steps),
                    "--task-name",
                    args.task_name,
                    "--out",
                    args.data_out,
                ]
            )
        _run(
            [
                sys.executable,
                os.path.join("scripts", "trl_sft_train.py"),
                "--data",
                args.data_out,
                "--model-id",
                args.model_id,
                "--epochs",
                str(args.epochs),
                "--output-dir",
                args.output_dir,
            ]
        )
        ckpt = resolve_hf_checkpoint_dir(args.output_dir)
        if not os.path.isdir(ckpt):
            raise FileNotFoundError(f"No checkpoint directory under {args.output_dir!r} (got {ckpt!r})")
        if not args.skip_push and hub_id and token:
            _push_checkpoint(ckpt, hub_id, private=args.private_repo, token=token)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(err, file=sys.stderr, flush=True)

    _serve_status(hub_model_id=hub_id if not err else None, error=err)


if __name__ == "__main__":
    main()
