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

Behaviour
---------
On boot the container immediately binds a FastAPI status page to ``PORT``
(7860) on a background thread, then in the foreground it runs
``collect -> trl_sft_train -> upload_folder`` of the latest checkpoint
(training-only shards omitted). The page reflects live training state so the
HF Space stays healthy and you can monitor progress without scraping logs.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
import threading
import traceback
from html import escape as _esc

os.environ.setdefault("PYTHONUTF8", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trl_data.eval_utils import resolve_hf_checkpoint_dir


_STATE_LOCK = threading.Lock()
_STATE: dict[str, object] = {
    "phase": "starting",
    "hub_model_id": None,
    "started_at": _dt.datetime.utcnow().isoformat() + "Z",
    "updated_at": _dt.datetime.utcnow().isoformat() + "Z",
    "error": None,
    "log_tail": [],
}


def _set_state(**kwargs: object) -> None:
    with _STATE_LOCK:
        _STATE.update(kwargs)
        _STATE["updated_at"] = _dt.datetime.utcnow().isoformat() + "Z"


def _push_log(line: str) -> None:
    with _STATE_LOCK:
        tail = list(_STATE.get("log_tail") or [])
        tail.append(line.rstrip())
        _STATE["log_tail"] = tail[-60:]
        _STATE["updated_at"] = _dt.datetime.utcnow().isoformat() + "Z"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    _push_log("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        _push_log(line)
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


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
    msg = f"Pushed model files from {folder} -> https://huggingface.co/{repo_id}"
    print(msg, flush=True)
    _push_log(msg)


def _start_status_server() -> None:
    """Bind FastAPI on PORT in a daemon thread so the Space is healthy from t=0."""
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    port = int(os.environ.get("PORT", "7860"))
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        with _STATE_LOCK:
            state = dict(_STATE)
        tail = "\n".join(state.get("log_tail") or [])  # type: ignore[arg-type]
        err = state.get("error")
        body = f"""
<!doctype html>
<html><head><meta http-equiv="refresh" content="10">
<title>metaOpenNV_V2-train</title>
<style>body{{font-family:system-ui,Arial,sans-serif;margin:24px;max-width:980px}}
pre{{background:#0b1020;color:#cce;padding:12px;border-radius:6px;max-height:520px;overflow:auto;font-size:12px}}
.k{{color:#666}} .v{{font-weight:600}} .err{{color:#b00}}</style></head>
<body>
<h1>Phase 3 TRL SFT - HF GPU Trainer</h1>
<p><span class="k">Phase:</span> <span class="v">{_esc(str(state.get('phase')))}</span></p>
<p><span class="k">Hub model:</span> <span class="v"><code>{_esc(str(state.get('hub_model_id') or 'pending'))}</code></span></p>
<p><span class="k">Started:</span> {_esc(str(state.get('started_at')))}<br>
<span class="k">Updated:</span> {_esc(str(state.get('updated_at')))}</p>
{('<p class="err"><b>Error:</b> ' + _esc(str(err)) + '</p>') if err else ''}
<h3>Log tail (auto-refresh 10s)</h3>
<pre>{_esc(tail)}</pre>
</body></html>
"""
        return body

    @app.get("/status")
    def status() -> JSONResponse:
        with _STATE_LOCK:
            return JSONResponse(dict(_STATE))

    def _serve() -> None:
        try:
            print(f"Status server on 0.0.0.0:{port}", flush=True)
            uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
        except Exception:
            traceback.print_exc()

    threading.Thread(target=_serve, daemon=True, name="status-server").start()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hub-model-id", default=None, help="Target Hub model repo (else env HF_HUB_MODEL_ID / HUB_MODEL_ID).")
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
    space_url = (args.space_url or "").strip() or "https://kj2461-metaopennv-v2.hf.space"
    os.environ["SPACE_URL"] = space_url.rstrip("/")

    _set_state(hub_model_id=hub_id or None, phase="starting status server")
    _start_status_server()

    if not args.skip_push and not hub_id:
        _set_state(phase="config error", error="Set --hub-model-id or HF_HUB_MODEL_ID")
        print("Set --hub-model-id or HF_HUB_MODEL_ID (target model repo id).", file=sys.stderr)
        # Keep the status server alive so logs are reachable.
        threading.Event().wait()
        return
    if not args.skip_push and not token:
        _set_state(phase="config error", error="Set HF_TOKEN secret on the Space")
        print("Set HF_TOKEN (Hub write token) for push.", file=sys.stderr)
        threading.Event().wait()
        return

    try:
        if not args.skip_collect:
            _set_state(phase="collecting SFT data from API")
            _run(
                [
                    sys.executable,
                    os.path.join("scripts", "collect_sft_dataset.py"),
                    "--episodes", str(args.collect_episodes),
                    "--max-steps", str(args.collect_max_steps),
                    "--task-name", args.task_name,
                    "--out", args.data_out,
                ]
            )
        _set_state(phase=f"TRL SFT on {args.model_id}")
        _run(
            [
                sys.executable,
                os.path.join("scripts", "trl_sft_train.py"),
                "--data", args.data_out,
                "--model-id", args.model_id,
                "--epochs", str(args.epochs),
                "--output-dir", args.output_dir,
            ]
        )
        _set_state(phase="locating checkpoint")
        ckpt = resolve_hf_checkpoint_dir(args.output_dir)
        if not os.path.isdir(ckpt):
            raise FileNotFoundError(f"No checkpoint directory under {args.output_dir!r} (got {ckpt!r})")
        if not args.skip_push and hub_id and token:
            _set_state(phase=f"pushing checkpoint to {hub_id}")
            _push_checkpoint(ckpt, hub_id, private=args.private_repo, token=token)
        _set_state(phase="done", error=None)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        _set_state(phase="failed", error=err)

    threading.Event().wait()


if __name__ == "__main__":
    main()
