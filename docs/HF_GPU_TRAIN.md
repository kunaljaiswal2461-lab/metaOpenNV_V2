# Train on Hugging Face GPU (no Colab)

Use a **second** Hugging Face Space with **paid GPU** so training runs on Hub infrastructure and the fine-tuned weights are pushed to a **Model** repo. Your submission **API Space** (`metaOpenNV_V2`) should stay on CPU/small hardware so judges always have a fast URL.

## 1. Create a model repo (empty)

On the Hub: **New model** → e.g. `Kj2461/metaOpenNV-sft-qwen05` (adjust owner/name). It can start empty; the job will upload files.

## 2. Duplicate the GitHub repo Space (or same repo, new Space)

- **New Space** → link your GitHub `metaOpenNV_V2` (same code is fine).
- Name it something like `metaOpenNV_V2-train` so it is obvious this is not the judge demo.

## 3. Point the build at `Dockerfile.train`

Space **Settings** → **Docker** (or Dev mode):

- **Dockerfile location**: `Dockerfile.train`  
  (not `server/Dockerfile`, which is the API image.)

## 4. Turn on GPU hardware

Space **Settings** → **Hardware**:

- Choose **Nvidia T4** (or L4 / A10G if you want faster runs).  
  This consumes **Hugging Face paid GPU / credits** (your $30 budget), not Colab.

Rebuild the Space after changing hardware.

## 5. Secrets (required for push)

Space **Settings** → **Variables and secrets** (repository secrets):

| Name | Value |
|------|--------|
| `HF_TOKEN` | A fine-grained or classic token with **write** access to **Models** (and this Space if private). |
| `HF_HUB_MODEL_ID` | The model repo id, e.g. `Kj2461/metaOpenNV-sft-qwen05`. |

Optional:

| Name | Value |
|------|--------|
| `SPACE_URL` | Defaults to `https://huggingface.co/spaces/Kj2461/metaOpenNV_V2`. Override if your trading API is elsewhere. |

## 6. What runs on boot

[`scripts/hf_train_and_push.py`](../scripts/hf_train_and_push.py):

1. Collects SFT JSONL from `SPACE_URL` (HTTP, same as Colab).
2. Runs `trl_sft_train.py` (default **Qwen2.5-0.5B-Instruct**, 1 epoch).
3. Uploads the latest `checkpoint-*` folder to `HF_HUB_MODEL_ID` (skips `optimizer.pt` / `scheduler.pt` etc. to save space).
4. Starts a tiny **FastAPI** page on port **7860** so the Space stays “Running” and you can read **Build logs** for full stdout.

## 7. After training

- Load the adapter in eval or inference with the Hub id, e.g.  
  `python scripts/eval_llm_on_env.py --models sft=Kj2461/metaOpenNV-sft-qwen05 ...`
- Download `results/trl_sft_loss.png` from the Space **Files** tab if you want to commit an updated plot (optional).

## 8. Cost tips

- Default collect is **15 × 400** steps; reduce in the script or add env knobs later if you need shorter smoke runs.
- Use **T4** first; scale hardware only if builds time out.
- Delete old model revisions on the Hub if you iterate often and care about storage.

## Local dry-run (no push)

```bash
export SPACE_URL=https://huggingface.co/spaces/Kj2461/metaOpenNV_V2
python scripts/hf_train_and_push.py --skip-push --hub-model-id dummy/dummy
```

`--hub-model-id` is still parsed; use any placeholder when `--skip-push` is set (push is skipped before Hub calls).
