# External materials (links only)

Hackathon rules ask for **URLs**, not large binaries in the Space repo. Add every artifact here and mirror the same links in the root `README.md`.

| Kind | URL | Status |
| :--- | :--- | :--- |
| OpenEnv API contract (README Phase 2) | *in-repo* [`README.md`](../README.md#phase-2-openenv-api-contract) | Live |
| Hugging Face Space | https://huggingface.co/spaces/Kj2461/metaOpenNV_V2 | Live |
| HF GPU training (no Colab) | [HF_GPU_TRAIN.md](HF_GPU_TRAIN.md) | Second Space + `Dockerfile.train` + `HF_HUB_MODEL_ID` |
| Source (GitHub) | https://github.com/kunaljaiswal2461-lab/metaOpenNV_V2 | Live |
| Training Colab (HF TRL SFT) | [Open in Colab](https://colab.research.google.com/github/kunaljaiswal2461-lab/metaOpenNV_V2/blob/main/colab/phase3_trl_sft.ipynb) | Same flow as README Phase 3 |
| Mini-blog (Hugging Face post) | *add in Phase 7* | TBD |
| Demo video (YouTube, under 2 minutes) | *add in Phase 7* | TBD |
| Phase 4 plots (PNG) | [`results/phase4_episode_return.png`](../results/phase4_episode_return.png), [`results/phase4_mean_return_bar.png`](../results/phase4_mean_return_bar.png) | Regenerate: `python -m eval.phase4_benchmark` |
| Phase 3 trained adapter (Hub model) | https://huggingface.co/Kj2461/metaOpenNV-sft-qwen15 | LoRA r=16 on Qwen2.5-1.5B, T4-medium, 1 epoch on 5,850 rows |
| Phase 3 TRL SFT loss | [`results/trl_sft_loss.png`](../results/trl_sft_loss.png) | Final loss 0.291, mean token acc 0.888 (initial 1.74 / 0.61) |
| Phase 3 LLM-on-env metrics | [`results/phase3_eval_metrics.md`](../results/phase3_eval_metrics.md), [`results/phase3_eval_bar.png`](../results/phase3_eval_bar.png) | Regenerate: `python scripts/eval_llm_on_env.py --models base=Qwen/Qwen2.5-1.5B-Instruct sft=Kj2461/metaOpenNV-sft-qwen15 --local` |
| Phase 4 metrics table | [`results/phase4_metrics.md`](../results/phase4_metrics.md) | Committed |
| Slides (Google Slides / PDF host) | *optional* | TBD |
| Weights & Biases (or other) run | *optional — link to a specific run* | TBD |

**Judging criteria reference:** [Apr ’26 OpenEnv Hackathon — themes & judging](https://docs.google.com/document/d/1AXXq9Mmjhjlwg2HmHiOLyQC9zee_MDmuqYXFy70fQuQ/edit?usp=sharing)
