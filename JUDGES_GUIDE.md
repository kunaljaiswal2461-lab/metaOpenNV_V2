# Judges' Guide — SPY OpenEnv Trading Environment

This is the technical companion to [`README.md`](README.md). The README owns the
single-source-of-truth scope (Phase 0), judging materials (Phase 1), and the API
contract (Phase 2). This file gives a deeper engineering walk-through.

> Single source of truth for thesis & artifacts: `README.md`. If anything below
> conflicts with `README.md`, the README wins.

---

## 1. Environment workflow (the engine)

1. **Observation.** A flattened vector of length **`WINDOW_SIZE × 7 + 3`**.
   Default Hugging Face Space uses `WINDOW_SIZE=20` → **143 dims**
   (140 market + 3 portfolio). Tasks `spy_trading | risk_aware_trading |
   multi_horizon_trading` map to windows **10 / 20 / 50** → **73 / 143 / 353**
   when `task_name` is sent on `POST /reset` or `WINDOW_SIZE` is set in the
   environment. The 7-feature schema (Apr 2026) replaced the previous 10-feature
   one — EMA-12, MACD-signal gap, and ATR% were dropped after a multicollinearity
   audit.
2. **Action.** Discrete `0=HOLD`, `1=BUY`, `2=SELL` with optional `amount`
   (0–1, default 1.0).
3. **State update.** Trade executes at next close; **0.1%** transaction cost on
   buy/sell; portfolio value, cash, holdings updated.
4. **Termination.** Episode ends when portfolio value falls below **40%** of
   initial cash, the sampled segment ends, or the data sequence is exhausted.

### REST surface (OpenEnv-style)

| Endpoint | Role |
| :--- | :--- |
| `POST /reset` | Start or restart an episode. Optional JSON body: `{"task_name": "..."}` selects a curriculum window. |
| `POST /step` | Apply `TradingAction`. Returns next `TradingObservation`. |
| `GET /state` | Side-effect-free snapshot (`TradingState`: cash, holdings, PV, step, transaction cost). |

Remote agents (and judges) should use **HTTP only** via
[`client.TradingEnv`](client.py). `server.trading_environment` is reserved for
local RL training and tests; do not import it in agent code.

---

## 2. Feature specification (what the agent sees)

**7** engineered features per bar over the rolling window, plus 3 portfolio
scalars. Implementation in [`data/preprocess.py`](data/preprocess.py). The
schema is multicollinearity-audited (Apr 2026); the redundant indicators
(EMA-12 dist ≈ SMA-5 dist, MACD-signal gap ≈ SMA-5 − SMA-20 dist, ATR% ≈
volatility) were dropped to give the agent a tighter, less correlated state.

| Indicator | Formula | Purpose |
| :--- | :--- | :--- |
| Log return | `ln(P_t / P_{t-1})` | Price momentum, scale-free. |
| SMA-5 distance | `(P − SMA5) / SMA5` | Short-term mean reversion. |
| SMA-20 distance | `(P − SMA20) / SMA20` | Medium-term trend strength. |
| RSI(14) | 14-period RSI normalized 0–1 | Momentum oscillator. |
| Norm volume | `V / SMA20(V)` (clipped 0–5) | Conviction of moves. |
| Volatility | `StdDev(log_return, 10)` | Risk regime. |
| VWAP distance | `(P − VWAP) / VWAP` | Position vs cumulative typical-price VWAP. |

Portfolio: `port_cash` (USD), `holdings` (shares), `port_val` (USD).

---

## 3. Baselines and trained agents (what we shipped)

The story for judges is *"on the same env physics, multiple baselines plus a
trained agent — and a trained LLM — are compared, with regenerable plots."*

### 3.1 Phase 4 (RL + heuristic baselines, in-repo eval)

Driver: [`eval/phase4_benchmark.py`](eval/phase4_benchmark.py). Outputs:
[`results/phase4_episode_return.png`](results/phase4_episode_return.png),
[`results/phase4_mean_return_bar.png`](results/phase4_mean_return_bar.png),
[`results/phase4_metrics.md`](results/phase4_metrics.md).

| Policy | Definition |
| :--- | :--- |
| `random` | Uniform over {HOLD, BUY, SELL}. |
| `always_hold` | Cash baseline. |
| `buy_once_then_hold` | Buy on step 1, hold thereafter. |
| `sma20_trend` | Buy if `sma20_dist > 0`, sell if `< −0.02`, else HOLD. |
| `DQN (greedy)` | Tiny PyTorch DQN trained briefly on the same env (`agent/dqn_agent.py`). |

Regenerate: `python -m eval.phase4_benchmark` (use `--train-episodes 120 --eval-episodes 30` for a longer DQN narrative).

### 3.2 Remote-LLM baseline ([`inference.py`](inference.py))

Calls an OpenAI-compatible chat endpoint (`MODEL_NAME`, default
`gpt-4o-mini`). It is a **remote-API baseline**, not the Phase 3 trained agent.
This file demonstrates that the env is callable from any LLM behind an
OpenAI-style API — judges can swap `MODEL_NAME` and `API_BASE_URL`.

### 3.3 Phase 3 — TRL supervised fine-tuning (judging-grade)

Pipeline (see README **Phase 3**):

1. [`scripts/collect_sft_dataset.py`](scripts/collect_sft_dataset.py) — rolls
   trajectories via HTTP `client.TradingEnv` (or `--local` for dev), labels
   each step with the **SMA20-distance teacher** in
   [`trl_data/prompt_utils.py`](trl_data/prompt_utils.py), writes
   `data/trl_sft_train.jsonl`.
2. [`scripts/trl_sft_train.py`](scripts/trl_sft_train.py) — TRL `SFTTrainer`
   on the JSONL `text` field. Default model
   `Qwen/Qwen2.5-0.5B-Instruct`; on CPU you can pass
   `--model-id distilgpt2 --no-gradient-checkpointing` for a smoke run.
   Outputs `results/trl_sft_loss.png` and an adapter directory.
3. GPU training: [`docs/HF_GPU_TRAIN.md`](docs/HF_GPU_TRAIN.md) (second HF Space,
   `Dockerfile.train`, `scripts/hf_train_and_push.py`) or the optional
   [`colab/phase3_trl_sft.ipynb`](colab/phase3_trl_sft.ipynb)
   ([Open in Colab](https://colab.research.google.com/github/kunaljaiswal2461-lab/metaOpenNV_V2/blob/main/colab/phase3_trl_sft.ipynb)).

### 3.4 Stretch — bigger model on paid GPU (optional)

Same pipeline with `--model-id Qwen/Qwen2.5-3B-Instruct` and 4-bit + LoRA on
an HF Space L4/A100 or Colab (README **Phase 3**). Optional for v1 and not
required for judging.

---

## 4. Evaluation metrics for judges

When reviewing the live dashboard, `inference.py` logs, or
`results/phase4_metrics.md`, watch for:

- **Mean episode reward** vs `always_hold` / `buy_once_then_hold` (cash and
  trend baselines).
- **Action distribution** — does the agent actually trade, or only HOLD?
- **Drawdown / liquidation** — termination at `< 40%` of initial capital.
- **Teacher-agreement rate** — fraction of steps where the agent matches
  `sma20_trend` (sanity check that SFT actually moved the model).

Phase 4 provides the regenerable bar/line plots; Phase 3 provides
`results/trl_sft_loss.png` (and, if added, a base-vs-fine-tuned bar).

---

## 5. Reproducibility checklist

- **Space:** https://huggingface.co/spaces/Kj2461/metaOpenNV_V2 (rebuild on
  the submission commit before judging).
- **GitHub:** https://github.com/kunaljaiswal2461-lab/metaOpenNV_V2.
- **Manifest:** [`openenv.yaml`](openenv.yaml) (`shape: [143]` matches
  default `WINDOW_SIZE=20` × **7** features + 3 portfolio scalars).
- **Tests:** `python -m pytest tests/test_env.py -q` (static),
  `python verify_shape.py` (HTTP smoke after `python server/app.py`).
- **Phase 4 eval:** `python -m eval.phase4_benchmark`.
- **Phase 3 SFT:** README Phase 3, [`docs/HF_GPU_TRAIN.md`](docs/HF_GPU_TRAIN.md), or optional Colab.

---

*Environment: SPY OpenEnv | Default model: `Qwen/Qwen2.5-0.5B-Instruct` (TRL SFT) | Optional remote-LLM baseline: OpenAI-compatible API via `inference.py` | Deployment: Hugging Face Spaces (Docker SDK)*
