# What I Actually Built (and Why)

I wanted to test one simple thing: can an LLM do more than talk about markets?

A lot of models can explain RSI, MACD, or risk management in words. But real trading is not a one-shot QA task. You have to keep making decisions step after step, while paying transaction costs, tracking your own position, and living with the consequences of earlier actions.

So I built this as a decision-making environment, not a "financial chatbot" demo.

---

## The Problem in Plain English

The agent gets market state + portfolio state and must repeatedly choose:

- `0` = HOLD
- `1` = BUY
- `2` = SELL

Every trade has friction (0.1% cost).  
If the portfolio blows up too much, the episode ends early (liquidation guard).

That setup forces the model to think in trajectories, not isolated responses.

---

## What the Environment Looks Like

Live Space: https://huggingface.co/spaces/Kj2461/metaOpenNV_V2

It is OpenEnv-style with:

- `POST /reset`
- `POST /step`
- `GET /state`

Observation = windowed technical indicators + portfolio state (`cash`, `holdings`, `portfolio_value`). The technical-indicator stack is **7 engineered features per bar** (log return, SMA-5 / SMA-20 distance, RSI, normalized volume, volatility, VWAP distance) — multicollinearity-audited in Apr 2026 (we dropped the EMA-12 / MACD-signal / ATR features that overlapped with SMA + volatility, so the agent sees a tighter, decorrelated state).

This means the model has to reason over:

1. short-term signal changes,
2. current exposure,
3. action cost,
4. risk over time.

---

## How I Trained It

I used a straightforward TRL SFT pipeline:

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Fine-tune method: LoRA (adapter only)
- Training infra: Hugging Face GPU Space (`metaOpenNV_V2-train`)
- Teacher labels: SMA20-style trend policy from the same environment rollouts

Trained adapter:

- https://huggingface.co/Kj2461/metaOpenNV-sft-qwen15

Repro notebook (judge can re-run):

- https://colab.research.google.com/github/kunaljaiswal2461-lab/metaOpenNV_V2/blob/main/colab/phase3_trl_sft.ipynb

---

## What Changed After Training (Real Numbers)

From `results/phase3_eval_metrics.md`:

- Mean episode reward: `-0.044` -> `+1.844`
- Final portfolio value: `$10,000.00` -> `$10,718.64`
- Teacher agreement: `0.201` -> `0.401`
- Action behavior: from `100% HOLD` to `77.2% HOLD / 22.6% BUY / 0.1% SELL`

So the tuned model did not just "sound smarter." It actually changed policy behavior and stopped freezing in HOLD-only mode.

Training curve also improved cleanly:

- Loss: `1.7436` -> `0.2914`
- Mean token accuracy: `0.6092` -> `0.8875`
- Loss plot: [`results/trl_sft_loss.png`](../results/trl_sft_loss.png)
- Eval bar: [`results/phase3_eval_bar.png`](../results/phase3_eval_bar.png)

---

## What I Learned Building This

1. If reward is vague, the model finds lazy behavior (usually HOLD forever).
2. Tight environment design matters more than prompt tricks.
3. The "agentic" part is memory + consistency over many steps, not just action classification.
4. Even one epoch of grounded SFT can move a model from passive to active behavior.

---

## Links (all public)

- Environment Space: https://huggingface.co/spaces/Kj2461/metaOpenNV_V2
- Source code: https://github.com/kunaljaiswal2461-lab/metaOpenNV_V2
- Training notebook: https://colab.research.google.com/github/kunaljaiswal2461-lab/metaOpenNV_V2/blob/main/colab/phase3_trl_sft.ipynb
- Trained adapter: https://huggingface.co/Kj2461/metaOpenNV-sft-qwen15
- Phase 3 metrics: [`results/phase3_eval_metrics.md`](../results/phase3_eval_metrics.md)
- Phase 4 baselines: [`results/phase4_metrics.md`](../results/phase4_metrics.md)

---
##Results benchmark of the model--
please refer to the given file in the  same repository:
link - https://github.com/kunaljaiswal2461-lab/metaOpenNV_V2/tree/main/results

## 20-Second Pitch I Would Say Live

"This is an OpenEnv trading world where the model has to take actions repeatedly under cost and risk, not just answer finance questions. I fine-tuned Qwen 1.5B on trajectories from this same environment. After training, the model moved from always HOLD to meaningful BUY/HOLD behavior, and both reward and final portfolio value improved on held evaluations. So the result is a reproducible benchmark showing better closed-loop decision-making, not just better financial text generation."


