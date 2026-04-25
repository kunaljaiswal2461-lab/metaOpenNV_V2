---
title: Trading Env
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
---

# 🛰️ AntiGravity: Strategic SPY Market Intelligence
**A high-fidelity Markov Decision Process (MDP) for Autonomous Financial Agents.**

---

## Submission thesis (Phase 0 — scope freeze)

This section is the **single source of truth** for hackathon scope and thesis. New **primary domains** require updating this section first. Criteria reference: [Apr ’26 OpenEnv Hackathon — themes & judging](https://docs.google.com/document/d/1AXXq9Mmjhjlwg2HmHiOLyQC9zee_MDmuqYXFy70fQuQ/edit?usp=sharing).

**Live Space (submission target):** [https://huggingface.co/spaces/Kj2461/metaOpenNV_V2](https://huggingface.co/spaces/Kj2461/metaOpenNV_V2)

### Theme alignment

| Role | Theme |
| :--- | :--- |
| **Primary** | **#3.1 Professional tasks** — a defined, partially observable **market world**: the agent consumes structured signals and tool-like state (prices, indicators, portfolio), acts through a strict API, and receives outcomes it cannot shortcut without maintaining internal state across steps. |
| **Secondary (pitch)** | **#2 Long-horizon planning & instruction following** — episodes unfold over many steps; good behavior depends on **sequences** of decisions under execution costs and shaped/delayed feedback, not a single-turn answer. |

### One sentence for judges

We give LLMs a **realistic SPY trading simulator** (OpenEnv-compliant) where they must map **multi-step market context + portfolio state** to **hold / buy / sell** under **transaction costs and composite risk-aware rewards**, so training improves **grounded decision-making under uncertainty** instead of generic financial chat.

### Problem → environment → reward → why an LLM gets better (five bullets)

1. **Problem:** Models are strong at financial prose but weak at **closed-loop control**: sizing actions, respecting friction, and staying coherent over long trajectories when the world pushes back bar-by-bar.
2. **Observation & action:** The agent sees a **fixed-length window** of engineered features (returns, trend distances, RSI, volume regime, volatility, VWAP distance, EMA/MACD/ATR-style signals) plus **cash, holdings, and portfolio value**; actions are **discrete** (hold / buy / sell) with an optional **continuous fraction** of cash or position to trade.
3. **Reward:** A **composite, dense-style** signal (portfolio change, downside pressure, risk-adjusted components, market-relative terms — see `reward.py`) so learning is guided **before** the episode ends, not only by a terminal label.
4. **Why an LLM gets better here:** The task forces **token-budget-friendly structured reasoning** over numeric state, **memory** of position and PnL path, and **incentive alignment** with a reward that penalizes naive churn and tail risk — capabilities that transfer to any agent that must **act** under constraints, not just describe markets.
5. **Scope freeze:** We **do not** add new primary domains (e.g. unrelated games or a second tradable asset) without updating this README section first. Docs, manifests, training Colab, and evaluation **must** stay aligned with the thesis above.

---

## Phase 1 (Judge materials)

Single place for reviewers (mirrored in [`docs/MATERIALS.md`](docs/MATERIALS.md)). **Do not** commit large video files; use URLs only.

| Artifact | URL | Notes |
| :--- | :--- | :--- |
| **Hugging Face Space** | [https://huggingface.co/spaces/Kj2461/metaOpenNV_V2](https://huggingface.co/spaces/Kj2461/metaOpenNV_V2) | Runnable environment |
| **GitHub source** | [https://github.com/kunaljaiswal2461-lab/metaOpenNV_V2](https://github.com/kunaljaiswal2461-lab/metaOpenNV_V2) | Version-controlled code |
| **OpenEnv manifest** | [`openenv.yaml`](openenv.yaml) | Default `observation_space.shape: [203]` matches `WINDOW_SIZE=20` in Docker |
| **Training Colab** (Unsloth or HF TRL) | *URL to be added in Phase 3* | Judges re-run training here |
| **Mini-blog** (HF post) | *URL to be added in Phase 7* | &lt; 2 min read OK |
| **Demo video** (YouTube) | *URL to be added in Phase 7* | **&lt; 2 minutes**; link only |
| **Plots / results** | *Paths under `results/` in Phase 4* | Loss + reward curves, baseline vs trained |

### 3-minute read for judges

1. **Problem:** LLMs need grounded, multi-step **decisions under market friction**, not one-shot financial text.  
2. **Environment:** SPY bars + engineered features + portfolio state; **reset/step** API; episode sampling over held-out-style train split.  
3. **Results:** *Phase 4 — placeholder; will embed committed PNGs and a short metrics table here.*  
4. **Why it matters:** Improves **tool-like control** and **long-horizon consistency** for professional / assistant-style agents (see Phase 0 themes).

---

## 🧭 SYSTEM NAVIGATION
[Phase 1](#phase-1-judge-materials) | [🏠 Overview](#-getting-started) | [⚙️ Specifications](#-environment-specification) | [📉 Market Dynamics](#-data--market-dynamics) | [🎯 Scoring](#-tasks--scoring) | [🤖 Agent Loop](#-agent--api) | [📁 Structure](#-file-structure)

---

## 💡 GETTING STARTED

### 🌟 Project Overview
AntiGravity is a specialized Reinforcement Learning environment designed for the **S&P 500 ETF (SPY)**. It simulates a high-frequency trading arena where agents must leverage technical signals to optimize risk-adjusted returns against a realistic "Market Physics" engine.

### 🚀 Quick Launch
```bash
# 1. Install Industry Standard Dependencies
pip install -r requirements.txt

# 2. Start the FastAPI + Gradio Orchestrator
python server/app.py

# 3. Launch the LLM Baseline Agent
python inference.py
```

---

## ⚙️ ENVIRONMENT SPECIFICATION

### 🎮 The Action Space
The environment utilizes a **Hybrid Discrete-Continuous** action space:
- **`0: HOLD`** — Maintains current position. Zero transaction cost.
- **`1: BUY`** — Allocates available cash into SPY (with 0.1% slippage penalty).
- **`2: SELL`** — Liquidates SPY holdings at the next 1-min candle close.

### 👁️ Observation stack (window × 10 + 3)

The flattened RL vector length is **`WINDOW_SIZE × 10 + 3`**: `market_features` has length **`WINDOW_SIZE × 10`**, plus **3** portfolio scalars (`port_cash`, `holdings`, `port_val` / `portfolio_value`). The default Space uses **`WINDOW_SIZE=20`** → **203** total (`openenv.yaml`). Tasks use windows **10 / 20 / 50** → **103 / 203 / 503** when you pass `task_name` on reset (see API).

| Layer | Dimensions (default) | Purpose |
| :--- | :--- | :--- |
| **Temporal window** | 200 in `market_features` (20 × 10) | Past 20 bars of 10 engineered features (see `data/preprocess.py`). |
| **Portfolio state** | 3 separate JSON fields | Cash (USD), holdings (shares), portfolio value (USD). |

### 🛠️ Market Physics Engine
- **Transaction Tax**: A 0.1% fixed cost per trade, modeling exchange fees and slippage.
- **Margin Safety**: Automatic liquidation if Portfolio Value < 40% of Initial Capital.
- **Execution**: All trades are executed at the **Closing Price** of the current minute.

---

## 📉 DATA & MARKET DYNAMICS

### 📊 Technical Indicator Reference
| Indicator | Key | Formula | Market Logic |
| :--- | :--- | :--- | :--- |
| **Log Return** | `log_ret` | `ln(P_t / P_t-1)` | Removes price scaling bias. |
| **SMA Dist (5)** | `sma_5` | `(P - SMA5)/SMA5` | Measures short-term mean-reversion. |
| **SMA Dist (20)**| `sma_20` | `(P - SMA20)/SMA20`| Measures medium-term trend strength. |
| **RSI (14)** | `rsi` | `14-period RSI` | Normalized oscillator (0-1). |
| **Volume Norm** | `vol` | `V / SMA_20(V)` | Validates strength of price moves. |
| **Volatility** | `volat` | `StdDev(Ret, 10)` | Identifies high-risk market regimes. |
| **VWAP distance** | `vwap_dist` | `(Close - VWAP) / VWAP` | Price vs cumulative-volume typical price. |
| **EMA12 distance** | `ema12_dist` | `(Close - EMA12) / EMA12` | Short trend alignment. |
| **MACD vs signal** | `macd_signal_gap` | `MACD_hist / Close` | Momentum vs signal, normalized. |
| **ATR%** | `atr_pct` | `ATR(14) / Close` | Volatility regime scaling. |

---

## 🎯 TASKS & SCORING

### 🏹 The Curriculum
We test agents across three progressive difficulty tiers (configured in `openenv.yaml`):
1. **`spy_trading`**: Basic trend following (10-min window).
2. **`risk_aware_trading`**: Standard industry benchmark (20-min window).
3. **`multi_horizon_trading`**: Deep sequence planning (50-min window).

### 🎁 Reward Shaping Architecture
Derived from the **Stanford GSB research (Moody & Saffell)**, the system uses a weighted composite reward (`α=0.4, β=0.25, γ=0.15`):
- **Log Return Component**: Direct profit maximization.
- **Downside Risk Penalty**: Exponential penalty for absolute losses.
- **Differential Sharpe Ratio**: A recursive gradient for risk-adjusted alpha.
- **Treynor Contribution**: Rewards performance relative to SPY's baseline beta.

---

## 🤖 AGENT & API

### 🧠 LLM Inference Cycle
The baseline `inference.py` follows a strict logic loop:
1. **Sanitize**: Observations are cleaned of `NaN` or `Inf` values for LLM stability.
2. **Context**: Prompting uses expert-trader personas to guide action selection.
3. **Decision**: The model consumes the full observation vector (length depends on `WINDOW_SIZE`) and outputs a discrete action.

### 📡 API Architecture
The Space exposes a standard REST interface for remote agent connectivity:
- `POST /reset`: Initialize a new episode. Optional JSON body: `{"task_name":"spy_trading"}` or `"risk_aware_trading"` or `"multi_horizon_trading"` (matches `openenv.yaml` tasks and sets the lookback window).
- `POST /step`: Execute action and return MDP state (JSON body: `TradingAction` — `action` 0/1/2, optional `amount` 0–1).

---

## 📁 FILE STRUCTURE
```text
rl-trading/
├── server/                     # Backend Orchestration
│   ├── app.py                  # Live Gateway (Gradio + FastAPI)
│   └── trading_environment.py  # Core Physics & Logic
├── data/                       # Intelligence Layer
│   ├── preprocess.py           # Technical Indicator Factory
│   └── spy_prices.csv          # High-Frequency CSV Feed
├── models.py                   # Protocol Buffers / Schemas
├── reward.py                   # Reward Shaping Engine
└── inference.py                # Agent Loop Baseline
```

---
*Developed for the Meta RL Trading Hackathon | Built with openenv-core*
