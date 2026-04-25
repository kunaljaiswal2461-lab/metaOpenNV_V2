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

## 🧭 SYSTEM NAVIGATION
[🏠 Overview](#-getting-started) | [⚙️ Specifications](#-environment-specification) | [📉 Market Dynamics](#-data--market-dynamics) | [🎯 Scoring](#-tasks--scoring) | [🤖 Agent Loop](#-agent--api) | [📁 Structure](#-file-structure)

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

### 👁️ Observation Stack (123 Dimensions)
To prevent the "Short-Term Memory" problem, we stack **20 minutes** of historical data:
| Layer | Dimensions | Purpose |
| :--- | :--- | :--- |
| **Temporal Window** | 120 (20x6) | Captures 1-min momentum, trend distance, and volatility. |
| **Portfolio State** | 3 | Real-time Cash, Holdings, and Net Worth. |

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
3. **Decision**: GPT-4o-mini parses 123 values to output raw discrete actions.

### 📡 API Architecture
The Space exposes a standard REST interface for remote agent connectivity:
- `POST /reset`: Initialize new session.
- `POST /step`: Execute action and return MDP state.

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
