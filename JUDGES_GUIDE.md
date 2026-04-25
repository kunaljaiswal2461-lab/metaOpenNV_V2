# ⚖️ Judges' Guide: Grok-Agent SPY Trading Environment

This document provides a deep technical breakdown of the Reinforcement Learning environment and the Grok-powered Agentic workflow for the Meta Hackathon judges.

---

## 🏗️ 1. Environment Workflow (The Engine)

The environment is a high-fidelity simulator for the **SPY (S&P 500 ETF)**. It follows a standard OpenAI Gym-style interaction loop but is optimized for LLM agents.

### The Loop:
1.  **Observation**: A flattened vector of length **`WINDOW_SIZE × 10 + 3`**. Default Hugging Face Space uses `WINDOW_SIZE=20` → **203** dimensions (200 market + 3 portfolio). Task presets (`spy_trading`, `risk_aware_trading`, `multi_horizon_trading`) map to windows **10 / 20 / 50** → **103 / 203 / 503** when `task_name` is sent on `POST /reset` (JSON body) or `WINDOW_SIZE` is set in the environment.
2.  **Inference**: An AI Agent (Grok-2 or GPT-4o-mini) processes this state.
3.  **Action**: The Agent returns a discrete action: `0 (HOLD)`, `1 (BUY)`, or `2 (SELL)`.
4.  **State Update**: The environment executes the trade, calculates transaction costs (0.1%), updates the Portfolio Value (PV), and computes the Reward.
5.  **Termination**: The session ends if portfolio value falls below **40%** of initial capital, the sampled episode segment ends, or the data sequence concludes.

---

## 📊 2. Feature Specification (What the Agent Sees)

The agent receives a **rolling window** of **10** engineered features per bar (past `WINDOW_SIZE` bars, exclusive of the current bar in the stacked slice used for history — see `server/trading_environment.py`), plus **3** portfolio scalars.

### Technical Indicators Explained:

| Indicator | Technical Formula | Purpose |
| :--- | :--- | :--- |
| **Log Return** | `ln(Price_t / Price_{t-1})` | Captures price momentum and volatility in a stationary format. |
| **SMA 5 Dist** | `(Price - SMA_5) / SMA_5` | Measures short-term overextension (Mean Reversion). |
| **SMA 20 Dist** | `(Price - SMA_20) / SMA_20` | Measures medium-term trend strength (Trend Following). |
| **RSI (14)** | `100 - [100 / (1 + RS)]` | Standard 14-period momentum oscillator (Normalized to 0-1). |
| **Norm Volume** | `Vol / SMA_20(Vol)` | Identifies high-conviction moves relative to recent activity. |
| **Volatility** | `StdDev(Log_Returns, 10)` | Captures market uncertainty and risk (10-period rolling). |
| **VWAP distance** | `(Price - VWAP) / VWAP` | Distance from cumulative typical-price VWAP (daily series on daily bars). |
| **EMA12 distance** | `(Price - EMA_12) / EMA_12` | Short-horizon trend alignment. |
| **MACD − signal** | `MACD_hist / Price` | Momentum vs signal line, price-normalized. |
| **ATR%** | `ATR(14) / Price` | Regime / volatility scaling. |

**Portfolio Features:**
- `port_cash`: Liquidity available for buying.
- `holdings`: Current quantity of SPY shares owned.
- `port_val`: Total net worth (Cash + Holdings * Price).

---

## 🧠 3. Agentic Workflow (Grok Integration)

The "Agentic" part of this project goes beyond simple classification. We use **Tool Interception** to let Grok bridge the gap between "Thinking" and "Acting."

1.  **System Consciousness**: We provide Grok with a "Trading Persona" and a JSON description of the tools available.
2.  **Reasoning Step**: Grok analyzes the observation. Instead of just outputting an action, it can **call internal tools** to fetch deeper data or execute complex orders.
3.  **Action Interception**:
    - Grok outputs a `tool_call` (e.g., `execute_trade(action='BUY', quantity=10)`).
    - Our Python backend intercepts this "intent."
    - The backend interacts with the Environment API.
    - The result (Success/Failure) is fed back into Grok's memory before the next decision.

---

## 🧪 4. Evaluation Metrics for Judges

When reviewing the Live Dashboard or logs, watch for:
- **Sharpe Ratio**: Reward vs. Volatility.
- **Max Drawdown**: How well the agent manages risk during market dips.
- **RSI Adherence**: Does the agent buy low (RSI < 0.3) and sell high (RSI > 0.7)?
- **Portfolio Growth**: Net profit relative to a "Buy and Hold" strategy.

---
*Environment: RL-Trading-v1 | Backend: Grok-2-Latest | Deployment: HF Spaces*
