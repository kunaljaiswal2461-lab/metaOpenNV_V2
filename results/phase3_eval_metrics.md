# Phase 3 - LLM on env evaluation

task=risk_aware_trading prompt=compact teacher=sma20 episodes=10

| Model | Mean episode reward | Std | Last final PV | HOLD | BUY | SELL | Teacher agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | -0.044400 | 0.000000 | 10000.00 | 100.0% | 0.0% | 0.0% | 0.201 |
| sft | 1.844074 | 1.190299 | 10718.64 | 77.2% | 22.6% | 0.1% | 0.401 |

