# Phase 3 - LLM on env evaluation

task=risk_aware_trading prompt=compact teacher=sma20 episodes=3

| Model | Mean episode reward | Std | Last final PV | HOLD | BUY | SELL | Teacher agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | -0.324662 | 0.542586 | 8840.13 | 99.6% | 0.4% | 0.0% | 0.196 |
| sft | 0.189311 | 0.989850 | 10363.94 | 0.0% | 100.0% | 0.0% | 0.596 |
