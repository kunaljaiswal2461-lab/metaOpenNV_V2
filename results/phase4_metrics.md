# Phase 4 metrics (auto-generated)

Eval episodes per policy: **20**. DQN train episodes: **70**. `random_episode_start=False` so policies are comparable on aligned rollouts.

| Policy | Mean episode reward | Std | Mean max drawdown | Mean # trades | Mean final PV |
| :--- | ---: | ---: | ---: | ---: | ---: |
| random | -1.8724 | 0.8110 | 0.2375 | 128.80 | 8573.46 |
| always_hold | -0.0579 | 0.0000 | 0.0000 | 0.00 | 10000.00 |
| buy_once_then_hold | 1.1635 | 0.0000 | 0.2450 | 1.00 | 9452.84 |
| sma20_trend | 0.9408 | 0.0000 | 0.2200 | 18.00 | 9725.27 |
| dqn_greedy | -0.0579 | 0.0000 | 0.0000 | 0.00 | 10000.00 |

Checkpoint (local, gitignored): `results/phase4_dqn.pt`

Interpretation: compare **mean episode reward** and **mean final PV** vs `random` and `always_hold`. Increase `--train-episodes` (e.g. 120+) for a stronger DQN signal on this env.