# DatathonFME2026 — Repte UPC Sostenible (team Low BaterIA)

An agent that decides, hour by hour, how much the campus battery charges or discharges,
to cut CO2 (70% of the score) and electricity cost (30%) against a campus with no battery.

## Repository layout

| Path | What it is |
|---|---|
| `agent.py`, `evaluate.py` | **Standard mode** (2,000 kWh / ±500 kW, 24 h of prices given): a 24 h linear-program planner, and the script that backs up its design. |
| `hard_mode/final_A_holidays/` | **Hard mode, final candidate A**, exactly as submitted (`agent.py` + 14 LightGBM models), plus `train_models.py`. |
| `hard_mode/final_B_rundrain/` | **Hard mode, final candidate B**, as submitted (same models; only one comment was updated). |
| `data/`, `starter_kit/`, `docs/` | Organisers' training data, simulator and statement. |

Each `final_*` folder can be zipped as it is (`agent.py` and `model_*.txt` at the root of the zip) and uploaded.

## Hard mode in one paragraph

Hard mode triples the battery (6,000 kWh, ±1,500 kW), multiplies solar by 12 and only reveals
the next 3 hours of price and CO2. Every hour the agent:

1. **Forecasts** with LightGBM: price and CO2 for hours +3..+23, campus load and solar for 0..23,
   and 10 quantiles of *this* hour's net load (load − solar), which is unknown when we decide.
2. **Plans** with a 24 h linear program minimising Σ (CO2ₜ + λ·priceₜ) · grid importₜ, with
   λ = 0.264 derived from the score formula (not tuned). For the current hour it averages over the
   10 net-load scenarios, so it never discharges more than the campus is likely to consume.
3. **Applies** only the first hour, clamped to the physical limits (0 infractions), and replans.

## The two final candidates

| | A — `final_A_holidays` | B — `final_B_rundrain` |
|---|---|---|
| Public holidays treated as Sundays in the load forecast | ✓ | ✓ |
| Ignore stale "last hour" readings after a jump to a new test week | ✓ | ✓ |
| Empty the battery before the very end of the run (energy left over is worth nothing) | – | ✓ |
| `SHRINK` (widen the forecast price/CO2 profile) | 1.0 | 1.15 |

## Validation (why we trust these, and not the leaderboard)

* `train_models.py` holds out 12 one-week blocks, retrains without them and scores the full agent
  in the official simulator: **18.13** points, against **5.69** for the same planner without forecasts.
* A wider offline backtest on 30 held-out weeks (three disjoint sets of 10), with a paired bootstrap
  over weeks, gave an honest baseline of **18.79**. Models scored on their own training weeks look
  0.6–1.1 points better, so in-sample numbers are optimistic.
* With perfect knowledge of the future the same planner reaches 20.43 (+1.64): about 1.0 of that gap is
  price/CO2 forecasting and 0.5 is the current hour's net load. The planner itself is not the bottleneck.
* Tuning knobs (λ, `SHRINK`, terminal value, scenario spread, 48 h horizon, stochastic planning) all moved
  the score by less than the week-to-week noise (about ±0.3). The run-end drain was the only change
  that was never negative (+0.06, 95% CI 0.00..+0.16).
* We also tested optimising only the first half of each week (the part the public leaderboard
  appears to score). It gained +4.5 there but lost −7.8 on the hidden half (−1.5 overall),
  so we did **not** choose it as a final candidate.

## How to run

```bash
pip install numpy pandas scipy lightgbm
cd hard_mode/final_A_holidays
python ../../starter_kit/check_submission.py agent.py   # official format check
python train_models.py ../.. --no-final                 # retrain without held-out weeks and validate
python train_models.py ../..                            # also train final models on all data
```

`train_models.py` writes to `retrained/` and never overwrites the submitted models.
