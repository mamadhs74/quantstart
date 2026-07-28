# quantstart

A systematic trading research pipeline built to be hard to fool.

The point of this repo is not a strategy. It is a **measuring instrument** —
one that tries honestly to kill every result it produces, and reports the
numbers that don't support the hypothesis alongside the ones that do.

---

## Quick start

```bash
pip install -r requirements.txt
python selftest.py        # no network needed
```

`selftest.py` runs the full pipeline on synthetic random walks — data with no
exploitable structure by construction — and verifies the engine correctly
finds nothing. Run this before trusting any result on real data.

Then, with internet:

```bash
python run_real.py            # baselines on 8 crypto pairs
python analyze_regimes.py     # when does it work?
python ml_regime.py           # does ML add anything?
python run_cross_sectional.py # survivorship bias, measured
python robustness.py          # parameter surface, sub-periods, walk-forward
python when_did_it_work.py    # rolling relative performance
python paper_trade.py         # daily live signal log
```

Data is public Binance archives via `data.binance.vision`. No account, no API
key, no KYC.

---

## What was found

Working through the pipeline on 2019–2025 daily crypto data, 25 assets
including five that collapsed:

| test | result |
|---|---|
| Engine on pure noise | Finds nothing. Deflated Sharpe 0.001. Correct. |
| Time-series momentum | Sharpe 1.11 vs buy-and-hold 1.33 — **loses** |
| ...but drawdown | −63.9% vs −79.8% — real risk reduction |
| ML regime filter (ridge) | Sharpe 0.64 vs 0.91 unfiltered — **hurts** |
| ML regime filter (boosting) | Sharpe 0.33 — **hurts more** |
| Cross-sectional momentum | Walk-forward Sharpe 1.50 vs 1.08 — **beats** |
| Overfitting gap | +0.01 — essentially none |
| Parameter surface (16 combos) | All positive, CV 0.12 — flat plateau |
| Sub-periods | Lost to benchmark in 0 of 3 |
| Rolling 12m windows won | 59.8% |
| Deflated Sharpe (n_trials=16) | **0.937** — below the 0.95 bar |

**Verdict: promising, not proven.** Cross-sectional momentum survives the
tests that catch parameter overfitting, but does not clear the significance
bar once the parameter search is honestly accounted for.

Two results worth highlighting because they run against intuition:

**Complexity monotonically hurt.** Buy-and-hold (0 parameters) → momentum
(6 fixed) → ridge (learned) → gradient boosting (learned, flexible) produced
Sharpes of 1.23 → 0.91 → 0.64 → 0.33. Every layer of sophistication made
out-of-sample performance worse. With ~98 effective independent samples, this
is what should happen.

**Trend following needs losers.** Adding collapsed assets (LUNA −99.9%,
FTT −99.4%) *improved* momentum's Sharpe by 0.10. Survivorship bias had been
hiding the assets the signal exists to avoid.

---

## The three rules the engine enforces structurally

1. **No lookahead.** `backtest.run()` applies `.shift(1)` to weights itself.
   Weights from bar *t* are held from *t+1*. You cannot forget, because you
   don't do it.
2. **Costs are mandatory.** `fee_bps` and `slippage_bps` are arguments with
   realistic defaults, not optional extras.
3. **Trailing statistics only.** Volatility scaling uses rolling windows. No
   full-sample normalisation anywhere.

Plus `walk_forward()` with an **embargo** gap so overlapping labels cannot
leak information backwards across the train/test boundary.

---

## Diagnostics that most retail backtests skip

- **Oracle strategy.** Perfect foresight, deliberately cheating. Establishes
  the ceiling. Any honest strategy landing near it has a bug, not an edge.
- **Noise floor.** 30 random-weight strategies. Anything inside that band is
  not a discovery.
- **Deflated Sharpe ratio** (Bailey & López de Prado 2014) with `n_trials`
  set to the true size of the parameter search.
- **Parameter surface.** A flat plateau across 16 combinations is evidence;
  a spike at one setting is a warning. *Note: a flat surface rules out
  parameter artifacts. It says nothing about persistence in time.*
- **Return concentration.** Remove the best 5/10/20 days. If the result
  vanishes, you have lucky days rather than a strategy.
- **Survivorship bias, quantified.** Same tests with and without assets that
  went to zero.
- **Rolling relative performance.** Locates exactly when an edge was earned
  and whether it still is.

---

## Files

| file | purpose |
|---|---|
| `data.py` | Binance archive loader; monthly for history, daily for the live tail |
| `backtest.py` | Engine, metrics, deflated Sharpe, walk-forward with embargo |
| `strategies.py` | Baselines: buy-and-hold, random, momentum, oracle |
| `selftest.py` | Verifies the engine on synthetic data |
| `run_real.py` | Baselines on real prices, matched-period comparison |
| `analyze_regimes.py` | Annual breakdown, up/down months, return concentration |
| `ml_regime.py` | Ridge and boosting regime filters, purged walk-forward |
| `run_cross_sectional.py` | Cross-sectional ranking; survivorship bias measured |
| `robustness.py` | Parameter surface, sub-periods, walk-forward, deflated Sharpe |
| `when_did_it_work.py` | Rolling relative performance vs benchmark |
| `paper_trade.py` | Daily signal log with frozen parameters |

---

## Current status

Backtesting is **exhausted**. The deflated Sharpe of 0.937 cannot be improved
by further testing — every additional variant raises `n_trials` and pushes the
number *down*. The historical data is spent.

The only remaining source of information is time. `paper_trade.py` logs daily
signals with frozen parameters and scores them against backtest expectations.
The failure mode to watch is not "live loses money" but "live and backtest
diverge" — turnover and drawdown diverge before returns do.

---

## Reading

- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*
- Bailey, Borwein, López de Prado & Zhu (2014), *Pseudo-Mathematics and
  Financial Charlatanism*
- López de Prado, *Advances in Financial Machine Learning* — ch. 7, purged CV
- Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*
- Jiang, Xu & Liang (2017), *Deep RL for the Financial Portfolio Management
  Problem*

---

## What this repo will not do

Make money on its own. It is an instrument, not a strategy. Its value is that
it tells you when an idea doesn't work before that discovery costs anything.

For reference, the SG Trend Index — the largest institutional trend-following
firms in the world — has compounded at roughly 5.3% annually with 20%
drawdowns. Any backtest that beats that by a wide margin deserves suspicion
first and celebration much later.
