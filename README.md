# quantstart

A backtesting engine built to be hard to fool, plus a free-data loader that
needs no account of any kind.

This is the piece of infrastructure every trading idea has to pass through.
Build it first, trust it, and then it will tell you the truth about every
strategy you try afterwards — including the ones you want to be true.

## Run it right now

```bash
pip install pandas numpy scipy requests pyarrow
python selftest.py
```

`selftest.py` needs no network. It runs the whole pipeline on synthetic random
walks — data with **no exploitable structure by construction** — and checks
that the engine correctly finds nothing. That is the test almost everyone
skips, and it is why almost everyone's backtester is silently wrong.

Then, with internet:

```python
from data import load_universe
prices = load_universe(["BTCUSDT", "ETHUSDT", "SOLUSDT"], "1d", "2020-01", "2025-12")
```

## What the self-test demonstrates

Running it on pure noise produces exactly the results it should:

| check | result | reading |
|---|---|---|
| Oracle (cheating) | Sharpe **33.4** | the engine *can* see a signal when one exists |
| Momentum, no costs | Sharpe −0.14 | inside the random-strategy band → no edge, correct |
| Momentum, 15 bps | Sharpe **−0.52** | costs alone cost 0.38 Sharpe |
| Deflated Sharpe | **0.001** | correctly says "this is luck" |

That third row is the lesson. Same signal, same data — **transaction costs
moved the result from flat to badly losing.** At 63× annual turnover, a
strategy needs a very large gross edge just to break even. Most published
retail strategies never model this, which is most of why they evaporate live.

## The three rules the engine enforces for you

1. **No lookahead, structurally.** `backtest.run()` applies `.shift(1)` to your
   weights itself. Weights computed from bar *t* are held from bar *t+1*. You
   cannot forget to do this, because you don't do it.
2. **Costs are mandatory.** `fee_bps` and `slippage_bps` are arguments with
   realistic defaults, not optional extras.
3. **Trailing statistics only.** Volatility scaling uses rolling windows. No
   full-sample normalisation anywhere.

Plus `walk_forward()` with an **embargo** gap, so overlapping labels can't leak
information backwards across the train/test boundary.

## Deflated Sharpe ratio

`n_trials` in `run()` is how many strategy variants you tried before the one
you're reporting. Be honest about it. Bailey & López de Prado's deflated Sharpe
then asks: *given that many attempts, and given these fat tails, what's the
probability this Sharpe is real rather than the best of N coin flips?*

Below ~0.95, it's noise. Most things are noise.

## Suggested order of work

1. Run `selftest.py`. Understand every number it prints.
2. Load real data. Run `oracle()` on it — note the Sharpe. That is the
   ceiling. Anything honest that lands near it is a bug in your code.
3. Run `buy_and_hold()`. That is the floor. Beat it or stop.
4. Run `ts_momentum()`. Compare to both.
5. Only then build something of your own — and add every variant you try to
   `n_trials`.

## What this will not do

It won't make money on its own, and it isn't meant to. It's a measuring
instrument. Its entire value is that it will tell you when an idea doesn't
work, before that costs you anything.

The published benchmark for the best trend-following firms in the world is
roughly **5.3% CAGR with 20% drawdowns**. Any backtest here that beats that by
a wide margin is almost certainly wrong, and this engine exists to help you
find out how.

## Reading

- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*
- López de Prado, *Advances in Financial Machine Learning* — ch. 7 on purged CV
- Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*
- Jiang, Xu & Liang (2017), *Deep RL for the Financial Portfolio Management Problem*
