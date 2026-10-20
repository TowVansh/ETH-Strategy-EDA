# ETHUSDT.P 30m Strategy: Backtest Engine, Trade EDA and Validated Optimization

EDA course project (Team 23). A textbook indicator-confluence strategy on Binance ETHUSDT perpetual futures,
30-minute bars, Nov 2019 to Aug 2026 (118,040 bars). Every number below is net of Binance taker fees
(0.05% per side) unless marked gross.

## Run it

Double-click `run_demo.bat` (needs Python 3.10+). It installs the requirements, runs both presets and the EDA,
and writes CSVs to `output/` and charts to `figures/`. The data is already in `data/`, so no internet is needed.

Manual:

```
cd code
pip install -r requirements.txt
python main.py --data ..\data\eth_usdt_perp_30m.csv --preset baseline
python main.py --data ..\data\eth_usdt_perp_30m.csv --preset optimized --out ..\output\optimized_trades.csv
python eda.py          # figures + output/eda_summary.txt
python optimize.py     # 200-candidate sweep -> output/optimize_sweep.csv
python fetch_binance_data.py   # optional: re-download fresh data
```

## Strategy

Standard indicators at standard settings: EMA 50/200, Bollinger 20/2, Stochastic %K 14, DMI 14, ATR 14.

- `bb_cross`: in an uptrend (EMA50 > EMA200) go long when price closes back above the lower band with %K < 25; mirror for shorts.
- `dmi_trend`: go long when +DI - -DI crosses above +20 with price above EMA50; mirror for shorts.
- Exit: fixed 1.5% take-profit / 1.5% stop-loss bracket. One position at a time, entries fill at the next bar's open.

## Results

| | Trades | Win rate | Gross | Net | Max DD | Calmar | Fit 2019-23 | Holdout 2024-26 | Positive years |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 3,692 | 52.1% | +227.5% | -141.7% | 198.1 | -0.72 | -0.71 | -0.51 | 2/8 |
| Optimized | 1,534 | 46.3% | +245.8% | +92.4% | 40.3 | 2.29 | 1.60 | 1.07 | 6/8 |

EDA findings that drove the four rules:
1. `bb_cross` loses 0.21% per trade after fees (`dmi_trend` makes +0.02%), so it is dropped.
2. Trades still open after 3 h lose on average, so a 24-bar (12 h) time-stop caps them.
3. The most volatile ATR% quintile is the worst (-0.10% per trade), so entries are skipped when ATR > 0.8% of price.
4. Fixed % brackets ignore the volatility regime, so TP/SL become 4x / 3x ATR.

Linear correlation between any entry indicator and the trade outcome is near zero (|r| < 0.05); the edge only
shows up in binned (quintile) and subgroup analysis.

The optimizer ranks 200 candidates by the worse of the two periods' Calmar ratios, so a rule set cannot win by
fitting one half of history.

## Files

- `code/fetch_binance_data.py`: downloads the 30m history from Binance's public API (resumable).
- `code/indicators.py`: TradingView-exact `ta.*` equivalents (ema, rma, sma, stdev, stoch, crossover, tick rounding).
- `code/signals.py`: strategy signals.
- `code/engine.py`: sequential backtest engine (next-bar-open fills, TradingView intrabar TP/SL rule, fees, rule variants).
- `code/main.py`: CLI, `--preset baseline|optimized`, writes one row per trade with the indicator snapshot at entry.
- `code/validate.py`: diffs the engine's trades against a TradingView "List of trades" CSV export.
- `code/eda.py`: figures and the text summary.
- `code/optimize.py`: the fit/holdout parameter sweep.

## Caveats

No funding rates or slippage. Calmar is computed on a fixed notional per trade (no compounding). 2023 and 2026
are still negative for the optimized version. This is a course project, not trading advice.
