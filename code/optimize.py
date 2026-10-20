"""
Systematic search for a better return/drawdown ratio than baseline, with an
out-of-sample discipline: every candidate is scored on 2019-2023 ("fit"
window) and on 2024-2026 ("holdout" window) separately, and ranked by the
WORSE of the two, so a config that only works on one slice can't win.

Metric: total_return_pct / max_drawdown_pct (Calmar), computed at TRADE
resolution from cumsum(net_return_pct) - fixed notional per trade, no
compounding, fees included.

Every grid axis comes from an EDA finding (eda.py), not blind tuning:
  skip_triggers   - bb_cross loses money in 7 of 8 years
  max_hold_bars   - long-held trades are mostly losers
  atr_pct_max     - the most volatile ATR% quintile is the worst
  atr_tp/sl_mult  - fixed % brackets ignore volatility regime
"""
import itertools

import numpy as np
import pandas as pd

import engine
import main as m
import signals

FIT_END = "2024-01-01"


def calmar(trades_df: pd.DataFrame, start=None, end=None, col="net_return_pct") -> dict:
    closed = trades_df[~trades_df["open"]].copy()
    closed["exit_time"] = pd.to_datetime(closed["exit_time"])
    closed = closed.sort_values("exit_time")
    if start is not None:
        closed = closed[closed["exit_time"] >= start]
    if end is not None:
        closed = closed[closed["exit_time"] < end]
    if len(closed) < 5:
        return {"trades": len(closed), "total_return": 0, "max_dd": 0, "calmar": 0, "win_rate": 0}

    cum = closed[col].cumsum().to_numpy()
    running_max = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    max_dd = (running_max - cum).max()
    total = cum[-1]
    return {
        "trades": len(closed),
        "total_return": round(float(total), 1),
        "max_dd": round(float(max_dd), 2),
        "calmar": round(float(total / max_dd), 3) if max_dd > 0.01 else float("inf"),
        "win_rate": round(float((closed[col] > 0).mean() * 100), 1),
    }


def main():
    ohlcv = m.load_ohlcv("../data/eth_usdt_perp_30m.csv")
    sig = signals.compute_signals(ohlcv)

    grid = itertools.product(
        [(), ("bb_cross",)],
        [None, 12, 24, 48],
        [None, 0.6, 0.8, 1.0, 1.3],
        [(None, None), (2.0, 2.0), (3.0, 2.0), (2.0, 3.0), (4.0, 3.0)],
    )
    rows = []
    for skip, hold, atr_cap, (tp_m, sl_m) in grid:
        cfg = engine.EngineConfig(
            skip_triggers=skip, max_hold_bars=hold, atr_pct_max=atr_cap, atr_tp_mult=tp_m, atr_sl_mult=sl_m
        )
        tdf = pd.DataFrame(engine.run_backtest(sig, cfg))
        full, fit, hold_out = calmar(tdf), calmar(tdf, end=FIT_END), calmar(tdf, start=FIT_END)
        rows.append({
            "skip_bb_cross": bool(skip),
            "max_hold_bars": hold,
            "atr_pct_max": atr_cap,
            "atr_tp_mult": tp_m,
            "atr_sl_mult": sl_m,
            "trades": full["trades"],
            "win_rate": full["win_rate"],
            "net_return": full["total_return"],
            "max_dd": full["max_dd"],
            "calmar_full": full["calmar"],
            "calmar_fit_19_23": fit["calmar"],
            "calmar_holdout_24_26": hold_out["calmar"],
            "min_calmar": min(fit["calmar"], hold_out["calmar"]),
        })
        print(f"  {len(rows)}/200", end="\r")

    res = pd.DataFrame(rows).sort_values("min_calmar", ascending=False)
    res.to_csv("../output/optimize_sweep.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\n=== Top 15 of {len(res)} by min_calmar (worst of fit/holdout) ===")
    print(res.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
