"""
CLI entry point: loads OHLCV data, computes the Pine-faithful signals, runs
the sequential backtest engine, and writes one row per trade (trade outcome
+ full indicator snapshot at entry) to a single output CSV.

Usage:
    python main.py --data ..\\data\\eth_usdt_perp_30m.csv --out ..\\output\\trades_with_indicators.csv
    python main.py --data ..\\data\\eth_usdt_perp_30m.csv --validate ..\\tv_trades.csv
    python main.py --data ..\\data\\eth_usdt_perp_30m.csv --preset optimized --out ..\\output\\optimized_trades.csv
"""
import argparse
import os

import pandas as pd

import engine
import signals
import validate as validate_mod


MIN_TRADES_THRESHOLD = 50

# Presets for engine.EngineConfig, selectable via --preset. "baseline" is the
# plain textbook strategy (both triggers, fixed 1.5%/1.5% bracket, no
# filters). "optimized" is the winner of optimize.py's sweep, ranked by the
# WORSE of its 2019-2023 and 2024-2026 Calmar ratios (net of fees) so it
# can't win by overfitting one half of history. Each rule traces back to an
# EDA finding (see eda.py): drop the bb_cross trigger (negative in 7 of 8
# year), a 24-bar (12h) time-stop, skip entries when ATR > 0.8% of price, and
# an ATR-scaled bracket (TP 4x ATR, SL 3x ATR) instead of fixed percentages.
PRESETS = {
    "baseline": lambda: engine.EngineConfig(),
    "optimized": lambda: engine.EngineConfig(
        skip_triggers=("bb_cross",), max_hold_bars=24, atr_pct_max=0.8, atr_tp_mult=4.0, atr_sl_mult=3.0
    ),
}


def _trim_thin_listing_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Drops leading bars from the very start of the series that have
    negligible trade count (e.g. a symbol's first few minutes on an
    exchange, before real liquidity arrives - Binance ETHUSDT.P's first six
    30m bars have 1/1/4/0/0/0 trades, all with degenerate flat/duplicate
    OHLC, before real trading with 300-600+ trades/bar begins). Only trims
    a leading run; a thin bar deep in real history is left alone."""
    if "num_trades" not in df.columns:
        return df
    thin = df["num_trades"] < MIN_TRADES_THRESHOLD
    if not thin.iloc[0]:
        return df
    first_real = thin.idxmin() if not thin.all() else None
    if first_real is None:
        return df
    return df.loc[first_real:]


def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    if "open_time_iso" in cols:
        time_col = cols["open_time_iso"]
    elif "timestamp" in cols:
        time_col = cols["timestamp"]
    elif "date and time" in cols:
        time_col = cols["date and time"]
    elif "date" in cols:
        time_col = cols["date"]
    else:
        raise ValueError(
            f"Couldn't find a timestamp column in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_localize(None)
    rename_map = {time_col: "time",
                  cols["open"]: "open",
                  cols["high"]: "high",
                  cols["low"]: "low",
                  cols["close"]: "close"}
    keep_cols = [time_col, cols["open"], cols["high"], cols["low"], cols["close"]]
    if "num_trades" in cols:
        keep_cols.append(cols["num_trades"])
        rename_map[cols["num_trades"]] = "num_trades"
    df = df.rename(columns=rename_map)
    df = df[[rename_map.get(c, c) for c in keep_cols]]
    df = df.dropna(subset=["time"]).drop_duplicates(subset="time").sort_values("time")
    df = df.set_index("time")
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df = _trim_thin_listing_bars(df)
    return df.drop(columns=["num_trades"], errors="ignore")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="OHLCV CSV path")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "..", "output", "trades_with_indicators.csv"),
        help="Output trades CSV path",
    )
    parser.add_argument(
        "--validate",
        default=None,
        help="Path to a TradingView List-of-trades CSV export to diff the generated trades against",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=25,
        help="Max mismatches to print when --validate is used",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="baseline",
        help="Engine configuration to run (see PRESETS docstring above)",
    )
    args = parser.parse_args()

    print(f"Loading OHLCV from {args.data} ...")
    ohlcv = load_ohlcv(args.data)
    print(f"  {len(ohlcv)} bars, {ohlcv.index[0]} .. {ohlcv.index[-1]}")

    print("Computing indicators/signals ...")
    sig_df = signals.compute_signals(ohlcv)

    print(f"Running backtest engine (preset={args.preset}) ...")
    trades = engine.run_backtest(sig_df, PRESETS[args.preset]())
    print(f"  {len(trades)} trades generated")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    if args.validate:
        print(f"\nValidating against {args.validate} ...")
        validate_mod.compare(trades, args.validate, max_report=args.max_report)


if __name__ == "__main__":
    main()
