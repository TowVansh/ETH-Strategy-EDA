"""
Compares the engine's generated trades against a TradingView "List of trades"
CSV export trade-by-trade, and reports the first mismatches with enough
context to debug quickly.

the TradingView export layout: each trade number contributes two rows - an "Exit *"
row and an "Entry *" row (in that order in the file, not chronological
order) - sharing the same Return %/Duration/Cumulative PnL values.
"""
import argparse

import pandas as pd

PRICE_TOL = 1e-4
PCT_TOL = 1e-3

TV_TIMEZONE = "America/New_York"


def load_tradingview_trades(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    # the TradingView export's "Date and time" is in TradingView's account timezone
    # (confirmed empirically: America/New_York, DST-aware - UTC-4 in summer,
    # UTC-5 in winter), not UTC. Convert to naive UTC to match the engine's
    # Binance-sourced (UTC) timestamps.
    raw["Date and time"] = (
        pd.to_datetime(raw["Date and time"])
        .dt.tz_localize(TV_TIMEZONE, ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )

    records = []
    for trade_no, grp in raw.groupby("Trade number"):
        entry_rows = grp[grp["Type"].str.startswith("Entry")]
        exit_rows = grp[grp["Type"].str.startswith("Exit")]
        if entry_rows.empty or exit_rows.empty:
            continue
        e = entry_rows.iloc[0]
        x = exit_rows.iloc[0]
        direction = "long" if "long" in e["Type"] else "short"
        records.append(
            {
                "trade_number": trade_no,
                "direction": direction,
                "entry_time": e["Date and time"],
                "entry_price": float(e["Price USDT"]),
                "exit_time": x["Date and time"],
                "exit_price": float(x["Price USDT"]),
                "return_pct": float(x["Return %"]),
                "duration_bars": int(x["Duration (bars)"]),
                "open": x["Signal"] == "Open",
            }
        )
    out = pd.DataFrame.from_records(records).sort_values("entry_time").reset_index(drop=True)
    return out


def compare(engine_trades: list, tv_csv_path: str, max_report: int = 25):
    tv = load_tradingview_trades(tv_csv_path)
    eng = pd.DataFrame(engine_trades).sort_values("entry_time").reset_index(drop=True)
    eng["entry_time"] = pd.to_datetime(eng["entry_time"])
    eng["exit_time"] = pd.to_datetime(eng["exit_time"])

    n = min(len(tv), len(eng))
    print(f"TradingView trades: {len(tv)}   Engine trades: {len(eng)}")
    if len(tv) != len(eng):
        print("!! Trade counts differ - alignment below is by position and will "
              "drift after the first divergence.")

    mismatches = 0
    for i in range(n):
        t = tv.iloc[i]
        e = eng.iloc[i]
        problems = []

        if t["direction"] != e["direction"]:
            problems.append(f"direction: tv={t['direction']} eng={e['direction']}")
        if t["entry_time"] != e["entry_time"]:
            problems.append(f"entry_time: tv={t['entry_time']} eng={e['entry_time']}")
        if abs(t["entry_price"] - e["entry_price"]) > PRICE_TOL:
            problems.append(f"entry_price: tv={t['entry_price']} eng={e['entry_price']}")
        if not t["open"]:
            if t["exit_time"] != e["exit_time"]:
                problems.append(f"exit_time: tv={t['exit_time']} eng={e['exit_time']}")
            if abs(t["exit_price"] - e["exit_price"]) > PRICE_TOL:
                problems.append(f"exit_price: tv={t['exit_price']} eng={e['exit_price']}")
            # TV's CSV only ever shows Return % rounded to 2 decimals (it's a
            # display artifact of the export, not the true internal value - see
            # the TradingView export's own Net PnL / Size(value) columns, which carry
            # more precision than the rounded Return % column does).
            if abs(round(t["return_pct"], 2) - round(e["return_pct"], 2)) > PCT_TOL:
                problems.append(f"return_pct: tv={t['return_pct']} eng={e['return_pct']}")

        if problems:
            mismatches += 1
            if mismatches <= max_report:
                print(f"\nTrade #{i+1} (tv trade_number={t['trade_number']}, "
                      f"engine trade_number={e['trade_number']}):")
                for p in problems:
                    print(f"  {p}")

    print(f"\n{mismatches} mismatched trade(s) out of {n} compared.")
    if mismatches == 0 and len(tv) == len(eng):
        print("MATCH: engine reproduces the TradingView export exactly.")
    return mismatches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("engine_trades_csv")
    parser.add_argument("tradingview_csv")
    parser.add_argument("--max-report", type=int, default=25)
    args = parser.parse_args()

    eng_df = pd.read_csv(args.engine_trades_csv)
    compare(eng_df.to_dict("records"), args.tradingview_csv, args.max_report)
