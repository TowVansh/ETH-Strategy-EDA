"""
Exploratory data analysis of trade outcomes. Runs the baseline and optimized
presets, then writes figures to ../figures and a text summary to
../output/eda_summary.txt.

Usage:
    python eda.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import engine
import main as m
import signals
from optimize import FIT_END, calmar

FIG = os.path.join("..", "figures")
OUT = os.path.join("..", "output")
BLUE, ORANGE, GREEN, RED, GREY = "#2a78d6", "#e8871e", "#1f9d55", "#d03b3b", "#888888"

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=180)
    plt.close(fig)


def closed(trades):
    df = pd.DataFrame(trades)
    df = df[~df["open"]].copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["year"] = df["exit_time"].dt.year
    df["win"] = df["net_return_pct"] > 0
    df["atr_pct"] = df["atr"] / df["entry_price"] * 100
    return df.sort_values("exit_time")


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    ohlcv = m.load_ohlcv("../data/eth_usdt_perp_30m.csv")
    sig = signals.compute_signals(ohlcv)
    base = closed(engine.run_backtest(sig, m.PRESETS["baseline"]()))
    opt = closed(engine.run_backtest(sig, m.PRESETS["optimized"]()))
    base.to_csv(os.path.join(OUT, "trades_with_indicators.csv"), index=False)
    opt.to_csv(os.path.join(OUT, "optimized_trades.csv"), index=False)
    lines = [f"Data: {len(ohlcv)} bars, {ohlcv.index[0]} .. {ohlcv.index[-1]}", ""]

    # 1) Equity curves, gross vs net.
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for df, name, color in [(base, "Baseline", GREY), (opt, "Optimized", BLUE)]:
        ax.plot(df["exit_time"], df["net_return_pct"].cumsum(), color=color, lw=1.8, label=f"{name} (net of fees)")
        ax.plot(df["exit_time"], df["return_pct"].cumsum(), color=color, lw=1, ls="--", alpha=0.7, label=f"{name} (gross)")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(pd.Timestamp(FIT_END), color=ORANGE, lw=1, ls=":")
    ax.text(pd.Timestamp(FIT_END), ax.get_ylim()[1] * 0.92, "  fit | holdout", color=ORANGE)
    ax.set_ylabel("Cumulative return (% of notional)")
    ax.set_title("Cumulative return per trade, baseline vs optimized")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "equity_curve.png")

    # 2) Yearly net return.
    yb = base.groupby("year")["net_return_pct"].sum()
    yo = opt.groupby("year")["net_return_pct"].sum().reindex(yb.index, fill_value=0)
    x = np.arange(len(yb))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - 0.2, yb.values, 0.4, color=GREY, label="Baseline")
    ax.bar(x + 0.2, yo.values, 0.4, color=BLUE, label="Optimized")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, yb.index)
    ax.set_ylabel("Net return (%)")
    ax.set_title("Net return by calendar year")
    ax.legend(frameon=False)
    save(fig, "yearly_return.png")

    # 3) Trigger x year: which entry family earns.
    tby = base.pivot_table(index="year", columns="trigger", values="net_return_pct", aggfunc="sum")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    tby.plot.bar(ax=ax, color=[RED, GREEN], width=0.75, rot=0)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Net return (%)")
    ax.set_xlabel("")
    ax.set_title("Baseline net return by entry trigger and year")
    ax.legend(frameon=False)
    save(fig, "trigger_by_year.png")

    # 4) Quintile analysis of indicators at entry.
    feats = [("atr_pct", "ATR % of price"), ("trendStrength", "DMI spread (+DI - -DI)"),
             ("stochK", "Stochastic %K"), ("bbWidthPct", "Bollinger width %")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    lines.append("Baseline mean net return per trade by indicator quintile (Q1 = lowest):")
    for ax, (col, label) in zip(axes, feats):
        q = pd.qcut(base[col], 5, labels=[f"Q{i}" for i in range(1, 6)], duplicates="drop")
        g = base.groupby(q, observed=True)["net_return_pct"].mean()
        ax.bar(g.index.astype(str), g.values, color=[GREEN if v > 0 else RED for v in g.values])
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(label, fontsize=11)
        lines.append(f"  {label:26s} " + "  ".join(f"{k}={v:+.3f}" for k, v in g.items()))
    axes[0].set_ylabel("Mean net return per trade (%)")
    save(fig, "quintiles.png")

    # 5) Correlation of entry indicators with outcome.
    cols = ["net_return_pct", "atr_pct", "trendStrength", "plusDi", "minusDi", "stochK", "bbWidthPct", "duration_bars"]
    corr = base[cols].astype(float).corr()
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation: indicators at entry vs trade outcome")
    save(fig, "correlation.png")

    # 6) Holding time vs outcome.
    bins = [0, 6, 12, 24, 48, np.inf]
    labels = ["<=3h", "3-6h", "6-12h", "12-24h", ">24h"]
    d = pd.cut(base["duration_bars"], bins, labels=labels)
    g = base.groupby(d, observed=True).agg(mean=("net_return_pct", "mean"), win=("win", "mean"), n=("win", "size"))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(g.index.astype(str), g["mean"], color=[GREEN if v > 0 else RED for v in g["mean"]])
    for i, (v, n) in enumerate(zip(g["mean"], g["n"])):
        ax.text(i, v, f"n={n}", ha="center", va="bottom" if v > 0 else "top", fontsize=9)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Mean net return per trade (%)")
    ax.set_title("Baseline: holding time vs outcome")
    save(fig, "holding_time.png")

    # Text summary.
    lines += ["", "Trigger summary (baseline):"]
    lines.append(base.groupby("trigger").agg(trades=("win", "size"), win_rate=("win", "mean"),
                                             mean_net=("net_return_pct", "mean")).round(3).to_string())
    lines += ["", "Holding time (baseline):", g.round(3).to_string(), ""]
    for name, df in [("Baseline", base), ("Optimized", opt)]:
        full, fit, ho = calmar(df), calmar(df, end=FIT_END), calmar(df, start=FIT_END)
        gross = calmar(df, col="return_pct")
        pos_years = int((df.groupby("year")["net_return_pct"].sum() > 0).sum())
        lines.append(f"{name}: trades={full['trades']} win%={full['win_rate']} gross={gross['total_return']}% "
                     f"net={full['total_return']}% maxDD={full['max_dd']} calmar={full['calmar']} "
                     f"fit={fit['calmar']} holdout={ho['calmar']} positive_years={pos_years}/{df['year'].nunique()}")
    lines += ["", "Yearly net return:", pd.DataFrame({"baseline": yb, "optimized": yo}).round(1).to_string()]
    text = "\n".join(lines)
    with open(os.path.join(OUT, "eda_summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
