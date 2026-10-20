"""
Signal logic for the course-project strategy: a textbook indicator-confluence
system on ETHUSDT.P 30m bars, built only from standard indicators at their
standard, widely published settings (EMA 50/200, Bollinger 20/2, Stochastic
%K 14, DMI 14, ATR 14).

Two independent entry families ("triggers"):

1. bb_cross (trend-aligned mean reversion): in an uptrend (EMA50 > EMA200),
   go long when price closes back above the lower Bollinger band while
   Stochastic %K is oversold. Mirror image for shorts.
2. dmi_trend (trend continuation): go long when the DMI spread (+DI - -DI)
   crosses up through +DMI_LEVEL with price above EMA50. Mirror for shorts.

Exits are a fixed take-profit / stop-loss bracket (see TP_PCT / SL_PCT).

Given a raw OHLC dataframe (one row per 30m bar, columns open/high/low/close,
ascending by time), `compute_signals` returns that dataframe augmented with
every intermediate indicator and the final buySignal/sellSignal columns.
"""
import pandas as pd

import indicators as ta

# --- Inputs (standard textbook settings) ------------------------------------
TREND_FAST = 50
TREND_SLOW = 200

BB_LENGTH = 20
BB_MULT = 2.0

STOCH_K_PERIOD = 14
STOCH_OVERSOLD = 25
STOCH_OVERBOUGHT = 75

DMI_PERIOD = 14
DMI_LEVEL = 20

ATR_PERIOD = 14

TP_PCT = 1.5 / 100
SL_PCT = 1.5 / 100


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """df must have columns: open, high, low, close (ascending by time)."""
    out = df.copy()
    c = out["close"]
    h = out["high"]
    l = out["low"]

    out["emaFast"] = ta.ema(c, TREND_FAST)
    out["emaSlow"] = ta.ema(c, TREND_SLOW)
    uptrend = out["emaFast"] > out["emaSlow"]
    downtrend = out["emaFast"] < out["emaSlow"]
    out["uptrend"] = uptrend
    out["downtrend"] = downtrend

    tr = ta.true_range(h, l, c.shift(1))
    out["atr"] = ta.rma(tr, ATR_PERIOD)
    plus_dm, minus_dm = ta.plus_minus_dm(h, l)
    out["plusDi"] = 100 * ta.rma(plus_dm, DMI_PERIOD) / out["atr"]
    out["minusDi"] = 100 * ta.rma(minus_dm, DMI_PERIOD) / out["atr"]
    out["trendStrength"] = out["plusDi"] - out["minusDi"]

    out["stochK"] = ta.stoch(c, h, l, STOCH_K_PERIOD)

    out["basis"] = ta.sma(c, BB_LENGTH)
    dev = BB_MULT * ta.stdev(c, BB_LENGTH)
    out["upperBB"] = out["basis"] + dev
    out["lowerBB"] = out["basis"] - dev
    out["bbWidthPct"] = (out["upperBB"] - out["lowerBB"]) / out["basis"] * 100

    bb_long = ta.crossover(c, out["lowerBB"]) & (out["stochK"] < STOCH_OVERSOLD) & uptrend
    bb_short = ta.crossunder(c, out["upperBB"]) & (out["stochK"] > STOCH_OVERBOUGHT) & downtrend
    out["bbLong"] = bb_long
    out["bbShort"] = bb_short

    level = pd.Series(DMI_LEVEL, index=out.index, dtype=float)
    dmi_long = ta.crossover(out["trendStrength"], level) & (c > out["emaFast"])
    dmi_short = ta.crossunder(out["trendStrength"], -level) & (c < out["emaFast"])
    out["dmiLong"] = dmi_long
    out["dmiShort"] = dmi_short

    out["buySignal"] = (bb_long | dmi_long).fillna(False)
    out["sellSignal"] = (bb_short | dmi_short).fillna(False)

    # Which trigger fired, for trade-level analysis.
    out["buyTrigger"] = pd.NA
    out.loc[dmi_long.fillna(False), "buyTrigger"] = "dmi_trend"
    out.loc[bb_long.fillna(False), "buyTrigger"] = "bb_cross"
    out["sellTrigger"] = pd.NA
    out.loc[dmi_short.fillna(False), "sellTrigger"] = "dmi_trend"
    out.loc[bb_short.fillna(False), "sellTrigger"] = "bb_cross"

    # Bracket levels are fixed from the signal bar's close and snapped to the
    # 0.01 tick the way TradingView does (limits toward a better price,
    # stops toward never triggering early).
    entry_buy = c.where(out["buySignal"])
    entry_sell = c.where(out["sellSignal"])
    out["tpBuy"] = ta.round_tick(entry_buy * (1 + TP_PCT), "up")
    out["slBuy"] = ta.round_tick(entry_buy * (1 - SL_PCT), "down")
    out["tpSell"] = ta.round_tick(entry_sell * (1 - TP_PCT), "down")
    out["slSell"] = ta.round_tick(entry_sell * (1 + SL_PCT), "up")

    return out
