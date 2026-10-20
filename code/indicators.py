"""
Pine Script v5 built-in ta.* equivalents, implemented to match Pine's exact
seeding/formula behavior (not just "close enough").

These are generic building blocks operating on whatever series are handed to
them. All bar-shifting (the close[1]/high[1]/close[2]/etc. lag pattern used
throughout the strategy) is done by the caller (signals.py) so that file
stays readable. Nothing here decides
*which* bars to look at - only how to compute the indicator once the caller
has chosen the right input series.
"""
import numpy as np
import pandas as pd


def ema(src: pd.Series, length: int) -> pd.Series:
    """Matches ta.ema: seeds with the first non-NaN source value, then
    recurses ema = alpha*src + (1-alpha)*ema[1]. pandas ewm(adjust=False)
    implements exactly this seeding/recursion."""
    alpha = 2.0 / (length + 1)
    return src.ewm(alpha=alpha, adjust=False).mean()


def sma(src: pd.Series, length: int) -> pd.Series:
    return src.rolling(length, min_periods=length).mean()


def stdev(src: pd.Series, length: int) -> pd.Series:
    """Population stdev (ddof=0), matching ta.stdev's default biased=true."""
    return src.rolling(length, min_periods=length).std(ddof=0)


def rma(src: pd.Series, length: int) -> pd.Series:
    """Matches ta.rma: rma = na(rma[1]) ? sma(src, length) : (src-rma[1])*alpha + rma[1].
    i.e. seeded with an SMA of the first `length` samples, then Wilder-smoothed
    recursively from there - NOT the same seeding as ta.ema."""
    alpha = 1.0 / length
    vals = src.to_numpy(dtype=float)
    seed = sma(src, length).to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan)

    prev = np.nan
    started = False
    for i in range(n):
        if not started:
            if not np.isnan(seed[i]):
                prev = seed[i]
                out[i] = prev
                started = True
        else:
            v = vals[i]
            if np.isnan(v):
                out[i] = np.nan
            else:
                prev = (v - prev) * alpha + prev
                out[i] = prev
    return pd.Series(out, index=src.index)


def stoch(source: pd.Series, high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    """Raw (unsmoothed) %K: ta.stoch(source, high, low, length)."""
    hh = high.rolling(length, min_periods=length).max()
    ll = low.rolling(length, min_periods=length).min()
    rng = hh - ll
    out = 100.0 * (source - ll) / rng
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    """Elementwise max(high-low, |high-prev_close|, |low-prev_close|).
    Caller supplies prev_close pre-shifted to whatever bar Pine's tr formula
    references."""
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def plus_minus_dm(high: pd.Series, low: pd.Series):
    """plusDm = high > high[1] ? max(high-high[1], 0) : 0
    minusDm = low < low[1] ? max(low[1]-low, 0) : 0
    (relative to whichever high/low series the caller passes in)."""
    high_prev = high.shift(1)
    low_prev = low.shift(1)

    up_move = high - high_prev
    down_move = low_prev - low

    plus_dm = np.where((high > high_prev) & up_move.notna(), up_move.clip(lower=0), 0.0)
    minus_dm = np.where((low < low_prev) & down_move.notna(), down_move.clip(lower=0), 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    # Preserve NaN where the shift makes the comparison undefined (first bar).
    plus_dm[high_prev.isna()] = np.nan
    minus_dm[low_prev.isna()] = np.nan
    return plus_dm, minus_dm


MIN_TICK = 0.01


def round_tick(price: pd.Series, direction: str, tick: float = MIN_TICK) -> pd.Series:
    """Snaps a price series to the instrument's tick size. direction='up'
    rounds toward +inf, 'down' toward -inf (never toward nearest) - matches
    how a real exchange/TradingView rounds submitted order prices."""
    scaled = (price / tick).round(6)  # kill float representation noise first
    if direction == "up":
        snapped = np.ceil(scaled)
    elif direction == "down":
        snapped = np.floor(scaled)
    else:
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
    return snapped * tick


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """ta.crossover(a, b): a[1] <= b[1] and a > b."""
    prev_le = a.shift(1) <= b.shift(1)
    now_gt = a > b
    return (prev_le & now_gt).fillna(False)


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    """ta.crossunder(a, b): a[1] >= b[1] and a < b."""
    prev_ge = a.shift(1) >= b.shift(1)
    now_lt = a < b
    return (prev_ge & now_lt).fillna(False)
