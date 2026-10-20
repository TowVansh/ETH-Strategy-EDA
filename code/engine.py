"""
Sequential, single-position backtest engine implementing TradingView's
broker-emulator order-execution model:

- strategy.entry (market order) fills at the OPEN of the bar AFTER the
  signal bar - never the signal bar itself.
- TP/SL price levels are fixed at signal time from that bar's `close`
  (tpBuy/slBuy etc. in signals.py), independent of the actual fill price.
- Exits are an OCA bracket (limit=TP, stop=SL). When a single bar's
  [low, high] range could satisfy both, TradingView's broker emulator (no
  Bar Magnifier) assumes one deterministic intrabar path per bar:
    open -> high -> low -> close   if |open-high| < |open-low|
    open -> low -> high -> close   otherwise
  Whichever level that path reaches first is the fill, at that level's exact
  price (the emulator assumes no gaps inside a bar, so it never "jumps past"
  a limit/stop once the bar's range covers it).
- Only one position open at a time (pyramiding=0 + strategy.opentrades==0
  gate in Pine). If both buySignal and sellSignal are true on the same
  bar, sellSignal wins (`and not sellSignal` guard).
- An entry fill and its bracket exit can both resolve within the same bar.
"""
import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class EngineConfig:
    """Every field defaults to the plain baseline behavior. Set fields to explore rule variants - each is an
    isolated, explainable change, not a curve-fit knob."""

    # Force-exit if neither TP nor SL has hit within this many bars of the
    # fill. Arms on the bar the threshold is reached and closes at the NEXT
    # bar's open (a plain strategy.close_all() market order - same timing as
    # every other order in this engine). None = no time-stop (baseline).
    max_hold_bars: Optional[int] = None

    # Skip taking a bb_cross-triggered entry when stochK at signal time falls
    # inside this (low, high) range - the "neutral zone, weak trigger" combo
    # identified in the trade-outcome analysis. None = no filter (baseline).
    suppress_bb_cross_stoch_range: Optional[tuple] = None

    # Replace the fixed-percentage TP/SL with ATR-multiples of the signal
    # bar's ATR (period 14), e.g. tp_mult=2.5, sl_mult=3.5. Both must be set
    # to activate; None/None = fixed-percentage baseline.
    atr_tp_mult: Optional[float] = None
    atr_sl_mult: Optional[float] = None

    # Skip ANY entry when the signal bar's ATR% (ATR / close * 100) exceeds
    # this threshold - the "most volatile quintile underperforms" finding.
    # None = no filter (baseline).
    atr_pct_max: Optional[float] = None

    # Entry triggers to ignore entirely, e.g. ("bb_cross",). Empty = take all.
    skip_triggers: tuple = ()

    # Round-trip trading cost in percent of notional (Binance USD-M taker fee
    # 0.05% per side). Reported as net_return_pct beside the gross return_pct.
    fee_pct: float = 0.10


def _round_tick(price: float, direction: str, tick: float = 0.01) -> float:
    scaled = round(price / tick, 6)
    return (math.ceil(scaled) if direction == "up" else math.floor(scaled)) * tick


@dataclass
class PendingOrder:
    side: str  # 'long' or 'short'
    tp: float
    sl: float
    signal_bar_idx: int
    trigger: str


@dataclass
class Position:
    side: str
    tp: float
    sl: float
    entry_price: float
    entry_bar_idx: int  # fill bar (bar after the signal bar)
    signal_bar_idx: int
    trigger: str
    time_stop_armed: bool = False  # duration threshold hit; closes at next bar's open


INDICATOR_COLUMNS = [
    "emaFast", "emaSlow", "uptrend", "downtrend",
    "atr", "plusDi", "minusDi", "trendStrength",
    "stochK", "basis", "upperBB", "lowerBB", "bbWidthPct",
    "bbLong", "bbShort", "dmiLong", "dmiShort",
]


def _resolve_intrabar_path(open_, high, low):
    """Returns 'high_first' or 'low_first' per TradingView's broker-emulator
    assumption."""
    if abs(open_ - high) <= abs(open_ - low):
        return "high_first"
    return "low_first"


def _check_exit(position: Position, open_, high, low):
    """Returns (exit_price, exit_reason) or None if neither TP nor SL is hit
    on this bar."""
    path = _resolve_intrabar_path(open_, high, low)

    if position.side == "long":
        tp_hit = high >= position.tp
        sl_hit = low <= position.sl
        if path == "high_first":
            if tp_hit:
                return position.tp, "tp"
            if sl_hit:
                return position.sl, "sl"
        else:
            if sl_hit:
                return position.sl, "sl"
            if tp_hit:
                return position.tp, "tp"
    else:  # short
        tp_hit = low <= position.tp
        sl_hit = high >= position.sl
        if path == "high_first":
            if sl_hit:
                return position.sl, "sl"
            if tp_hit:
                return position.tp, "tp"
        else:
            if tp_hit:
                return position.tp, "tp"
            if sl_hit:
                return position.sl, "sl"
    return None


def run_backtest(df: pd.DataFrame, config: Optional[EngineConfig] = None) -> list:
    """df must be the output of signals.compute_signals (has OHLC + all
    signal/indicator columns), indexed by an ascending timestamp.

    Returns a list of trade dicts (one per closed trade, plus a final
    unclosed trade dict if a position is still open at the end of data)."""
    if config is None:
        config = EngineConfig()

    idx = df.index
    open_ = df["open"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    atr = df["atr"].to_numpy()
    stoch_k = df["stochK"].to_numpy()
    buy_signal = df["buySignal"].to_numpy()
    sell_signal = df["sellSignal"].to_numpy()
    tp_buy = df["tpBuy"].to_numpy()
    sl_buy = df["slBuy"].to_numpy()
    tp_sell = df["tpSell"].to_numpy()
    sl_sell = df["slSell"].to_numpy()
    buy_trigger = df["buyTrigger"].to_numpy()
    sell_trigger = df["sellTrigger"].to_numpy()

    atr_scaled = config.atr_tp_mult is not None and config.atr_sl_mult is not None

    n = len(df)
    trades = []
    trade_num = 0

    position: Optional[Position] = None
    pending: Optional[PendingOrder] = None

    def snapshot_indicators(bar_idx):
        row = df.iloc[bar_idx]
        return {col: row[col] for col in INDICATOR_COLUMNS}

    for i in range(n):
        # 1) Fill any pending entry order at this bar's open.
        if pending is not None and position is None:
            position = Position(
                side=pending.side,
                tp=pending.tp,
                sl=pending.sl,
                entry_price=float(open_[i]),
                entry_bar_idx=i,
                signal_bar_idx=pending.signal_bar_idx,
                trigger=pending.trigger,
            )
            pending = None

        # 1.5) A time-stop armed on a PRIOR bar closes at THIS bar's open - a
        # plain strategy.close_all() market order, same next-bar-open timing
        # as every other order in this engine (Pine has no way to force a
        # same-bar-close fill either, so this matches Pine's strategy.close_all() timing).
        exit_price = exit_reason = None
        if position is not None and position.time_stop_armed:
            exit_price, exit_reason = float(open_[i]), "time_stop"

        # 2) Check TP/SL against this bar (whether just filled or already
        #    open), unless already closed by the time-stop above.
        if position is not None and exit_price is None:
            result = _check_exit(position, open_[i], high[i], low[i])
            if result is not None:
                exit_price, exit_reason = result
            elif (
                config.max_hold_bars is not None
                and (i - position.entry_bar_idx + 1) >= config.max_hold_bars
            ):
                # Duration threshold reached with neither TP nor SL hit this
                # bar - arm the time-stop to close at the NEXT bar's open.
                position.time_stop_armed = True

        if position is not None and exit_price is not None:
            trade_num += 1
            ret_pct = (
                (exit_price - position.entry_price) / position.entry_price * 100
                if position.side == "long"
                else (position.entry_price - exit_price) / position.entry_price * 100
            )
            trade = {
                "trade_number": trade_num,
                "direction": position.side,
                "trigger": position.trigger,
                "signal_time": idx[position.signal_bar_idx],
                "entry_time": idx[position.entry_bar_idx],
                "entry_price": position.entry_price,
                "exit_time": idx[i],
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "return_pct": ret_pct,
                "net_return_pct": ret_pct - config.fee_pct,
                "duration_bars": i - position.entry_bar_idx + 1,
                "open": bool(False),
            }
            trade.update(snapshot_indicators(position.signal_bar_idx))
            trades.append(trade)
            position = None

        # 3) If flat, decide whether this bar's signal queues a new order for
        #    the NEXT bar's open (noOpenOrders gate).
        if position is None and pending is None:
            side = None
            if sell_signal[i]:
                side, trigger = "short", str(sell_trigger[i])
            elif buy_signal[i]:
                side, trigger = "long", str(buy_trigger[i])

            if side is not None and trigger in config.skip_triggers:
                side = None

            if side is not None and config.suppress_bb_cross_stoch_range is not None and trigger == "bb_cross":
                lo, hi = config.suppress_bb_cross_stoch_range
                if lo <= stoch_k[i] <= hi:
                    side = None  # filtered out - skip this entry entirely

            if side is not None and config.atr_pct_max is not None:
                atr_pct = atr[i] / close[i] * 100
                if atr_pct > config.atr_pct_max:
                    side = None  # too volatile - skip this entry entirely

            if side == "short":
                if atr_scaled:
                    ref = float(close[i])
                    tp = _round_tick(ref - atr[i] * config.atr_tp_mult, "down")
                    sl = _round_tick(ref + atr[i] * config.atr_sl_mult, "up")
                else:
                    tp, sl = float(tp_sell[i]), float(sl_sell[i])
                pending = PendingOrder(side="short", tp=tp, sl=sl, signal_bar_idx=i, trigger=trigger)
            elif side == "long":
                if atr_scaled:
                    ref = float(close[i])
                    tp = _round_tick(ref + atr[i] * config.atr_tp_mult, "up")
                    sl = _round_tick(ref - atr[i] * config.atr_sl_mult, "down")
                else:
                    tp, sl = float(tp_buy[i]), float(sl_buy[i])
                pending = PendingOrder(side="long", tp=tp, sl=sl, signal_bar_idx=i, trigger=trigger)

    # If a position is still open at the end of history, report it unrealized
    # against the last available close (mirrors TradingView's trailing
    # "Open" row in a trade-list export).
    if position is not None:
        last_close = float(close[-1])
        ret_pct = (
            (last_close - position.entry_price) / position.entry_price * 100
            if position.side == "long"
            else (position.entry_price - last_close) / position.entry_price * 100
        )
        trade_num += 1
        trade = {
            "trade_number": trade_num,
            "direction": position.side,
            "trigger": position.trigger,
            "signal_time": idx[position.signal_bar_idx],
            "entry_time": idx[position.entry_bar_idx],
            "entry_price": position.entry_price,
            "exit_time": idx[-1],
            "exit_price": last_close,
            "exit_reason": "open",
            "return_pct": ret_pct,
            "net_return_pct": ret_pct - config.fee_pct,
            "duration_bars": (n - 1) - position.entry_bar_idx + 1,
            "open": True,
        }
        trade.update(snapshot_indicators(position.signal_bar_idx))
        trades.append(trade)

    return trades
