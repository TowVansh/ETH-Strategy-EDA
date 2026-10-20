"""
Fetches the complete ETHUSDT Perpetual Futures 30m kline history from Binance.

Pagination: starts at (or before) ETHUSDT.P's listing date, requests the max
batch size per call, and advances startTime to the last candle's open_time +
one interval after each batch, until Binance returns a short/empty batch
(caught up to "now"). Resumable: if the output CSV already exists, resumes
from its last timestamp instead of re-downloading everything.
"""
import argparse
import csv
import os
import time
from datetime import datetime, timezone

import requests

BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = "ETHUSDT"
INTERVAL = "30m"
INTERVAL_MS = 30 * 60 * 1000
LIMIT = 1500
# ETHUSDT perpetual futures launched 2019-11-27 on Binance; start a little
# earlier to be safe, Binance will just return the earliest available bars.
DEFAULT_START_MS = int(datetime(2019, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)

COLUMNS = [
    "open_time_ms",
    "open_time_iso",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms",
    "num_trades",
]


def fetch_batch(session, start_ms, retries=6):
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": start_ms,
        "limit": LIMIT,
    }
    delay = 1.0
    for attempt in range(retries):
        try:
            resp = session.get(BASE_URL, params=params, timeout=15)
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            print(f"  request error ({exc}), retrying in {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code in (429, 418):
            wait = float(resp.headers.get("Retry-After", delay))
            print(f"  rate limited (HTTP {resp.status_code}), waiting {wait:.1f}s")
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue

        if 500 <= resp.status_code < 600:
            if attempt == retries - 1:
                resp.raise_for_status()
            print(f"  server error (HTTP {resp.status_code}), retrying in {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue

        # 4xx other than rate limit - not retryable
        resp.raise_for_status()

    raise RuntimeError("exhausted retries fetching batch")


def row_from_kline(k):
    open_time_ms = int(k[0])
    return [
        open_time_ms,
        datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).isoformat(),
        k[1],
        k[2],
        k[3],
        k[4],
        k[5],
        int(k[6]),
        int(k[8]),
    ]


def last_open_time_ms(csv_path):
    last_row = None
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if row:
                last_row = row
    if last_row is None:
        return None
    return int(last_row[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "eth_usdt_perp_30m.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Override start time in epoch ms (default: 2019-11-01, or resume point)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.35,
        help="Seconds to sleep between requests",
    )
    args = parser.parse_args()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    resuming = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    if args.start is not None:
        start_ms = args.start
    elif resuming:
        last_ms = last_open_time_ms(out_path)
        start_ms = (last_ms + INTERVAL_MS) if last_ms is not None else DEFAULT_START_MS
        print(f"Resuming from existing file, last open_time={last_ms} -> starting at {start_ms}")
    else:
        start_ms = DEFAULT_START_MS

    mode = "a" if resuming else "w"
    session = requests.Session()

    total_written = 0
    with open(out_path, mode, newline="") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(COLUMNS)

        cur_start = start_ms
        seen_last_ms = None
        while True:
            klines = fetch_batch(session, cur_start)
            if not klines:
                print("Empty batch received - reached the end of available history.")
                break

            rows = [row_from_kline(k) for k in klines]

            # Guard against an infinite loop if Binance ever returns the same
            # candle again without advancing.
            if seen_last_ms is not None and rows[-1][0] <= seen_last_ms:
                print("No forward progress in returned batch - stopping.")
                break

            writer.writerows(rows)
            f.flush()
            total_written += len(rows)
            seen_last_ms = rows[-1][0]

            first_iso = rows[0][1]
            last_iso = rows[-1][1]
            print(f"  fetched {len(rows)} candles [{first_iso} .. {last_iso}] (total written this run: {total_written})")

            if len(klines) < LIMIT:
                print("Received a short batch - caught up to the present.")
                break

            cur_start = seen_last_ms + INTERVAL_MS
            time.sleep(args.sleep)

    print(f"Done. Wrote {total_written} new rows to {out_path}")


if __name__ == "__main__":
    main()
