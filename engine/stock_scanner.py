#!/usr/bin/env python3
"""Daily stock scanner and paper-trading simulator (Python 3.9+).

Try it without an account:
    python3 stock_scanner.py --demo

Use historical market data:
    Create an Alpaca paper-only account and generate paper API credentials.
    Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your local environment.
    Run: python3 stock_scanner.py --start YYYY-MM-DD
    Use the same start date for later runs to replay the same paper portfolio.

Optional: --symbols AAPL,MSFT,...  --cash 10000  --output trades.csv
Prices are split/dividend adjusted. Simulated fills use the next session's
opening price plus 0.10% friction. This program uses market-data requests only.
"""

import argparse
import csv
from datetime import date, datetime, timedelta
import json
import math
import os
import random
from statistics import mean
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

DEFAULT = (
    "AAPL,MSFT,NVDA,AMZN,GOOGL,META,AVGO,JPM,V,UNH,COST,XOM,HD,PG,KO,PEP,ABBV,AMD,CRM,WMT,"
    "ADBE,ORCL,CAT,GE,UPS,NEE,DUK,COP,BAC,MCD"
)
BENCHMARK = "SPY"
SLOTS = 3
FRICTION = 0.001


def get_bars(symbols, start, end):
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise ValueError("Set your Alpaca paper API credentials locally, or try --demo.")
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params = {
        "symbols": ",".join(symbols), "timeframe": "1Day", "feed": "iex",
        "adjustment": "all", "start": start.isoformat(), "end": end.isoformat(),
        "limit": 10000,
    }
    result = {symbol: {} for symbol in symbols}
    while True:
        request = Request("https://data.alpaca.markets/v2/stocks/bars?" + urlencode(params), headers=headers)
        with urlopen(request, timeout=30) as response:
            page = json.load(response)
        for symbol, bars in page["bars"].items():
            for bar in bars:
                session = date.fromisoformat(bar["t"][:10])
                if session < end and all(math.isfinite(bar[k]) and bar[k] > 0 for k in ("o", "c")):
                    result[symbol][session] = bar["o"], bar["c"]
        token = page.get("next_page_token")
        if not token:
            return result
        params["page_token"] = token


def demo_bars(symbols, start, end):
    rng = random.Random(42)
    result = {symbol: {} for symbol in symbols}
    for symbol in symbols:
        price = 100
        for offset in range((end - start).days):
            day = start + timedelta(days=offset)
            if day.weekday() < 5:
                opening = price * rng.uniform(0.995, 1.005)
                price = opening * (1 + 0.002 * math.sin(offset / 25) + rng.uniform(-0.018, 0.019))
                result[symbol][day] = opening, price
    return result


def simulate(bars, symbols, start, initial_cash):
    dates = sorted(bars[BENCHMARK])
    if len(dates) < 51:
        raise ValueError("At least 51 benchmark sessions are needed.")
    cash = initial_cash
    positions = {}
    last_close = {}
    pending = None
    benchmark_open = None
    trades = []
    peak = initial_cash
    drawdown = 0
    ranked = []

    for index, session in enumerate(dates):
        todays = {symbol: bars[symbol][session] for symbol in symbols if session in bars[symbol]}
        if session < start:
            continue
        if benchmark_open is None:
            benchmark_open = bars[BENCHMARK][session][0]

        if pending is not None:
            for symbol in sorted(positions.keys() - pending):
                if symbol in todays:
                    fill = todays[symbol][0] * (1 - FRICTION)
                    shares = positions.pop(symbol)
                    cash += shares * fill
                    trades.append((session, symbol, "SELL", shares, fill))
            slots = SLOTS - len(positions)
            for symbol in (item[0] for item in ranked if item[0] in pending):
                if slots <= 0:
                    break
                if symbol in positions or symbol not in todays or cash <= 0.01:
                    continue
                fill = todays[symbol][0] * (1 + FRICTION)
                budget = cash / slots
                positions[symbol] = budget / fill
                cash -= budget
                slots -= 1
                trades.append((session, symbol, "BUY", positions[symbol], fill))

        last_close.update({symbol: prices[1] for symbol, prices in todays.items()})
        equity = cash + sum(shares * last_close[symbol] for symbol, shares in positions.items())
        peak = max(peak, equity)
        drawdown = max(drawdown, 1 - equity / peak)

        ranked = []
        if index >= 49:
            window = dates[index - 49:index + 1]
            for symbol in symbols:
                series = bars[symbol]
                if any(day not in series for day in window):
                    continue
                closes = [series[day][1] for day in window]
                momentum = closes[-1] / closes[-21] - 1
                if closes[-1] > mean(closes) and momentum > 0:
                    ranked.append((symbol, momentum))
            ranked.sort(key=lambda item: (-item[1], item[0]))
        # Keep held positions with incomplete data until we can assess them again.
        protected = {symbol for symbol in positions
                     if index >= 49 and any(day not in bars[symbol] for day in window)}
        pending = set(protected)
        candidates = [symbol for symbol, _ in ranked if symbol not in protected]
        pending.update(candidates[:SLOTS - len(protected)])

    if benchmark_open is None:
        raise ValueError("No completed market sessions on or after the start date.")
    benchmark_return = bars[BENCHMARK][dates[-1]][1] / benchmark_open - 1
    return dates[-1], equity, benchmark_return, drawdown, positions, pending, ranked, trades


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--cash", type=float, default=10000)
    parser.add_argument("--symbols", default=DEFAULT)
    parser.add_argument("--output", help="Optional path for the simulated trade log (CSV)")
    args = parser.parse_args()
    end = datetime.now(ZoneInfo("America/New_York")).date()
    if not args.demo and args.start is None:
        parser.error("Provide --start YYYY-MM-DD or try --demo.")
    start = args.start or end - timedelta(days=90)
    if not math.isfinite(args.cash) or args.cash <= 0 or start >= end:
        parser.error("Cash must be positive and the start date must precede today.")
    symbols = tuple(dict.fromkeys(s.strip().upper() for s in args.symbols.split(",") if s.strip()))
    if not symbols:
        parser.error("Provide at least one stock symbol.")
    universe = tuple(dict.fromkeys((*symbols, BENCHMARK)))
    warmup = start - timedelta(days=120)
    try:
        fetch = demo_bars if args.demo else get_bars
        bars = fetch(universe, warmup, end)
        missing = [symbol for symbol in universe if not bars[symbol]]
        if missing:
            raise ValueError("No data for " + ", ".join(missing))
        day, equity, benchmark, drawdown, positions, targets, ranked, trades = simulate(
            bars, symbols, start, args.cash
        )
        print(f"{'SYNTHETIC DEMO' if args.demo else 'HISTORICAL PAPER SIMULATION'} | Through {day}")
        print(f"Scanned {len(symbols)} symbols | Simulated fills: {len(trades)}")
        print(f"Portfolio: ${equity:,.2f} ({equity / args.cash - 1:+.2%})")
        print(f"SPY buy and hold: {benchmark:+.2%} | Max drawdown: {drawdown:.2%}")
        for label, group in (("BUY", targets - positions.keys()),
                             ("SELL", positions.keys() - targets),
                             ("HOLD", targets & positions.keys())):
            print(f"{label} next session: {', '.join(sorted(group)) or 'None'}")
        print("Top eligible stocks (20-session return):")
        for symbol, momentum in ranked[:5]:
            print(f"  {symbol:<6} {momentum:+.2%}")
        if args.output:
            with open(args.output, "w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("session", "symbol", "side", "simulated_shares", "simulated_fill"))
                writer.writerows(trades)
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, HTTPError):
            parser.exit(1, f"Market data request failed (HTTP {exc.code}).\n")
        if isinstance(exc, URLError):
            parser.exit(1, f"Market data connection failed: {exc.reason}\n")
        parser.exit(1, f"Scanner error: {exc}\n")


if __name__ == "__main__":
    main()
