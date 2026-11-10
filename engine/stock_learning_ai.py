#!/usr/bin/env python3
"""Train a stock-ranking model and preview its picks. Requires stock_scanner.py."""
import argparse
from datetime import datetime, timedelta
from getpass import GetPassWarning, getpass
import os
from statistics import mean, pstdev
import sys
from urllib.error import HTTPError, URLError
import warnings
from zoneinfo import ZoneInfo

from stock_scanner import BENCHMARK, DEFAULT, FRICTION, SLOTS, demo_bars, get_bars

HORIZON = 5
WINDOW = 50


def features(series, dates, i):
    window = dates[i - WINDOW + 1:i + 1]
    if len(window) != WINDOW or any(day not in series for day in window):
        return None
    close = [series[day][1] for day in window]
    daily = [b / a - 1 for a, b in zip(close, close[1:])]
    return (
        close[-1] / close[-6] - 1,
        close[-1] / close[-21] - 1,
        close[-1] / mean(close) - 1,
        pstdev(daily[-20:]),
        mean(daily[-5:]) - mean(daily[-20:]),
    )


def fit(examples):
    """Standardized ridge regression fitted by batch gradient descent."""
    if len(examples) < 100:
        raise ValueError("Too little training data.")
    columns = list(zip(*(x for x, _ in examples)))
    centers = [mean(c) for c in columns]
    scales = [pstdev(c) or 1 for c in columns]

    def scale(x):
        return [(value - center) / spread for value, center, spread in zip(x, centers, scales)]

    intercept = mean(y for _, y in examples)
    rows = [(scale(x), y - intercept) for x, y in examples]
    weights = [0.0] * len(columns)
    for _ in range(140):
        gradient = [0.0] * len(weights)
        for x, y in rows:
            error = sum(w * v for w, v in zip(weights, x)) - y
            for j, v in enumerate(x):
                gradient[j] += error * v
        weights = [w - 0.08 * (g / len(rows) + 0.2 * w)
                   for w, g in zip(weights, gradient)]

    def predict(x):
        return intercept + sum(w * v for w, v in zip(weights, scale(x)))

    return predict


def score_test(bars, dates, first, pick):
    """Run the same five-session trading schedule for both ranking methods."""
    cash = 1.0
    holdings = {}
    last_close = {}
    peak = 1.0
    drawdown = 0.0
    fills = 0
    for i in range(first + 1, len(dates)):
        session = dates[i]
        if (i - first - 1) % HORIZON == 0:
            desired = set(pick(i - 1))
            for symbol in sorted(holdings.keys() - desired):
                if session in bars[symbol]:
                    cash += holdings.pop(symbol) * bars[symbol][session][0] * (1 - FRICTION)
                    fills += 1
            for symbol in sorted(desired - holdings.keys()):
                slots = SLOTS - len(holdings)
                if slots and cash > 1e-10 and session in bars[symbol]:
                    budget = cash / slots
                    holdings[symbol] = budget / (bars[symbol][session][0] * (1 + FRICTION))
                    cash -= budget
                    fills += 1
        for symbol in holdings:
            if session in bars[symbol]:
                last_close[symbol] = bars[symbol][session][1]
        equity = cash + sum(shares * last_close[symbol] for symbol, shares in holdings.items())
        peak = max(peak, equity)
        drawdown = max(drawdown, 1 - equity / peak)
    return equity - 1, drawdown, fills


def compare(bars, symbols):
    dates = sorted(bars[BENCHMARK])
    if len(dates) < 180:
        raise ValueError("At least 180 benchmark sessions are required.")
    cutoff = int(len(dates) * 0.75)
    rows = {}
    training = []
    for i in range(WINDOW - 1, len(dates)):
        rows[i] = {}
        for symbol in symbols:
            series = bars[symbol]
            x = features(series, dates, i)
            if x is None:
                continue
            rows[i][symbol] = x
            exit_index = i + 1 + HORIZON
            if exit_index < cutoff and dates[i + 1] in series and dates[exit_index] in series:
                entry = series[dates[i + 1]][0] * (1 + FRICTION)
                exit_price = series[dates[exit_index]][0] * (1 - FRICTION)
                training.append((x, exit_price / entry - 1))

    predict = fit(training)

    def ai_rank(i):
        return sorted(((s, predict(x)) for s, x in rows[i].items()),
                      key=lambda item: (-item[1], item[0]))

    def ai_pick(i):
        return [s for s, estimate in ai_rank(i) if estimate > 0][:SLOTS]

    def rules_pick(i):
        eligible = [(s, x[1]) for s, x in rows[i].items() if x[1] > 0 and x[2] > 0]
        return [s for s, _ in sorted(eligible, key=lambda item: (-item[1], item[0]))[:SLOTS]]

    print(f"Training examples: {len(training):,} | Training data ends {dates[cutoff - 1]}")
    print(f"Held-out test: {dates[cutoff + 1]} to {dates[-1]}")
    spy = bars[BENCHMARK][dates[-1]][1] / bars[BENCHMARK][dates[cutoff + 1]][0] - 1
    print(f"SPY buy and hold: {spy:+.2%}")
    for label, pick in (("Learning model", ai_pick), ("Current ranking rules", rules_pick)):
        returns, drawdown, fills = score_test(bars, dates, cutoff, pick)
        score = returns - spy - drawdown
        print(f"{label}: {returns:+.2%} return | {drawdown:.2%} max drawdown | "
              f"{fills} simulated fills | risk score {score * 100:+.2f} points")
    last = len(dates) - 1
    print(f"Latest completed session: {dates[last]}")
    ranked = ai_rank(last)[:SLOTS]
    for symbol, estimate in ranked:
        print(f"  {symbol}: model estimate {estimate:+.2%} over five sessions after modeled costs")
    print("AI preview picks: " + (", ".join(ai_pick(last)) or "Cash"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Use synthetic prices")
    args = parser.parse_args()
    warnings.simplefilter("error", GetPassWarning)
    end = datetime.now(ZoneInfo("America/New_York")).date()
    symbols = DEFAULT.split(",")
    try:
        if not args.demo:
            for name, label in (("APCA_API_KEY_ID", "Key ID"),
                                ("APCA_API_SECRET_KEY", "Secret Key")):
                if not os.getenv(name):
                    os.environ[name] = getpass(f"Paste Alpaca Paper {label}: ").strip()
        fetch = demo_bars if args.demo else get_bars
        bars = fetch([*symbols, BENCHMARK], end - timedelta(days=1461), end)
        if not all(bars.values()):
            raise ValueError("One or more stocks returned no price history.")
        print("SYNTHETIC DEMO" if args.demo else "HISTORICAL MARKET DATA")
        compare(bars, symbols)
    except HTTPError as exc:
        parser.exit(1, f"Alpaca returned HTTP {exc.code}. Check the paper keys and market data access.\n")
    except (GetPassWarning, OSError, URLError, ValueError, KeyError) as exc:
        parser.exit(1, f"Learning model stopped: {exc}\n")
    except KeyboardInterrupt:
        sys.exit("\nStopped.")


if __name__ == "__main__":
    main()
