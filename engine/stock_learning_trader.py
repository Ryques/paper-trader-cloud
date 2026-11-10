#!/usr/bin/env python3
"""Learning-model trader for a separate Alpaca Paper account."""
import argparse
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_DOWN
import fcntl
from getpass import GetPassWarning, getpass
import json
import os
import sqlite3
from pathlib import Path
from functools import lru_cache
import time
import warnings
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from stock_scanner import BENCHMARK, DEFAULT, FRICTION, SLOTS, get_bars
from stock_learning_ai import HORIZON, WINDOW, features, fit

PAPER_URL = "https://paper-api.alpaca.markets"
ORDER_SIZE = Decimal("100")
DAILY_LOSS_LIMIT = Decimal("5")
ORDER_PREFIX = "scanner-ai-"
EASTERN = ZoneInfo("America/New_York")
FEEDBACK_WEIGHT = 20
SYMBOLS = DEFAULT.split(",")
DECISION_LOG = Path.home() / ".stock_paper_decisions.jsonl"
SCORECARD = Path.home() / "paper_trade_scorecard.sqlite3"


def paper(path, order=None):
    headers = {
        "APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"],
        "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"],
        "Content-Type": "application/json",
    }
    payload = None if order is None else json.dumps(order).encode()
    with urlopen(Request(PAPER_URL + path, data=payload, headers=headers), timeout=30) as response:
        return json.load(response)


def reward(net_return):
    return net_return - max(0, -net_return)


def owned_positions(orders, session, closed_trades=None):
    positions = {}
    feedback = []
    entries = {}
    entry_orders = {}
    fills = sorted((order for order in orders if Decimal(order["filled_qty"]) > 0),
                   key=lambda order: (order["filled_at"] or order["updated_at"], order["id"]))
    for order in fills:
        if not order["client_order_id"].startswith(ORDER_PREFIX):
            raise ValueError("Use a separate Alpaca Paper account with no other trading history.")
        symbol = order["symbol"]
        quantity = Decimal(order["filled_qty"])
        value = quantity * Decimal(order["filled_avg_price"])
        if order["side"] == "buy":
            if symbol not in entries:
                entry_day = date.fromisoformat(order["client_order_id"][len(ORDER_PREFIX):][:10])
                entries[symbol] = entry_day, Decimal(0), Decimal(0)
                entry_orders[symbol] = order["client_order_id"]
            entry_day, cost, proceeds = entries[symbol]
            entries[symbol] = entry_day, cost + value, proceeds
            positions[symbol] = positions.get(symbol, Decimal(0)) + quantity
        else:
            if symbol not in entries or quantity > positions[symbol]:
                raise ValueError("Order history does not reconcile. Check the Paper account.")
            entry_day, cost, proceeds = entries[symbol]
            positions[symbol] -= quantity
            proceeds += value
            if positions[symbol] == 0:
                del positions[symbol], entries[symbol]
                exit_stamp = datetime.fromisoformat((order["filled_at"] or order["updated_at"]).replace("Z", "+00:00"))
                entry_order = entry_orders.pop(symbol)
                if closed_trades is not None:
                    friction = Decimal(str(FRICTION))
                    net_return = proceeds * (1 - friction) / (cost * (1 + friction)) - 1
                    points = net_return * (100 if net_return >= 0 else 200)
                    closed_trades.append((order["id"], symbol, entry_order,
                                          exit_stamp.astimezone(EASTERN).isoformat(),
                                          str(net_return), str(points)))
                if exit_stamp.astimezone(EASTERN).date() < session:
                    feedback.append((symbol, entry_day, reward(float(proceeds / cost - 1))))
            else:
                entries[symbol] = entry_day, cost, proceeds
    return positions, feedback


def update_scorecard(account_id, closed_trades):
    if not closed_trades:
        return
    new_scores = []
    with sqlite3.connect(SCORECARD) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                account_id TEXT NOT NULL, exit_order TEXT NOT NULL,
                symbol TEXT NOT NULL, entry_order TEXT NOT NULL,
                closed_at TEXT NOT NULL, net_return TEXT NOT NULL,
                points TEXT NOT NULL,
                PRIMARY KEY (account_id, exit_order)
            )
        """)
        db.execute("BEGIN IMMEDIATE")
        total = sum((Decimal(row[0]) for row in db.execute(
            "SELECT points FROM scores WHERE account_id = ?", (account_id,)
        )), Decimal(0))
        for trade in closed_trades:
            inserted = db.execute(
                "INSERT OR IGNORE INTO scores VALUES (?, ?, ?, ?, ?, ?, ?)",
                (account_id, *trade),
            )
            if inserted.rowcount:
                total += Decimal(trade[-1])
                new_scores.append((trade, total))
    for (_, symbol, entry_order, closed_at, net_return, points), total in new_scores:
        print(f"Scorecard | {symbol} | closed {closed_at} | "
              f"net {Decimal(net_return):+.2%} | {Decimal(points):+.2f} points | "
              f"total {total:+.2f} | entry {entry_order}", flush=True)


@lru_cache(maxsize=1)
def completed_bars(session):
    bars = get_bars([*SYMBOLS, BENCHMARK], session - timedelta(days=1095), session)
    dates = sorted(bars[BENCHMARK])
    calendar = paper("/v2/calendar?" + urlencode({
        "start": session - timedelta(days=14), "end": session - timedelta(days=1),
    }))
    if len(dates) < 180 or not calendar or dates[-1] != date.fromisoformat(calendar[-1]["date"]):
        raise ValueError("Previous session data is missing or there is too little history.")
    return bars, dates


@lru_cache(maxsize=2)
def targets(session, feedback):
    bars, dates = completed_bars(session)
    training = []
    for i in range(WINDOW - 1, len(dates) - HORIZON - 1):
        for symbol in SYMBOLS:
            series = bars[symbol]
            x = features(series, dates, i)
            exit_day = dates[i + 1 + HORIZON]
            if x is None or dates[i + 1] not in series or exit_day not in series:
                continue
            buy = series[dates[i + 1]][0] * (1 + FRICTION)
            sell = series[exit_day][0] * (1 - FRICTION)
            training.append((x, reward(sell / buy - 1)))
    indices = {day: i for i, day in enumerate(dates)}
    learned = 0
    for symbol, entry_day, result in feedback:
        i = indices.get(entry_day, 0) - 1
        if i < WINDOW - 1 or symbol not in SYMBOLS:
            continue
        x = features(bars[symbol], dates, i)
        if x is not None:
            training.extend([(x, result)] * FEEDBACK_WEIGHT)
            learned += 1
    predict = fit(training)
    scores = {}
    incomplete = set()
    for symbol in SYMBOLS:
        x = features(bars[symbol], dates, len(dates) - 1)
        if x is None:
            incomplete.add(symbol)
        else:
            scores[symbol] = predict(x)
    ranked = sorted((s for s in scores if scores[s] > 0), key=lambda s: (-scores[s], s))
    return ranked, incomplete, dates[-1], scores, learned


def orders_for(positions, ranked, incomplete, cash):
    protected = positions.keys() & incomplete
    wanted = protected | set(ranked[:max(0, SLOTS - len(protected))])
    sells = [(symbol, "sell", {"qty": str(positions[symbol])})
             for symbol in sorted(positions.keys() - wanted)]
    if sells:
        return sells  # Wait for exits to fill before reserving cash for entries.
    orders = []
    slots = SLOTS - len(positions)
    for symbol in ranked:
        if slots <= 0:
            break
        if symbol not in wanted or symbol in positions:
            continue
        amount = min(ORDER_SIZE, cash).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        if amount < 1:
            break
        orders.append((symbol, "buy", {"notional": str(amount)}))
        cash -= amount
        slots -= 1
    return orders


def premarket_model(session):
    history = paper("/v2/orders?status=closed&limit=500&direction=asc")
    if len(history) >= 500:
        raise ValueError("Order history exceeds the safety limit.")
    _, feedback = owned_positions(history, session)
    _, _, signal_date, scores, _ = targets(session, tuple(feedback))
    return scores, signal_date


def run(submit):
    clock = paper("/v2/clock")
    session = date.fromisoformat(clock["timestamp"][:10])
    if submit and not clock["is_open"]:
        if os.getenv("PAPER_PREMARKET_PREVIEW") == "1":
            from stock_premarket import run_comparison
            try:
                run_comparison(clock, paper, premarket_model, completed_bars)
            except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
                print(f"PREMARKET | comparison unavailable ({type(exc).__name__}): {exc}", flush=True)
        print("Market closed. No orders submitted.", flush=True)
        return
    account = paper("/v2/account")
    if account["trading_blocked"] or account["account_blocked"]:
        raise ValueError("Alpaca has blocked this paper account from trading.")
    if paper("/v2/orders?status=open&limit=1"):
        print("An order is open. Waiting for it to resolve.", flush=True)
        return
    positions = {}
    for position in paper("/v2/positions"):
        symbol = position["symbol"]
        if symbol not in SYMBOLS or position["side"] != "long":
            raise ValueError("Use a dedicated Paper account without other positions.")
        positions[symbol] = Decimal(position["qty"])
    if len(positions) > SLOTS:
        raise ValueError("Use a dedicated Paper account with at most three positions.")
    history = paper("/v2/orders?status=closed&limit=500&direction=asc")
    if len(history) >= 500:
        raise ValueError("Order history exceeds the safety limit. Review the account before continuing.")
    closed_trades = [] if submit else None
    owned, feedback = owned_positions(history, session, closed_trades)
    if owned != positions:
        raise ValueError("Positions do not match this AI's fills. Use a separate empty Paper account.")
    if closed_trades:
        try:
            update_scorecard(account["id"], closed_trades)
        except sqlite3.Error as exc:
            print(f"Scorecard unavailable: {exc}", flush=True)
    ranked, incomplete, signal_date, scores, learned = targets(session, tuple(feedback))
    orders = orders_for(positions, ranked, incomplete, Decimal(account["cash"]))
    daily_change = Decimal(account["equity"]) - Decimal(account["last_equity"])
    if daily_change <= -DAILY_LOSS_LIMIT:
        print(f"Loss brake active (paper equity change ${daily_change:.2f}). New buys paused today.",
              flush=True)
        orders = [order for order in orders if order[1] == "sell"]
    print(f"AI learning | Prior closed paper trades used in training: {learned}", flush=True)
    print(f"{session} | {'PAPER ORDERS' if submit else 'PREVIEW'} | "
          f"Candidates: {', '.join(ranked[:SLOTS]) or 'none'}", flush=True)
    if not orders:
        print("No orders needed.", flush=True)
    for symbol, side, sizing in orders:
        size = next(iter(sizing.values()))
        print(f"{side.upper()} {symbol} {'$' + size if side == 'buy' else size + ' shares'}",
              flush=True)
        if not submit:
            continue
        client_id = f"{ORDER_PREFIX}{session}-{symbol}-{side}"
        try:
            paper("/v2/orders:by_client_order_id?" + urlencode({"client_order_id": client_id}))
        except HTTPError as exc:
            if exc.code != 404:
                raise
        else:
            print("Already submitted for this session.", flush=True)
            continue
        if not paper("/v2/clock")["is_open"]:
            print("Market closed. Stopping submissions.", flush=True)
            break
        rank = ranked.index(symbol) + 1 if symbol in ranked else None
        protected = len(positions.keys() & incomplete)
        slots = max(0, SLOTS - protected)
        estimate = scores[symbol]
        if side == "buy":
            reason = f"AI ranked it #{rank} with a positive risk-adjusted score ({estimate:+.2%})."
        elif rank is None:
            reason = f"AI estimated a nonpositive risk-adjusted score ({estimate:+.2%})."
        else:
            reason = f"AI ranked it #{rank}, outside the {slots} available ranking slots."
        record = {
            "session": str(session), "signal_date": str(signal_date), "symbol": symbol,
            "side": side, "rank": rank, "reason": reason, "client_order_id": client_id,
        }
        # Record the decision before sending the request so an interrupted submission
        # can still be reconciled with Alpaca by its client order ID.
        fd = os.open(DECISION_LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as log:
            log.write(json.dumps(record) + "\n")
            log.flush()
            os.fsync(log.fileno())
        order = paper("/v2/orders", {
            "symbol": symbol, "side": side, "type": "market", "time_in_force": "day",
            "client_order_id": client_id, **sizing,
        })
        print(f"Alpaca paper order status: {order['status']}", flush=True)


def main():
    warnings.simplefilter("error", GetPassWarning)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true", help="Send practice orders")
    parser.add_argument("--watch", action="store_true", help="Check every five minutes until stopped")
    args = parser.parse_args()
    if args.watch and not args.submit:
        parser.error("--watch requires --submit.")
    try:
        with (Path.home() / ".stock_paper_bot.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.exit(1, "Another copy of the paper bot is already running.\n")
            for name, label in (("APCA_API_KEY_ID", "Key ID"),
                                ("APCA_API_SECRET_KEY", "Secret Key")):
                if not os.getenv(name):
                    os.environ[name] = getpass(f"Paste Alpaca Paper {label}: ").strip()
                if not os.environ[name]:
                    raise ValueError(f"Missing Alpaca Paper {label}.")
            while True:
                run(args.submit)
                if not args.watch:
                    break
                time.sleep(300)
    except KeyboardInterrupt:
        print("\nStopped.")
    except HTTPError as exc:
        parser.exit(1, f"Alpaca returned HTTP {exc.code}. Check the Paper dashboard before restarting.\n")
    except (URLError, OSError, ValueError, KeyError, GetPassWarning) as exc:
        parser.exit(1, f"Paper bot stopped: {exc}\n")


if __name__ == "__main__":
    main()
