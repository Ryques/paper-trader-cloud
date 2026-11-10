"""One pre-open comparison per session. All broker requests are reads."""
from datetime import datetime, time, timedelta
import json
import os
from pathlib import Path
import sqlite3
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from stock_scanner import FRICTION, SLOTS

DB = Path.home() / "premarket_comparisons.sqlite3"
EASTERN = ZoneInfo("America/New_York")
DELAY = timedelta(minutes=16)
MIN_VOLUME = 10_000
MAX_BONUS = 0.0025


def data_pages(path, params, field):
    headers = {
        "APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"],
        "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"],
    }
    params = dict(params)
    for _ in range(10):
        req = Request("https://data.alpaca.markets" + path + "?" + urlencode(params), headers=headers)
        with urlopen(req, timeout=10) as response:
            page = json.load(response)
        yield page[field]
        token = page.get("next_page_token")
        if not token:
            return
        params["page_token"] = token
    raise ValueError("Market data pagination exceeded the comparison limit.")


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def bars(symbols, start, end):
    result = {}
    for page in data_pages("/v2/stocks/bars", {
        "symbols": ",".join(symbols), "timeframe": "5Min", "feed": "sip",
        "adjustment": "split", "start": start.isoformat(), "end": end.isoformat(),
        "limit": 10000,
    }, "bars"):
        for symbol, items in page.items():
            result.setdefault(symbol, []).extend(
                row for row in items if start <= timestamp(row["t"]) < end
            )
    return result


def news(symbols, start, end):
    result = {}
    for page in data_pages("/v1beta1/news", {
        "symbols": ",".join(symbols), "start": start.isoformat(), "end": end.isoformat(),
        "limit": 50, "sort": "desc",
    }, "news"):
        for item in page:
            link = urlsplit(item.get("url") or "")
            published = timestamp(item["created_at"])
            if (link.scheme != "https" or not link.hostname or link.username or link.password
                    or not start <= published <= end or not item.get("headline")):
                continue
            for symbol in set(item["symbols"]).intersection(symbols):
                result.setdefault(symbol, {
                    "headline": " ".join(item["headline"].split()),
                    "url": item["url"], "published_at": published.isoformat(),
                })
    return result


def compare(scores, prior, pre):
    adjusted = scores.copy()
    evidence = {}
    for symbol in scores.keys() & prior.keys() & pre.keys():
        if not prior[symbol] or not pre[symbol]:
            continue
        close = max(prior[symbol], key=lambda row: row["t"])["c"]
        last = max(pre[symbol], key=lambda row: row["t"])["c"]
        if close <= 0 or last <= 0:
            continue
        volume = sum(row["v"] for row in pre[symbol])
        gap = last / close - 1
        bonus = max(-MAX_BONUS, min(MAX_BONUS, gap * 0.1)) if volume >= MIN_VOLUME else 0
        adjusted[symbol] += bonus
        evidence[symbol] = {"gap": gap, "volume": volume, "adjustment": bonus}
    return adjusted, evidence


def picks(scores):
    return sorted((s for s in scores if scores[s] > 0), key=lambda s: (-scores[s], s))[:SLOTS]


def evaluate(db, bars_by_symbol, dates):
    for session, payload in db.execute("SELECT session, snapshot FROM comparisons WHERE outcome IS NULL"):
        record = json.loads(payload)
        signal = datetime.fromisoformat(session).date()
        if signal not in dates or len(dates) - dates.index(signal) < 5:
            continue
        exit_day = dates[dates.index(signal) + 4]
        outcomes = {}
        for name in ("baseline", "premarket"):
            returns = []
            for symbol in record[name]:
                series = bars_by_symbol.get(symbol, {})
                if signal not in series or exit_day not in series:
                    break
                buy = series[signal][0] * (1 + FRICTION)
                sell = series[exit_day][1] * (1 - FRICTION)
                returns.append(sell / buy - 1)
            else:
                outcomes[name] = sum(returns) / SLOTS
        if len(outcomes) == 2:
            db.execute("UPDATE comparisons SET outcome = ? WHERE session = ?",
                       (json.dumps(outcomes), session))
            print(f"PREMARKET OUTCOME | {session} | baseline {outcomes['baseline']:+.2%} "
                  f"| premarket {outcomes['premarket']:+.2%}", flush=True)


def run_comparison(clock, paper, get_model, get_daily):
    now = timestamp(clock["timestamp"]).astimezone(EASTERN)
    market_open = timestamp(clock["next_open"]).astimezone(EASTERN)
    if now.date() != market_open.date() or not timedelta(0) < market_open - now <= timedelta(minutes=20):
        return
    session = now.date()
    with sqlite3.connect(DB) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS comparisons
                      (session TEXT PRIMARY KEY, snapshot TEXT NOT NULL, outcome TEXT)""")
        if db.execute("SELECT 1 FROM comparisons WHERE session = ?", (str(session),)).fetchone():
            return
        scores, signal_date = get_model(session)
        daily, dates = get_daily(session)
        evaluate(db, daily, dates)
        calendar = paper("/v2/calendar?" + urlencode({"start": signal_date, "end": signal_date}))
        if not calendar:
            raise ValueError("Previous session calendar is unavailable.")
        prior_close = datetime.combine(signal_date, time.fromisoformat(calendar[0]["close"]), now.tzinfo)
        cutoff = (now - DELAY).replace(second=0, microsecond=0)
        cutoff -= timedelta(minutes=cutoff.minute % 5)
        symbols = sorted(scores)
        if not symbols:
            raise ValueError("The baseline model produced no scores.")
        prior = bars(symbols, prior_close - timedelta(minutes=15), prior_close)
        pre = bars(symbols, datetime.combine(session, time(4), now.tzinfo), cutoff)
        adjusted, evidence = compare(scores, prior, pre)
        if not evidence:
            raise ValueError("No comparable delayed premarket bars were available.")
        try:
            headlines = news(symbols, prior_close, cutoff)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"PREMARKET | headline feed unavailable ({type(exc).__name__})", flush=True)
            headlines = {}
        record = {"generated_at": now.isoformat(), "data_cutoff": cutoff.isoformat(),
                  "baseline": picks(scores), "premarket": picks(adjusted),
                  "scores": scores, "adjusted_scores": adjusted,
                  "evidence": evidence, "headlines": headlines}
        inserted = db.execute("INSERT OR IGNORE INTO comparisons VALUES (?, ?, NULL)",
                              (str(session), json.dumps(record)))
    if inserted.rowcount:
        print(f"PREMARKET PREVIEW | {session} | delayed data through {cutoff.isoformat()} | "
              f"baseline: {', '.join(record['baseline']) or 'Cash'} | "
              f"premarket: {', '.join(record['premarket']) or 'Cash'} | "
              f"{len(evidence)} symbols with bars | {len(headlines)} symbols with headlines", flush=True)
