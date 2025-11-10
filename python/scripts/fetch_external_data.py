#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from io import StringIO

import pandas as pd
import requests

from fx_intraday_ai.utils.config import load_config

STOOQ_BASE = "https://stooq.com/q/d/l/"
CALENDAR_BASES = [
    "https://cdn-nfs.faireconomy.media/",
    "https://nfs.faireconomy.media/",
]
DEFAULT_CALENDAR_FEEDS = [
    "ff_calendar_thisweek.json",
    "ff_calendar_nextweek.json",
    "ff_calendar_lastweek.json",
]
FMP_CALENDAR_ENDPOINT = "https://financialmodelingprep.com/api/v3/economic_calendar"
SENTIMENT_URL = "https://api.alternative.me/fng/?limit=0&format=json"


@dataclass(frozen=True)
class MacroSeriesSpec:
    symbols: List[str]
    column: str


MACRO_SERIES: List[MacroSeriesSpec] = [
    MacroSeriesSpec(symbols=["usdidx", "usdidxus", "dxyusd"], column="usd_index_close"),
    MacroSeriesSpec(symbols=["xauusd", "xauusd"], column="gold_xauusd_close"),
    MacroSeriesSpec(symbols=["vix"], column="vix_close"),
]


def _to_timestamp(value: Optional[str], *, tz: str = "UTC") -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.Timestamp(value, tz=tz)
    return ts


def _fetch_stooq_series(symbol: str) -> pd.DataFrame:
    url = f"{STOOQ_BASE}?s={symbol}&i=d"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    if df.empty:
        raise ValueError(f"Stooq returned no data for {symbol}")
    df = df.rename(columns=str.lower)
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"Unexpected schema for {symbol}: {df.columns}")
    df["timestamp"] = pd.to_datetime(df["date"], utc=True)
    return df[["timestamp", "close"]].sort_values("timestamp")


def download_macro_series(start: Optional[pd.Timestamp], end: Optional[pd.Timestamp], output: Path) -> None:
    frames: List[pd.DataFrame] = []
    for spec in MACRO_SERIES:
        df = None
        last_error: Optional[Exception] = None
        for symbol in spec.symbols:
            try:
                df = _fetch_stooq_series(symbol)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if df is None:
            print(f"[macro] Warning: unable to fetch {spec.column}: {last_error}", file=sys.stderr)
            continue
        if start is not None:
            df = df[df["timestamp"] >= start]
        if end is not None:
            df = df[df["timestamp"] <= end]
        df = df.rename(columns={"close": spec.column})
        frames.append(df.set_index("timestamp"))
    if not frames:
        raise ValueError("No macro series downloaded. Check symbols or network connectivity.")
    combined = pd.concat(frames, axis=1).sort_index().ffill()
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output)
    print(f"[macro] Saved {len(combined):,} rows to {output}")


def download_sentiment_series(pairs: Iterable[str], output: Path) -> None:
    resp = requests.get(SENTIMENT_URL, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data", [])
    if not data:
        raise ValueError("Alternative.me returned no sentiment entries.")
    records = []
    for entry in data:
        ts_raw = entry.get("timestamp")
        try:
            timestamp = pd.to_datetime(int(ts_raw), unit="s", utc=True)
        except Exception:  # noqa: BLE001
            timestamp = pd.to_datetime(ts_raw, utc=True)
        score = float(entry.get("value", 0.0))
        for pair in pairs:
            records.append(
                {
                    "timestamp": timestamp,
                    "pair": pair,
                    "sentiment_score": score,
                }
            )
    df = pd.DataFrame(records).sort_values("timestamp")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"[sentiment] Saved {len(df):,} rows to {output}")


def _fetch_calendar_feed(slug: str) -> List[Dict[str, object]]:
    last_exc: Optional[Exception] = None
    for base in CALENDAR_BASES:
        url = f"{base}{slug}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            continue
    if last_exc:
        raise last_exc
    return []


def _parse_calendar_entry(entry: Dict[str, object]) -> Optional[Dict[str, object]]:
    timestamp = entry.get("timestamp") or entry.get("date")
    if timestamp is None:
        return None
    try:
        if isinstance(timestamp, (int, float)):
            ts = pd.to_datetime(int(timestamp), unit="s", utc=True)
        else:
            ts = pd.to_datetime(str(timestamp), utc=True)
    except Exception:  # noqa: BLE001
        return None
    importance_raw = str(entry.get("impact") or entry.get("importance") or "").lower()
    importance_map = {"low": 1, "medium": 2, "med": 2, "high": 3}
    importance = importance_map.get(importance_raw, 0)
    record = {
        "timestamp": ts,
        "event": entry.get("title") or entry.get("event") or "",
        "region": entry.get("country") or entry.get("region") or entry.get("currency") or "",
        "importance": importance,
        "actual": entry.get("actual"),
        "forecast": entry.get("forecast"),
        "previous": entry.get("previous"),
    }
    return record


def download_calendar(
    feeds: List[str],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    output: Path,
    provider: str = "forexfactory",
    fmp_api_key: str = "demo",
) -> None:
    if provider == "fmp":
        download_calendar_fmp(start, end, output, fmp_api_key)
        return
    df = download_calendar_forexfactory(feeds, start, end)
    if df.empty:
        print("[calendar] Forexfactory returned no rows; falling back to FMP feed.", file=sys.stderr)
        try:
            download_calendar_fmp(start, end, output, fmp_api_key)
        except requests.HTTPError as exc:
            print(
                f"[calendar] FMP request failed ({exc}); provide --fmp-apikey or use --skip-calendar.",
                file=sys.stderr,
            )
            _write_calendar_frame(pd.DataFrame(columns=["timestamp", "event", "region", "importance"]), output, source="fallback-empty")
        return
    _write_calendar_frame(df, output, source="Forexfactory")


def download_calendar_forexfactory(
    feeds: List[str],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    entries: List[Dict[str, object]] = []
    for slug in feeds:
        try:
            payload = _fetch_calendar_feed(slug)
        except requests.HTTPError as exc:
            print(f"[calendar] Failed to fetch {slug}: {exc}", file=sys.stderr)
            continue
        if isinstance(payload, dict) and "data" in payload:
            payload = payload["data"]
        if not isinstance(payload, list):
            continue
        for item in payload:
            parsed = _parse_calendar_entry(item)
            if parsed:
                entries.append(parsed)
    if not entries:
        print(
            "[calendar] Warning: no entries downloaded from Forexfactory feeds.",
            file=sys.stderr,
        )
        return pd.DataFrame(columns=["timestamp", "event", "region", "importance"])
    df = pd.DataFrame(entries).dropna(subset=["timestamp"]).sort_values("timestamp")
    if start is not None:
        df = df[df["timestamp"] >= start]
    if end is not None:
        df = df[df["timestamp"] <= end]
    df = df.drop_duplicates(subset=["timestamp", "event", "region"])
    return df


def download_calendar_fmp(
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    output: Path,
    api_key: str = "demo",
) -> None:
    params = {
        "from": (start.date().isoformat() if start is not None else pd.Timestamp.utcnow().date().isoformat()),
        "to": (end.date().isoformat() if end is not None else pd.Timestamp.utcnow().date().isoformat()),
        "apikey": api_key or "demo",
    }
    resp = requests.get(FMP_CALENDAR_ENDPOINT, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected FMP payload: {payload}")
    records = []
    for item in payload:
        ts_value = item.get("date") or item.get("timestamp")
        if not ts_value:
            continue
        timestamp = pd.to_datetime(ts_value, utc=True)
        if start is not None and timestamp < start:
            continue
        if end is not None and timestamp > end:
            continue
        records.append(
            {
                "timestamp": timestamp,
                "event": item.get("event"),
                "region": item.get("country") or item.get("region"),
                "importance": item.get("importance", 0),
                "actual": item.get("actual"),
                "previous": item.get("previous"),
                "forecast": item.get("estimate"),
            }
        )
    df = pd.DataFrame(records)
    _write_calendar_frame(df, output, source="FinancialModelingPrep")


def _write_calendar_frame(df: pd.DataFrame, output: Path, source: str = "Forexfactory") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"[calendar] Saved {len(df):,} rows to {output} (source: {source})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download macro/sentiment/calendar data for FX Intraday AI pipeline.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Experiment config path.")
    parser.add_argument("--start", type=str, default=None, help="Inclusive UTC start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="Inclusive UTC end date (YYYY-MM-DD).")
    parser.add_argument("--macro-path", type=str, default=None, help="Override macro parquet destination.")
    parser.add_argument("--sentiment-path", type=str, default=None, help="Override sentiment parquet destination.")
    parser.add_argument("--calendar-path", type=str, default=None, help="Override calendar CSV destination.")
    parser.add_argument(
        "--calendar-feeds",
        type=str,
        nargs="*",
        default=DEFAULT_CALENDAR_FEEDS,
        help="List of calendar JSON feeds to aggregate (relative to nfs.faireconomy.media).",
    )
    parser.add_argument(
        "--calendar-provider",
        choices=["forexfactory", "fmp"],
        default="forexfactory",
        help="Source for calendar data.",
    )
    parser.add_argument(
        "--fmp-apikey",
        type=str,
        default="demo",
        help="FinancialModelingPrep API key (used when provider=fmp or as fallback).",
    )
    parser.add_argument("--skip-macro", action="store_true", help="Skip macro download.")
    parser.add_argument("--skip-sentiment", action="store_true", help="Skip sentiment download.")
    parser.add_argument("--skip-calendar", action="store_true", help="Skip calendar download.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    start = _to_timestamp(args.start or cfg.data.start_date)
    end = _to_timestamp(args.end or cfg.data.end_date)

    if not args.skip_macro and cfg.data.macro_path:
        macro_path = Path(args.macro_path) if args.macro_path else cfg.data.macro_path
        download_macro_series(start, end, macro_path)
    else:
        print("[macro] skipped")

    if not args.skip_sentiment and cfg.data.sentiment_path:
        sentiment_path = Path(args.sentiment_path) if args.sentiment_path else cfg.data.sentiment_path
        download_sentiment_series(cfg.pairs, sentiment_path)
    else:
        print("[sentiment] skipped")

    if not args.skip_calendar and cfg.data.calendar_path:
        calendar_path = Path(args.calendar_path) if args.calendar_path else cfg.data.calendar_path
        feeds = args.calendar_feeds or DEFAULT_CALENDAR_FEEDS
        download_calendar(
            feeds,
            start,
            end,
            calendar_path,
            provider=args.calendar_provider,
            fmp_api_key=args.fmp_apikey or "demo",
        )
    else:
        print("[calendar] skipped")


if __name__ == "__main__":
    main()
