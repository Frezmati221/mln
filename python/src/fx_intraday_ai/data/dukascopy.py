from __future__ import annotations

import logging
import lzma
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_HOUR = pd.Timedelta(hours=1)
_STRUCT_DTYPE = np.dtype(
    [
        ("millis", ">i4"),
        ("ask", ">i4"),
        ("bid", ">i4"),
        ("ask_volume", ">f4"),
        ("bid_volume", ">f4"),
    ]
)


ProgressCallback = Callable[[str, int, int], None]  # pair, completed, total


def _coerce_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


@dataclass(slots=True)
class DukascopyDownloader:
    """Utility for pulling FX tick data from Dukascopy and converting to OHLC bars."""

    timeframe_minutes: int = 5
    price_path: Optional[Path] = None
    session: requests.Session = field(default_factory=requests.Session)

    BASE_URL: str = "http://datafeed.dukascopy.com/datafeed"

    def download(
        self,
        pairs: Iterable[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        start_ts = _coerce_timestamp(start)
        end_ts = _coerce_timestamp(end)
        if end_ts <= start_ts:
            raise ValueError("End timestamp must be after start timestamp.")

        pair_frames: List[pd.DataFrame] = []
        for pair in pairs:
            logger.info("Downloading %s from %s to %s", pair, start_ts, end_ts)
            pair_df = self._download_pair(pair.upper(), start_ts, end_ts, progress_cb)
            pair_frames.append(pair_df)

        combined = pd.concat(pair_frames, ignore_index=True)
        if self.price_path:
            self.price_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(self.price_path, index=False)
            logger.info("Saved %s bars to %s", len(combined), self.price_path)
        return combined

    def _download_pair(
        self,
        pair: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        progress_cb: Optional[ProgressCallback],
    ) -> pd.DataFrame:
        current = start.floor("H")
        end_hour = end.ceil("H")
        total_hours = int(((end_hour - current) / _HOUR)) + 1
        hour_frames: List[pd.DataFrame] = []
        completed = 0

        while current <= end_hour:
            frame = self._download_hour(pair, current)
            if frame is not None:
                hour_frames.append(frame)
            current += _HOUR
            completed += 1
            if progress_cb:
                progress_cb(pair, completed, total_hours)

        if not hour_frames:
            raise RuntimeError(f"No data downloaded for {pair}. Check the date window.")

        raw = (
            pd.concat(hour_frames)
            .set_index("timestamp")
            .sort_index()
        )
        freq = f"{self.timeframe_minutes}T"
        ohlc = raw["mid"].resample(freq, label="right", closed="right").ohlc()
        volume = raw["volume"].resample(freq, label="right", closed="right").sum()
        spread = raw["spread"].resample(freq, label="right", closed="right").mean()
        merged = pd.concat([ohlc, volume, spread], axis=1)
        merged.columns = ["open", "high", "low", "close", "volume", "spread"]
        merged = merged.dropna(subset=["open", "high", "low", "close"])
        merged = merged.loc[(merged.index >= start) & (merged.index <= end)]
        result = merged.reset_index()
        result["pair"] = pair
        return result[["timestamp", "pair", "open", "high", "low", "close", "volume", "spread"]]

    def _download_hour(self, pair: str, ts: pd.Timestamp) -> Optional[pd.DataFrame]:
        url = self._hour_url(pair, ts)
        try:
            response = self.session.get(url, timeout=15)
        except requests.RequestException as exc:
            logger.warning("Network error for %s: %s", url, exc)
            return None

        if response.status_code == 404 or not response.content:
            return None
        response.raise_for_status()

        try:
            payload = lzma.decompress(response.content)
        except lzma.LZMAError as exc:
            logger.error("Failed to decompress %s: %s", url, exc)
            return None
        if not payload:
            return None

        records = np.frombuffer(payload, dtype=_STRUCT_DTYPE)
        if records.size == 0:
            return None

        base = ts.tz_convert("UTC")
        timestamps = base + pd.to_timedelta(records["millis"], unit="ms")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "ask": records["ask"] / 100000.0,
                "bid": records["bid"] / 100000.0,
                "ask_volume": records["ask_volume"].astype(float),
                "bid_volume": records["bid_volume"].astype(float),
            }
        )
        df["mid"] = (df["ask"] + df["bid"]) / 2.0
        df["spread"] = df["ask"] - df["bid"]
        df["volume"] = df["ask_volume"] + df["bid_volume"]
        return df[["timestamp", "mid", "volume", "spread"]]

    def _hour_url(self, pair: str, ts: pd.Timestamp) -> str:
        return (
            f"{self.BASE_URL}/{pair}/"
            f"{ts.strftime('%Y')}/"
            f"{ts.strftime('%m')}/"
            f"{ts.strftime('%d')}/"
            f"{ts.strftime('%H')}h_ticks.bi5"
        )
