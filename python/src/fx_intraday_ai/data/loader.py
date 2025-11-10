from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .schemas import PriceFrame
from ..utils.config import ExperimentConfig

REQUIRED_COLUMNS = ["timestamp", "pair", "open", "high", "low", "close", "volume"]


def _read_market_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format for {path}")


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input price data missing columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "spread" not in df.columns:
        df["spread"] = 0.0
    return df


def _resample_to_timeframe(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    rule = f"{timeframe_minutes}T"
    grouped = (
        df.set_index("timestamp")
        .groupby("pair")
        .resample(rule, label="right", closed="right")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            spread=("spread", "mean"),
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return grouped


def _filter_window(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    return df


def load_price_frames(cfg: ExperimentConfig, refresh_cache: bool = False) -> Dict[str, PriceFrame]:
    """Load and cache 5-minute bars for each configured pair."""
    cache_file = cfg.data.cache_dir / f"prices_{cfg.timeframe_minutes}m.parquet"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and not refresh_cache:
        df = pd.read_parquet(cache_file)
        if "pair" not in df.columns:
            # Cached file was created with an older schema; rebuild.
            refresh_cache = True
    if not cache_file.exists() or refresh_cache:
        raw = _read_market_file(cfg.data.price_path)
        raw = _ensure_schema(raw)
        resampled = _resample_to_timeframe(raw, cfg.timeframe_minutes)
        resampled.to_parquet(cache_file, index=False)
        df = resampled

    df = _filter_window(df, cfg.data.start_date, cfg.data.end_date)

    frames: Dict[str, PriceFrame] = {}
    for pair in cfg.pairs:
        pair_df = (
            df[df["pair"] == pair]
            .drop(columns=["pair"])
            .set_index("timestamp")
            .sort_index()
        )
        if pair_df.empty:
            raise ValueError(f"No price data available for pair {pair}")
        pair_df.index = pair_df.index.tz_convert("UTC")
        frames[pair] = PriceFrame(pair=pair, data=pair_df)
    return frames
