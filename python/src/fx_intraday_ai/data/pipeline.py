from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .loader import load_price_frames
from ..features.engineer import FeatureEngineer
from ..labels.intraday import IntradayLabeler
from ..utils.config import ExperimentConfig
from .schemas import PriceFrame, FeatureFrame


@dataclass(slots=True)
class DataPipeline:
    cfg: ExperimentConfig
    _feature_engineer: FeatureEngineer = field(init=False, repr=False)
    _labeler: IntradayLabeler = field(init=False, repr=False)
    _macro_cache: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _sentiment_cache: Optional[Dict[str, pd.DataFrame]] = field(default=None, init=False, repr=False)
    _calendar_cache: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._feature_engineer = FeatureEngineer(self.cfg.features)
        self._labeler = IntradayLabeler(self.cfg)

    def run(self, refresh_cache: bool = False) -> Dict[str, pd.DataFrame]:
        price_frames = load_price_frames(self.cfg, refresh_cache=refresh_cache)
        feature_frames = self.build_feature_frames(price_frames)
        label_frames = self._labeler.build(price_frames)

        merged: Dict[str, pd.DataFrame] = {}
        for pair, feature_frame in feature_frames.items():
            if pair not in label_frames:
                raise KeyError(f"Missing labels for {pair}")
            df = feature_frame.data.join(label_frames[pair].data, how="inner")
            label_cols = [
                "direction",
                "tp_pips",
                "sl_pips",
                "holding_minutes",
                "edge_pips",
                "volatility_target",
                "regime_target",
            ]
            drop_cols = [col for col in label_cols if col in df.columns]
            if drop_cols:
                df.dropna(subset=drop_cols, inplace=True)
            df.ffill(inplace=True)
            df.bfill(inplace=True)
            df["volatility_target"] = (df["realized_vol_24"] > df["realized_vol_24"].median()).astype(float)
            df["regime_target"] = df.index.tz_convert("UTC").hour.map(lambda h: 1.0 if 12 <= h < 21 else 0.0)
            merged[pair] = df
        return merged

    def build_feature_frames(self, price_frames: Dict[str, PriceFrame]) -> Dict[str, FeatureFrame]:
        macro_frame = self._load_macro_frame()
        sentiment_frames = self._load_sentiment_frames()
        calendar_flags = self._load_calendar_flags()
        return self._feature_engineer.transform(
            price_frames,
            macro_frame=macro_frame,
            sentiment_frames=sentiment_frames,
            calendar_flags=calendar_flags,
        )

    def load_prices(self, refresh_cache: bool = False) -> Dict[str, PriceFrame]:
        return load_price_frames(self.cfg, refresh_cache=refresh_cache)

    def _load_macro_frame(self) -> Optional[pd.DataFrame]:
        if self._macro_cache is not None:
            return self._macro_cache
        path = self.cfg.data.macro_path
        if path is None or not path.exists():
            return None
        df = self._read_external_table(path)
        if "timestamp" not in df.columns:
            if df.index.name is not None:
                df = df.reset_index().rename(columns={df.index.name: "timestamp"})
            else:
                raise ValueError(f"Macro file {path} missing 'timestamp' column.")
        macro = df.copy()
        macro["timestamp"] = pd.to_datetime(macro["timestamp"], utc=True)
        macro = macro.set_index("timestamp").sort_index()
        macro.columns = [str(col).lower() for col in macro.columns]
        macro = macro.ffill()
        self._macro_cache = macro
        return macro

    def _load_sentiment_frames(self) -> Dict[str, pd.DataFrame]:
        if self._sentiment_cache is not None:
            return self._sentiment_cache
        path = self.cfg.data.sentiment_path
        if path is None or not path.exists():
            self._sentiment_cache = {}
            return {}
        df = self._read_external_table(path)
        required = {"timestamp", "pair"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Sentiment file {path} missing columns: {missing}")
        score_col = next(
            (col for col in df.columns if col.lower() in {"score", "sentiment", "sentiment_score", "value"}),
            None,
        )
        if score_col is None:
            raise ValueError(f"Sentiment file {path} missing score column.")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        frames: Dict[str, pd.DataFrame] = {}
        for pair, group in df.groupby("pair"):
            pair_df = (
                group.sort_values("timestamp")
                .set_index("timestamp")[[score_col]]
                .rename(columns={score_col: "sentiment_score"})
            )
            frames[str(pair)] = pair_df
        self._sentiment_cache = frames
        return frames

    def _load_calendar_flags(self) -> Optional[pd.DataFrame]:
        if self._calendar_cache is not None:
            return self._calendar_cache
        path = self.cfg.data.calendar_path
        if path is None or not path.exists():
            return None
        df = self._read_external_table(path)
        if "timestamp" not in df.columns:
            raise ValueError(f"Calendar file {path} missing 'timestamp' column.")
        calendar = df.copy()
        calendar["timestamp"] = pd.to_datetime(calendar["timestamp"], utc=True)
        calendar["event"] = calendar["event"].astype(str) if "event" in calendar.columns else ""
        calendar["region"] = calendar["region"].astype(str) if "region" in calendar.columns else ""
        importance_series = calendar["importance"] if "importance" in calendar.columns else 0.0
        calendar["importance"] = pd.to_numeric(importance_series, errors="coerce").fillna(0.0)
        calendar["event_day"] = calendar["timestamp"].dt.normalize()
        calendar["is_fed_event"] = (
            calendar["event"].str.contains("fed|fomc", case=False, na=False)
            | calendar["region"].str.contains("us", case=False, na=False)
        ).astype(float)
        calendar["is_ecb_event"] = (
            calendar["event"].str.contains("ecb", case=False, na=False)
            | calendar["region"].str.contains("eu", case=False, na=False)
        ).astype(float)
        grouped = (
            calendar.groupby("event_day")
            .agg(
                fed_event=("is_fed_event", "max"),
                ecb_event=("is_ecb_event", "max"),
                high_impact=("importance", lambda x: float((x >= 3).any())),
                macro_event_count=("timestamp", "count"),
            )
            .astype(float)
        )
        grouped.index.name = "event_day"
        self._calendar_cache = grouped
        return grouped

    @staticmethod
    def _read_external_table(path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix in {".csv", ".txt"}:
            return pd.read_csv(path)
        raise ValueError(f"Unsupported external data format: {path.suffix}")
