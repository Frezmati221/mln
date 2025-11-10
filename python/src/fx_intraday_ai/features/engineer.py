from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from ..data.schemas import PriceFrame, FeatureFrame
from ..utils.config import FeatureConfig


@dataclass(slots=True)
class FeatureEngineer:
    cfg: FeatureConfig

    def transform(self, price_frames: Dict[str, PriceFrame]) -> Dict[str, FeatureFrame]:
        feature_frames: Dict[str, FeatureFrame] = {}
        for pair, frame in price_frames.items():
            data = self._build_features(frame.data.copy())
            feature_frames[pair] = FeatureFrame(pair=pair, data=data.dropna())
        return feature_frames

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        spread = df.get("spread", pd.Series(0, index=df.index))

        # Returns & volatility
        feats["log_ret_1"] = np.log(close / close.shift(1))
        feats["log_ret_3"] = np.log(close / close.shift(3))
        feats["log_ret_12"] = np.log(close / close.shift(12))
        feats["realized_vol_24"] = feats["log_ret_1"].rolling(24).std().fillna(0)

        # EMAs
        for window in self.cfg.ema_windows:
            feats[f"ema_{window}"] = close.ewm(span=window, adjust=False).mean()
            feats[f"ema_gap_{window}"] = close / feats[f"ema_{window}"] - 1.0

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        for window in self.cfg.rsi_windows:
            avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            feats[f"rsi_{window}"] = 100 - (100 / (1 + rs))

        # ATR
        tr = pd.concat(
            [
                (high - low),
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        for window in self.cfg.atr_windows:
            feats[f"atr_{window}"] = tr.rolling(window).mean()

        # Bollinger
        bb_window = int(self.cfg.bollinger.get("window", 20))
        bb_std = float(self.cfg.bollinger.get("num_std", 2.0))
        ma = close.rolling(bb_window).mean()
        std = close.rolling(bb_window).std()
        feats["bollinger_upper"] = ma + bb_std * std
        feats["bollinger_lower"] = ma - bb_std * std
        feats["bollinger_width"] = (feats["bollinger_upper"] - feats["bollinger_lower"]) / close

        # Volume & spread features
        feats["volume_zscore_48"] = (volume - volume.rolling(48).mean()) / (volume.rolling(48).std() + 1e-9)
        feats["spread_sma_12"] = spread.rolling(12).mean()

        # Session encodings
        idx = df.index.tz_convert("UTC")
        feats["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
        feats["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
        feats["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

        # Lag features for directionality
        feats["close_pct_rank_288"] = close.rolling(288).rank(pct=True)
        feats["range_pct_12"] = (close - low.rolling(12).min()) / (high.rolling(12).max() - low.rolling(12).min() + 1e-9)

        return feats
