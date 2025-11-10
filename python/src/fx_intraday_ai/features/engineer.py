from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..data.schemas import PriceFrame, FeatureFrame
from ..utils.config import FeatureConfig


@dataclass(slots=True)
class FeatureEngineer:
    cfg: FeatureConfig

    def transform(
        self,
        price_frames: Dict[str, PriceFrame],
        macro_frame: Optional[pd.DataFrame] = None,
        sentiment_frames: Optional[Dict[str, pd.DataFrame]] = None,
        calendar_flags: Optional[pd.DataFrame] = None,
    ) -> Dict[str, FeatureFrame]:
        feature_frames: Dict[str, FeatureFrame] = {}
        for pair, frame in price_frames.items():
            sentiment = sentiment_frames.get(pair) if sentiment_frames else None
            data = self._build_features(
                frame.data.copy(),
                macro_frame=macro_frame,
                sentiment_frame=sentiment,
                calendar_flags=calendar_flags,
            )
            feature_frames[pair] = FeatureFrame(pair=pair, data=data.dropna())
        return feature_frames

    def _build_features(
        self,
        df: pd.DataFrame,
        macro_frame: Optional[pd.DataFrame] = None,
        sentiment_frame: Optional[pd.DataFrame] = None,
        calendar_flags: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)
        open_prices = df["open"] if "open" in df else df["close"]
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        spread = df.get("spread", pd.Series(0, index=df.index))

        # Returns & volatility
        feats["log_ret_1"] = np.log(close / close.shift(1))
        feats["log_ret_3"] = np.log(close / close.shift(3))
        feats["log_ret_12"] = np.log(close / close.shift(12))
        feats["log_ret_24"] = np.log(close / close.shift(24))
        feats["realized_vol_24"] = feats["log_ret_1"].rolling(24).std().fillna(0)

        # EMAs
        for window in self.cfg.ema_windows:
            feats[f"ema_{window}"] = close.ewm(span=window, adjust=False).mean()
            feats[f"ema_gap_{window}"] = close / feats[f"ema_{window}"] - 1.0

        # MACD
        fast_ema = close.ewm(span=12, adjust=False).mean()
        slow_ema = close.ewm(span=26, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        feats["macd_line"] = macd_line
        feats["macd_signal"] = signal_line
        feats["macd_hist"] = macd_line - signal_line

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

        # Stochastic oscillator
        stoch_window = 14
        lowest_low = low.rolling(stoch_window).min()
        highest_high = high.rolling(stoch_window).max()
        percent_k = (close - lowest_low) / (highest_high - lowest_low + 1e-9)
        feats["stoch_k"] = percent_k
        feats["stoch_d"] = percent_k.rolling(3).mean()

        # CCI
        typical_price = (high + low + close) / 3.0
        cci_window = 20
        tp_sma = typical_price.rolling(cci_window).mean()
        tp_dev = (typical_price - tp_sma).abs().rolling(cci_window).mean()
        feats["cci_20"] = (typical_price - tp_sma) / (0.015 * (tp_dev + 1e-9))

        # ADX & directional indicators
        adx_window = 14
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
        tr_ewm = tr.ewm(alpha=1 / adx_window, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / adx_window, adjust=False).mean() / (tr_ewm + 1e-9)
        minus_di = 100 * minus_dm.ewm(alpha=1 / adx_window, adjust=False).mean() / (tr_ewm + 1e-9)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) * 100
        feats["plus_di_14"] = plus_di
        feats["minus_di_14"] = minus_di
        feats["adx_14"] = dx.ewm(alpha=1 / adx_window, adjust=False).mean()

        # Normalized volatility & spread
        atr_ref = feats.get("atr_14", tr.rolling(14).mean())
        feats["atr_ratio_14"] = atr_ref / (close + 1e-9)
        feats["spread_pct"] = spread / (close + 1e-9)

        # Volume dynamics
        feats["volume_ret_3"] = volume.pct_change(3)
        feats["volume_ret_12"] = volume.pct_change(12)
        feats["price_volume_corr_48"] = feats["log_ret_1"].rolling(48).corr(volume.pct_change().rolling(48).mean())

        # Skewness & kurtosis of returns
        feats["ret_skew_48"] = feats["log_ret_1"].rolling(48).skew()
        feats["ret_kurt_48"] = feats["log_ret_1"].rolling(48).kurt()

        # Momentum and acceleration
        feats["momentum_24"] = close / close.shift(24) - 1.0
        feats["momentum_72"] = close / close.shift(72) - 1.0
        feats["acceleration_24"] = feats["momentum_24"].diff(24)

        # Heikin-Ashi derived signals
        ha_close = (open_prices + high + low + close) / 4.0
        ha_open = ha_close.copy()
        ha_open.iloc[0] = (close.iloc[0] + open_prices.iloc[0]) / 2
        for idx_pos in range(1, len(ha_open)):
            ha_open.iloc[idx_pos] = (ha_open.iloc[idx_pos - 1] + ha_close.iloc[idx_pos - 1]) / 2
        feats["ha_close"] = ha_close
        feats["ha_open"] = ha_open
        feats["ha_body"] = ha_close - ha_open

        # VWAP ratio (using cumulative intraday volume)
        typical_price = (high + low + close) / 3.0
        cum_volume = volume.cumsum()
        cum_vwap = (typical_price * volume).cumsum() / (cum_volume + 1e-9)
        feats["vwap_ratio"] = close / (cum_vwap + 1e-9) - 1.0

        # Rolling quantiles
        feats["close_p90_48"] = close.rolling(48).quantile(0.9)
        feats["close_p10_48"] = close.rolling(48).quantile(0.1)
        feats["close_spread_p9010"] = feats["close_p90_48"] - feats["close_p10_48"]

        self._inject_macro_features(feats, macro_frame)
        self._inject_sentiment(feats, sentiment_frame)
        self._inject_event_flags(feats, calendar_flags)

        return feats

    def _inject_macro_features(self, feats: pd.DataFrame, macro_frame: Optional[pd.DataFrame]) -> None:
        if macro_frame is None or macro_frame.empty:
            return
        aligned = macro_frame.reindex(feats.index, method="ffill").fillna(method="ffill")
        for col in aligned.columns:
            if col == "timestamp":
                continue
            base_name = f"macro_{col}"
            feats[base_name] = aligned[col]
            feats[f"{base_name}_ret_12"] = aligned[col].pct_change(12)

        if self.cfg.regime_features.get("enable_vix_proxy"):
            vix_col = next((col for col in aligned.columns if "vix" in col), None)
            if vix_col:
                vix_series = aligned[vix_col]
                feats["vix_zscore_96"] = (vix_series - vix_series.rolling(96).mean()) / (
                    vix_series.rolling(96).std() + 1e-9
                )

        dxy_col = next((col for col in aligned.columns if "dxy" in col), None)
        if dxy_col:
            dxy_ret = aligned[dxy_col].pct_change()
            feats["dxy_ret_12"] = dxy_ret.rolling(12).mean()
            feats["dxy_corr_96"] = feats["log_ret_1"].rolling(96).corr(dxy_ret)

        gold_col = next((col for col in aligned.columns if "xau" in col or "gold" in col), None)
        if gold_col:
            gold_ret = aligned[gold_col].pct_change()
            feats["gold_ret_12"] = gold_ret.rolling(12).mean()
            feats["gold_corr_96"] = feats["log_ret_1"].rolling(96).corr(gold_ret)

    def _inject_sentiment(self, feats: pd.DataFrame, sentiment_frame: Optional[pd.DataFrame]) -> None:
        if sentiment_frame is None or sentiment_frame.empty:
            return
        aligned = sentiment_frame.reindex(feats.index, method="ffill").fillna(method="ffill")
        score = aligned["sentiment_score"]
        feats["sentiment_score"] = score
        feats["sentiment_change_12"] = score.diff(12)
        feats["sentiment_zscore_96"] = (score - score.rolling(96).mean()) / (score.rolling(96).std() + 1e-9)

    def _inject_event_flags(self, feats: pd.DataFrame, calendar_flags: Optional[pd.DataFrame]) -> None:
        if (
            calendar_flags is None
            or calendar_flags.empty
            or not self.cfg.regime_features.get("enable_macro_event_flags", False)
        ):
            return
        daily_index = feats.index.normalize()
        aligned = calendar_flags.reindex(daily_index).fillna(0.0)
        aligned.index = feats.index
        for col in aligned.columns:
            feats[f"event_{col}"] = aligned[col]
