from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.schemas import PriceFrame, LabelFrame
from ..utils.config import ExperimentConfig


def _pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def _session_close_timestamp(ts: pd.Timestamp, session_close: str) -> pd.Timestamp:
    hour, minute = [int(x) for x in session_close.split(":")]
    close = ts.normalize() + pd.Timedelta(hours=hour, minutes=minute)
    if ts >= close:
        close += pd.Timedelta(days=1)
    return close


@dataclass(slots=True)
class IntradayLabeler:
    cfg: ExperimentConfig

    def build(self, price_frames: Dict[str, PriceFrame]) -> Dict[str, LabelFrame]:
        label_frames: Dict[str, LabelFrame] = {}
        for pair, frame in price_frames.items():
            label_frames[pair] = LabelFrame(pair=pair, data=self._label_pair(pair, frame.data))
        return label_frames

    def _label_pair(self, pair: str, df: pd.DataFrame) -> pd.DataFrame:
        pip = _pip_size(pair)
        max_hold = pd.Timedelta(minutes=self.cfg.max_holding_minutes)
        timestamps = df.index
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()

        rows = []
        tp_grid = sorted(set(self.cfg.labels.tp_grid_pips))
        sl_grid = sorted(set(self.cfg.labels.sl_grid_pips))

        for idx in range(len(df)):
            entry_time = timestamps[idx]
            cutoff = min(entry_time + max_hold, _session_close_timestamp(entry_time, self.cfg.session_close_utc))
            end_pos = timestamps.searchsorted(cutoff, side="right")
            if end_pos - idx <= 1:
                continue

            future_slice = slice(idx + 1, end_pos)
            future_highs = highs[future_slice]
            future_lows = lows[future_slice]
            future_closes = closes[future_slice]

            long_reward, long_tp, long_sl, long_hold = self._evaluate_direction(
                entry_price=closes[idx],
                highs=future_highs,
                lows=future_lows,
                closes=future_closes,
                pip=pip,
                tp_grid=tp_grid,
                sl_grid=sl_grid,
                timeframe_minutes=self.cfg.timeframe_minutes,
                direction="long",
            )
            short_reward, short_tp, short_sl, short_hold = self._evaluate_direction(
                entry_price=closes[idx],
                highs=future_highs,
                lows=future_lows,
                closes=future_closes,
                pip=pip,
                tp_grid=tp_grid,
                sl_grid=sl_grid,
                timeframe_minutes=self.cfg.timeframe_minutes,
                direction="short",
            )

            direction = 0
            tp_pips = 0.0
            sl_pips = 0.0
            hold_minutes = 0.0
            edge = 0.0

            if max(long_reward, short_reward) > 0:
                if long_reward >= short_reward:
                    direction = 1
                    tp_pips, sl_pips, hold_minutes, edge = long_tp, long_sl, long_hold, long_reward
                else:
                    direction = -1
                    tp_pips, sl_pips, hold_minutes, edge = short_tp, short_sl, short_hold, short_reward

            rows.append(
                {
                    "timestamp": entry_time,
                    "direction": direction,
                    "tp_pips": tp_pips,
                    "sl_pips": sl_pips,
                    "holding_minutes": hold_minutes,
                    "edge_pips": edge,
                }
            )

        if not rows:
            raise ValueError(f"Unable to construct labels for {pair}; check data window size.")

        label_df = pd.DataFrame(rows).set_index("timestamp")
        return label_df

    def _evaluate_direction(
        self,
        entry_price: float,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        pip: float,
        tp_grid: list[float],
        sl_grid: list[float],
        timeframe_minutes: int,
        direction: str,
    ) -> Tuple[float, float, float, float]:
        best_reward = -np.inf
        best_tp = 0.0
        best_sl = 0.0
        best_hold = 0.0

        for tp in tp_grid:
            for sl in sl_grid:
                reward, hold_minutes = self._simulate_path(
                    entry_price=entry_price,
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    pip=pip,
                    tp=tp,
                    sl=sl,
                    timeframe_minutes=timeframe_minutes,
                    direction=direction,
                )
                if reward > best_reward:
                    best_reward = reward
                    best_tp, best_sl, best_hold = tp, sl, hold_minutes

        if not np.isfinite(best_reward):
            return -np.inf, 0.0, 0.0, 0.0
        return best_reward, best_tp, best_sl, best_hold

    def _simulate_path(
        self,
        entry_price: float,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        pip: float,
        tp: float,
        sl: float,
        timeframe_minutes: int,
        direction: str,
    ) -> Tuple[float, float]:
        cost = self.cfg.labels.transaction_cost_pips
        tp_hit_idx: Optional[int] = None
        sl_hit_idx: Optional[int] = None
        tp_price: float
        sl_price: float

        if direction == "long":
            tp_price = entry_price + tp * pip
            sl_price = entry_price - sl * pip
            for idx, (high, low) in enumerate(zip(highs, lows)):
                if tp_hit_idx is None and high >= tp_price:
                    tp_hit_idx = idx
                if sl_hit_idx is None and low <= sl_price:
                    sl_hit_idx = idx
                if tp_hit_idx is not None or sl_hit_idx is not None:
                    break
        else:
            tp_price = entry_price - tp * pip
            sl_price = entry_price + sl * pip
            for idx, (high, low) in enumerate(zip(highs, lows)):
                if tp_hit_idx is None and low <= tp_price:
                    tp_hit_idx = idx
                if sl_hit_idx is None and high >= sl_price:
                    sl_hit_idx = idx
                if tp_hit_idx is not None or sl_hit_idx is not None:
                    break

        if tp_hit_idx is not None and (sl_hit_idx is None or tp_hit_idx <= sl_hit_idx):
            reward = tp - cost
            hold_minutes = (tp_hit_idx + 1) * timeframe_minutes
        elif sl_hit_idx is not None:
            reward = -sl - cost
            hold_minutes = (sl_hit_idx + 1) * timeframe_minutes
        else:
            final_diff = closes[-1] - entry_price
            reward = (final_diff / pip) - cost if direction == "long" else (-final_diff / pip) - cost
            hold_minutes = len(highs) * timeframe_minutes

        return reward, float(hold_minutes)
