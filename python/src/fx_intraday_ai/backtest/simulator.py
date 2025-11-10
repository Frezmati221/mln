from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd

from ..data.schemas import PriceFrame
from ..utils.config import ExperimentConfig
from ..labels.intraday import _pip_size, _session_close_timestamp


@dataclass(slots=True)
class TradeRecord:
    timestamp: pd.Timestamp
    exit_time: pd.Timestamp
    pair: str
    direction: int
    tp_pips: float
    sl_pips: float
    holding_minutes: float
    outcome: str
    pnl_pips: float


class Backtester:
    def __init__(self, cfg: ExperimentConfig, price_frames: Dict[str, PriceFrame]):
        self.cfg = cfg
        self.price_frames = price_frames

    def simulate(self, signals: pd.DataFrame) -> pd.DataFrame:
        if signals is None or signals.empty:
            raise ValueError("No signals provided for backtest.")
        trades: List[TradeRecord] = []
        grouped = signals.groupby("pair")
        for pair, pair_signals in grouped:
            if pair not in self.price_frames:
                continue
            trades.extend(self._simulate_pair(pair, pair_signals.sort_index()))
        trade_df = pd.DataFrame([asdict(trade) for trade in trades])
        trade_df.set_index("timestamp", inplace=True)
        return trade_df

    def summarize(self, trades: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for pair, pair_trades in trades.groupby("pair"):
            summary[pair] = _compute_metrics(pair_trades)
        summary["aggregate"] = _compute_metrics(trades)
        return summary

    def _simulate_pair(self, pair: str, signals: pd.DataFrame) -> List[TradeRecord]:
        frame = self.price_frames[pair]
        df = frame.data
        pip = _pip_size(pair)
        cost = self.cfg.labels.transaction_cost_pips
        trades: List[TradeRecord] = []

        for ts, signal in signals.iterrows():
            direction = int(signal["direction"])
            if direction == 0 or ts not in df.index:
                continue

            entry_price = df.loc[ts, "close"]
            tp_pips = float(signal["tp_pips"])
            sl_pips = float(signal["sl_pips"])
            holding_minutes = float(signal["holding_minutes"])
            holding_minutes = max(holding_minutes, self.cfg.timeframe_minutes)
            exit_cutoff = ts + pd.Timedelta(minutes=holding_minutes)
            exit_cutoff = min(exit_cutoff, _session_close_timestamp(ts, self.cfg.session_close_utc))
            exit_idx = df.index.searchsorted(exit_cutoff, side="right")
            entry_idx = df.index.get_loc(ts)
            if exit_idx <= entry_idx + 1:
                continue

            highs = df["high"].to_numpy()[entry_idx + 1 : exit_idx]
            lows = df["low"].to_numpy()[entry_idx + 1 : exit_idx]
            closes = df["close"].to_numpy()[entry_idx + 1 : exit_idx]

            pnl_pips, exit_time, outcome = self._evaluate_trade(
                direction, entry_price, highs, lows, closes, tp_pips, sl_pips, pip, cost, ts, df.index[entry_idx + 1 : exit_idx]
            )

            trades.append(
                TradeRecord(
                    timestamp=ts,
                    exit_time=exit_time,
                    pair=pair,
                    direction=direction,
                    tp_pips=tp_pips,
                    sl_pips=sl_pips,
                    holding_minutes=holding_minutes,
                    outcome=outcome,
                    pnl_pips=pnl_pips,
                )
            )

        return trades

    def _evaluate_trade(
        self,
        direction: int,
        entry_price: float,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        tp_pips: float,
        sl_pips: float,
        pip: float,
        cost: float,
        entry_time: pd.Timestamp,
        timestamps: pd.Index,
    ):
        tp_price = entry_price + (tp_pips * pip if direction == 1 else -tp_pips * pip)
        sl_price = entry_price - (sl_pips * pip if direction == 1 else -sl_pips * pip)
        tp_hit_idx = None
        sl_hit_idx = None

        for idx, (high, low) in enumerate(zip(highs, lows)):
            if tp_hit_idx is None:
                if direction == 1 and high >= tp_price:
                    tp_hit_idx = idx
                elif direction == -1 and low <= tp_price:
                    tp_hit_idx = idx
            if sl_hit_idx is None:
                if direction == 1 and low <= sl_price:
                    sl_hit_idx = idx
                elif direction == -1 and high >= sl_price:
                    sl_hit_idx = idx
            if tp_hit_idx is not None or sl_hit_idx is not None:
                break

        if tp_hit_idx is not None and (sl_hit_idx is None or tp_hit_idx <= sl_hit_idx):
            pnl = tp_pips - cost
            exit_time = timestamps[tp_hit_idx]
            outcome = "tp"
        elif sl_hit_idx is not None:
            pnl = -sl_pips - cost
            exit_time = timestamps[sl_hit_idx]
            outcome = "sl"
        else:
            final_close = closes[-1]
            diff = (final_close - entry_price) / pip
            pnl = diff - cost if direction == 1 else -diff - cost
            exit_time = timestamps[-1]
            outcome = "timeout"

        return pnl, exit_time, outcome


def _compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    pnl = df["pnl_pips"]
    wins = (pnl > 0).sum()
    losses = (pnl < 0).sum()
    pnl_sum = pnl.sum()
    pnl_pos = pnl[pnl > 0].sum()
    pnl_neg = -pnl[pnl < 0].sum()
    profit_factor = pnl_pos / pnl_neg if pnl_neg > 0 else float("inf")
    sharpe = pnl.mean() / (pnl.std() + 1e-9) * np.sqrt(len(pnl)) if len(pnl) > 1 else 0.0
    dd = _max_drawdown(pnl.cumsum())
    return {
        "trades": float(len(df)),
        "win_rate": wins / len(df) if len(df) else 0.0,
        "avg_pnl": pnl.mean() if len(df) else 0.0,
        "pnl_total": pnl_sum,
        "profit_factor": profit_factor,
        "sharpe_like": sharpe,
        "max_drawdown": dd,
    }


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity - peak
    return drawdown.min()
