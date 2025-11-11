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
    pnl_raw_pips: float
    direction_conf: float
    risk_fraction: float
    session: str
    adx_14: float
    cost_pips: float
    edge_pred_pips: float
    sentiment_zscore: float
    event_high_impact: float
    event_macro_event_count: float
    vix_zscore: float


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
        if not trades:
            raise ValueError("No trades were generated from the supplied predictions.")
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
        base_cost = self.cfg.labels.transaction_cost_pips + self.cfg.backtest.slippage_pips
        pair_spread = self.cfg.backtest.spread_pips.get(pair, self.cfg.backtest.spread_pips.get("default", 0.0))
        session_spreads = self.cfg.backtest.session_spreads or {}
        min_adx = self.cfg.backtest.min_adx
        min_edge = self.cfg.backtest.min_edge_pips
        comfort_ratio = getattr(self.cfg.labels, "min_tp_sl_ratio", 1.0)
        max_concurrent = max(1, self.cfg.backtest.max_concurrent_trades)
        dynamic_risk = self.cfg.backtest.dynamic_risk
        risk_power = max(self.cfg.backtest.confidence_risk_power, 1e-6)
        base_risk = self.cfg.backtest.risk_per_trade
        sentiment_min = self.cfg.backtest.sentiment_zscore_min
        sentiment_max = self.cfg.backtest.sentiment_zscore_max
        max_event_importance = self.cfg.backtest.max_event_importance
        max_event_count = self.cfg.backtest.max_macro_event_count
        vix_min = self.cfg.backtest.vix_zscore_min
        vix_max = self.cfg.backtest.vix_zscore_max
        trades: List[TradeRecord] = []
        active_exits: List[pd.Timestamp] = []

        for ts, signal in signals.iterrows():
            direction = int(signal["direction"])
            if direction == 0 or ts not in df.index:
                continue

            conf = self._signal_confidence(signal)
            edge_pred = float(signal.get("edge_pred_pips", np.nan))
            if min_edge is not None and (np.isnan(edge_pred) or edge_pred < min_edge):
                continue
            session = signal.get("session") or self._session_label(ts)
            adx_val = float(signal.get("adx_14", np.nan))
            if min_adx > 0 and (np.isnan(adx_val) or adx_val < min_adx):
                continue
            sentiment_z = float(signal.get("sentiment_zscore_96", np.nan))
            if sentiment_min is not None and (np.isnan(sentiment_z) or sentiment_z < sentiment_min):
                continue
            if sentiment_max is not None and (np.isnan(sentiment_z) or sentiment_z > sentiment_max):
                continue
            event_importance = float(signal.get("event_high_impact", 0.0))
            if max_event_importance is not None and event_importance > max_event_importance:
                continue
            event_count = float(signal.get("event_macro_event_count", 0.0))
            if max_event_count is not None and event_count > max_event_count:
                continue
            vix_z = float(signal.get("vix_zscore_96", np.nan))
            if vix_min is not None and (np.isnan(vix_z) or vix_z < vix_min):
                continue
            if vix_max is not None and (np.isnan(vix_z) or vix_z > vix_max):
                continue
            entry_price = df.loc[ts, "close"]
            tp_pips = float(signal["tp_pips"])
            sl_pips = float(signal["sl_pips"])
            if sl_pips <= 0 or tp_pips <= 0:
                continue
            if tp_pips / (sl_pips + 1e-9) < comfort_ratio:
                continue
            active_exits = [et for et in active_exits if et > ts]
            if len(active_exits) >= max_concurrent:
                continue

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

            session_multiplier = session_spreads.get(session, session_spreads.get("default", 1.0))
            spread_cost = pair_spread * session_multiplier
            trade_cost = base_cost + spread_cost

            risk_scale = conf**risk_power if dynamic_risk else 1.0
            risk_fraction = base_risk * risk_scale

            pnl_raw, exit_time, outcome = self._evaluate_trade(
                direction,
                entry_price,
                highs,
                lows,
                closes,
                tp_pips,
                sl_pips,
                pip,
                trade_cost,
                ts,
                df.index[entry_idx + 1 : exit_idx],
            )
            pnl_scaled = pnl_raw * risk_scale

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
                    pnl_pips=pnl_scaled,
                    pnl_raw_pips=pnl_raw,
                    direction_conf=conf,
                    risk_fraction=risk_fraction,
                    session=session,
                    adx_14=adx_val,
                    cost_pips=trade_cost,
                    edge_pred_pips=edge_pred,
                    sentiment_zscore=sentiment_z,
                    event_high_impact=event_importance,
                    event_macro_event_count=event_count,
                    vix_zscore=vix_z,
                )
            )
            active_exits.append(exit_time)

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
    ) -> tuple[float, pd.Timestamp, str]:
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

    @staticmethod
    def _session_label(ts: pd.Timestamp) -> str:
        hour = ts.tz_convert("UTC").hour
        if 23 <= hour or hour < 7:
            return "asia"
        if 7 <= hour < 16:
            return "europe"
        return "us"

    @staticmethod
    def _signal_confidence(signal: pd.Series) -> float:
        conf = signal.get("direction_conf", np.nan)
        if pd.isna(conf):
            probs = signal.get("direction_prob")
            if isinstance(probs, (list, tuple)) and probs:
                conf = max(float(p) for p in probs)
            else:
                conf = 1.0
        return float(max(min(conf, 1.0), 0.0))


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
