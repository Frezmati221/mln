from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Dict

import pandas as pd

BarColumns = Literal["open", "high", "low", "close", "volume", "spread"]


@dataclass(slots=True)
class PriceFrame:
    """Container describing cleaned 5-minute price bars for a single pair."""

    pair: str
    data: pd.DataFrame  # index: timestamp (UTC), columns: BarColumns

    def validate(self) -> None:
        expected = {"open", "high", "low", "close", "volume"}
        missing = expected.difference(self.data.columns)
        if missing:
            raise ValueError(f"{self.pair} missing columns: {missing}")
        if not self.data.index.is_monotonic_increasing:
            self.data = self.data.sort_index()
        if self.data.index.tzinfo is None:
            raise ValueError("PriceFrame data must be timezone-aware (UTC).")


@dataclass(slots=True)
class FeatureFrame:
    pair: str
    data: pd.DataFrame  # index aligned to PriceFrame


@dataclass(slots=True)
class LabelFrame:
    pair: str
    data: pd.DataFrame  # columns: direction, tp_pips, sl_pips, hold_minutes


PredictionTargets = Dict[str, Literal["direction", "tp_pips", "sl_pips", "hold_minutes"]]
