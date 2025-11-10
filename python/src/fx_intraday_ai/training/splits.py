from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import pandas as pd


@dataclass(slots=True)
class WalkForwardSplit:
    train: Dict[str, pd.DataFrame]
    valid: Dict[str, pd.DataFrame]
    start: pd.Timestamp
    end: pd.Timestamp


def walk_forward(frames: Dict[str, pd.DataFrame], window_days: int, step_days: int) -> Iterator[WalkForwardSplit]:
    min_start = min(df.index.min() for df in frames.values())
    max_end = max(df.index.max() for df in frames.values())
    cursor = min_start
    window = pd.Timedelta(days=window_days)
    step = pd.Timedelta(days=step_days)

    while cursor + window < max_end:
        train_end = cursor + window
        valid_end = train_end + step
        train_split: Dict[str, pd.DataFrame] = {}
        valid_split: Dict[str, pd.DataFrame] = {}
        for pair, df in frames.items():
            train_split[pair] = df[(df.index >= cursor) & (df.index < train_end)]
            valid_split[pair] = df[(df.index >= train_end) & (df.index < valid_end)]
        yield WalkForwardSplit(train=train_split, valid=valid_split, start=train_end, end=valid_end)
        cursor += step
