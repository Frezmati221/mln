from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd

from .loader import load_price_frames
from ..features.engineer import FeatureEngineer
from ..labels.intraday import IntradayLabeler
from ..utils.config import ExperimentConfig
from .schemas import PriceFrame


@dataclass(slots=True)
class DataPipeline:
    cfg: ExperimentConfig
    _feature_engineer: FeatureEngineer = field(init=False, repr=False)
    _labeler: IntradayLabeler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._feature_engineer = FeatureEngineer(self.cfg.features)
        self._labeler = IntradayLabeler(self.cfg)

    def run(self, refresh_cache: bool = False) -> Dict[str, pd.DataFrame]:
        price_frames = load_price_frames(self.cfg, refresh_cache=refresh_cache)
        feature_frames = self._feature_engineer.transform(price_frames)
        label_frames = self._labeler.build(price_frames)

        merged: Dict[str, pd.DataFrame] = {}
        for pair, feature_frame in feature_frames.items():
            if pair not in label_frames:
                raise KeyError(f"Missing labels for {pair}")
            df = feature_frame.data.join(label_frames[pair].data, how="inner")
            df.dropna(inplace=True)
            merged[pair] = df
        return merged

    def load_prices(self, refresh_cache: bool = False) -> Dict[str, PriceFrame]:
        return load_price_frames(self.cfg, refresh_cache=refresh_cache)
