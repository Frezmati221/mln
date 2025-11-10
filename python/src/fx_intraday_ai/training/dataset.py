from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils.config import AugmentationConfig

TARGET_COLUMNS = ["direction", "tp_pips", "sl_pips", "holding_minutes", "edge_pips"]
AUX_TARGET_COLUMNS = ["volatility_target", "regime_target"]
ALL_TARGET_COLUMNS = TARGET_COLUMNS + AUX_TARGET_COLUMNS
DIR_TO_CLASS = {-1: 0, 0: 1, 1: 2}


@dataclass(slots=True)
class SequenceNormalizer:
    mean: pd.Series | None = None
    std: pd.Series | None = None
    feature_columns: List[str] | None = None

    def fit(self, frames: Dict[str, pd.DataFrame]) -> None:
        non_empty = [df for df in frames.values() if not df.empty]
        if not non_empty:
            raise ValueError("No data provided to fit normalizer.")
        sample = non_empty[0]
        self.feature_columns = [col for col in sample.columns if col not in ALL_TARGET_COLUMNS]
        concat = pd.concat([df[self.feature_columns] for df in frames.values()])
        self.mean = concat.mean()
        self.std = concat.std().replace(0, 1.0)

    def transform(self, frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if self.mean is None or self.std is None or self.feature_columns is None:
            raise ValueError("Normalizer must be fit before transform.")
        normalized: Dict[str, pd.DataFrame] = {}
        for pair, df in frames.items():
            norm_df = df.copy()
            norm_df[self.feature_columns] = (norm_df[self.feature_columns] - self.mean) / self.std
            normalized[pair] = norm_df
        return normalized


class SequenceDataset(Dataset):
    def __init__(
        self,
        frames: Dict[str, pd.DataFrame],
        seq_len: int,
        feature_columns: List[str],
        augmentation: Optional[AugmentationConfig] = None,
        is_training: bool = False,
        seed: Optional[int] = None,
        raw_frames: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        self.seq_len = seq_len
        self.feature_columns = feature_columns
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.metadata: List[Tuple[str, pd.Timestamp]] = []
        self.frames = frames
        self.meta_frames = raw_frames if raw_frames is not None else frames
        self.is_training = is_training
        self.augmentation = augmentation if is_training and augmentation and augmentation.enabled else None
        self.rng = np.random.default_rng(seed) if self.augmentation is not None else None
        self._build_samples(frames)

    def _build_samples(self, frames: Dict[str, pd.DataFrame]) -> None:
        for pair_idx, pair in enumerate(sorted(frames.keys())):
            df = frames[pair]
            feature_values = df[self.feature_columns].to_numpy(dtype=np.float32)
            labels = df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
            aux = df[AUX_TARGET_COLUMNS].to_numpy(dtype=np.float32)
            for idx in range(self.seq_len, len(df)):
                window = feature_values[idx - self.seq_len : idx]
                direction_raw = int(labels[idx, 0])
                direction_class = DIR_TO_CLASS.get(direction_raw, 1)
                vol_label = int(aux[idx, 0] > 0.5)
                regime_label = int(aux[idx, 1] > 0.5)
                target = np.array(
                    [
                        direction_class,
                        labels[idx, 1],  # tp
                        labels[idx, 2],  # sl
                        labels[idx, 3],  # holding_minutes
                        labels[idx, 4],  # edge
                        pair_idx,
                        vol_label,
                        regime_label,
                    ],
                    dtype=np.float32,
                )
                self.samples.append((torch.from_numpy(window), torch.from_numpy(target)))
                self.metadata.append((pair, df.index[idx]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        features, target = self.samples[index]
        features = features.clone()
        target = target.clone()
        pair, timestamp = self.metadata[index]
        if self.is_training and self.augmentation is not None and self.rng is not None:
            features, target = self._apply_augmentation(features, target, pair, timestamp)

        x = features  # shape [seq_len, num_features]
        y = {
            "direction": target[0].long(),
            "tp": target[1].float(),
            "sl": target[2].float(),
            "holding": target[3].float(),
            "edge": target[4].float(),
            "pair_idx": target[5].long(),
            "sample_id": torch.tensor(index, dtype=torch.long),
            "volatility_target": target[6].long(),
            "regime_target": target[7].long(),
        }
        return x, y

    def sample_metadata(self, index: int) -> Tuple[str, pd.Timestamp]:
        return self.metadata[index]

    def _apply_augmentation(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
        pair: str,
        timestamp: pd.Timestamp,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        aug = self.augmentation
        assert aug is not None and self.rng is not None
        if aug.time_shift_range > 0 and aug.time_shift_prob > 0:
            if self.rng.random() < aug.time_shift_prob:
                shift = int(self.rng.integers(-aug.time_shift_range, aug.time_shift_range + 1))
                if shift != 0:
                    features = self._time_shift(features, shift)
        if aug.feature_noise_std > 0:
            noise = torch.from_numpy(
                self.rng.normal(0.0, aug.feature_noise_std, size=features.shape).astype(np.float32)
            )
            features = features + noise
        if aug.atr_jitter_pct > 0:
            atr_val = self._lookup_indicator(pair, timestamp, "atr_14")
            if not np.isnan(atr_val):
                scale = float(1.0 + self.rng.normal(0.0, aug.atr_jitter_pct))
                scale = float(np.clip(scale, 0.5, 1.5))
                target[1] = torch.relu(target[1] * scale)
                target[2] = torch.relu(target[2] * scale)
                target[4] = target[4] * scale
        return features, target

    @staticmethod
    def _time_shift(features: torch.Tensor, shift: int) -> torch.Tensor:
        seq_len = features.shape[0]
        steps = int(min(abs(shift), max(seq_len - 1, 1)))
        if steps == 0:
            return features
        if shift > 0:
            pad = features[:1].repeat(steps, 1)
            shifted = torch.cat([pad, features[:-steps]], dim=0)
        else:
            pad = features[-1:].repeat(steps, 1)
            shifted = torch.cat([features[steps:], pad], dim=0)
        return shifted

    def _lookup_indicator(self, pair: str, timestamp: pd.Timestamp, column: str) -> float:
        frame = self.meta_frames.get(pair)
        if frame is None or column not in frame.columns:
            return float("nan")
        try:
            value = frame.at[timestamp, column]
        except KeyError:
            return float("nan")
        return float(value) if pd.notna(value) else float("nan")

    def meta_value(self, pair: str, timestamp: pd.Timestamp, column: str) -> float:
        return self._lookup_indicator(pair, timestamp, column)
