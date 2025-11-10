from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TARGET_COLUMNS = ["direction", "tp_pips", "sl_pips", "holding_minutes", "edge_pips"]
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
        self.feature_columns = [col for col in sample.columns if col not in TARGET_COLUMNS]
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
    def __init__(self, frames: Dict[str, pd.DataFrame], seq_len: int, feature_columns: List[str]):
        self.seq_len = seq_len
        self.feature_columns = feature_columns
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.metadata: List[Tuple[str, pd.Timestamp]] = []
        self._build_samples(frames)

    def _build_samples(self, frames: Dict[str, pd.DataFrame]) -> None:
        for pair_idx, pair in enumerate(sorted(frames.keys())):
            df = frames[pair]
            feature_values = df[self.feature_columns].to_numpy(dtype=np.float32)
            labels = df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
            for idx in range(self.seq_len, len(df)):
                window = feature_values[idx - self.seq_len : idx]
                direction_raw = int(labels[idx, 0])
                direction_class = DIR_TO_CLASS.get(direction_raw, 1)
                target = np.array(
                    [
                        direction_class,
                        labels[idx, 1],  # tp
                        labels[idx, 2],  # sl
                        labels[idx, 3],  # holding_minutes
                        labels[idx, 4],  # edge
                        pair_idx,
                    ],
                    dtype=np.float32,
                )
                self.samples.append((torch.from_numpy(window), torch.from_numpy(target)))
                self.metadata.append((pair, df.index[idx]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        features, target = self.samples[index]
        x = features  # shape [seq_len, num_features]
        y = {
            "direction": target[0].long(),
            "tp": target[1].float(),
            "sl": target[2].float(),
            "holding": target[3].float(),
            "edge": target[4].float(),
            "pair_idx": target[5].long(),
            "sample_id": torch.tensor(index, dtype=torch.long),
        }
        return x, y

    def sample_metadata(self, index: int) -> Tuple[str, pd.Timestamp]:
        return self.metadata[index]
