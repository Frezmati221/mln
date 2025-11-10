from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch
from torch import nn
import torch.nn.functional as F

from ..utils.config import ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiTaskTransformer(nn.Module):
    def __init__(self, input_dim: int, cfg: ModelConfig, num_pairs: int):
        super().__init__()
        self.cfg = cfg
        self.token_proj = nn.Linear(input_dim, cfg.hidden_dim)
        self.positional_encoding = PositionalEncoding(cfg.hidden_dim, cfg.max_seq_len)
        self.pair_embedding = nn.Embedding(num_pairs, cfg.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.hidden_dim)

        self.direction_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, 3),
        )
        self.tp_head = nn.Linear(cfg.hidden_dim, 1)
        self.sl_head = nn.Linear(cfg.hidden_dim, 1)
        self.holding_head = nn.Linear(cfg.hidden_dim, 1)
        self.uncertainty_head = nn.Linear(cfg.hidden_dim, 1) if cfg.uncertainty_head else None

    def forward(self, x: torch.Tensor, pair_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [batch, seq_len, input_dim]
        tokens = self.token_proj(x)
        tokens = self.positional_encoding(tokens)
        tokens = tokens + self.pair_embedding(pair_idx).unsqueeze(1)
        encoded = self.encoder(tokens)
        pooled = self.norm(encoded[:, -1])

        outputs = {
            "direction_logits": self.direction_head(pooled),
            "tp": torch.relu(self.tp_head(pooled)),
            "sl": torch.relu(self.sl_head(pooled)),
            "holding": torch.relu(self.holding_head(pooled)),
        }
        if self.uncertainty_head is not None:
            outputs["uncertainty"] = F.softplus(self.uncertainty_head(pooled))
        return outputs
