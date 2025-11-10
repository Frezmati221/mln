from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml
from pydantic import BaseModel, Field, validator


class DataConfig(BaseModel):
    price_path: Path
    macro_path: Optional[Path] = None
    sentiment_path: Optional[Path] = None
    calendar_path: Optional[Path] = None
    cache_dir: Path = Path("data/cache")
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    @validator("price_path", "macro_path", "sentiment_path", "calendar_path", pre=True, always=True)
    def _expand_path(cls, value: Optional[str]) -> Optional[Path]:
        if value is None:
            return None
        return Path(value).expanduser().resolve()

    @validator("cache_dir", pre=True, always=True)
    def _expand_cache(cls, value: str) -> Path:
        return Path(value).expanduser().resolve()


class FeatureConfig(BaseModel):
    ema_windows: List[int]
    rsi_windows: List[int]
    atr_windows: List[int]
    bollinger: Dict[str, float] = Field(default_factory=dict)
    regime_features: Dict[str, bool] = Field(default_factory=dict)
    multi_scale_windows: Optional[List[int]] = None


class PairGrid(BaseModel):
    default: List[float]
    USDJPY: Optional[List[float]] = None


class LabelConfig(BaseModel):
    tp_grid_pips: Any
    sl_grid_pips: Any
    transaction_cost_pips: float = 1.0
    min_tp_sl_ratio: float = 1.5
    reward_metric: str = "sharpe"


class ModelLossWeights(BaseModel):
    direction: float = 1.0
    take_profit: float = 1.0
    stop_loss: float = 1.0
    holding_minutes: float = 0.5
    direction_balance: float = 0.0
    volatility: float = 0.0
    regime: float = 0.0


class ModelConfig(BaseModel):
    type: str = "multitask_transformer"
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 256
    multi_scale: Optional[List[int]] = None
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    loss_weights: ModelLossWeights = Field(default_factory=ModelLossWeights)
    uncertainty_head: bool = True
    auxiliary_heads: Dict[str, bool] = Field(default_factory=dict)


class WalkForwardConfig(BaseModel):
    window_days: int = 90
    step_days: int = 30


class AugmentationConfig(BaseModel):
    enabled: bool = False
    time_shift_range: int = 0
    time_shift_prob: float = 0.0
    feature_noise_std: float = 0.0
    atr_jitter_pct: float = 0.0


class TrainingConfig(BaseModel):
    batch_size: int = 128
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    max_epochs: int = 50
    gradient_clip_norm: float = 1.0
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    early_stopping_patience: Optional[int] = None
    mixed_precision: bool = False
    num_workers: int = 4
    ensemble_members: int = 1
    base_seed: int = 1337
    augmentation: Optional[AugmentationConfig] = None
    direction_focal_gamma: float = 0.0
    direction_class_weights: Optional[List[float]] = None


class BacktestConfig(BaseModel):
    slippage_pips: float = 0.1
    spread_pips: Dict[str, float]
    risk_per_trade: float = 0.005
    dynamic_risk: bool = False
    confidence_risk_power: float = 1.0
    session_spreads: Optional[Dict[str, float]] = None
    min_adx: float = 0.0
    max_concurrent_trades: int = 1
    max_daily_drawdown: float = 0.02
    sentiment_zscore_min: Optional[float] = None
    sentiment_zscore_max: Optional[float] = None
    max_event_importance: Optional[float] = None
    max_macro_event_count: Optional[int] = None
    vix_zscore_min: Optional[float] = None
    vix_zscore_max: Optional[float] = None


class ExperimentConfig(BaseModel):
    pairs: List[str]
    timeframe_minutes: int = 5
    session_close_utc: str = "21:59"
    max_holding_minutes: int = 240
    data: DataConfig
    features: FeatureConfig
    labels: LabelConfig
    model: ModelConfig
    training: TrainingConfig
    backtest: BacktestConfig


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload: Dict[str, Any] = yaml.safe_load(handle)
    return ExperimentConfig(**payload)
