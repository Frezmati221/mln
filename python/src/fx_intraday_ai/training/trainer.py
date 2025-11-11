from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from contextlib import nullcontext

import pandas as pd
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TextColumn,
)
from rich.console import Console

from ..data.pipeline import DataPipeline
from ..training.dataset import SequenceDataset, SequenceNormalizer
from ..training.splits import walk_forward, WalkForwardSplit
from ..models.multitask_transformer import MultiTaskTransformer
from ..utils.config import ExperimentConfig

CLASS_TO_DIR = {0: -1, 1: 0, 2: 1}
console = Console()


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean_probabilities(series: pd.Series) -> Optional[List[float]]:
    arrays: List[np.ndarray] = []
    for value in series:
        if isinstance(value, (list, tuple)):
            arrays.append(np.asarray(value, dtype=np.float32))
        elif isinstance(value, np.ndarray):
            arrays.append(value.astype(np.float32))
    if not arrays:
        return None
    stacked = np.stack(arrays, axis=0)
    return stacked.mean(axis=0).tolist()


def _aggregate_ensemble_predictions(predictions: List[pd.DataFrame]) -> pd.DataFrame:
    if not predictions:
        return pd.DataFrame()
    if len(predictions) == 1:
        df = predictions[0].copy()
        return df.drop(columns=["ensemble_member"], errors="ignore")
    concat = pd.concat(predictions)
    concat = concat.drop(columns=["ensemble_member"], errors="ignore")
    concat = concat.copy()
    concat = concat.reset_index()
    grouped = concat.groupby(["timestamp", "pair"], sort=True)
    records: List[Dict[str, object]] = []
    for (timestamp, pair), group in grouped:
        prob_list = _mean_probabilities(group["direction_prob"])
        if prob_list is None:
            counts = group["direction_class"].value_counts()
            total = counts.sum()
            prob_vec = np.zeros(3, dtype=np.float32)
            if total > 0:
                for cls, count in counts.items():
                    prob_vec[int(cls)] = count / total
            else:
                prob_vec[:] = 1.0 / 3.0
            prob_list = prob_vec.tolist()
        prob_arr = np.asarray(prob_list, dtype=np.float32)
        direction_class = int(np.argmax(prob_arr))
        direction_conf = float(prob_arr[direction_class])
        record: Dict[str, object] = {
            "timestamp": timestamp,
            "pair": pair,
            "direction_class": direction_class,
            "direction": CLASS_TO_DIR.get(direction_class, 0),
            "direction_prob": prob_arr.tolist(),
            "direction_conf": direction_conf,
            "tp_pips": group["tp_pips"].mean(),
            "sl_pips": group["sl_pips"].mean(),
            "holding_minutes": group["holding_minutes"].mean(),
            "adx_14": group["adx_14"].mean(),
            "session": group["session"].iloc[0],
        }
        if "edge_pred_pips" in group.columns:
            record["edge_pred_pips"] = group["edge_pred_pips"].mean()
        if "volatility_prob" in group.columns:
            record["volatility_prob"] = _mean_probabilities(group["volatility_prob"])
        if "regime_prob" in group.columns:
            record["regime_prob"] = _mean_probabilities(group["regime_prob"])
        records.append(record)
    result = pd.DataFrame(records)
    if result.empty:
        return result
    result.set_index("timestamp", inplace=True)
    return result.sort_index()


def _min_length(frames: Dict[str, pd.DataFrame]) -> int:
    lengths = [len(df) for df in frames.values() if len(df) > 0]
    return min(lengths) if lengths else 0


@dataclass(slots=True)
class EpochLog:
    epoch: int
    train_loss: float
    valid_loss: float
    valid_accuracy: float


@dataclass(slots=True)
class TrainingArtifacts:
    history: List[EpochLog]
    predictions: Optional[pd.DataFrame] = None
    model_states: List[Tuple[str, Dict[str, torch.Tensor]]] = field(default_factory=list)


class ModelTrainer:
    def __init__(self, cfg: ExperimentConfig, input_dim: int, num_pairs: int):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiTaskTransformer(
            input_dim=input_dim,
            cfg=cfg.model,
            num_pairs=num_pairs,
            base_timeframe=cfg.timeframe_minutes,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        self.use_amp = cfg.training.mixed_precision and torch.cuda.is_available()
        self._amp_backend = None
        if self.use_amp and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            self._amp_backend = "torch.amp"
            try:
                self.grad_scaler = torch.amp.GradScaler("cuda")
            except TypeError:
                self.grad_scaler = torch.amp.GradScaler("cuda", enabled=True)
        elif self.use_amp and hasattr(torch.cuda, "amp"):
            self._amp_backend = "torch.cuda.amp"
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.grad_scaler = None
            self.use_amp = False
        self.criterion_dir = nn.CrossEntropyLoss()
        self.criterion_reg = nn.SmoothL1Loss(reduction="none")
        self.criterion_aux = nn.CrossEntropyLoss()
        self.history: List[EpochLog] = []
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.best_metric: float = float("inf")
        self.direction_gamma = max(float(cfg.training.direction_focal_gamma or 0.0), 0.0)
        weights = cfg.training.direction_class_weights or None
        if weights and len(weights) == 3:
            self.direction_weight = torch.tensor(weights, dtype=torch.float32)
        else:
            self.direction_weight = None

    def fit(self, train_ds: SequenceDataset, valid_ds: SequenceDataset, progress_desc: str | None = None) -> None:
        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.training.num_workers,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        label = progress_desc or "Training"
        patience = self.cfg.training.early_stopping_patience
        patience_counter = 0
        with progress:
            epoch_task = progress.add_task(f"{label} | epochs", total=self.cfg.training.max_epochs)
            for epoch in range(1, self.cfg.training.max_epochs + 1):
                train_total = len(train_loader) or 1
                train_task = progress.add_task(f"{label} | train e{epoch}", total=train_total)
                train_loss, _ = self._run_epoch(train_loader, training=True, progress=progress, task_id=train_task)
                progress.remove_task(train_task)

                valid_total = len(valid_loader) or 1
                valid_task = progress.add_task(f"{label} | valid e{epoch}", total=valid_total)
                valid_loss, valid_acc = self._run_epoch(valid_loader, training=False, progress=progress, task_id=valid_task)
                progress.remove_task(valid_task)

                self.history.append(
                    EpochLog(epoch=epoch, train_loss=train_loss, valid_loss=valid_loss, valid_accuracy=valid_acc)
                )
                if valid_loss < self.best_metric:
                    self.best_metric = valid_loss
                    self.best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                progress.advance(epoch_task)
                console.log(
                    f"{label} | epoch {epoch}: train_loss={train_loss:.4f} "
                    f"valid_loss={valid_loss:.4f} acc={valid_acc:.3f}"
                )
                if patience is not None and patience_counter >= patience:
                    console.log(f"{label} | early stopping at epoch {epoch}")
                    break

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        progress: Progress | None = None,
        task_id: int | None = None,
    ) -> tuple[float, float]:
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
        if training:
            self.model.train()
        else:
            self.model.eval()
        for batch in loader:
            features, targets = batch
            features = features.to(self.device)
            target_tensors = {k: v.to(self.device) for k, v in targets.items()}
            if self._amp_backend == "torch.amp":
                autocast_cm = torch.amp.autocast("cuda")
            elif self._amp_backend == "torch.cuda.amp":
                autocast_cm = torch.cuda.amp.autocast()
            else:
                autocast_cm = nullcontext()
            with autocast_cm:
                outputs = self.model(features, target_tensors["pair_idx"])
                loss, direction_logits = self._compute_loss(outputs, target_tensors)

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                if self.grad_scaler is not None:
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clip_norm)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clip_norm)
                    self.optimizer.step()

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            preds = direction_logits.argmax(dim=-1)
            total_accuracy += (preds == target_tensors["direction"]).sum().item()
            total_samples += batch_size
            if progress is not None and task_id is not None:
                progress.advance(task_id)

        return total_loss / max(total_samples, 1), total_accuracy / max(total_samples, 1)

    def _compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        loss_weights = self.cfg.model.loss_weights
        dir_logits = outputs["direction_logits"]
        weight_tensor = None
        if self.direction_weight is not None:
            weight_tensor = self.direction_weight.to(dir_logits.device)
        ce = F.cross_entropy(dir_logits, targets["direction"], weight=weight_tensor, reduction="none")
        if self.direction_gamma > 0:
            probs = torch.softmax(dir_logits, dim=-1)
            pt = probs.gather(1, targets["direction"].unsqueeze(-1)).squeeze(-1)
            focal_factor = (1.0 - pt).pow(self.direction_gamma)
            ce = ce * focal_factor
        dir_weights = 1.0 + torch.relu(targets["edge"])
        weight_norm = torch.clamp(dir_weights.sum(), min=1.0)
        direction_loss = (ce * dir_weights).sum() / weight_norm

        sample_weight = dir_weights.unsqueeze(-1)
        tp_loss = self.criterion_reg(outputs["tp"].squeeze(-1), targets["tp"])
        sl_loss = self.criterion_reg(outputs["sl"].squeeze(-1), targets["sl"])
        hold_loss = self.criterion_reg(outputs["holding"].squeeze(-1), targets["holding"])

        tp_loss = (tp_loss.unsqueeze(-1) * sample_weight).mean()
        sl_loss = (sl_loss.unsqueeze(-1) * sample_weight).mean()
        hold_loss = (hold_loss.unsqueeze(-1) * sample_weight).mean()

        total_loss = (
            loss_weights.direction * direction_loss
            + loss_weights.take_profit * tp_loss
            + loss_weights.stop_loss * sl_loss
            + loss_weights.holding_minutes * hold_loss
        )
        if "volatility_logits" in outputs and loss_weights.volatility > 0:
            vol_loss = self.criterion_aux(outputs["volatility_logits"], targets["volatility_target"])
            total_loss += loss_weights.volatility * vol_loss
        if "regime_logits" in outputs and loss_weights.regime > 0:
            regime_loss = self.criterion_aux(outputs["regime_logits"], targets["regime_target"])
            total_loss += loss_weights.regime * regime_loss
        if "edge" in outputs and loss_weights.edge > 0:
            edge_loss = self.criterion_reg(outputs["edge"].squeeze(-1), targets["edge"])
            edge_loss = (edge_loss * dir_weights).mean()
            total_loss += loss_weights.edge * edge_loss
        if loss_weights.direction_balance > 0:
            probs = torch.softmax(outputs["direction_logits"], dim=-1)
            balance_penalty = (probs[:, 0] - probs[:, 2]).mean().abs()
            total_loss += loss_weights.direction_balance * balance_penalty
        return total_loss, outputs["direction_logits"]

    def predict(self, dataset: SequenceDataset) -> pd.DataFrame:
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
        )
        self.model.eval()
        records: List[Dict[str, object]] = []
        with torch.no_grad():
            for features, targets in loader:
                meta_ids = targets["sample_id"].tolist()
                features = features.to(self.device)
                pair_idx = targets["pair_idx"].to(self.device)
                outputs = self.model(features, pair_idx)
                probs = torch.softmax(outputs["direction_logits"], dim=-1).cpu()
                tp_pred = outputs["tp"].squeeze(-1).cpu()
                sl_pred = outputs["sl"].squeeze(-1).cpu()
                hold_pred = outputs["holding"].squeeze(-1).cpu()
                direction_pred = probs.argmax(dim=-1)
                volatility_logits = outputs.get("volatility_logits")
                regime_logits = outputs.get("regime_logits")
                vol_prob = torch.softmax(volatility_logits, dim=-1).cpu() if volatility_logits is not None else None
                regime_prob = torch.softmax(regime_logits, dim=-1).cpu() if regime_logits is not None else None
                edge_pred = outputs.get("edge")
                edge_values = edge_pred.squeeze(-1).cpu() if edge_pred is not None else None

                for batch_pos, sample_id in enumerate(meta_ids):
                    pair, timestamp = dataset.sample_metadata(int(sample_id))
                    conf = float(probs[batch_pos].max().item())
                    adx_val = dataset.meta_value(pair, timestamp, "adx_14")
                    sentiment_val = dataset.meta_value(pair, timestamp, "sentiment_score")
                    sentiment_z = dataset.meta_value(pair, timestamp, "sentiment_zscore_96")
                    event_high = dataset.meta_value(pair, timestamp, "event_high_impact")
                    event_count = dataset.meta_value(pair, timestamp, "event_macro_event_count")
                    vix_z = dataset.meta_value(pair, timestamp, "vix_zscore_96")
                    session = self._session_label(timestamp)
                    pred_edge = (
                        float(max(edge_values[batch_pos].item(), 0.0)) if edge_values is not None else float("nan")
                    )
                    records.append(
                        {
                            "timestamp": timestamp,
                            "pair": pair,
                            "direction_class": int(direction_pred[batch_pos].item()),
                            "direction_prob": probs[batch_pos].tolist(),
                            "direction_conf": conf,
                            "tp_pips": float(tp_pred[batch_pos].item()),
                            "sl_pips": float(sl_pred[batch_pos].item()),
                            "holding_minutes": float(hold_pred[batch_pos].item()),
                            "adx_14": float(adx_val),
                            "session": session,
                            "edge_pred_pips": pred_edge,
                            "sentiment_score": float(sentiment_val),
                            "sentiment_zscore_96": float(sentiment_z),
                            "event_high_impact": float(event_high),
                            "event_macro_event_count": float(event_count),
                            "vix_zscore_96": float(vix_z),
                            "volatility_prob": vol_prob[batch_pos].tolist() if vol_prob is not None else None,
                            "regime_prob": regime_prob[batch_pos].tolist() if regime_prob is not None else None,
                        }
                    )
        pred_df = pd.DataFrame(records).set_index("timestamp").sort_index()
        pred_df["direction"] = pred_df["direction_class"].map(CLASS_TO_DIR)
        return pred_df

    @staticmethod
    def _session_label(ts: pd.Timestamp) -> str:
        hour = ts.tz_convert("UTC").hour
        if 23 <= hour or hour < 7:
            return "asia"
        if 7 <= hour < 16:
            return "europe"
        return "us"


class TrainingManager:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.data_pipeline = DataPipeline(cfg)

    def run(self, refresh_cache: bool = False) -> TrainingArtifacts:
        frames = self.data_pipeline.run(refresh_cache=refresh_cache)
        predictions: List[pd.DataFrame] = []
        history: List[EpochLog] = []
        model_states: List[Tuple[str, Dict[str, torch.Tensor]]] = []
        splits = list(
            walk_forward(
                frames,
                window_days=self.cfg.training.walk_forward.window_days,
                step_days=self.cfg.training.walk_forward.step_days,
            )
        )
        ensemble_members = max(1, getattr(self.cfg.training, "ensemble_members", 1))
        base_seed = getattr(self.cfg.training, "base_seed", 1337)

        if not splits:
            fallback = _build_fallback_split(frames)
            if fallback is None:
                return TrainingArtifacts(history=[], predictions=None)
            splits = [fallback]

        for split_id, split in enumerate(splits, start=1):
            normalizer = SequenceNormalizer()
            normalizer.fit(split.train)
            train_frames = normalizer.transform(split.train)
            valid_frames = normalizer.transform(split.valid)
            feature_columns = normalizer.feature_columns or []
            if not feature_columns:
                continue

            min_train_len = _min_length(train_frames)
            min_valid_len = _min_length(valid_frames)
            seq_cap = self.cfg.model.max_seq_len
            seq_len = min(seq_cap, max(min_train_len - 1, 1), max(min_valid_len - 1, 1))
            if min_train_len <= seq_len or min_valid_len <= seq_len or seq_len < 5:
                continue

            valid_ds = SequenceDataset(
                valid_frames,
                seq_len=seq_len,
                feature_columns=feature_columns,
                is_training=False,
                raw_frames=split.valid,
            )
            if len(valid_ds) == 0:
                continue

            split_preds: List[pd.DataFrame] = []
            for member_idx in range(ensemble_members):
                seed = base_seed + split_id * 100 + member_idx
                train_ds = SequenceDataset(
                    train_frames,
                    seq_len=seq_len,
                    feature_columns=feature_columns,
                    augmentation=self.cfg.training.augmentation,
                    is_training=True,
                    seed=seed,
                    raw_frames=split.train,
                    min_edge_pips=self.cfg.training.min_edge_pips,
                    flat_class_dropout=self.cfg.training.flat_class_dropout,
                )
                if len(train_ds) == 0:
                    continue
                _set_random_seed(seed)
                trainer = ModelTrainer(self.cfg, input_dim=len(feature_columns), num_pairs=len(self.cfg.pairs))
                if ensemble_members > 1:
                    desc = f"Split {split_id} | member {member_idx + 1}/{ensemble_members}"
                else:
                    desc = f"Split {split_id}"
                trainer.fit(train_ds, valid_ds, progress_desc=desc)
                history.extend(trainer.history)
                member_preds = trainer.predict(valid_ds)
                if not member_preds.empty:
                    member_preds = member_preds.copy()
                    member_preds["ensemble_member"] = member_idx + 1
                    split_preds.append(member_preds)
                if trainer.best_state_dict is not None:
                    suffix = "" if ensemble_members == 1 else f"_m{member_idx + 1}"
                    model_id = f"split_{split_id}{suffix}"
                    model_states.append((model_id, trainer.best_state_dict))

            if not split_preds:
                continue
            agg_preds = _aggregate_ensemble_predictions(split_preds)
            if agg_preds.empty:
                continue
            agg_preds["split_id"] = split_id
            agg_preds["ensemble_size"] = ensemble_members
            predictions.append(agg_preds)

        combined_preds = pd.concat(predictions).sort_index() if predictions else None
        return TrainingArtifacts(history=history, predictions=combined_preds, model_states=model_states)


def _build_fallback_split(frames: Dict[str, pd.DataFrame]) -> WalkForwardSplit | None:
    train_split: Dict[str, pd.DataFrame] = {}
    valid_split: Dict[str, pd.DataFrame] = {}
    for pair, df in frames.items():
        if len(df) < 20:
            continue
        split_idx = max(int(len(df) * 0.8), 10)
        if split_idx >= len(df):
            split_idx = len(df) - 5
        train_split[pair] = df.iloc[:split_idx]
        valid_split[pair] = df.iloc[split_idx:]
    if not train_split or not valid_split:
        return None
    start = min(v.index.min() for v in valid_split.values())
    end = max(v.index.max() for v in valid_split.values())
    return WalkForwardSplit(train=train_split, valid=valid_split, start=start, end=end)
