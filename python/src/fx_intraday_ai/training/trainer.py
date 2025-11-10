from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch import nn
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
    model_states: List[Tuple[int, Dict[str, torch.Tensor]]] = field(default_factory=list)


class ModelTrainer:
    def __init__(self, cfg: ExperimentConfig, input_dim: int, num_pairs: int):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiTaskTransformer(input_dim=input_dim, cfg=cfg.model, num_pairs=num_pairs).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.mixed_precision and torch.cuda.is_available())
        self.criterion_dir = nn.CrossEntropyLoss()
        self.criterion_reg = nn.SmoothL1Loss(reduction="none")
        self.history: List[EpochLog] = []
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.best_metric: float = float("inf")

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
                progress.advance(epoch_task)
                console.log(
                    f"{label} | epoch {epoch}: train_loss={train_loss:.4f} "
                    f"valid_loss={valid_loss:.4f} acc={valid_acc:.3f}"
                )

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
            with torch.cuda.amp.autocast(enabled=self.cfg.training.mixed_precision and torch.cuda.is_available()):
                outputs = self.model(features, target_tensors["pair_idx"])
                loss, direction_logits = self._compute_loss(outputs, target_tensors)

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                self.grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clip_norm)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

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
        direction_loss = self.criterion_dir(outputs["direction_logits"], targets["direction"])

        sample_weight = (1.0 + torch.relu(targets["edge"])).unsqueeze(-1)
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

                for batch_pos, sample_id in enumerate(meta_ids):
                    pair, timestamp = dataset.sample_metadata(int(sample_id))
                    conf = float(probs[batch_pos].max().item())
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
                        }
                    )
        pred_df = pd.DataFrame(records).set_index("timestamp").sort_index()
        pred_df["direction"] = pred_df["direction_class"].map(CLASS_TO_DIR)
        return pred_df


class TrainingManager:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.data_pipeline = DataPipeline(cfg)

    def run(self, refresh_cache: bool = False) -> TrainingArtifacts:
        frames = self.data_pipeline.run(refresh_cache=refresh_cache)
        predictions: List[pd.DataFrame] = []
        history: List[EpochLog] = []
        model_states: List[Tuple[int, Dict[str, torch.Tensor]]] = []
        splits = list(
            walk_forward(
                frames,
                window_days=self.cfg.training.walk_forward.window_days,
                step_days=self.cfg.training.walk_forward.step_days,
            )
        )

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

            train_ds = SequenceDataset(train_frames, seq_len=seq_len, feature_columns=feature_columns)
            valid_ds = SequenceDataset(valid_frames, seq_len=seq_len, feature_columns=feature_columns)
            if len(train_ds) == 0 or len(valid_ds) == 0:
                continue

            trainer = ModelTrainer(self.cfg, input_dim=len(feature_columns), num_pairs=len(self.cfg.pairs))
            desc = f"Split {split_id}"
            trainer.fit(train_ds, valid_ds, progress_desc=desc)
            history.extend(trainer.history)
            predictions.append(trainer.predict(valid_ds).assign(split_id=split_id))
            if trainer.best_state_dict is not None:
                model_states.append((split_id, trainer.best_state_dict))

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
