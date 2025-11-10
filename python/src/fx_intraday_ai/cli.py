from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict

import click
import pandas as pd
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, TextColumn

from .utils.config import load_config
from .data.pipeline import DataPipeline
from .data.loader import load_price_frames
from .training.dataset import SequenceNormalizer
from .training.trainer import TrainingManager, CLASS_TO_DIR
from .backtest.simulator import Backtester
from .data.dukascopy import DukascopyDownloader
from .models.multitask_transformer import MultiTaskTransformer

console = Console()


def _latest_checkpoint(model_dir: Path) -> Optional[Path]:
    if not model_dir.exists():
        return None
    checkpoints = list(model_dir.glob("split_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


@click.group(help="Intraday FX AI orchestration CLI.")
def cli() -> None:
    pass


@cli.command("prepare", help="Validate data sources and compute feature snapshots.")
@click.option(
    "--config",
    "config_path",
    default="configs/default.yaml",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--refresh-cache", is_flag=True, help="Force rebuild of cached price data.")
def prepare_command(config_path: str, refresh_cache: bool) -> None:
    cfg = load_config(Path(config_path))
    pipeline = DataPipeline(cfg)
    frames = pipeline.run(refresh_cache=refresh_cache)
    table = Table(title="Prepared Feature Frames")
    table.add_column("Pair")
    table.add_column("Rows")
    table.add_column("Start", overflow="fold")
    table.add_column("End", overflow="fold")
    for pair, df in frames.items():
        table.add_row(pair, f"{len(df):,}", str(df.index.min()), str(df.index.max()))
    console.print(table)


@cli.command("train", help="Train the multi-task model with walk-forward validation.")
@click.option(
    "--config",
    "config_path",
    default="configs/default.yaml",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--refresh-cache", is_flag=True, help="Force rebuild of cached price data.")
@click.option(
    "--predictions-path",
    type=click.Path(dir_okay=False),
    default="data/cache/predictions.parquet",
    show_default=True,
    help="Path to store validation predictions for reuse.",
)
@click.option(
    "--model-dir",
    type=click.Path(file_okay=False),
    default="artifacts/models",
    show_default=True,
    help="Directory to save trained model checkpoints.",
)
def train_command(
    config_path: str,
    refresh_cache: bool,
    predictions_path: str,
    model_dir: str,
) -> None:
    cfg = load_config(Path(config_path))
    manager = TrainingManager(cfg)
    artifacts = manager.run(refresh_cache=refresh_cache)
    if artifacts.history:
        last = artifacts.history[-1]
        console.print(f"[green]Training complete[/green] (epoch {last.epoch})")
    else:
        console.print("[yellow]Training skipped. Verify walk-forward windows.[/yellow]")
    if artifacts.predictions is not None:
        console.print(f"Generated {len(artifacts.predictions):,} validation predictions.")
        pred_path = Path(predictions_path)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        save_df = artifacts.predictions.reset_index()
        save_df.to_parquet(pred_path, index=False)
        console.print(f"Predictions saved to {pred_path}")
    if artifacts.model_states:
        model_dir_path = Path(model_dir)
        model_dir_path.mkdir(parents=True, exist_ok=True)
        for model_id, state_dict in artifacts.model_states:
            model_file = model_dir_path / f"{model_id}.pt"
            torch.save(state_dict, model_file)
        console.print(f"Saved {len(artifacts.model_states)} checkpoint(s) to {model_dir_path}")


@cli.command("backtest", help="Train and backtest the validation predictions.")
@click.option(
    "--config",
    "config_path",
    default="configs/default.yaml",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--refresh-cache", is_flag=True, help="Force rebuild of cached price data.")
@click.option(
    "--predictions-path",
    type=click.Path(dir_okay=False),
    default="data/cache/predictions.parquet",
    show_default=True,
    help="Path to load cached validation predictions.",
)
@click.option(
    "--retrain",
    is_flag=True,
    help="Force retraining even if cached predictions exist.",
)
@click.option(
    "--min-confidence",
    type=float,
    default=0.0,
    show_default=True,
    help="Minimum direction confidence required to include a signal in backtest.",
)
def backtest_command(
    config_path: str,
    refresh_cache: bool,
    predictions_path: str,
    retrain: bool,
    min_confidence: float,
) -> None:
    cfg = load_config(Path(config_path))
    pred_path = Path(predictions_path)
    predictions: Optional[pd.DataFrame] = None

    if not retrain and pred_path.exists():
        predictions = pd.read_parquet(pred_path)
        predictions = _ensure_timestamp_index(predictions)
        console.print(f"Loaded predictions from {pred_path}")
    else:
        manager = TrainingManager(cfg)
        artifacts = manager.run(refresh_cache=refresh_cache)
        if artifacts.predictions is None or artifacts.predictions.empty:
            console.print("[red]No predictions available for backtest.[/red]")
            raise SystemExit(1)
        predictions = artifacts.predictions.copy()
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.reset_index().to_parquet(pred_path, index=False)
        console.print(f"Predictions saved to {pred_path}")
    predictions = predictions.copy()
    predictions = _ensure_timestamp_index(predictions)
    if min_confidence > 0:
        if "direction_conf" not in predictions.columns:
            predictions["direction_conf"] = predictions["direction_prob"].apply(lambda probs: max(probs) if isinstance(probs, list) else 1.0)
        before = len(predictions)
        predictions = predictions[predictions["direction_conf"] >= min_confidence]
        console.print(f"Applied confidence filter {min_confidence}; {before}->{len(predictions)} rows.")

    pipeline = DataPipeline(cfg)
    price_frames = pipeline.load_prices(refresh_cache=refresh_cache)
    backtester = Backtester(cfg, price_frames)
    try:
        trades = backtester.simulate(predictions)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc
    summary = backtester.summarize(trades)

    table = Table(title="Backtest Summary")
    table.add_column("Scope")
    table.add_column("Trades")
    table.add_column("Win %")
    table.add_column("Avg PnL (pips)")
    table.add_column("Total PnL")
    table.add_column("Profit Factor")
    table.add_column("Sharpe-like")
    table.add_column("Max Drawdown")
    for scope, metrics in summary.items():
        table.add_row(
            scope,
            f"{metrics['trades']:.0f}",
            f"{metrics['win_rate']*100:.1f}",
            f"{metrics['avg_pnl']:.2f}",
            f"{metrics['pnl_total']:.2f}",
            f"{metrics['profit_factor']:.2f}",
            f"{metrics['sharpe_like']:.2f}",
            f"{metrics['max_drawdown']:.2f}",
        )
    console.print(table)


def _ensure_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    elif df.index.name != "timestamp":
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True)
    return df


@cli.command("download-data", help="Download historical FX data from Dukascopy.")
@click.option(
    "--config",
    "config_path",
    default="configs/default.yaml",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--start", "start_date", required=True, help="UTC start datetime (e.g., 2018-01-01).")
@click.option("--end", "end_date", required=True, help="UTC end datetime (e.g., 2020-12-31).")
@click.option(
    "--pair",
    "pair_overrides",
    multiple=True,
    help="Override the configured FX pairs. Provide multiple --pair values for more than one instrument.",
)
@click.option(
    "--timeframe",
    type=int,
    default=None,
    help="Override bar timeframe in minutes (defaults to config value).",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Custom parquet destination (defaults to config.data.price_path).",
)
def download_data_command(
    config_path: str,
    start_date: str,
    end_date: str,
    pair_overrides: tuple[str, ...],
    timeframe: int | None,
    output_path: str | None,
) -> None:
    cfg = load_config(Path(config_path))
    pairs = list(pair_overrides) if pair_overrides else cfg.pairs
    tf_minutes = timeframe or cfg.timeframe_minutes
    target_path = Path(output_path) if output_path else cfg.data.price_path
    downloader = DukascopyDownloader(timeframe_minutes=tf_minutes, price_path=target_path)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}h"),
        TimeElapsedColumn(),
    )
    task_map: dict[str, int] = {}

    with progress:
        for pair in pairs:
            task_map[pair.upper()] = progress.add_task(f"[cyan]{pair.upper()}[/cyan]", total=1)

        def _progress_cb(pair: str, completed: int, total: int) -> None:
            task_id = task_map[pair]
            progress.update(task_id, total=total, completed=completed)

        df = downloader.download(pairs, start_date, end_date, progress_cb=_progress_cb)

    console.print(
        f"[green]Downloaded[/green] {len(df):,} bars across {len(pairs)} pairs "
        f"into {target_path}"
    )


@cli.command("status", help="Show current project state and suggested next step.")
@click.option(
    "--config",
    "config_path",
    default="configs/default.yaml",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--predictions-path",
    default="data/cache/predictions.parquet",
    show_default=True,
    type=click.Path(dir_okay=False),
)
@click.option(
    "--model-dir",
    default="artifacts/models",
    show_default=True,
    type=click.Path(file_okay=False),
)
def status_command(config_path: str, predictions_path: str, model_dir: str) -> None:
    cfg = load_config(Path(config_path))
    suggestions: List[str] = []

    # Data coverage
    price_path = cfg.data.price_path
    data_rows = 0
    if price_path.exists():
        try:
            df = pd.read_parquet(price_path, columns=["timestamp", "pair"])
            data_rows = len(df)
            start = df["timestamp"].min()
            end = df["timestamp"].max()
            console.print(
                f"[cyan]Data[/cyan]: {price_path} ({data_rows:,} rows, {start} → {end})"
            )
            if data_rows < 5000:
                suggestions.append("Download more history via download-data (current sample is tiny).")
        except Exception as exc:  # pragma: no cover - diagnostic only
            console.print(f"[red]Failed to read price data:[/red] {exc}")
            suggestions.append("Fix price parquet or re-run download-data.")
    else:
        console.print(f"[red]Data file missing:[/red] {price_path}")
        suggestions.append("Run download-data to populate price history.")

    # Predictions
    pred_path = Path(predictions_path)
    if pred_path.exists():
        try:
            preds = pd.read_parquet(pred_path, columns=["pair"])
            console.print(
                f"[cyan]Predictions[/cyan]: {pred_path} ({len(preds):,} rows)"
            )
        except Exception as exc:  # pragma: no cover - diagnostic only
            console.print(f"[red]Failed to read predictions:[/red] {exc}")
            suggestions.append("Re-run train to regenerate predictions.")
    else:
        console.print(f"[yellow]Predictions not found[/yellow]: {pred_path}")
        suggestions.append("Run train to generate predictions and checkpoints.")

    # Model checkpoints
    model_dir_path = Path(model_dir)
    checkpoints = list(model_dir_path.glob("split_*.pt")) if model_dir_path.exists() else []
    if checkpoints:
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        console.print(
            f"[cyan]Models[/cyan]: {len(checkpoints)} checkpoint(s) in {model_dir_path}, "
            f"latest {latest.name}"
        )
    else:
        console.print(f"[yellow]No checkpoints in[/yellow] {model_dir_path}")
        suggestions.append("Run train to export model weights for real-time use.")

    # Suggestions summary
    if suggestions:
        console.print("\n[bold]Next steps:[/bold]")
        for item in dict.fromkeys(suggestions):
            console.print(f"- {item}")
    else:
        console.print("All artifacts present. You can backtest or wire real-time inference.")


@cli.command("infer", help="Generate latest trade instructions for all configured pairs.")
@click.option(
    "--config",
    "config_path",
    default="configs/default.yaml",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Checkpoint to use. Defaults to most recent split_*.pt.",
)
@click.option("--seq-len", type=int, default=None, help="Override sequence length for inference.")
@click.option("--refresh-cache", is_flag=True, help="Regenerate cached prices before inference.")
def infer_command(config_path: str, model_path: Optional[str], seq_len: Optional[int], refresh_cache: bool) -> None:
    cfg = load_config(Path(config_path))
    pipeline = DataPipeline(cfg)
    price_frames = pipeline.load_prices(refresh_cache=refresh_cache)
    feature_frames = pipeline.build_feature_frames(price_frames)
    frames = {pair: frame.data.dropna().copy() for pair, frame in feature_frames.items() if not frame.data.empty}
    if not frames:
        console.print("[red]No feature data available. Ensure price history exists.[/red]")
        raise SystemExit(1)

    normalizer = SequenceNormalizer()
    normalizer.fit(frames)
    norm_frames = normalizer.transform(frames)
    feature_cols = normalizer.feature_columns or []
    if not feature_cols:
        console.print("[red]No features detected for inference.[/red]")
        raise SystemExit(1)

    checkpoint_path: Optional[Path]
    if model_path:
        checkpoint_path = Path(model_path)
    else:
        checkpoint_path = _latest_checkpoint(Path("artifacts/models"))
    if checkpoint_path is None or not checkpoint_path.exists():
        console.print("[red]No checkpoint found. Run train first or provide --model-path.[/red]")
        raise SystemExit(1)

    model = MultiTaskTransformer(input_dim=len(feature_cols), cfg=cfg.model, num_pairs=len(cfg.pairs))
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    seq_cap = seq_len or cfg.model.max_seq_len
    table = Table(title="Latest Signals")
    table.add_column("Pair")
    table.add_column("Timestamp")
    table.add_column("Direction")
    table.add_column("Confidence", justify="right")
    table.add_column("TP (pips)", justify="right")
    table.add_column("SL (pips)", justify="right")
    table.add_column("Hold (min)", justify="right")

    dir_text = {1: "LONG", -1: "SHORT", 0: "FLAT"}
    produced = 0
    for pair_idx, pair in enumerate(cfg.pairs):
        df = norm_frames.get(pair)
        if df is None or df.empty:
            console.print(f"[yellow]Skipping {pair}: insufficient data.[/yellow]")
            continue
        effective_len = min(seq_cap, len(df))
        if effective_len < 5:
            console.print(f"[yellow]Skipping {pair}: need at least 5 rows (got {len(df)}).[/yellow]")
            continue
        window = torch.from_numpy(df[feature_cols].to_numpy(dtype=np.float32)[-effective_len:])
        window = window.unsqueeze(0)
        pair_tensor = torch.tensor([pair_idx], dtype=torch.long)
        with torch.no_grad():
            outputs = model(window, pair_tensor)
            logits = outputs["direction_logits"].squeeze(0)
            probs = torch.softmax(logits, dim=0)
        dir_class = int(torch.argmax(probs).item())
        direction = CLASS_TO_DIR.get(dir_class, 0)
        conf = float(torch.max(probs).item())
        tp = float(outputs["tp"].squeeze().item())
        sl = float(outputs["sl"].squeeze().item())
        hold = float(outputs["holding"].squeeze().item())
        ts = df.index[-1]
        table.add_row(
            pair,
            str(ts),
            dir_text.get(direction, "FLAT"),
            f"{conf:.2f}",
            f"{tp:.2f}",
            f"{sl:.2f}",
            f"{hold:.1f}",
        )
        produced += 1

    if produced == 0:
        console.print("[red]No valid signals generated. Check data availability.[/red]")
        raise SystemExit(1)
    console.print(f"Using checkpoint {checkpoint_path}")
    console.print(table)

if __name__ == "__main__":
    cli()
