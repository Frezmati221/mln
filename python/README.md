# Intraday FX Trade Generator

This project scaffolds a multi-task AI pipeline that issues intraday trade instructions
(direction, stop-loss, take-profit) for the `EURUSD`, `USDJPY`, and `GBPUSD` forex pairs
on 5-minute bars. The system is designed to stay flat by the daily session close and
optimize TP/SL placement based on current microstructure and regime context.

## Key Capabilities

1. **Data ingest & labeling** – Loads multi-source market data, aligns it on 5-minute
   intervals, and derives optimal intraday TP/SL targets under cost constraints.
2. **Feature engineering** – Produces a rich technical, microstructure, and macro feature
   set while encoding session and regime characteristics.
3. **Multi-task modeling** – Shares a temporal encoder across heads that predict trade
   direction, TP distance, SL distance, and expected holding time.
4. **Backtesting & risk** – Simulates realistic execution with slippage/spread,
   enforces intraday flat rules, and reports trading KPIs.

## Repository Layout

```
python/
├── requirements.txt        # Python dependencies
├── README.md               # This guide
└── src/
    └── fx_intraday_ai/     # Source package (data, models, training, backtests)
```

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you prefer not to install the package, export `PYTHONPATH=src` before invoking the CLI.

Detailed usage instructions and configuration options are documented inline within the
package modules.

## Downloading Historical Data

Use the built-in downloader to fetch Dukascopy tick data and aggregate it into 5-minute
bars compatible with the rest of the pipeline:

```bash
python -m fx_intraday_ai.cli download-data \
  --start 2022-01-01 \
  --end 2023-01-01 \
  --pair EURUSD --pair USDJPY --pair GBPUSD
```

The parquet output defaults to `data/raw/prices.parquet` (from `configs/default.yaml`).
Override it with `--output path/to/prices.parquet` or change the bar size via
`--timeframe 15`.

## Training → Backtest Workflow

1. **Train & save artifacts**
   ```bash
   python -m fx_intraday_ai.cli train --config configs/default.yaml
   ```
   This command now:
   - Writes validation predictions to `data/cache/predictions.parquet`
   - Stores per-split checkpoints under `artifacts/models/`
2. **Backtest without retraining**
   ```bash
   python -m fx_intraday_ai.cli backtest --config configs/default.yaml
   ```
   By default it loads `data/cache/predictions.parquet`. Pass `--retrain` if you want to
   refresh predictions before simulating. You can also override paths via
   `--predictions-path` on both commands.

### Edge-aware filtering

The transformer now learns to predict the expected edge (in pips) for every trade
candidate. During backtests we ignore signals whose predicted edge is below
`backtest.min_edge_pips` (default `3.0`) or a custom `--min-edge` CLI override. This
keeps the model focused on imbalance regimes where the TP/SL grid historically offered
positive expectancy, which materially improves hit rate versus taking every signal.

### High-edge curriculum

Low-edge timestamps tend to label as flat, which overwhelms the direction head. You can
now focus training on the most actionable slices via `training.min_edge_pips` (skip
examples whose historical reward was weaker than the threshold) and
`training.flat_class_dropout` (probability of discarding flat labels during training).
Together these act as a curriculum that doubles down on favorable opportunities and
materially improves walk-forward win rate without adding latency at inference.

## Quick Status Check

Need to know what to do next? Run:

```bash
python -m fx_intraday_ai.cli status
```

The command inspects your price data, cached predictions, and model checkpoints, and
prints concrete next steps (e.g., run `download-data`, `train`, or `backtest`).

## Instant Signals (Experimental)

Need current instructions without running the full pipeline? Use the new inference
command. It pulls the latest cached bars, rebuilds indicators, loads your most recent
checkpoint, and prints direction/TP/SL per pair:

```bash
python -m fx_intraday_ai.cli infer \
  --config configs/default.yaml \
  --model-path artifacts/models/split_3.pt   # optional
```

If `--model-path` is omitted, the CLI uses the newest `split_*.pt` in `artifacts/models`.
Override `--seq-len` to control how many recent bars feed the transformer.
