#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize_predictions(pred_path: Path, min_conf: float) -> None:
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    df = pd.read_parquet(pred_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    df = df.sort_index()
    if min_conf > 0 and "direction_conf" in df.columns:
        before = len(df)
        df = df[df["direction_conf"] >= min_conf]
        print(f"Applied confidence filter {min_conf:.2f}: {before} -> {len(df)} rows")
    print(f"Total rows: {len(df):,}")
    if len(df) == 0:
        return
    pivot = df.groupby("pair")
    for pair, group in pivot:
        print(f"\nPair: {pair}")
        counts = group["direction_class"].value_counts().to_dict()
        print(" direction counts:", counts)
        print(" mean conf:", group.get("direction_conf", pd.Series()).mean())
        print(" avg tp/sl:", group["tp_pips"].mean(), group["sl_pips"].mean())
        if "sentiment_zscore_96" in group.columns:
            print(
                " sentiment z range:",
                float(group["sentiment_zscore_96"].min()),
                float(group["sentiment_zscore_96"].max()),
            )
    conf = df.get("direction_conf")
    if conf is not None:
        print("\nConfidence quartiles:\n", conf.quantile([0.25, 0.5, 0.75]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze cached validation predictions")
    parser.add_argument("--pred-path", default="data/cache/predictions.parquet")
    parser.add_argument("--min-conf", type=float, default=0.0)
    args = parser.parse_args()
    summarize_predictions(Path(args.pred_path), args.min_conf)


if __name__ == "__main__":
    main()
