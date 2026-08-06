#!/usr/bin/env python3
"""
Convert focused rolling row-level predictions into long-only nested prediction JSONL.

Recommended first submission setting:
- target_name = rl_long_current_pnl
- models = RandomForest, LightGBM
- prediction = signal_score
- confidence = 1.0 neutral placeholder
- direction = long if signal_score >= 0.90 else no_trade

Notes:
- IndexReference is preserved from the original model-ready dataset and is written as `index`.
- signal_score is a training-window prediction-distribution percentile, not an exact test-set top-10% rank.
- Low long scores are treated as no_trade, not short, because rl.long.current_pnl is a long-side target.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


LABEL_MAP = {
    "rl_long_current_pnl": "rl.long.current_pnl",
    "rl_long_action_quality": "rl.long.action_quality",
    "rl_long_action_binary": "rl.long.action_label",
    "rl_expert_action": "rl.expert_action",
    "pi_hindsight_entry_long": "pi_hindsight_entry_long",
    "pi_hindsight_entry_positive": "pi_hindsight_entry_long.positive",
    "pi_hindsight_entry_original": "pi_hindsight_entry_long.original_entry_binary",
    "pi_hindsight_entry_6bins": "pi_hindsight_entry_long.6bins",
}


def safe_float(x):
    if pd.isna(x):
        return None
    try:
        value = float(x)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def iso_z(ts):
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(t):
        return None
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def read_predictions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def export_long_only_jsonl(
    df: pd.DataFrame,
    output_path: Path,
    target_name: str,
    model_name: str,
    threshold: float,
    experiment_id: str,
    split: str,
    include_actual: bool,
) -> tuple[int, int]:
    sub = df[(df["target_name"] == target_name) & (df["model_name"] == model_name)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for target_name={target_name}, model_name={model_name}")

    sub = sub.sort_values(["attr__timestamp", "IndexReference"]).reset_index(drop=True)

    ticker_values = sorted(sub["ticker"].dropna().astype(str).unique().tolist())
    ticker_str = "_".join(ticker_values) if len(ticker_values) <= 3 else f"{len(ticker_values)}tickers"
    dotted_label = LABEL_MAP.get(target_name, target_name)

    metadata = {
        "prediction_schema_version": "1.0",
        "experiment_id": experiment_id,
        "split": split,
        "model_id": f"{ticker_str}_{model_name}_{target_name}_long_only_signal_score_{threshold:.2f}",
        "model_type": model_name.lower(),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "training_design": "single-ticker rolling 3-year train window with next-year test window",
        "target_name": target_name,
        "label": dotted_label,
        "prediction_field": "signal_score",
        "raw_prediction_field": "prediction_score",
        "confidence_rule": "confidence is set to 1.0 as a neutral placeholder because the benchmark model does not produce an independent confidence estimate",
        "direction_rule": f"long if signal_score >= {threshold:.2f}; otherwise no_trade",
        "notes": "Long-only benchmark submission. Low-ranked long scores are treated as no_trade, not short. signal_score is based on the training-window prediction distribution and is not an exact test-set top-decile rank."
    }

    long_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"section": "metadata", "metadata": metadata}, ensure_ascii=False) + "\n")

        for _, r in sub.iterrows():
            sig = safe_float(r.get("signal_score"))
            raw = safe_float(r.get("prediction_score"))
            direction = "long" if (sig is not None and sig >= threshold) else "no_trade"
            long_count += int(direction == "long")

            pred_obj = {
                "label": dotted_label,
                "prediction": sig,
                "confidence": 1.0,
                "raw_prediction": raw,
                "signal_score": sig,
                "direction": direction,
            }
            if include_actual:
                pred_obj["actual_value"] = safe_float(r.get("y_true"))

            record = {
                "section": "prediction",
                "data": {
                    "index": int(r["IndexReference"]),
                    "ticker": str(r["ticker"]),
                    "timestamp": iso_z(r["attr__timestamp"]),
                    "timeframe": "d",
                    "predictions": {
                        "trade.long": pred_obj
                    },
                    "model_metrics": {
                        "model_name": str(r["model_name"]),
                        "target_name": str(r["target_name"]),
                        "task": str(r["task"]),
                        "feature_set": str(r["feature_set"]),
                        "walkforward_scheme": str(r["walkforward_scheme"]),
                        "fold_id": str(r["fold_id"]),
                        "train_start_year": int(r["train_start_year"]),
                        "train_end_year": int(r["train_end_year"]),
                        "test_year": int(r["test_year"]),
                    },
                },
            }
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    return len(sub), long_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to row-level predictions CSV or parquet")
    parser.add_argument("--output-dir", default="prediction_jsonl_exports")
    parser.add_argument("--target", default="rl_long_current_pnl")
    parser.add_argument("--models", nargs="+", default=["RandomForest", "LightGBM"])
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--experiment-id", default="ROLLING-SINGLE-TICKER")
    parser.add_argument("--split", default="test")
    parser.add_argument("--include-actual", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    df = read_predictions(input_path)

    for model_name in args.models:
        ticker_values = sorted(df["ticker"].dropna().astype(str).unique().tolist())
        ticker_str = "_".join(ticker_values) if len(ticker_values) <= 3 else f"{len(ticker_values)}tickers"
        output_path = output_dir / f"{ticker_str}_{model_name}_{args.target}_long_only_signal_score_{args.threshold:.2f}.jsonl"
        n_rows, n_long = export_long_only_jsonl(
            df=df,
            output_path=output_path,
            target_name=args.target,
            model_name=model_name,
            threshold=args.threshold,
            experiment_id=args.experiment_id,
            split=args.split,
            include_actual=args.include_actual,
        )
        print(f"Saved {output_path} ({n_rows} rows, {n_long} long signals)")


if __name__ == "__main__":
    main()
