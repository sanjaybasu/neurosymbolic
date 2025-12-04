#!/usr/bin/env python3
"""
Fairness and usability audit scaffold for neurosymbolic evaluation outputs.

Usage:
  python 07_fairness_usability_audit.py --results-path <csv> --group-cols race,language

If the requested group columns are present, computes alert/block rates by group.
Otherwise, emits a warning that demographics are unavailable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd


def compute_group_metrics(df: pd.DataFrame, group_cols: List[str]) -> Dict:
    """Compute per-group contraindication and elevated-risk rates."""
    metrics = {}
    for col in group_cols:
        if col not in df.columns:
            metrics[col] = {"error": "column_missing"}
            continue
        grouped = []
        for group, subset in df.groupby(col):
            total = len(subset)
            contra = int((subset["contraindications_detected"] > 0).sum()) if "contraindications_detected" in subset else 0
            risk = int((subset.get("has_elevated_risk", False)).sum()) if "has_elevated_risk" in subset else 0
            grouped.append(
                {
                    "group": group,
                    "n": total,
                    "contra_rate": contra / total if total else 0,
                    "risk_rate": risk / total if total else 0,
                }
            )
        metrics[col] = grouped
    return metrics


def main(args):
    results_path = Path(args.results_path)
    df = pd.read_csv(results_path)

    if args.group_cols:
        group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    else:
        group_cols = []

    has_groups = any(col in df.columns for col in group_cols)
    output = {
        "source": str(results_path),
        "n_rows": len(df),
        "group_columns_requested": group_cols,
        "group_columns_present": [c for c in group_cols if c in df.columns],
    }

    if has_groups:
        output["group_metrics"] = compute_group_metrics(df, group_cols)
    else:
        output["group_metrics"] = "no_requested_group_columns_present"

    # Simple usability proxy: explanation length distribution if present
    if "explanation_length" in df.columns:
        output["explanation_length_mean"] = float(df["explanation_length"].mean())
        output["explanation_length_median"] = float(df["explanation_length"].median())

    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fairness/usability audit scaffold.")
    parser.add_argument("--results-path", type=str, required=True, help="CSV of evaluation outputs")
    parser.add_argument("--group-cols", type=str, default="", help="Comma-separated demographic columns to stratify")
    main(parser.parse_args())
