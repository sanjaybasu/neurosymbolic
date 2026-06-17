"""
Build mixed-source train/val/test splits for physician-created vignettes and real-world population health notes.

Inputs:
  - Vignette CSV: notebooks/rl_vs_llm_safety/data/scenario_library.csv (n=432)
  - Real-world JSON splits: notebooks/rl_vs_llm_safety_v2/data_final_v3/realworld_{train,val,test}.json (700/150/150)

Strategy:
  - Stratify vignettes to ~42/10/48% (train/val/test) to yield ~200 held-out physician cases.
  - Use provided real-world splits (train/val/test) without downsampling.
  - Save mixed splits under packaging/neurosymbolic_github/data/mixed_splits/:
      mixed_train.json, mixed_val.json, mixed_physician_test.json, mixed_realworld_val.json, mixed_realworld_test.json

Notes:
  - Assumes fields: prompt/text, hazard_type or hazard_category, severity (if available), label 1=hazard/0=benign.
  - No data leakage: physician test and real-world val/test remain disjoint from training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "notebooks" / "rl_vs_llm_safety" / "data"
# Use the full real-world set (n=1000) with 700/150/150 train/val/test splits.
RW_DIR = REPO_ROOT / "notebooks" / "rl_vs_llm_safety_v2" / "data_final_v3"
OUT_DIR = REPO_ROOT / "packaging" / "neurosymbolic_github" / "data" / "mixed_splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Target splits: physician ~ 42/10/48% (train/val/test) to yield ~200 held-out,
# real-world 700/150/150 (train/val/test) from provided files.
V_TRAIN_FRAC = 0.42
V_VAL_FRAC = 0.10
V_TEST_FRAC = 0.48


def load_vignettes() -> List[Dict]:
    df = pd.read_csv(DATA_DIR / "scenario_library.csv")
    records = []
    for _, row in df.iterrows():
        hazard = row["category"] if row["severity"] != "none" else "benign"
        records.append(
            {
                "text": row["prompt"],
                "label": 0 if hazard == "benign" else 1,
                "hazard_type": hazard,
                "severity": row.get("severity", "none"),
                "source": "vignette",
            }
        )
    return records


def stratified_split(records: List[Dict], train_size=0.7, val_size=0.15, test_size=0.15, seed=42):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6
    labels = np.array([r["hazard_type"] for r in records])
    indices = np.arange(len(records))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=(val_size + test_size), random_state=seed)
    train_idx, tmp_idx = next(sss.split(indices, labels))
    tmp_labels = labels[tmp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size / (val_size + test_size), random_state=seed)
    val_idx, test_idx = next(sss2.split(tmp_idx, tmp_labels))
    return (
        [records[i] for i in train_idx],
        [records[tmp_idx[i]] for i in val_idx],
        [records[tmp_idx[i]] for i in test_idx],
    )


def load_realworld(split: str) -> List[Dict]:
    with open(RW_DIR / f"realworld_{split}.json", "r") as f:
        data = json.load(f)
    records = []
    for row in data:
        records.append(
            {
                "text": row.get("message", ""),
                "label": 0 if row.get("detection_truth") == 0 else 1,
                "hazard_type": row.get("hazard_category", "unknown"),
                "severity": row.get("severity", "unknown"),
                "age": row.get("age"),
                "sex": row.get("sex"),
                "race_ethnicity": row.get("race_ethnicity"),
                "source": "realworld",
            }
        )
    return records


def main():
    vignettes = load_vignettes()
    v_train, v_val, v_test = stratified_split(
        vignettes, train_size=V_TRAIN_FRAC, val_size=V_VAL_FRAC, test_size=V_TEST_FRAC
    )

    rw_train = load_realworld("train")  # n=700
    rw_val = load_realworld("val")      # n=150
    rw_test = load_realworld("test")    # n=150

    mixed_train = v_train + rw_train
    mixed_val = v_val + rw_val

    with open(OUT_DIR / "mixed_train.json", "w") as f:
        json.dump(mixed_train, f, indent=2)
    with open(OUT_DIR / "mixed_val.json", "w") as f:
        json.dump(mixed_val, f, indent=2)
    with open(OUT_DIR / "mixed_physician_test.json", "w") as f:
        json.dump(v_test, f, indent=2)
    with open(OUT_DIR / "mixed_realworld_val.json", "w") as f:
        json.dump(rw_val, f, indent=2)
    with open(OUT_DIR / "mixed_realworld_test.json", "w") as f:
        json.dump(rw_test, f, indent=2)

    print(f"Saved mixed splits to {OUT_DIR}")
    print(f"Train: {len(mixed_train)} (vignettes {len(v_train)}, real-world {len(rw_train)})")
    print(f"Val:   {len(mixed_val)} (vignettes {len(v_val)}, real-world {len(rw_val)})")
    print(f"Physician test: {len(v_test)} | Real-world val/test: {len(rw_val)}/{len(rw_test)}")


if __name__ == "__main__":
    main()
