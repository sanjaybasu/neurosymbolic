#!/usr/bin/env python3
"""
Binary harm vs. benign evaluation using TF–IDF + logistic regression.

Why: the prospective evaluation CSVs lack reliable harm_type annotations, which
breaks the multiclass hazard detector. This script trains a binary classifier on
the physician scenarios plus prospective data (with an 80/20 stratified split)
and reports held-out performance.
"""

from __future__ import annotations

import json
import sys
import math
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data_utils import parse_multiline_csv


BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = Path(__file__).resolve().parents[3]
RL_DATA_DIR = REPO_ROOT / "notebooks" / "rl_vs_llm_safety" / "data"
SCENARIO_LIBRARY = RL_DATA_DIR / "scenario_library.csv"
PROSPECTIVE_DIR = RL_DATA_DIR / "prospective_eval"
BENIGN_CASES = PROSPECTIVE_DIR / "benign_cases_500.csv"
HARM_CASES = PROSPECTIVE_DIR / "harm_cases_500.csv"

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "trained_models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_prospective_binary() -> pd.DataFrame:
    """Load prospective benign/harm cases and assign binary labels."""
    benign_df = parse_multiline_csv(BENIGN_CASES, record_prefixes=["benign_candidate"])
    benign_df["label"] = 0
    benign_df["text"] = benign_df["context_text"]

    harm_df = parse_multiline_csv(HARM_CASES, record_prefixes=["harm_candidate"])
    harm_df["label"] = 1
    harm_df["text"] = harm_df["context_text"]
    return pd.concat([benign_df, harm_df], ignore_index=True)[["text", "label"]]


def load_physician_scenarios() -> pd.DataFrame:
    """Load physician scenarios and collapse to binary harm/benign labels."""
    df = pd.read_csv(SCENARIO_LIBRARY)
    df["label"] = (df["category"] != "benign").astype(int)
    df["text"] = df["prompt"]
    return df[["text", "label"]]


def train_and_evaluate(seed: int = 42) -> Tuple[Pipeline, Dict]:
    """Train binary classifier and return fitted pipeline + metrics."""
    # Assemble dataset
    physician_df = load_physician_scenarios()
    prospective_df = load_prospective_binary()
    dataset = pd.concat([physician_df, prospective_df], ignore_index=True)

    train_df, test_df = train_test_split(
        dataset,
        test_size=0.2,
        random_state=seed,
        stratify=dataset["label"],
    )

    pipeline: Pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=15000,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    pipeline.fit(train_df["text"], train_df["label"])
    probas = pipeline.predict_proba(test_df["text"])[:, 1]
    preds = (probas >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_df["label"], preds, average="binary"
    )
    acc = accuracy_score(test_df["label"], preds)
    auc = roc_auc_score(test_df["label"], probas)
    pr_auc = average_precision_score(test_df["label"], probas)
    brier = brier_score_loss(test_df["label"], probas)
    tn, fp, fn, tp = confusion_matrix(test_df["label"], preds).ravel()

    def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
        if n == 0:
            return (0.0, 0.0)
        z = 1.96  # 95% approx
        p = k / n
        denom = 1 + z**2 / n
        center = p + z**2 / (2 * n)
        margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
        return ((center - margin) / denom, (center + margin) / denom)

    def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            if i < n_bins - 1:
                mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            else:
                mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
            if mask.sum() == 0:
                continue
            bin_conf = y_prob[mask].mean()
            bin_acc = y_true[mask].mean()
            ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
        return float(ece)

    ece = expected_calibration_error(test_df["label"].values, probas)

    # Bootstrap CIs for AUROC/PR-AUC/F1/Brier
    rng = np.random.default_rng(seed)
    n_boot = 1000
    boot_aucs, boot_praucs, boot_f1s, boot_briers = [], [], [], []
    y_true = test_df["label"].values
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        y_b = y_true[idx]
        p_b = probas[idx]
        preds_b = (p_b >= 0.5).astype(int)
        try:
            boot_aucs.append(roc_auc_score(y_b, p_b))
        except ValueError:
            continue
        boot_praucs.append(average_precision_score(y_b, p_b))
        _, _, f1_b, _ = precision_recall_fscore_support(y_b, preds_b, average="binary", zero_division=0)
        boot_f1s.append(f1_b)
        boot_briers.append(brier_score_loss(y_b, p_b))

    def ci_from_samples(samples):
        if not samples:
            return (None, None)
        lower = float(np.percentile(samples, 2.5))
        upper = float(np.percentile(samples, 97.5))
        return (lower, upper)

    metrics = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "brier": float(brier),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "confidence_intervals": {
            "accuracy": wilson_ci(tn + tp, len(test_df)),
            "precision": wilson_ci(tp, tp + fp),
            "recall": wilson_ci(tp, tp + fn),
        },
        "ece": ece,
        "bootstrap_cis": {
            "auroc": ci_from_samples(boot_aucs),
            "pr_auc": ci_from_samples(boot_praucs),
            "f1": ci_from_samples(boot_f1s),
            "brier": ci_from_samples(boot_briers),
        },
    }
    # Baseline keyword heuristic (bag-of-keywords hazard flag)
    keywords = [
        "overdose",
        "suicidal",
        "pregnant",
        "ace inhibitor",
        "nsaid",
        "ibuprofen",
        "naproxen",
        "beta blocker",
        "metformin",
        "valproate",
        "opioid",
        "abuse",
        "violence",
    ]
    baseline_preds = []
    for text in test_df["text"]:
        low = str(text).lower()
        baseline_preds.append(int(any(kw in low for kw in keywords)))
    base_prec, base_rec, base_f1, _ = precision_recall_fscore_support(
        test_df["label"], baseline_preds, average="binary", zero_division=0
    )
    base_acc = accuracy_score(test_df["label"], baseline_preds)
    base_auc = roc_auc_score(test_df["label"], baseline_preds)

    metrics["baseline_keyword"] = {
        "accuracy": float(base_acc),
        "precision": float(base_prec),
        "recall": float(base_rec),
        "f1": float(base_f1),
        "auc": float(base_auc),
    }
    return pipeline, metrics


def main():
    model, metrics = train_and_evaluate(seed=42)

    # Persist artifacts
    with open(RESULTS_DIR / "binary_hazard_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    import pickle

    with open(MODELS_DIR / "hazard_binary.pkl", "wb") as f:
        pickle.dump(model, f)

    print("\n[Binary harm detection]")
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
