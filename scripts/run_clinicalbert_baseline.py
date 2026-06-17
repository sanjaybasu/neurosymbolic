"""
ClinicalBERT baseline (no symbolic routing).

Trains a lightweight classifier on pooled [CLS] embeddings from Bio_ClinicalBERT
and evaluates on physician-created vignettes and prospective operational notes.

Inputs:
  - notebooks/rl_vs_llm_safety/data/scenario_library.csv
  - notebooks/rl_vs_llm_safety/data/prospective_eval/benign_cases_500.csv
  - notebooks/rl_vs_llm_safety/data/prospective_eval/harm_cases_500.csv

Outputs:
  - packaging/neurosymbolic_github/results/clinicalbert_baseline_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = REPO_ROOT / "notebooks" / "rl_vs_llm_safety" / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vignettes() -> List[Dict]:
    df = pd.read_csv(DATA_DIR / "scenario_library.csv")
    scenarios = []
    for _, row in df.iterrows():
        hazard = row["category"] if row["severity"] != "none" else "benign"
        scenarios.append({"text": row["prompt"], "label": hazard})
    return scenarios


def load_prospective() -> List[Dict]:
    from data_utils import parse_multiline_csv

    benign_df = parse_multiline_csv(
        DATA_DIR / "prospective_eval" / "benign_cases_500.csv",
        record_prefixes=["benign_candidate"],
    )
    harm_df = parse_multiline_csv(
        DATA_DIR / "prospective_eval" / "harm_cases_500.csv",
        record_prefixes=["harm_candidate"],
    )
    scenarios = []
    for _, row in benign_df.iterrows():
        scenarios.append({"text": row["context_text"], "label": "benign"})
    for _, row in harm_df.iterrows():
        scenarios.append({"text": row["context_text"], "label": row.get("harm_type") or "harm"})
    return scenarios


def encode_texts(texts: List[str], tokenizer, model) -> np.ndarray:
    """Return pooled CLS embeddings."""
    all_embeddings: List[np.ndarray] = []
    batch_size = 16
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            outputs = model(**encoded)
            cls = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
            all_embeddings.append(cls.cpu().numpy())
    return np.vstack(all_embeddings)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, y_pred) if (tp + fp) > 0 else 0.0
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "tp": int(tp),
        "fn": int(fn),
        "tn": int(tn),
        "fp": int(fp),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "accuracy": float(accuracy),
    }


def main():
    print("Loading data...")
    vign = load_vignettes()
    pros = load_prospective()

    print("Loading ClinicalBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True, trust_remote_code=True).to(DEVICE)

    # Prepare labels: hazard vs benign
    vign_texts = [s["text"] for s in vign]
    vign_labels = np.array([0 if s["label"] == "benign" else 1 for s in vign])
    pros_texts = [s["text"] for s in pros]
    pros_labels = np.array([0 if s["label"] == "benign" else 1 for s in pros])

    print("Encoding vignettes...")
    vign_emb = encode_texts(vign_texts, tokenizer, model)

    print("Training logistic regression on vignettes...")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(vign_emb, vign_labels)

    print("Evaluating on vignettes...")
    vign_pred = clf.predict(vign_emb)
    vign_metrics = compute_metrics(vign_labels, vign_pred)

    print("Encoding prospective notes...")
    pros_emb = encode_texts(pros_texts, tokenizer, model)

    print("Evaluating on prospective notes...")
    pros_pred = clf.predict(pros_emb)
    pros_metrics = compute_metrics(pros_labels, pros_pred)

    summary = {
        "model": MODEL_NAME,
        "train_data": "physician-created vignettes",
        "vignettes": vign_metrics,
        "prospective": pros_metrics,
    }

    out_path = RESULTS_DIR / "clinicalbert_baseline_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {out_path}")
    print("Vignettes metrics:", vign_metrics)
    print("Prospective metrics:", pros_metrics)


if __name__ == "__main__":
    main()
