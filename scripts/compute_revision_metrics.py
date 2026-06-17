#!/usr/bin/env python3
"""
Compute additional metrics for JMIR revision:
1. AUROC on n=150 real-world test set with bootstrap CIs
2. Fairness analysis by sex and race on n=150 test set
3. OR-fusion ablation (ClinicalBERT alone vs ClinicalBERT + symbolic)
4. BioBERT and PubMedBERT baselines (train + evaluate)

Outputs:
  results/revision_metrics.json
"""

import json
import math
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from transformers import AutoModel, AutoTokenizer
from transformers.utils import import_utils

# Patch for dev torch builds
_original_check = import_utils.check_torch_load_is_safe
def _compat_check():
    if import_utils.is_torch_greater_or_equal("2.6", accept_dev=True):
        return
    _original_check()
import_utils.check_torch_load_is_safe = _compat_check
import transformers.modeling_utils as modeling_utils
modeling_utils.check_torch_load_is_safe = _compat_check

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "mixed_splits"
MODEL_DIR = REPO_ROOT / "trained_models"
RESULTS_DIR = REPO_ROOT / "results"
KG_DIR = REPO_ROOT / "knowledge_graphs"
RULES_PATH = REPO_ROOT / "data" / "rules" / "clinical_rules_expanded.json"
VOCAB_DIR = REPO_ROOT / "data"

sys.path.insert(0, str(REPO_ROOT / "models"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

MAX_LEN = 256
BATCH_SIZE = 16
SEED = 42


class ClinicalBERTClassifier(nn.Module):
    def __init__(self, base_model: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls).squeeze(-1)
        return logits


class TextDataset(Dataset):
    def __init__(self, records: List[Dict], tokenizer, max_len=256):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["text"],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(rec["label"], dtype=torch.float)
        return item


def load_json(path: Path) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def wilson_ci(k: int, n: int) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return (round((center - margin) / denom, 4), round((center + margin) / denom, 4))


def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=2000, seed=42):
    """Bootstrap 95% CI for a metric function."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        y_b = y_true[idx]
        p_b = y_prob[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            vals.append(metric_fn(y_b, p_b))
        except ValueError:
            continue
    if not vals:
        return (None, None)
    return (round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4))


def expected_calibration_error(labels, probs, n_bins=10):
    labels = np.array(labels)
    probs = np.array(probs)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
        else:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        if not mask.any():
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(avg_acc - avg_conf)
    return float(ece)


def get_predictions(model, tokenizer, data, device, max_len=256):
    """Run model inference and return labels, probabilities."""
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for rec in data:
            enc = tokenizer(
                rec["text"],
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            prob = torch.sigmoid(logits).cpu().numpy().item()
            all_probs.append(prob)
            all_labels.append(rec["label"])
    return np.array(all_labels), np.array(all_probs)


def compute_full_metrics(labels, probs, threshold=0.5):
    """Compute all classification metrics with CIs."""
    preds = (probs >= threshold).astype(int)
    tp = int(((labels == 1) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    n = len(labels)

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    acc = (tp + tn) / n if n > 0 else 0

    auroc = roc_auc_score(labels, probs)
    auroc_ci = bootstrap_ci(labels, probs, roc_auc_score)
    auprc = average_precision_score(labels, probs)
    auprc_ci = bootstrap_ci(labels, probs, average_precision_score)
    brier = brier_score_loss(labels, probs)
    ece = expected_calibration_error(labels, probs)

    return {
        "n": n,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "sensitivity": round(sens, 4),
        "sensitivity_ci": wilson_ci(tp, tp + fn),
        "specificity": round(spec, 4),
        "specificity_ci": wilson_ci(tn, tn + fp),
        "precision": round(prec, 4),
        "precision_ci": wilson_ci(tp, tp + fp),
        "accuracy": round(acc, 4),
        "accuracy_ci": wilson_ci(tp + tn, n),
        "auroc": round(auroc, 4),
        "auroc_ci": auroc_ci,
        "auprc": round(auprc, 4),
        "auprc_ci": auprc_ci,
        "brier": round(brier, 4),
        "ece": round(ece, 4),
    }


def compute_fairness(labels, probs, demographics, threshold=0.5):
    """Compute per-group metrics for fairness analysis."""
    preds = (probs >= threshold).astype(int)
    groups = {}
    unique_vals = sorted(set(demographics))
    for val in unique_vals:
        mask = np.array([d == val for d in demographics])
        n_group = mask.sum()
        if n_group < 3:
            continue
        y_true = labels[mask]
        y_pred = preds[mask]
        y_prob = probs[mask]

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())

        tpr = tp / (tp + fn) if (tp + fn) > 0 else None
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None
        sens = tpr
        spec = tn / (tn + fp) if (tn + fp) > 0 else None

        group_auroc = None
        if len(np.unique(y_true)) == 2 and n_group >= 5:
            try:
                group_auroc = round(roc_auc_score(y_true, y_prob), 4)
            except ValueError:
                pass

        groups[str(val)] = {
            "n": int(n_group),
            "n_positive": int(y_true.sum()),
            "n_negative": int(n_group - y_true.sum()),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sensitivity": round(tpr, 4) if tpr is not None else None,
            "sensitivity_ci": wilson_ci(tp, tp + fn) if (tp + fn) > 0 else None,
            "specificity": round(spec, 4) if spec is not None else None,
            "specificity_ci": wilson_ci(tn, tn + fp) if (tn + fp) > 0 else None,
            "fpr": round(fpr, 4) if fpr is not None else None,
            "auroc": group_auroc,
        }

    # Compute equalized odds gap
    tprs = [g["sensitivity"] for g in groups.values() if g["sensitivity"] is not None]
    fprs = [g["fpr"] for g in groups.values() if g["fpr"] is not None]
    eq_odds = {
        "tpr_gap": round(max(tprs) - min(tprs), 4) if len(tprs) >= 2 else None,
        "fpr_gap": round(max(fprs) - min(fprs), 4) if len(fprs) >= 2 else None,
    }

    return {"groups": groups, "equalized_odds_gap": eq_odds}


def train_transformer_baseline(model_name, train_data, val_data, model_save_name,
                                epochs=3, lr=2e-5, batch_size=16, max_len=256):
    """Train a transformer baseline and return trained model + tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = ClinicalBERTClassifier(model_name).to(device)

    train_ds = TextDataset(train_data, tokenizer, max_len)
    val_ds = TextDataset(val_data, tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    labels_arr = np.array([r["label"] for r in train_data])
    pos_weight = torch.tensor(
        (len(labels_arr) - labels_arr.sum()) / max(labels_arr.sum(), 1),
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_t = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_t = batch["labels"].to(device)
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels_t)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"  Epoch {epoch+1}: train_loss={avg_loss:.4f}, val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / model_save_name)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Reload best model
    model.load_state_dict(torch.load(MODEL_DIR / model_save_name, map_location=device))
    return model, tokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    train_data = load_json(DATA_DIR / "mixed_train.json")
    val_data = load_json(DATA_DIR / "mixed_val.json")
    rw_test = load_json(DATA_DIR / "mixed_realworld_test.json")
    rw_val = load_json(DATA_DIR / "mixed_realworld_val.json")
    phys_test = load_json(DATA_DIR / "mixed_physician_test.json")

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"RW Test: {len(rw_test)} (labels: {sum(r['label'] for r in rw_test)} pos)")
    print(f"RW Val: {len(rw_val)}")
    print(f"Physician Test: {len(phys_test)}")

    results = {}

    # =========================================================================
    # 1. AUROC on n=150 test set using existing ClinicalBERT
    # =========================================================================
    print("\n" + "="*60)
    print("1. Computing AUROC for ClinicalBERT on n=150 test set")
    print("="*60)

    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = ClinicalBERTClassifier(MODEL_NAME).to(device)
    state = torch.load(MODEL_DIR / "clinicalbert_mixed.pt", map_location=device)
    model.load_state_dict(state)

    labels_test, probs_test = get_predictions(model, tokenizer, rw_test, device)
    labels_val, probs_val = get_predictions(model, tokenizer, rw_val, device)
    labels_phys, probs_phys = get_predictions(model, tokenizer, phys_test, device)

    clinicalbert_test_metrics = compute_full_metrics(labels_test, probs_test, threshold=0.5)
    clinicalbert_val_metrics = compute_full_metrics(labels_val, probs_val, threshold=0.5)
    clinicalbert_phys_metrics = compute_full_metrics(labels_phys, probs_phys, threshold=0.5)

    results["clinicalbert"] = {
        "model": "emilyalsentzer/Bio_ClinicalBERT",
        "rw_test": clinicalbert_test_metrics,
        "rw_val": clinicalbert_val_metrics,
        "physician_test": clinicalbert_phys_metrics,
    }

    print(f"  RW Test AUROC: {clinicalbert_test_metrics['auroc']} "
          f"(95% CI: {clinicalbert_test_metrics['auroc_ci']})")
    print(f"  RW Test Sensitivity: {clinicalbert_test_metrics['sensitivity']}")
    print(f"  RW Test Specificity: {clinicalbert_test_metrics['specificity']}")

    # =========================================================================
    # 2. Fairness analysis on n=150 test set
    # =========================================================================
    print("\n" + "="*60)
    print("2. Fairness analysis on n=150 test set")
    print("="*60)

    sex_labels = [r.get("sex") or "Unknown" for r in rw_test]
    race_labels = [r.get("race_ethnicity") or "Unknown" for r in rw_test]

    fairness_sex = compute_fairness(labels_test, probs_test, sex_labels, threshold=0.5)
    fairness_race = compute_fairness(labels_test, probs_test, race_labels, threshold=0.5)

    results["fairness_n150"] = {
        "sex": fairness_sex,
        "race_ethnicity": fairness_race,
    }

    print("  Sex breakdown:")
    for group, metrics in fairness_sex["groups"].items():
        print(f"    {group}: n={metrics['n']}, sens={metrics['sensitivity']}, "
              f"spec={metrics['specificity']}, auroc={metrics['auroc']}")
    print(f"  Sex equalized odds gap: {fairness_sex['equalized_odds_gap']}")

    print("  Race/ethnicity breakdown:")
    for group, metrics in fairness_race["groups"].items():
        print(f"    {group}: n={metrics['n']}, sens={metrics['sensitivity']}, "
              f"spec={metrics['specificity']}, auroc={metrics['auroc']}")
    print(f"  Race equalized odds gap: {fairness_race['equalized_odds_gap']}")

    # =========================================================================
    # 3. OR-fusion ablation
    # =========================================================================
    print("\n" + "="*60)
    print("3. OR-fusion ablation")
    print("="*60)

    from neurosymbolic_reasoner import SymbolicReasoner, ClinicalContext
    from extraction import ClinicalExtractor

    extractor = ClinicalExtractor(str(VOCAB_DIR))
    symbolic = SymbolicReasoner(str(KG_DIR), rules_path=str(RULES_PATH))

    RISK_UPGRADE = 1.2

    # ClinicalBERT alone (no symbolic) vs ClinicalBERT + symbolic OR fusion
    for threshold_name, threshold in [("t050", 0.50), ("t010", 0.10)]:
        for fusion_name, use_fusion in [("no_fusion", False), ("or_fusion", True)]:
            tp = fn = tn = fp = 0
            upgraded_tp = upgraded_fp = 0
            for i, rec in enumerate(rw_test):
                prob = probs_test[i]
                hazard_neural = prob >= threshold

                if use_fusion:
                    extraction = extractor.extract(rec["text"])
                    clinical_context = ClinicalContext(
                        patient_id="unknown",
                        conditions=extraction.conditions,
                        medications=extraction.medications,
                        recent_encounters=[],
                        goals=[],
                        demographics={},
                        risk_factors=[],
                        current_state={},
                    )
                    is_safe, violations = symbolic.check_contraindications(
                        clinical_context, "medication_review"
                    )
                    risk_scores = symbolic.compute_risk_cascade(clinical_context)
                    max_risk = max(risk_scores.values()) if risk_scores else 0.0
                    required = symbolic.find_required_interventions(clinical_context)

                    high_severity = any(
                        c in {"med_access_issue", "med_access_issue_insulin",
                               "med_access_issue_bronchodilator",
                               "med_access_issue_glp1", "med_access_issue_immuno"}
                        for c in extraction.conditions
                    )
                    symbolic_flag = (not is_safe) or max_risk > 0 or len(required) > 0
                    allow_upgrade = symbolic_flag and high_severity and (
                        max_risk >= RISK_UPGRADE or (not is_safe) or
                        (len(required) > 0 and max_risk > 0)
                    )
                    hazard_pred = hazard_neural or allow_upgrade
                    if (not hazard_neural) and hazard_pred:
                        if rec["label"] == 1:
                            upgraded_tp += 1
                        else:
                            upgraded_fp += 1
                else:
                    hazard_pred = hazard_neural

                hazard_true = rec["label"] == 1
                if hazard_true and hazard_pred:
                    tp += 1
                elif hazard_true and not hazard_pred:
                    fn += 1
                elif not hazard_true and not hazard_pred:
                    tn += 1
                else:
                    fp += 1

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            prec_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            acc = (tp + tn) / len(rw_test)

            config_name = f"{fusion_name}_{threshold_name}"
            results[f"ablation_{config_name}"] = {
                "threshold": threshold,
                "or_fusion": use_fusion,
                "tp": tp, "fn": fn, "tn": tn, "fp": fp,
                "sensitivity": round(sens, 4),
                "sensitivity_ci": wilson_ci(tp, tp + fn),
                "specificity": round(spec, 4),
                "specificity_ci": wilson_ci(tn, tn + fp),
                "precision": round(prec_val, 4),
                "accuracy": round(acc, 4),
                "upgraded_tp": upgraded_tp,
                "upgraded_fp": upgraded_fp,
            }
            print(f"  {config_name}: sens={sens:.4f}, spec={spec:.4f}, "
                  f"acc={acc:.4f}, upgraded_tp={upgraded_tp}, upgraded_fp={upgraded_fp}")

    # =========================================================================
    # 4. BioBERT and PubMedBERT baselines
    # =========================================================================
    print("\n" + "="*60)
    print("4. Training alternative transformer baselines")
    print("="*60)

    alternative_models = {
        "biobert": {
            "name": "dmis-lab/biobert-v1.1",
            "save": "biobert_mixed.pt",
        },
        "pubmedbert": {
            "name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
            "save": "pubmedbert_mixed.pt",
        },
    }

    for key, config in alternative_models.items():
        try:
            alt_model, alt_tokenizer = train_transformer_baseline(
                model_name=config["name"],
                train_data=train_data,
                val_data=val_data,
                model_save_name=config["save"],
                epochs=3,
                lr=2e-5,
                batch_size=BATCH_SIZE,
                max_len=MAX_LEN,
            )

            alt_labels_test, alt_probs_test = get_predictions(
                alt_model, alt_tokenizer, rw_test, device
            )
            alt_labels_val, alt_probs_val = get_predictions(
                alt_model, alt_tokenizer, rw_val, device
            )
            alt_labels_phys, alt_probs_phys = get_predictions(
                alt_model, alt_tokenizer, phys_test, device
            )

            results[key] = {
                "model": config["name"],
                "rw_test": compute_full_metrics(alt_labels_test, alt_probs_test),
                "rw_val": compute_full_metrics(alt_labels_val, alt_probs_val),
                "physician_test": compute_full_metrics(alt_labels_phys, alt_probs_phys),
            }

            print(f"  {key} RW Test AUROC: {results[key]['rw_test']['auroc']} "
                  f"(CI: {results[key]['rw_test']['auroc_ci']})")
            print(f"  {key} RW Test Sens: {results[key]['rw_test']['sensitivity']}, "
                  f"Spec: {results[key]['rw_test']['specificity']}")

        except Exception as e:
            print(f"  ERROR training {key}: {e}")
            results[key] = {"error": str(e)}

    # =========================================================================
    # Save all results
    # =========================================================================
    output_path = RESULTS_DIR / "revision_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved all revision metrics to {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nClinicalBERT (n=150 test):")
    m = results["clinicalbert"]["rw_test"]
    print(f"  AUROC: {m['auroc']} (95% CI: {m['auroc_ci'][0]}--{m['auroc_ci'][1]})")
    print(f"  AUPRC: {m['auprc']} (95% CI: {m['auprc_ci'][0]}--{m['auprc_ci'][1]})")
    print(f"  Sensitivity: {m['sensitivity']} (CI: {m['sensitivity_ci']})")
    print(f"  Specificity: {m['specificity']} (CI: {m['specificity_ci']})")
    print(f"  Brier: {m['brier']}, ECE: {m['ece']}")

    for key in ["biobert", "pubmedbert"]:
        if key in results and "error" not in results[key]:
            m = results[key]["rw_test"]
            print(f"\n{key.upper()} (n=150 test):")
            print(f"  AUROC: {m['auroc']} (95% CI: {m['auroc_ci'][0]}--{m['auroc_ci'][1]})")
            print(f"  Sensitivity: {m['sensitivity']}, Specificity: {m['specificity']}")

    print(f"\nFairness (sex equalized odds gap): {results['fairness_n150']['sex']['sex']['equalized_odds_gap']}")
    print(f"Fairness (race equalized odds gap): {results['fairness_n150']['race_ethnicity']['race_ethnicity']['equalized_odds_gap']}")


if __name__ == "__main__":
    main()
