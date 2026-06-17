"""
Evaluate hybrid vs baselines on mixed-source splits.

Reads:
  - data/mixed_splits/mixed_train.json (not used here), mixed_val.json, mixed_physician_test.json, mixed_realworld_val.json, mixed_realworld_test.json
Uses:
  - trained_models/hazard_detector.pkl (TF-IDF baseline)
  - trained_models/clinicalbert_mixed.pt (optional ClinicalBERT baseline)

Outputs:
  - results/mixed_evaluation_summary.json
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from sklearn.metrics import brier_score_loss

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
from neurosymbolic_reasoner_v2 import HybridNeurosymbolicReasoner
from hazard_detection import HazardDetector
from train_clinicalbert_hybrid import ClinicalBERTClassifier, MODEL_NAME, MAX_LEN
from neurosymbolic_reasoner import SymbolicReasoner, ClinicalContext
from extraction import ClinicalExtractor
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "packaging" / "neurosymbolic_github" / "data" / "mixed_splits"
KG_DIR = REPO_ROOT / "packaging" / "neurosymbolic_github" / "knowledge_graphs"
RULES_PATH = REPO_ROOT / "packaging" / "neurosymbolic_github" / "data" / "rules" / "clinical_rules_expanded.json"
VOCAB_DIR = REPO_ROOT / "packaging" / "neurosymbolic_github" / "data"
RESULTS_DIR = REPO_ROOT / "packaging" / "neurosymbolic_github" / "results"
MODEL_DIR = REPO_ROOT / "packaging" / "neurosymbolic_github" / "trained_models"


def load_json(path: Path) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def eval_hybrid(detector: HazardDetector, data: List[Dict], threshold=0.35, or_fusion=False):
    reasoner = HybridNeurosymbolicReasoner(
        kg_dir=str(KG_DIR),
        hazard_detector=detector,
        confidence_threshold=threshold,
        rules_path=str(RULES_PATH),
        vocab_dir=str(VOCAB_DIR),
        or_fusion=or_fusion,
    )
    tp=fn=tn=fp=0
    for rec in data:
        res = reasoner.evaluate_scenario({"prompt": rec["text"], "context": {}, "hazard_type": "hazard" if rec["label"]==1 else "benign"})
        hazard_pred = not res["is_safe_pred"]
        hazard_true = rec["label"]==1
        if hazard_true and hazard_pred:
            tp+=1
        elif hazard_true and not hazard_pred:
            fn+=1
        elif not hazard_true and not hazard_pred:
            tn+=1
        else:
            fp+=1
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    acc = (tp+tn)/(tp+tn+fp+fn)
    return {"tp":tp,"fn":fn,"tn":tn,"fp":fp,"sens":sens,"spec":spec,"prec":prec,"acc":acc}


def eval_clinicalbert(data: List[Dict]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = ClinicalBERTClassifier(MODEL_NAME).to(device)
    state = torch.load(MODEL_DIR / "clinicalbert_mixed.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    all_probs=[]
    labels=[]
    with torch.no_grad():
        for rec in data:
            enc = tokenizer(rec["text"], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
            logits = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
            prob = torch.sigmoid(logits).cpu().numpy().item()
            all_probs.append(prob)
            labels.append(rec["label"])
    labels = np.array(labels)
    probs = np.array(all_probs)
    preds = (probs>=0.5).astype(int)
    tp = int(((labels==1) & (preds==1)).sum())
    fn = int(((labels==1) & (preds==0)).sum())
    tn = int(((labels==0) & (preds==0)).sum())
    fp = int(((labels==0) & (preds==1)).sum())
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    acc = (tp+tn)/(tp+tn+fp+fn)
    brier = brier_score_loss(labels, probs)
    ece = expected_calibration_error(labels, probs)
    return {"tp":tp,"fn":fn,"tn":tn,"fp":fp,"sens":sens,"spec":spec,"prec":prec,"acc":acc,"brier":brier,"ece_10bin":ece}


def symbolic_firing(detector: HazardDetector, data: List[Dict], threshold=0.35, or_fusion=False):
    """Count symbolic firings (contraindications or risk cascades) for a dataset."""
    reasoner = HybridNeurosymbolicReasoner(
        kg_dir=str(KG_DIR),
        hazard_detector=detector,
        confidence_threshold=threshold,
        rules_path=str(RULES_PATH),
        vocab_dir=str(VOCAB_DIR),
        or_fusion=or_fusion,
    )
    counts = {
        "total": len(data),
        "contraindication": 0,
        "risk": 0,
        "any_symbolic": 0,
    }
    for rec in data:
        res = reasoner.evaluate_scenario({"prompt": rec["text"], "context": {}, "hazard_type": "hazard" if rec["label"]==1 else "benign"})
        contra = res.get("contraindications_detected", 0) > 0
        risk = res.get("risk_score", 0) > 0
        if contra:
            counts["contraindication"] += 1
        if risk:
            counts["risk"] += 1
        if contra or risk:
            counts["any_symbolic"] += 1
    return counts


def eval_clinicalbert_hybrid(data: List[Dict], threshold=0.5, or_symbolic=True):
    """ClinicalBERT primary detector with symbolic OR fusion."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = ClinicalBERTClassifier(MODEL_NAME).to(device)
    state = torch.load(MODEL_DIR / "clinicalbert_mixed.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    extractor = ClinicalExtractor(str(VOCAB_DIR))
    symbolic = SymbolicReasoner(str(KG_DIR), rules_path=str(RULES_PATH))

    tp=fn=tn=fp=0
    counts = {"total": len(data), "contraindication": 0, "risk": 0, "any_symbolic": 0}
    recoveries = {"upgraded_tp": 0, "upgraded_fp": 0, "examples": []}
    RISK_UPGRADE = 1.2

    for rec in data:
        enc = tokenizer(rec["text"], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        logits = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
        prob = torch.sigmoid(logits).detach().cpu().numpy().item()
        hazard_pred_neural = prob >= threshold

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
        is_safe_symbolic, violations = symbolic.check_contraindications(clinical_context, "medication_review")
        risk_scores = symbolic.compute_risk_cascade(clinical_context)
        max_risk = max(risk_scores.values()) if risk_scores else 0.0
        required = symbolic.find_required_interventions(clinical_context)

        # Allow symbolic OR upgrades for high-severity cues (any med_access_issue) but require non-trivial risk/requirements
        high_severity_symbolic = any(
            cond
            in {
                "med_access_issue",
                "med_access_issue_insulin",
                "med_access_issue_bronchodilator",
                "med_access_issue_glp1",
                "med_access_issue_immuno",
            }
            for cond in extraction.conditions
        )

        symbolic_flag = (not is_safe_symbolic) or max_risk > 0 or len(required) > 0
        if not is_safe_symbolic and violations:
            counts["contraindication"] += 1
        if max_risk > 0:
            counts["risk"] += 1
        if symbolic_flag:
            counts["any_symbolic"] += 1

        allow_symbolic_upgrade = symbolic_flag and high_severity_symbolic and (
            (max_risk >= RISK_UPGRADE)
            or (not is_safe_symbolic)
            or (len(required) > 0 and max_risk > 0)
        )

        hazard_pred = hazard_pred_neural or (allow_symbolic_upgrade if or_symbolic else False)
        if (not hazard_pred_neural) and hazard_pred:
            if rec["label"] == 1:
                recoveries["upgraded_tp"] += 1
                if len(recoveries["examples"]) < 5:
                    recoveries["examples"].append(
                        {
                            "label": rec["label"],
                            "prob": prob,
                            "conds": extraction.conditions,
                            "meds": extraction.medications,
                            "risk": risk_scores,
                            "required": [r[0] for r in required],
                            "text": rec["text"][:200].replace("\n", " "),
                        }
                    )
            else:
                recoveries["upgraded_fp"] += 1
        hazard_true = rec["label"] == 1
        if hazard_true and hazard_pred:
            tp += 1
        elif hazard_true and not hazard_pred:
            fn += 1
        elif not hazard_true and not hazard_pred:
            tn += 1
        else:
            fp += 1

    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    acc = (tp+tn)/(tp+tn+fp+fn)
    return {"tp":tp,"fn":fn,"tn":tn,"fp":fp,"sens":sens,"spec":spec,"prec":prec,"acc":acc,"symbolic":counts,"recoveries":recoveries}


def expected_calibration_error(labels: List[int], probs: List[float], n_bins: int = 10) -> float:
    """Compute a simple Expected Calibration Error (ECE) with equal-width bins."""
    labels = np.array(labels)
    probs = np.array(probs)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1]) if i < n_bins - 1 else (probs >= bins[i]) & (probs <= bins[i + 1])
        if not mask.any():
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(avg_acc - avg_conf)
    return float(ece)


def main():
    # Load splits
    phys_test = load_json(DATA_DIR / "mixed_physician_test.json")
    rw_val = load_json(DATA_DIR / "mixed_realworld_val.json")
    rw_test = load_json(DATA_DIR / "mixed_realworld_test.json")

    # Load hazard detector (TF-IDF baseline)
    with open(REPO_ROOT / "packaging" / "neurosymbolic_github" / "trained_models" / "hazard_detector.pkl", "rb") as f:
        detector = pickle.load(f)

    # Thresholds: keep 0.15 for physician, use 0.10 for mixed real-world to balance recall/specificity
    results = {
        "hybrid_physician": eval_hybrid(detector, phys_test, threshold=0.15),
        "hybrid_rw_val": eval_hybrid(detector, rw_val, threshold=0.10, or_fusion=True),
        "hybrid_rw_test": eval_hybrid(detector, rw_test, threshold=0.10, or_fusion=True),
        "hybrid_minus_rules_physician": eval_hybrid(detector, phys_test, threshold=0.15, or_fusion=False),
        "hybrid_minus_rules_rw_val": eval_hybrid(detector, rw_val, threshold=0.10, or_fusion=False),
        "hybrid_minus_rules_rw_test": eval_hybrid(detector, rw_test, threshold=0.10, or_fusion=False),
        "clinicalbert_physician": eval_clinicalbert(phys_test),
        "clinicalbert_rw_val": eval_clinicalbert(rw_val),
        "clinicalbert_rw_test": eval_clinicalbert(rw_test),
        "clinicalbert_hybrid_rw_val": eval_clinicalbert_hybrid(rw_val, threshold=0.10, or_symbolic=True),
        "clinicalbert_hybrid_rw_test": eval_clinicalbert_hybrid(rw_test, threshold=0.10, or_symbolic=True),
    }

    # Symbolic firing rates (for hybrid configs)
    results["symbolic_firing"] = {
        "hybrid_physician": symbolic_firing(detector, phys_test, threshold=0.15, or_fusion=True),
        "hybrid_rw_val": symbolic_firing(detector, rw_val, threshold=0.10, or_fusion=True),
        "hybrid_rw_test": symbolic_firing(detector, rw_test, threshold=0.10, or_fusion=True),
    }

    with open(RESULTS_DIR / "mixed_evaluation_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved evaluation summary to results/mixed_evaluation_summary.json")


if __name__ == "__main__":
    main()
