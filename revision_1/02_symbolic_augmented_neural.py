"""
Reviewer EZ #1b — does a neural classifier trained on symbolic-generated labels
absorb the symbolic component, making a separate symbolic layer unnecessary?

Design (internally controlled; same seed/config, differ only in training labels):
  baseline:  ClinicalBERT trained on original labels y
  augmented: ClinicalBERT trained on y_aug = y OR (symbolic contraindication fires on the training note)
Both evaluated on the held-out real-world test (n=150) and on the 1,519 pharmacy
reviews (does the symbolic-informed neural model now catch the symbolic-only notes?).

Establishes and documents ONE training config (reconciles Methods vs code: epochs,
pos_weight) used for this controlled experiment.
"""
import json, sys
from pathlib import Path
import numpy as np

WL = Path("/Users/sanjaybasu/waymark-local")
REPO = WL / "packaging" / "neurosymbolic_github"
SPLITS = REPO / "data" / "mixed_splits"
OUT = WL / "notebooks" / "neurosymbolic" / "submission" / "jmir" / "revision_1" / "audit" / "derived"
PR_CSV = OUT / "pharmacy_reviews_per_note.csv"   # written by script 01 (stage C)
sys.path.append(str(REPO / "models")); sys.path.append(str(REPO / "scripts"))

SEED = 42
EPOCHS = 3          # documented canonical config for this experiment
LR = 2e-5
MAX_LEN = 256
BATCH = 16


def set_seed():
    import torch
    np.random.seed(SEED); torch.manual_seed(SEED)
    import random; random.seed(SEED)


def symbolic_fires(texts):
    from extraction import ClinicalExtractor
    from neurosymbolic_reasoner import SymbolicReasoner, ClinicalContext
    ext = ClinicalExtractor(str(REPO / "data"))
    sym = SymbolicReasoner(str(WL / "notebooks" / "neurosymbolic" / "knowledge_graphs"),
                           rules_path=str(REPO / "data" / "rules" / "clinical_rules_expanded.json"))
    flags = []
    for t in texts:
        e = ext.extract(str(t))
        ctx = ClinicalContext(patient_id="x", conditions=e.conditions, medications=e.medications,
                              recent_encounters=[], goals=[], demographics={}, risk_factors=[], current_state={})
        safe, _ = sym.check_contraindications(ctx, "medication_review")
        flags.append(0 if safe else 1)
    return np.array(flags)


def train_model(train_recs, labels, dev):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    import train_clinicalbert_hybrid as T
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(T.MODEL_NAME, use_fast=True)

    class DS(Dataset):
        def __init__(self, recs, y):
            self.recs = recs; self.y = y
        def __len__(self): return len(self.recs)
        def __getitem__(self, i):
            enc = tok(self.recs[i]["text"], truncation=True, max_length=MAX_LEN,
                      padding="max_length", return_tensors="pt")
            it = {k: v.squeeze(0) for k, v in enc.items()}
            it["labels"] = torch.tensor(self.y[i], dtype=torch.float)
            return it

    set_seed()
    model = T.ClinicalBERTClassifier(T.MODEL_NAME).to(dev)
    dl = DataLoader(DS(train_recs, labels), batch_size=BATCH, shuffle=True)
    y = np.asarray(labels)
    pos_weight = torch.tensor((len(y) - y.sum()) / max(y.sum(), 1), device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for ep in range(EPOCHS):
        for b in dl:
            opt.zero_grad()
            logit = model(b["input_ids"].to(dev), b["attention_mask"].to(dev))
            loss = crit(logit, b["labels"].to(dev)); loss.backward(); opt.step()
    model.eval()
    return tok, model, float(pos_weight.item())


def predict(texts, tok, model, dev):
    import torch
    out = []
    with torch.no_grad():
        for t in texts:
            enc = tok(str(t)[:512], truncation=True, max_length=MAX_LEN,
                      padding="max_length", return_tensors="pt")
            out.append(torch.sigmoid(model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))).cpu().item())
    return np.array(out)


def metrics(y, p, thr=0.5):
    y = np.asarray(y); pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    from sklearn.metrics import roc_auc_score
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sens": tp / (tp + fn) if tp + fn else 0, "spec": tn / (tn + fp) if tn + fp else 0,
            "auroc": float(roc_auc_score(y, p)) if y.min() != y.max() else None}


def main():
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    train = json.load(open(SPLITS / "mixed_train.json"))
    test = json.load(open(SPLITS / "mixed_realworld_test.json"))
    y0 = np.array([r["label"] for r in train])
    print(f"train n={len(train)} pos={int(y0.sum())}")

    print("computing symbolic fires on training notes...")
    sf = symbolic_fires([r["text"] for r in train])
    y_aug = ((y0 == 1) | (sf == 1)).astype(int)
    n_flipped = int(((y0 == 0) & (sf == 1)).sum())
    print(f"symbolic fired on {int(sf.sum())} training notes; flipped {n_flipped} benign->hazard")

    yt = np.array([r["label"] for r in test])

    print("training BASELINE (original labels)...")
    tokB, mB, pwB = train_model(train, y0, dev)
    pB_test = predict([r["text"] for r in test], tokB, mB, dev)

    print("training AUGMENTED (symbolic-augmented labels)...")
    tokA, mA, pwA = train_model(train, y_aug, dev)
    pA_test = predict([r["text"] for r in test], tokA, mA, dev)

    res = {"config": {"epochs": EPOCHS, "lr": LR, "max_len": MAX_LEN, "batch": BATCH, "seed": SEED,
                      "pos_weight_baseline": pwB, "pos_weight_augmented": pwA,
                      "n_train": len(train), "n_symbolic_fired_train": int(sf.sum()),
                      "n_benign_flipped_to_hazard": n_flipped},
           "test_baseline": metrics(yt, pB_test), "test_augmented": metrics(yt, pA_test)}

    # Pharmacy review re-detection: can augmented neural catch symbolic-only notes?
    import pandas as pd
    if PR_CSV.exists():
        pr = pd.read_csv(PR_CSV)
        # need note text -> reload from per-note? per-note csv has no text; reload raw subset deterministically
        # Instead, reuse symbolic flags from pr; recompute neural probs requires text -> read raw notes
        notes = pd.read_csv(WL / "data" / "real_inputs" / "notes" / "encounter notes.csv", low_memory=False)
        notes = notes[notes["text"].str.len() > 50].reset_index(drop=True)
        prn = notes[notes["encounterType"].isin(["PHARMACY_NEW_PATIENT_REVIEW", "PHARMACIST_CONSULTATION"])].reset_index(drop=True)
        texts = prn["text"].astype(str).tolist()
        sym_pr = symbolic_fires(texts)
        pA_pr = predict(texts, tokA, mA, dev)
        pB_pr = predict(texts, mBtok if False else tokB, mB, dev)
        def overlap(p, s):
            s = s.astype(bool)
            d = {}
            for t in (0.10, 0.50):
                bf = p >= t
                d[f"{t:.2f}"] = {"symbolic_only": int((~bf & s).sum()),
                                 "both": int((bf & s).sum()),
                                 "catch_pct_of_contra": round(100 * (bf & s).sum() / max(s.sum(), 1), 2),
                                 "bert_flag_pct": round(100 * bf.mean(), 2)}
            return d
        res["pharmacy_overlap_baseline"] = overlap(pB_pr, sym_pr)
        res["pharmacy_overlap_augmented"] = overlap(pA_pr, sym_pr)
        res["n_symbolic_contra_pharmacy"] = int(sym_pr.sum())

    (OUT / "stage_EZ1b_symbolic_augmented.json").write_text(json.dumps(res, indent=2, default=str))
    print("wrote", OUT / "stage_EZ1b_symbolic_augmented.json")
    print(json.dumps(res, indent=2, default=str)[:1500])


if __name__ == "__main__":
    main()
