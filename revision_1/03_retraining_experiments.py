"""Training-based revision analyses (single background job, incremental saves):
  S1  OR-fusion ablation on n=150 test (saved canonical ClinicalBERT + symbolic firing)  [#13, EZ-6]
  S2  Reviewer EZ #1b: baseline vs symbolic-augmented ClinicalBERT (test + pharmacy reviews)
  S3  Strict patient-level re-evaluation of ClinicalBERT (#21)
  S4  Canonical TF-IDF baseline (test + physician)  [#14 baseline]
  S5  Encoder-generalization: BioBERT + PubMedBERT retrained under canonical config (appendix)
Outputs to revision_1/audit/derived/. PHI-derived; stays local.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

WL = Path("/Users/sanjaybasu/waymark-local")
REPO = WL / "packaging" / "neurosymbolic_github"
SPLITS = REPO / "data" / "mixed_splits"
OUT = WL / "notebooks/neurosymbolic/submission/jmir/revision_1/audit/derived"
SAVED_MODEL = WL / "packaging/packaging/neurosymbolic_github/trained_models/clinicalbert_mixed.pt"
sys.path.append(str(REPO / "models")); sys.path.append(str(REPO / "scripts"))

SEED = 42; EPOCHS = 3; LR = 2e-5; MAX_LEN = 256; BATCH = 16
RES = {}


def seed_all():
    import torch, random
    np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)


def dev():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def symbolic_flags(texts):
    from extraction import ClinicalExtractor
    from neurosymbolic_reasoner import SymbolicReasoner, ClinicalContext
    ext = ClinicalExtractor(str(REPO / "data"))
    sym = SymbolicReasoner(str(WL / "notebooks/neurosymbolic/knowledge_graphs"), rules_path=str(REPO / "data/rules/clinical_rules_expanded.json"))
    out = []
    for t in texts:
        e = ext.extract(str(t))
        ctx = ClinicalContext(patient_id="x", conditions=e.conditions, medications=e.medications,
                              recent_encounters=[], goals=[], demographics={}, risk_factors=[], current_state={})
        safe, _ = sym.check_contraindications(ctx, "medication_review")
        out.append(0 if safe else 1)
    return np.array(out)


def make_tok(name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, use_fast=True)


def train_bert(base_name, train_recs, labels, d):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    import train_clinicalbert_hybrid as T
    tok = make_tok(base_name)

    class DS(Dataset):
        def __init__(s, recs, y): s.r = recs; s.y = y
        def __len__(s): return len(s.r)
        def __getitem__(s, i):
            enc = tok(s.r[i]["text"], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
            it = {k: v.squeeze(0) for k, v in enc.items()}; it["labels"] = torch.tensor(s.y[i], dtype=torch.float); return it
    seed_all()
    model = T.ClinicalBERTClassifier(base_name).to(d)
    dl = DataLoader(DS(train_recs, labels), batch_size=BATCH, shuffle=True)
    y = np.asarray(labels); pw = torch.tensor(float((len(y) - y.sum()) / max(y.sum(), 1)), dtype=torch.float32, device=d)
    opt = torch.optim.AdamW(model.parameters(), lr=LR); crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    model.train()
    for _ in range(EPOCHS):
        for b in dl:
            opt.zero_grad(); logit = model(b["input_ids"].to(d), b["attention_mask"].to(d))
            crit(logit, b["labels"].to(d)).backward(); opt.step()
    model.eval()
    return tok, model, float(pw.item())


def bert_predict(texts, tok, model, d):
    import torch
    out = []
    with torch.no_grad():
        for t in texts:
            enc = tok(str(t)[:512], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
            out.append(torch.sigmoid(model(enc["input_ids"].to(d), enc["attention_mask"].to(d))).cpu().item())
    return np.array(out)


def load_saved_clinicalbert(d):
    import torch
    import train_clinicalbert_hybrid as T
    tok = make_tok(T.MODEL_NAME)
    m = T.ClinicalBERTClassifier(T.MODEL_NAME).to(d)
    m.load_state_dict(torch.load(SAVED_MODEL, map_location=d)); m.eval()
    return tok, m


def metr(y, p, thr=0.5):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.asarray(y); pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sens": round(tp / (tp + fn), 4) if tp + fn else 0, "spec": round(tn / (tn + fp), 4) if tn + fp else 0,
            "prec": round(tp / (tp + fp), 4) if tp + fp else 0,
            "auroc": round(float(roc_auc_score(y, p)), 4) if y.min() != y.max() else None,
            "auprc": round(float(average_precision_score(y, p)), 4) if y.min() != y.max() else None}


def flush():
    (OUT / "stage03_retraining.json").write_text(json.dumps(RES, indent=2, default=str))
    print("  [saved]")


def main():
    d = dev(); print("device", d)
    train = json.load(open(SPLITS / "mixed_train.json"))
    test = json.load(open(SPLITS / "mixed_realworld_test.json"))
    val = json.load(open(SPLITS / "mixed_realworld_val.json"))
    phys = json.load(open(SPLITS / "mixed_physician_test.json"))
    yt = np.array([r["label"] for r in test]); yp = np.array([r["label"] for r in phys])

    # ---- S1 OR-fusion ablation (saved canonical ClinicalBERT) ----
    print("S1 OR-fusion ablation")
    tokC, mC = load_saved_clinicalbert(d)
    pC_test = bert_predict([r["text"] for r in test], tokC, mC, d)
    sym_test = symbolic_flags([r["text"] for r in test])
    ab = {}
    for thr in (0.10, 0.50):
        base = metr(yt, pC_test, thr)
        fused = ((pC_test >= thr) | (sym_test == 1)).astype(int)
        fm = metr(yt, fused.astype(float), 0.5)  # fused is already 0/1
        ab[f"{thr:.2f}"] = {"neural_only": base, "or_fusion": fm,
                            "tp_recovered": fm["tp"] - base["tp"], "fp_added": fm["fp"] - base["fp"]}
    RES["S1_or_fusion_ablation_test"] = {"symbolic_fires_on_test": int(sym_test.sum()), "by_threshold": ab}
    RES["S1_canonical_test_metrics"] = metr(yt, pC_test, 0.50)
    flush()

    # ---- S2 EZ #1b symbolic-augmented neural ----
    print("S2 EZ#1b symbolic-augmented")
    import train_clinicalbert_hybrid as T
    y0 = np.array([r["label"] for r in train])
    sf_train = symbolic_flags([r["text"] for r in train])
    y_aug = ((y0 == 1) | (sf_train == 1)).astype(int)
    flipped = int(((y0 == 0) & (sf_train == 1)).sum())
    tokB, mB, pwB = train_bert(T.MODEL_NAME, train, y0.tolist(), d)
    pB_test = bert_predict([r["text"] for r in test], tokB, mB, d)
    tokA, mA, pwA = train_bert(T.MODEL_NAME, train, y_aug.tolist(), d)
    pA_test = bert_predict([r["text"] for r in test], tokA, mA, d)
    # pharmacy reviews re-detection
    notes = pd.read_csv(WL / "data/real_inputs/notes/encounter notes.csv", low_memory=False)
    notes = notes[notes["text"].str.len() > 50].reset_index(drop=True)
    prn = notes[notes["encounterType"].isin(["PHARMACY_NEW_PATIENT_REVIEW", "PHARMACIST_CONSULTATION"])].reset_index(drop=True)
    pr_texts = prn["text"].astype(str).tolist()
    sym_pr = symbolic_flags(pr_texts)
    pB_pr = bert_predict(pr_texts, tokB, mB, d); pA_pr = bert_predict(pr_texts, tokA, mA, d)

    def pr_overlap(p, s):
        s = s.astype(bool); o = {}
        for thr in (0.10, 0.50):
            bf = p >= thr
            o[f"{thr:.2f}"] = {"symbolic_only": int((~bf & s).sum()), "both": int((bf & s).sum()),
                               "catch_pct": round(100 * (bf & s).sum() / max(s.sum(), 1), 2), "bert_flag_pct": round(100 * bf.mean(), 2)}
        return o
    RES["S2_ez1b"] = {"config": {"epochs": EPOCHS, "lr": LR, "pos_weight_baseline": round(pwB, 3), "pos_weight_augmented": round(pwA, 3),
                                 "n_train": len(train), "symbolic_fired_train": int(sf_train.sum()), "benign_flipped": flipped},
                      "test_baseline": metr(yt, pB_test), "test_augmented": metr(yt, pA_test),
                      "n_symbolic_contra_pharmacy": int(sym_pr.sum()),
                      "pharmacy_overlap_baseline": pr_overlap(pB_pr, sym_pr),
                      "pharmacy_overlap_augmented": pr_overlap(pA_pr, sym_pr)}
    flush()

    # ---- S3 strict patient-level re-eval ----
    print("S3 patient-level re-eval")
    notes["text"] = notes["text"].astype(str)
    text2pt = dict(zip(notes["text"].str.strip(), notes["WaymarkId"]))
    def pid(r): return text2pt.get(str(r["text"]).strip())
    test_pts = set(filter(None, (pid(r) for r in test)))
    clean_train = [r for r in train if pid(r) not in test_pts]  # drop train notes from test patients
    removed = len(train) - len(clean_train)
    yc = np.array([r["label"] for r in clean_train])
    tokP, mP, _ = train_bert(T.MODEL_NAME, clean_train, yc.tolist(), d)
    pP_test = bert_predict([r["text"] for r in test], tokP, mP, d)
    RES["S3_patient_level"] = {"train_notes_removed_for_overlap": removed,
                               "n_test_patients_linked": len(test_pts),
                               "metrics_patient_disjoint": metr(yt, pP_test, 0.50),
                               "metrics_baseline_same_config": metr(yt, pB_test, 0.50),
                               "note": "baseline uses full train (S2 baseline); patient_disjoint removes train notes sharing a test patient"}
    flush()

    # ---- S4 TF-IDF canonical baseline ----
    print("S4 TF-IDF baseline")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    Xtr = vec.fit_transform([r["text"] for r in train])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, y0)
    p_tf_test = clf.predict_proba(vec.transform([r["text"] for r in test]))[:, 1]
    p_tf_phys = clf.predict_proba(vec.transform([r["text"] for r in phys]))[:, 1]
    RES["S4_tfidf"] = {"test": metr(yt, p_tf_test), "physician": metr(yp, p_tf_phys)}
    flush()

    # ---- S5 encoder generalization (appendix) ----
    for name, key in [("dmis-lab/biobert-v1.1", "biobert"),
                      ("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", "pubmedbert")]:
        print("S5", key)
        try:
            tk, mm, _ = train_bert(name, train, y0.tolist(), d)
            pt = bert_predict([r["text"] for r in test], tk, mm, d)
            pp = bert_predict([r["text"] for r in phys], tk, mm, d)
            RES.setdefault("S5_encoder_generalization", {})[key] = {"test": metr(yt, pt), "physician": metr(yp, pp)}
            flush()
        except Exception as e:
            RES.setdefault("S5_encoder_generalization", {})[key] = {"error": str(e)}
            flush()

    print("DONE")


if __name__ == "__main__":
    main()
