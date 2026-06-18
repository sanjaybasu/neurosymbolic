"""UNIFIED single-source pipeline (addresses this editor's "full end-to-end audit" demand).

One freshly-trained primary model (seed 42) underlies EVERY model-dependent number:
Table 2 (test + physician), Table 6 overlap, fairness, alert-fatigue, the OR-fusion
ablation, AND it IS the EZ#1b baseline and the patient-split baseline. So the EZ#1b
"baseline catch rate" equals the Table 6 catch rate and the patient-split "baseline AUROC"
equals the Table 2 AUROC by construction -- no cross-table discrepancy.

Reproducibility: seed 42 for training and bootstrap; the trained model weights and the
per-note prediction CSVs are saved as the authoritative source (MPS training is not
guaranteed bit-reproducible, so numbers are pinned to saved predictions, and bootstrap
CIs are exactly reproducible from those predictions + seed).

Model-independent results (symbolic doc-type Tables 4/5, note-length regression, adjudication)
are NOT recomputed here; they carry over from stageB/stageD2 and the adjudication scoring.
"""
import os
import json, sys, random
from pathlib import Path
import numpy as np

WL = Path(os.environ.get("NEUROSYMBOLIC_ROOT", "."))
REPO = WL / "packaging" / "neurosymbolic_github"
SPLITS = REPO / "data" / "mixed_splits"
OUT = WL / "notebooks/neurosymbolic/submission/jmir/revision_1/audit/derived"
MODELS = OUT / "models"; MODELS.mkdir(parents=True, exist_ok=True)
sys.path.append(str(REPO / "models")); sys.path.append(str(REPO / "scripts"))

SEED = 42; EPOCHS = 3; LR = 2e-5; MAX_LEN = 256; BATCH = 16
R = {"config": {"epochs": EPOCHS, "lr": LR, "max_len": MAX_LEN, "batch": BATCH, "seed": SEED}}


def set_seed():
    import torch
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.backends.mps.is_available():
        try: torch.mps.manual_seed(SEED)
        except Exception: pass
    try: torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception: pass


def dev():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def tok_for(name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, use_fast=True)


def train_model(base_name, recs, labels, d):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    import train_clinicalbert_hybrid as T
    tok = tok_for(base_name)

    class DS(Dataset):
        def __init__(s, r, y): s.r = r; s.y = y
        def __len__(s): return len(s.r)
        def __getitem__(s, i):
            e = tok(s.r[i]["text"], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
            it = {k: v.squeeze(0) for k, v in e.items()}; it["labels"] = torch.tensor(s.y[i], dtype=torch.float); return it
    set_seed()
    g = torch.Generator(); g.manual_seed(SEED)
    model = T.ClinicalBERTClassifier(base_name).to(d)
    dl = DataLoader(DS(recs, labels), batch_size=BATCH, shuffle=True, generator=g)
    y = np.asarray(labels); pw = torch.tensor(float((len(y) - y.sum()) / max(y.sum(), 1)), dtype=torch.float32, device=d)
    opt = torch.optim.AdamW(model.parameters(), lr=LR); crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    model.train()
    for _ in range(EPOCHS):
        for b in dl:
            opt.zero_grad(); loss = crit(model(b["input_ids"].to(d), b["attention_mask"].to(d)), b["labels"].to(d))
            loss.backward(); opt.step()
    model.eval(); return tok, model, float(pw)


def probs(texts, tok, model, d):
    import torch
    out = []
    with torch.no_grad():
        for t in texts:
            e = tok(str(t)[:512], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
            out.append(torch.sigmoid(model(e["input_ids"].to(d), e["attention_mask"].to(d))).cpu().item())
    return np.array(out)


def symbolic_flags(texts):
    from extraction import ClinicalExtractor
    from neurosymbolic_reasoner import SymbolicReasoner, ClinicalContext
    ext = ClinicalExtractor(str(REPO / "data"))
    sym = SymbolicReasoner(str(WL / "notebooks/neurosymbolic/knowledge_graphs"), rules_path=str(REPO / "data/rules/clinical_rules_expanded.json"))
    f = []
    for t in texts:
        e = ext.extract(str(t))
        ctx = ClinicalContext(patient_id="x", conditions=e.conditions, medications=e.medications,
                              recent_encounters=[], goals=[], demographics={}, risk_factors=[], current_state={})
        safe, _ = sym.check_contraindications(ctx, "medication_review"); f.append(0 if safe else 1)
    return np.array(f)


def wilson(k, n, z=1.96):
    if n == 0: return (0, 0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - h) / d, (c + h) / d)


def boot(y, s, fn, n=2000):
    rng = np.random.default_rng(SEED); y = np.asarray(y); s = np.asarray(s); idx = np.arange(len(y)); v = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        try: v.append(fn(y[b], s[b]))
        except Exception: pass
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def cls(y, p, thr=0.5):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.asarray(y); pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = wilson(tp, tp + fn); spec = wilson(tn, tn + fp); prec = wilson(tp, tp + fp); acc = wilson(tp + tn, len(y))
    o = {"tp": tp, "fn": fn, "tn": tn, "fp": fp,
         "sensitivity": round(sens[0]*100,1), "sensitivity_ci": [round(sens[1]*100,1), round(sens[2]*100,1)],
         "specificity": round(spec[0]*100,1), "specificity_ci": [round(spec[1]*100,1), round(spec[2]*100,1)],
         "precision": round(prec[0]*100,1), "precision_ci": [round(prec[1]*100,1), round(prec[2]*100,1)],
         "accuracy": round(acc[0]*100,1), "accuracy_ci": [round(acc[1]*100,1), round(acc[2]*100,1)]}
    if y.min() != y.max():
        a = roc_auc_score(y, p); al, ah = boot(y, p, lambda yy, ss: roc_auc_score(yy, ss))
        ap = average_precision_score(y, p); pl, ph = boot(y, p, lambda yy, ss: average_precision_score(yy, ss))
        o.update({"auroc": round(a, 4), "auroc_ci": [round(al, 4), round(ah, 4)],
                  "auprc": round(ap, 4), "auprc_ci": [round(pl, 4), round(ph, 4)]})
    return o


def ece(y, p, bins=10):
    y = np.asarray(y); p = np.asarray(p); ed = np.linspace(0, 1, bins+1); e = 0
    for i in range(bins):
        m = (p > ed[i]) & (p <= ed[i+1]) if i else (p >= ed[i]) & (p <= ed[i+1])
        if m.sum(): e += m.sum()/len(y) * abs(y[m].mean() - p[m].mean())
    return round(float(e), 4)


def main():
    import pandas as pd
    d = dev(); print("device", d)
    train = json.load(open(SPLITS / "mixed_train.json"))
    test = json.load(open(SPLITS / "mixed_realworld_test.json"))
    val = json.load(open(SPLITS / "mixed_realworld_val.json"))
    phys = json.load(open(SPLITS / "mixed_physician_test.json"))
    yt = np.array([r["label"] for r in test]); yv = np.array([r["label"] for r in val]); yp = np.array([r["label"] for r in phys])

    # ---------- M_primary (THE model) ----------
    print("training M_primary...")
    tokP, mP, pwP = train_model("emilyalsentzer/Bio_ClinicalBERT", train, [r["label"] for r in train], d)
    import torch; torch.save(mP.state_dict(), MODELS / "clinicalbert_primary_seed42.pt")
    R["config"]["pos_weight_primary"] = round(pwP, 4)
    pP_test = probs([r["text"] for r in test], tokP, mP, d)
    pP_val = probs([r["text"] for r in val], tokP, mP, d)
    pP_phys = probs([r["text"] for r in phys], tokP, mP, d)
    # save canonical per-note predictions
    pd.DataFrame({"label": yt, "prob": pP_test, "sex": [r.get("sex") for r in test], "race_ethnicity": [r.get("race_ethnicity") for r in test]}).to_csv(OUT/"v2_test_predictions.csv", index=False)
    pd.DataFrame({"label": yv, "prob": pP_val, "sex": [r.get("sex") for r in val], "race_ethnicity": [r.get("race_ethnicity") for r in val]}).to_csv(OUT/"v2_val_predictions.csv", index=False)
    pd.DataFrame({"label": yp, "prob": pP_phys}).to_csv(OUT/"v2_physician_predictions.csv", index=False)

    R["table2_clinicalbert_test"] = {**cls(yt, pP_test), "brier": round(float(np.mean((pP_test-yt)**2)),4), "ece": ece(yt, pP_test)}
    R["table2_clinicalbert_physician"] = {**cls(yp, pP_phys), "brier": round(float(np.mean((pP_phys-yp)**2)),4), "ece": ece(yp, pP_phys)}
    R["clinicalbert_val"] = cls(yv, pP_val)
    R["threshold_sweep_test"] = {f"{t:.2f}": {k: cls(yt, pP_test, t)[k] for k in ("sensitivity","specificity","tp","fp")} for t in (0.10,0.30,0.50)}

    # ---------- fairness pooled 300 ----------
    pooled = pd.concat([
        pd.DataFrame({"label": yv, "prob": pP_val, "sex":[r.get("sex") for r in val], "race":[r.get("race_ethnicity") for r in val]}),
        pd.DataFrame({"label": yt, "prob": pP_test, "sex":[r.get("sex") for r in test], "race":[r.get("race_ethnicity") for r in test]})], ignore_index=True)
    def fair(col):
        rows=[]; sub=pooled.dropna(subset=[col])
        from sklearn.metrics import roc_auc_score
        for g,gd in sub.groupby(col):
            if len(gd)<5: rows.append({"group":g,"n":len(gd),"note":"n<5"}); continue
            y=gd["label"].values; p=gd["prob"].values; m=cls(y,p)
            au=round(float(roc_auc_score(y,p)),2) if y.min()!=y.max() else None
            rows.append({"group":g,"n":len(gd),"pos":int(y.sum()),"sens":m["sensitivity"],"spec":m["specificity"],"fpr":round(100-m["specificity"],1),"auroc":au})
        ev=[r for r in rows if "sens" in r]
        return {"groups":rows,"tpr_gap":round((max(r["sens"] for r in ev)-min(r["sens"] for r in ev))/100,3),
                "fpr_gap":round((max(r["fpr"] for r in ev)-min(r["fpr"] for r in ev))/100,3),"n":len(sub)}
    R["fairness_pooled_300"]={"by_sex":fair("sex"),"by_race":fair("race")}

    # ---------- pharmacy reviews: M_primary probs + symbolic (aligned) ----------
    print("pharmacy reviews...")
    notes = pd.read_csv(Path(os.environ.get("NEUROSYMBOLIC_NOTES", "data/notes.csv")), low_memory=False)
    notes = notes[notes["text"].str.len()>50].reset_index(drop=True)
    prn = notes[notes["encounterType"].isin(["PHARMACY_NEW_PATIENT_REVIEW","PHARMACIST_CONSULTATION"])].reset_index(drop=True)
    pr_texts = prn["text"].astype(str).tolist()
    sym_pr = symbolic_flags(pr_texts)
    pP_pr = probs(pr_texts, tokP, mP, d)
    pd.DataFrame({"prob": pP_pr, "symbolic_contra": sym_pr}).to_csv(OUT/"v2_pharmacy_predictions.csv", index=False)
    nC = int(sym_pr.sum())
    overlap={}
    for t in (0.10,0.30,0.50):
        bf=pP_pr>=t; s=sym_pr.astype(bool)
        both=int((bf&s).sum()); bo=int((bf&~s).sum()); so=int((~bf&s).sum()); nei=int((~bf&~s).sum())
        cat=wilson(both,nC); mis=wilson(so,nC)
        overlap[f"{t:.2f}"]={"both":both,"bert_only":bo,"symbolic_only":so,"neither":nei,
            "neural_flag_pct":round(100*bf.mean(),1),"catch_pct":round(cat[0]*100,1),"catch_ci":[round(cat[1]*100,1),round(cat[2]*100,1)],
            "miss_pct":round(mis[0]*100,1),"miss_ci":[round(mis[1]*100,1),round(mis[2]*100,1)],
            "or_fusion_flag":int((bf|s).sum()),"or_fusion_pct":round(100*(bf|s).mean(),1)}
    R["overlap"]=overlap; R["n_symbolic_contra_pharmacy"]=nC
    R["bert_on_contra_mean"]=round(float(pP_pr[sym_pr.astype(bool)].mean()),3)

    # ---------- OR-fusion ablation on test ----------
    sym_test = symbolic_flags([r["text"] for r in test])
    R["or_fusion_ablation_test"]={"symbolic_fires_on_test":int(sym_test.sum()),
        "tp_recovered_0.50": int(((pP_test>=0.5)|(sym_test==1)).sum().item()) - cls(yt,pP_test,0.5)["tp"] if False else 0}
    # compute properly
    for t in (0.10,0.50):
        base=cls(yt,pP_test,t); fused=((pP_test>=t)|(sym_test==1)).astype(int)
        ftp=int(((fused==1)&(yt==1)).sum()); ffp=int(((fused==1)&(yt==0)).sum())
        R["or_fusion_ablation_test"][f"thr_{t:.2f}"]={"neural_tp":base["tp"],"neural_fp":base["fp"],"fused_tp":ftp,"fused_fp":ffp,"tp_recovered":ftp-base["tp"]}

    # ---------- EZ1b: baseline = M_primary (SAME), augmented = retrain ----------
    print("EZ1b augmented...")
    sf_train = symbolic_flags([r["text"] for r in train])
    y0 = np.array([r["label"] for r in train]); y_aug = ((y0==1)|(sf_train==1)).astype(int)
    tokA, mA, _ = train_model("emilyalsentzer/Bio_ClinicalBERT", train, y_aug.tolist(), d)
    pA_pr = probs(pr_texts, tokA, mA, d)
    def ov_simple(p):
        out={}
        for t in (0.10,0.50):
            bf=p>=t; s=sym_pr.astype(bool)
            out[f"{t:.2f}"]={"catch":int((bf&s).sum()),"catch_pct":round(100*(bf&s).sum()/nC,1),"flag_pct":round(100*bf.mean(),1)}
        return out
    R["ez1b"]={"baseline_is_primary": ov_simple(pP_pr), "augmented": ov_simple(pA_pr),
               "n_symbolic_fired_train": int(sf_train.sum()), "n_benign_flipped": int(((y0==0)&(sf_train==1)).sum())}

    # ---------- patient-split: baseline = M_primary (SAME), disjoint = retrain ----------
    print("patient-split disjoint...")
    notes["text"]=notes["text"].astype(str); t2p=dict(zip(notes["text"].str.strip(), notes["WaymarkId"]))
    test_pts=set(filter(None,(t2p.get(str(r["text"]).strip()) for r in test)))
    clean=[r for r in train if t2p.get(str(r["text"]).strip()) not in test_pts]
    tokD, mD, _ = train_model("emilyalsentzer/Bio_ClinicalBERT", clean, [r["label"] for r in clean], d)
    pD_test = probs([r["text"] for r in test], tokD, mD, d)
    R["patient_split"]={"train_notes_removed": len(train)-len(clean), "test_patients_linked": len(test_pts),
        "baseline_is_primary_auroc": R["table2_clinicalbert_test"]["auroc"], "baseline_sens": R["table2_clinicalbert_test"]["sensitivity"],
        "disjoint": {k: cls(yt,pD_test)[k] for k in ("sensitivity","specificity","auroc","tp","fp")}}

    # ---------- encoder generalization + TF-IDF ----------
    print("encoders + tfidf...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=2); Xtr=vec.fit_transform([r["text"] for r in train])
    clf=LogisticRegression(max_iter=1000,class_weight="balanced").fit(Xtr,y0)
    R["tfidf"]={"test":cls(yt,clf.predict_proba(vec.transform([r["text"] for r in test]))[:,1]),
                "physician":cls(yp,clf.predict_proba(vec.transform([r["text"] for r in phys]))[:,1])}
    R["encoders"]={}
    for name,key in [("dmis-lab/biobert-v1.1","biobert"),("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext","pubmedbert")]:
        tk,mm,_=train_model(name,train,y0.tolist(),d)
        R["encoders"][key]={"test":cls(yt,probs([r["text"] for r in test],tk,mm,d)),"physician":cls(yp,probs([r["text"] for r in phys],tk,mm,d))}
        (OUT/"canonical_v2.json").write_text(json.dumps(R,indent=2,default=str))  # incremental

    (OUT/"canonical_v2.json").write_text(json.dumps(R,indent=2,default=str))
    print("DONE -> canonical_v2.json")
    # quick assertions for coherence
    print("EZ1b baseline catch@0.50 =", R["ez1b"]["baseline_is_primary"]["0.50"]["catch"], "== Table6 both@0.50 =", overlap["0.50"]["both"])
    print("patient-split baseline AUROC =", R["patient_split"]["baseline_is_primary_auroc"], "== Table2 AUROC =", R["table2_clinicalbert_test"]["auroc"])


if __name__ == "__main__":
    main()
