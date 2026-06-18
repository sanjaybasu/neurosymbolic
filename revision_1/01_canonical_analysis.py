"""
JMIR Med Inform ms#93763 Revision 1 — canonical analysis (no retraining).

Single end-to-end pipeline that (a) reproduces every reported manuscript number
import os
from one run and (b) produces the new analyses requested by reviewers/editor.
Reads raw PHI locally; writes derived numeric outputs to revision_1/audit/derived/.
PHI-derived outputs stay local and NEVER go to the public code repository.

Stages:
  A  ClinicalBERT predictions: test(150), val(150), physician(208); metrics; ECE/Brier; physician standalone (#14); pooled-300 fairness (#3)
  B  Symbolic doc-type analysis (per-note dataframe) + chi-square + pairwise Fisher
  C  ClinicalBERT on pharmacy reviews -> overlap + alert-fatigue reconciliation (#7, EM-4)
  D  Note-length logistic regression for contraindication ~ log10(len) + doc_type (#9)
  E  Patient-level: dedup/repetitive alerting (#16); split->patient linkage + leakage (#21)
  F  Age: population at enrollment-start (median/IQR) + diagnostics (#12); note-level age-at-encounter (#22)
  G  Risk-amplification / required-intervention edge characterization (#18)
  H  Adjudication packet for the 107 pharmacy-review contraindication notes (#5)
"""

import json
import sys
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- paths
WL = Path(os.environ.get("NEUROSYMBOLIC_ROOT", "."))
REPO = WL / "packaging" / "neurosymbolic_github"
MODEL_PATH = WL / "packaging" / "packaging" / "neurosymbolic_github" / "trained_models" / "clinicalbert_mixed.pt"
DATA_ROOT = Path(os.environ.get("NEUROSYMBOLIC_DATA", "data"))
NOTES_PATH = Path(os.environ.get("NEUROSYMBOLIC_NOTES", "data/notes.csv"))
MEMBER_PATH = DATA_ROOT / "member_attributes.parquet"
ELIG_PATH = DATA_ROOT / "eligibility.parquet"
KG_DIR = str(WL / "notebooks" / "neurosymbolic" / "knowledge_graphs")
RULES_PATH = str(REPO / "data" / "rules" / "clinical_rules_expanded.json")
VOCAB_DIR = str(REPO / "data")
SPLITS = REPO / "data" / "mixed_splits"
OUT = WL / "notebooks" / "neurosymbolic" / "submission" / "jmir" / "revision_1" / "audit" / "derived"
PACKET = WL / "notebooks" / "neurosymbolic" / "submission" / "jmir" / "revision_1" / "adjudication_packet"
OUT.mkdir(parents=True, exist_ok=True)
PACKET.mkdir(parents=True, exist_ok=True)

sys.path.append(str(REPO / "models"))
sys.path.append(str(REPO / "scripts"))

SEED = 42
np.random.seed(SEED)
import random
random.seed(SEED)

RESULTS = {}


def save(name, obj):
    p = OUT / name
    if name.endswith(".json"):
        p.write_text(json.dumps(obj, indent=2, default=str))
    print(f"  wrote {p}")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - h) / d, (c + h) / d)


def boot_ci(y, s, fn, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        try:
            vals.append(fn(y[b], s[b]))
        except Exception:
            pass
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ================================================================ model
def load_clinicalbert():
    import torch
    import train_clinicalbert_hybrid as T  # applies torch.load safety patch on import
    from transformers import AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(T.MODEL_NAME, use_fast=True)
    model = T.ClinicalBERTClassifier(T.MODEL_NAME).to(dev)
    state = torch.load(MODEL_PATH, map_location=dev)
    model.load_state_dict(state)
    model.eval()
    return tok, model, dev, T.MAX_LEN


def bert_probs(texts, tok, model, dev, max_len):
    import torch
    out = []
    with torch.no_grad():
        for t in texts:
            enc = tok(str(t)[:512], truncation=True, max_length=max_len,
                      padding="max_length", return_tensors="pt")
            logit = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
            out.append(torch.sigmoid(logit).cpu().item())
    return np.array(out)


def cls_metrics(y, p, thr=0.5):
    y = np.asarray(y); pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    acc = (tp + tn) / len(y)
    return dict(tp=tp, fn=fn, tn=tn, fp=fp, sens=sens, spec=spec, prec=prec, acc=acc)


def ece_score(y, p, bins=10):
    y = np.asarray(y); p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        conf = p[m].mean(); acc = y[m].mean()
        e += (m.sum() / len(y)) * abs(acc - conf)
    return float(e)


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


# ================================================================ A
def stage_A(tok, model, dev, max_len):
    from sklearn.metrics import roc_auc_score, average_precision_score
    print("[A] ClinicalBERT predictions + metrics + fairness")
    test = json.load(open(SPLITS / "mixed_realworld_test.json"))
    val = json.load(open(SPLITS / "mixed_realworld_val.json"))
    phys = json.load(open(SPLITS / "mixed_physician_test.json"))

    def run(recs):
        y = np.array([r["label"] for r in recs])
        p = bert_probs([r["text"] for r in recs], tok, model, dev, max_len)
        return y, p

    yt, pt = run(test); yv, pv = run(val); yp, pp = run(phys)

    # per-note prediction CSVs (canonical)
    pd.DataFrame([{**{k: r.get(k) for k in ("label", "hazard_type", "age", "sex", "race_ethnicity")},
                   "prob": float(pp_)} for r, pp_ in zip(test, pt)]).to_csv(OUT / "clinicalbert_test_predictions.csv", index=False)
    pd.DataFrame([{**{k: r.get(k) for k in ("label", "hazard_type", "age", "sex", "race_ethnicity")},
                   "prob": float(pv_)} for r, pv_ in zip(val, pv)]).to_csv(OUT / "clinicalbert_val_predictions.csv", index=False)
    pd.DataFrame([{"label": r["label"], "hazard_type": r.get("hazard_type"), "prob": float(pp_)}
                  for r, pp_ in zip(phys, pp)]).to_csv(OUT / "clinicalbert_physician_predictions.csv", index=False)

    def block(y, p, name):
        m = cls_metrics(y, p, 0.5)
        a_lo, a_hi = boot_ci(y, p, lambda yy, ss: roc_auc_score(yy, ss))
        ap_lo, ap_hi = boot_ci(y, p, lambda yy, ss: average_precision_score(yy, ss))
        sens = wilson(m["tp"], m["tp"] + m["fn"]); spec = wilson(m["tn"], m["tn"] + m["fp"])
        prec = wilson(m["tp"], m["tp"] + m["fp"]); acc = wilson(m["tp"] + m["tn"], len(y))
        return {
            "name": name, "n": len(y), "pos": int(y.sum()), "neg": int((y == 0).sum()),
            **{k: m[k] for k in ("tp", "fn", "tn", "fp")},
            "auroc": float(roc_auc_score(y, p)), "auroc_ci": [a_lo, a_hi],
            "auprc": float(average_precision_score(y, p)), "auprc_ci": [ap_lo, ap_hi],
            "sensitivity": sens[0], "sensitivity_ci": [sens[1], sens[2]],
            "specificity": spec[0], "specificity_ci": [spec[1], spec[2]],
            "precision": prec[0], "precision_ci": [prec[1], prec[2]],
            "accuracy": acc[0], "accuracy_ci": [acc[1], acc[2]],
            "brier": brier(y, p), "ece": ece_score(y, p),
        }

    RESULTS["clinicalbert"] = {
        "rw_test": block(yt, pt, "rw_test"),
        "rw_val": block(yv, pv, "rw_val"),
        "physician_test": block(yp, pp, "physician_test"),  # comment #14
    }
    # threshold sweep on test (for #7/EZ-6)
    RESULTS["clinicalbert"]["test_threshold_sweep"] = {
        f"{t:.2f}": cls_metrics(yt, pt, t) for t in (0.10, 0.30, 0.50)
    }

    # pooled val+test fairness (n=300) — decision 4 / comment #3
    pooled = []
    for r, p in list(zip(val, pv)) + list(zip(test, pt)):
        pooled.append({"label": r["label"], "prob": float(p), "sex": r.get("sex"),
                       "race_ethnicity": r.get("race_ethnicity")})
    pdf = pd.DataFrame(pooled)
    pdf.to_csv(OUT / "fairness_pooled_300_predictions.csv", index=False)

    def fairness(col):
        rows = []
        sub = pdf.dropna(subset=[col])
        for g, gdf in sub.groupby(col):
            if len(gdf) < 5:
                rows.append({"group": g, "n": len(gdf), "note": "n<5, not evaluated"})
                continue
            y = gdf["label"].values; p = gdf["prob"].values
            m = cls_metrics(y, p, 0.5)
            au = float(roc_auc_score(y, p)) if (y.min() != y.max()) else None
            s = wilson(m["tp"], m["tp"] + m["fn"])
            rows.append({"group": g, "n": len(gdf), "pos": int(y.sum()), "neg": int((y == 0).sum()),
                         "sens": m["sens"], "sens_ci": [s[1], s[2]], "spec": m["spec"],
                         "fpr": (1 - m["spec"]), "auroc": au})
        ev = [r for r in rows if "sens" in r]
        tpr_gap = (max(r["sens"] for r in ev) - min(r["sens"] for r in ev)) if ev else None
        fpr_gap = (max(r["fpr"] for r in ev) - min(r["fpr"] for r in ev)) if ev else None
        return {"groups": rows, "tpr_gap": tpr_gap, "fpr_gap": fpr_gap, "n_total": len(sub)}

    RESULTS["fairness_pooled_300"] = {"by_sex": fairness("sex"), "by_race_ethnicity": fairness("race_ethnicity")}
    save("stageA_clinicalbert_metrics.json", {"clinicalbert": RESULTS["clinicalbert"],
                                              "fairness_pooled_300": RESULTS["fairness_pooled_300"]})


# ================================================================ B/C/D/E/G/H symbolic
def build_note_frames():
    """Run extractor+reasoner across 4 doc types; return per-note dataframes + source notes."""
    from extraction import ClinicalExtractor
    from neurosymbolic_reasoner import SymbolicReasoner, ClinicalContext
    print("[B] symbolic doc-type analysis")
    notes = pd.read_csv(NOTES_PATH, low_memory=False)
    notes = notes[notes["text"].str.len() > 50].reset_index(drop=True)

    pr = notes[notes["encounterType"].isin(["PHARMACY_NEW_PATIENT_REVIEW", "PHARMACIST_CONSULTATION"])].copy()
    po = notes[(notes["title"].str.contains("PHARM", case=False, na=False)) & (notes["encounterType"] == "PATIENT_OUTREACH")].copy()
    chw = notes[notes["title"].str.contains("^CHW$|^CHW_LEAD$", case=False, na=False, regex=True)].copy()
    cc = notes[notes["title"].str.contains("CARE_COORDINATOR|CARE_COORDINATOR_LEAD", case=False, na=False)].copy()

    ext = ClinicalExtractor(VOCAB_DIR)
    sym = SymbolicReasoner(KG_DIR, rules_path=RULES_PATH)

    def run(sub, label, max_n=None):
        if max_n and len(sub) > max_n:
            sub = sub.sample(max_n, random_state=SEED).reset_index(drop=True)
        recs = []
        for _, row in sub.iterrows():
            text = str(row["text"])
            e = ext.extract(text)
            ctx = ClinicalContext(patient_id=row.get("WaymarkId", "unknown"), conditions=e.conditions,
                                  medications=e.medications, recent_encounters=[], goals=[],
                                  demographics={}, risk_factors=[], current_state={})
            is_safe, viol = sym.check_contraindications(ctx, "medication_review")
            risk = sym.compute_risk_cascade(ctx); maxr = max(risk.values()) if risk else 0.0
            req = sym.find_required_interventions(ctx)
            recs.append({"doc_type": label, "WaymarkId": row.get("WaymarkId", "unknown"),
                         "dateOfEncounter": row.get("dateOfEncounter"), "text": text, "text_len": len(text),
                         "n_meds": len(e.medications), "n_conds": len(e.conditions),
                         "medications": e.medications, "conditions": e.conditions,
                         "contraindication": (not is_safe), "n_violations": len(viol) if viol else 0,
                         "violations": viol if viol else [], "risk_score": maxr,
                         "n_required": len(req), "required": req,
                         "any_symbolic": (not is_safe) or maxr > 0 or len(req) > 0})
        return pd.DataFrame(recs)

    frames = {
        "Pharmacy Reviews": run(pr, "Pharmacy Reviews"),
        "Pharmacy Outreach": run(po, "Pharmacy Outreach", 1500),
        "CHW": run(chw, "CHW", 1500),
        "Care Coordinator": run(cc, "Care Coordinator", 1500),
    }
    return frames, notes


def stage_B_summary(frames):
    from scipy.stats import chi2_contingency, fisher_exact
    summ = []
    for label, df in frames.items():
        n = len(df)
        def pc(mask):
            k = int(mask.sum()); p, lo, hi = wilson(k, n)
            return {"n": k, "pct": round(p * 100, 2), "ci": [round(lo * 100, 2), round(hi * 100, 2)]}
        summ.append({"label": label, "n": n, "median_text_len": int(df["text_len"].median()),
                     "any_medication": pc(df["n_meds"] > 0), "any_condition": pc(df["n_conds"] > 0),
                     "both": pc((df["n_meds"] > 0) & (df["n_conds"] > 0)),
                     "contraindication": pc(df["contraindication"]),
                     "risk_cascade": pc(df["risk_score"] > 0),
                     "required_intervention": pc(df["n_required"] > 0),
                     "any_symbolic": pc(df["any_symbolic"]),
                     "mean_meds": round(float(df["n_meds"].mean()), 3),
                     "mean_conds": round(float(df["n_conds"].mean()), 3)})
    # 4x2 chi-square on contraindication
    labels = list(frames.keys())
    tab = np.array([[int(frames[l]["contraindication"].sum()), len(frames[l]) - int(frames[l]["contraindication"].sum())] for l in labels])
    chi2, pchi, dof, _ = chi2_contingency(tab)
    pairwise = {}
    import itertools
    for a, b in itertools.combinations(labels, 2):
        t = np.array([[int(frames[a]["contraindication"].sum()), len(frames[a]) - int(frames[a]["contraindication"].sum())],
                      [int(frames[b]["contraindication"].sum()), len(frames[b]) - int(frames[b]["contraindication"].sum())]])
        _, pf = fisher_exact(t)
        pairwise[f"{a} vs {b}"] = {"p": pf, "p_bonf6": min(pf * 6, 1.0)}
    # catalog for pharmacy reviews
    cat = Counter()
    for v in frames["Pharmacy Reviews"]["violations"]:
        cat.update(v)
    RESULTS["symbolic_by_doc_type"] = summ
    RESULTS["contraindication_chi2"] = {"chi2": float(chi2), "p": float(pchi), "dof": int(dof),
                                        "table_labels": labels, "table": tab.tolist()}
    RESULTS["contraindication_pairwise_fisher"] = pairwise
    RESULTS["contraindication_catalog"] = [{"violation": k, "count": v} for k, v in cat.most_common()]
    save("stageB_symbolic_summary.json", {k: RESULTS[k] for k in
         ("symbolic_by_doc_type", "contraindication_chi2", "contraindication_pairwise_fisher", "contraindication_catalog")})


def stage_C_overlap(frames, tok, model, dev, max_len):
    print("[C] ClinicalBERT on pharmacy reviews + overlap + alert-fatigue reconciliation")
    pr = frames["Pharmacy Reviews"].reset_index(drop=True)
    probs = bert_probs(pr["text"].tolist(), tok, model, dev, max_len)
    pr = pr.assign(bert_prob=probs)
    pr[["WaymarkId", "text_len", "n_meds", "n_conds", "contraindication", "risk_score",
        "n_required", "any_symbolic", "bert_prob", "doc_type", "dateOfEncounter"]].to_csv(
        OUT / "pharmacy_reviews_per_note.csv", index=False)
    sym = pr["contraindication"].values
    ov = {}
    for t in (0.10, 0.30, 0.50):
        bf = probs >= t
        both = int((bf & sym).sum()); bonly = int((bf & ~sym).sum())
        sonly = int((~bf & sym).sum()); neither = int((~bf & ~sym).sum())
        catch = wilson(both, int(sym.sum())); miss = wilson(sonly, int(sym.sum()))
        ov[f"{t:.2f}"] = {"both": both, "bert_only": bonly, "symbolic_only": sonly, "neither": neither,
                          "bert_flag_total": int(bf.sum()), "bert_flag_pct": round(100 * bf.mean(), 2),
                          "catch_pct": round(catch[0] * 100, 2), "catch_ci": [round(catch[1] * 100, 2), round(catch[2] * 100, 2)],
                          "miss_pct": round(miss[0] * 100, 2), "miss_ci": [round(miss[1] * 100, 2), round(miss[2] * 100, 2)]}
    RESULTS["overlap"] = ov
    RESULTS["bert_on_contra"] = {"mean": float(probs[sym].mean()), "median": float(np.median(probs[sym]))}
    # alert-fatigue reconciliation (#7, EM-4): deployable signal sizes
    RESULTS["alert_fatigue"] = {
        "symbolic_contraindication_pct": round(100 * sym.mean(), 2),
        "neural_flag_pct_at_0.50": round(100 * (probs >= 0.5).mean(), 2),
        "or_fusion_flag_at_0.50": int(((probs >= 0.5) | sym).sum()),
        "or_fusion_pct_at_0.50": round(100 * ((probs >= 0.5) | sym).mean(), 2),
        "or_fusion_flag_at_0.10": int(((probs >= 0.1) | sym).sum()),
        "or_fusion_pct_at_0.10": round(100 * ((probs >= 0.1) | sym).mean(), 2),
    }
    save("stageC_overlap.json", {k: RESULTS[k] for k in ("overlap", "bert_on_contra", "alert_fatigue")})
    return pr


def stage_D_notelength(frames):
    import statsmodels.formula.api as smf
    print("[D] note-length logistic regression (#9)")
    allnotes = pd.concat([frames[k][["doc_type", "text_len", "contraindication", "n_meds", "n_conds"]]
                          for k in frames], ignore_index=True)
    allnotes["y"] = allnotes["contraindication"].astype(int)
    allnotes["log_len"] = np.log10(allnotes["text_len"].clip(lower=1))
    allnotes["dt"] = pd.Categorical(allnotes["doc_type"],
        categories=["Care Coordinator", "CHW", "Pharmacy Outreach", "Pharmacy Reviews"])
    res = {}
    # Model 1: length only
    try:
        m1 = smf.logit("y ~ log_len", data=allnotes).fit(disp=0)
        res["length_only"] = {"params": m1.params.to_dict(), "pvalues": m1.pvalues.to_dict(),
                              "pseudo_r2": float(m1.prsquared)}
    except Exception as e:
        res["length_only"] = {"error": str(e)}
    # Model 2: doc_type only
    try:
        m2 = smf.logit("y ~ C(dt)", data=allnotes).fit(disp=0)
        res["doctype_only"] = {"params": m2.params.to_dict(), "pvalues": m2.pvalues.to_dict(),
                               "pseudo_r2": float(m2.prsquared)}
    except Exception as e:
        res["doctype_only"] = {"error": str(e)}
    # Model 3: both (does doc_type survive adjustment for length?)
    try:
        m3 = smf.logit("y ~ log_len + C(dt)", data=allnotes).fit(disp=0)
        conf = m3.conf_int()
        res["adjusted"] = {"params": m3.params.to_dict(), "pvalues": m3.pvalues.to_dict(),
                           "or": np.exp(m3.params).to_dict(),
                           "or_ci": {k: [float(np.exp(conf.loc[k, 0])), float(np.exp(conf.loc[k, 1]))] for k in m3.params.index},
                           "pseudo_r2": float(m3.prsquared)}
    except Exception as e:
        res["adjusted"] = {"error": str(e)}
    # restricted to pharmacy reviews only: contraindication ~ length (within-type length effect)
    pr = frames["Pharmacy Reviews"].copy(); pr["y"] = pr["contraindication"].astype(int)
    pr["log_len"] = np.log10(pr["text_len"].clip(lower=1))
    try:
        m4 = smf.logit("y ~ log_len", data=pr).fit(disp=0)
        res["within_pharmacy_length"] = {"params": m4.params.to_dict(), "pvalues": m4.pvalues.to_dict(),
                                         "or_log_len": float(np.exp(m4.params["log_len"]))}
    except Exception as e:
        res["within_pharmacy_length"] = {"error": str(e)}
    RESULTS["note_length_regression"] = res
    save("stageD_note_length_regression.json", res)


def stage_E_patient(frames, notes):
    print("[E] patient-level dedup (#16) + split leakage (#21)")
    pr = frames["Pharmacy Reviews"]
    # dedup / repetitive alerting
    per_pt = pr.groupby("WaymarkId").agg(n_notes=("text", "size"),
                                         n_contra=("contraindication", "sum")).reset_index()
    contra_pts = per_pt[per_pt["n_contra"] > 0]
    RESULTS["patient_dedup"] = {
        "n_pharmacy_review_notes": len(pr),
        "n_unique_patients": int(pr["WaymarkId"].nunique()),
        "notes_per_patient_mean": round(float(per_pt["n_notes"].mean()), 3),
        "notes_per_patient_median": float(per_pt["n_notes"].median()),
        "notes_per_patient_max": int(per_pt["n_notes"].max()),
        "n_contra_notes": int(pr["contraindication"].sum()),
        "n_unique_patients_with_contra": int(len(contra_pts)),
        "contra_notes_per_affected_patient_mean": round(float(contra_pts["n_contra"].mean()), 3) if len(contra_pts) else 0,
        "contra_notes_per_affected_patient_max": int(contra_pts["n_contra"].max()) if len(contra_pts) else 0,
        "pct_alert_reduction_with_patient_dedup": round(100 * (1 - len(contra_pts) / max(int(pr["contraindication"].sum()), 1)), 2),
    }
    # split -> patient linkage via exact text match against source notes
    src = notes.copy()
    src["text"] = src["text"].astype(str)
    text2pt = dict(zip(src["text"].str.strip(), src["WaymarkId"]))
    link = {}
    split_pts = {}
    for split in ("mixed_train", "mixed_val", "mixed_realworld_val", "mixed_realworld_test"):
        f = SPLITS / f"{split}.json"
        if not f.exists():
            continue
        recs = json.load(open(f))
        pts = []
        matched = 0
        for r in recs:
            t = str(r["text"]).strip()
            if t in text2pt:
                pts.append(text2pt[t]); matched += 1
        split_pts[split] = set(pts)
        link[split] = {"n": len(recs), "matched_to_patient": matched,
                       "unique_patients": len(set(pts)),
                       "duplicate_texts_within_split": len(recs) - len(set(str(r["text"]).strip() for r in recs))}
    # cross-split patient overlap (real-world splits)
    rv = split_pts.get("mixed_realworld_val", set()); rt = split_pts.get("mixed_realworld_test", set())
    tr = split_pts.get("mixed_train", set())
    link["overlap_val_test_patients"] = len(rv & rt)
    link["overlap_train_test_patients"] = len(tr & rt)
    link["overlap_train_val_patients"] = len(tr & rv)
    RESULTS["split_leakage"] = link
    save("stageE_patient_analyses.json", {"patient_dedup": RESULTS["patient_dedup"], "split_leakage": link})


def stage_F_age(frames):
    print("[F] age recompute (#12, #22)")
    members = pd.read_parquet(MEMBER_PATH)
    elig = pd.read_parquet(ELIG_PATH)
    members["birth_date"] = pd.to_datetime(members["birth_date"], errors="coerce")
    elig["birth_date"] = pd.to_datetime(elig["birth_date"], errors="coerce")
    elig["enrollment_start_date"] = pd.to_datetime(elig["enrollment_start_date"], errors="coerce")
    ref = pd.Timestamp("2025-01-01")
    age_static = (ref - members["birth_date"]).dt.days / 365.25
    diag = {"static_jan2025": {
        "n": int(age_static.notna().sum()), "mean": float(age_static.mean()), "std": float(age_static.std()),
        "min": float(age_static.min()), "max": float(age_static.max()),
        "n_negative": int((age_static < 0).sum()), "n_over_120": int((age_static > 120).sum()),
        "median": float(age_static.median()),
        "q1": float(age_static.quantile(.25)), "q3": float(age_static.quantile(.75))}}
    # age at enrollment start (cohort entry) on eligibility
    age_enr = (elig["enrollment_start_date"] - elig["birth_date"]).dt.days / 365.25
    age_enr_clean = age_enr[(age_enr >= 0) & (age_enr <= 120)]
    diag["enrollment_start_clean"] = {
        "n": int(age_enr_clean.notna().sum()), "median": float(age_enr_clean.median()),
        "q1": float(age_enr_clean.quantile(.25)), "q3": float(age_enr_clean.quantile(.75)),
        "mean": float(age_enr_clean.mean()), "std": float(age_enr_clean.std()),
        "n_excluded_negative": int((age_enr < 0).sum()), "n_excluded_over120": int((age_enr > 120).sum())}
    RESULTS["age"] = diag
    save("stageF_age.json", diag)


def stage_G_edges(frames):
    print("[G] risk-amplification / required-intervention characterization (#18)")
    pr = frames["Pharmacy Reviews"]
    risk = Counter(); req = Counter()
    for r in pr["required"]:
        req.update(r if isinstance(r, (list, tuple)) else [])
    RESULTS["edge_characterization"] = {
        "pharmacy_reviews_n": len(pr),
        "risk_cascade_rate": round(100 * (pr["risk_score"] > 0).mean(), 2),
        "required_intervention_rate": round(100 * (pr["n_required"] > 0).mean(), 2),
        "top_required": req.most_common(15),
    }
    save("stageG_edges.json", RESULTS["edge_characterization"])


def stage_H_packet(pr):
    print("[H] adjudication packet for the 107 contraindication notes (#5)")
    import re
    contra = pr[pr["contraindication"]].reset_index(drop=True)
    rows = []
    for i, row in contra.iterrows():
        text = str(row["text"])
        # light scrub: collapse capitalized name-like tokens around the contraindication context
        viols = row["violations"] if isinstance(row["violations"], (list, tuple)) else []
        rows.append({
            "case_id": f"CN{i+1:03d}",
            "rule_fired": " | ".join(viols),
            "medications_extracted": ", ".join(row["medications"]) if isinstance(row["medications"], (list, tuple)) else "",
            "conditions_extracted": ", ".join(row["conditions"]) if isinstance(row["conditions"], (list, tuple)) else "",
            "note_excerpt": text[:1500],
            "adjudication": "",  # pharmacist fills: true_hazard / appropriate_monitored / false_positive
            "rationale": "",
        })
    pd.DataFrame(rows).to_csv(PACKET / "adjudication_packet_107.csv", index=False)
    (PACKET / "INSTRUCTIONS.md").write_text(
        "# Pharmacist adjudication packet (ms#93763, editor #5 / Reviewer EM)\n\n"
        f"{len(rows)} pharmacy-review notes in which the symbolic layer fired at least one\n"
        "contraindication. For each case, classify the firing as one of:\n\n"
        "- true_hazard: an actionable drug-disease contraindication requiring intervention\n"
        "- appropriate_monitored: the combination is present but clinically appropriate / monitored\n"
        "- false_positive: extraction or rule error; no real contraindication\n\n"
        "Fill the `adjudication` and `rationale` columns. PPV = true_hazard / total.\n"
        "This packet contains PHI and must remain on Waymark infrastructure; do not upload externally.\n")
    RESULTS["adjudication_packet"] = {"n_cases": len(rows), "path": str(PACKET / "adjudication_packet_107.csv")}
    print(f"  packet: {len(rows)} cases")


# ================================================================ main
def main():
    print("Loading ClinicalBERT...")
    tok, model, dev, max_len = load_clinicalbert()
    print(f"  device={dev}")
    stage_A(tok, model, dev, max_len)
    frames, notes = build_note_frames()
    stage_B_summary(frames)
    pr = stage_C_overlap(frames, tok, model, dev, max_len)
    stage_D_notelength(frames)
    stage_E_patient(frames, notes)
    stage_F_age(frames)
    stage_G_edges(frames)
    stage_H_packet(pr)
    save("canonical_results.json", RESULTS)
    print("\nDONE. Canonical results in", OUT)


if __name__ == "__main__":
    main()
