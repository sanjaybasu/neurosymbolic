"""Comment #9 — does note length explain the pharmacy-review contraindication rate?
Handles the perfect separation (Care Coordinator has 0 events) with a binary
pharmacy indicator + L2-penalized logistic, and reports within-type length effects.
Reads the per-note frames already produced by script 01 (pharmacy_reviews_per_note.csv)
and rebuilds the other doc types' length+contraindication from stageB (recompute light)."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

WL = Path("/Users/sanjaybasu/waymark-local")
REPO = WL / "packaging" / "neurosymbolic_github"
OUT = WL / "notebooks/neurosymbolic/submission/jmir/revision_1/audit/derived"
sys.path.append(str(REPO / "models")); sys.path.append(str(REPO / "scripts"))

SEED = 42
np.random.seed(SEED)
import random; random.seed(SEED)


def build_all():
    from extraction import ClinicalExtractor
    from neurosymbolic_reasoner import SymbolicReasoner, ClinicalContext
    notes = pd.read_csv(WL / "data/real_inputs/notes/encounter notes.csv", low_memory=False)
    notes = notes[notes["text"].str.len() > 50].reset_index(drop=True)
    subsets = {
        "Pharmacy Reviews": notes[notes["encounterType"].isin(["PHARMACY_NEW_PATIENT_REVIEW", "PHARMACIST_CONSULTATION"])],
        "Pharmacy Outreach": notes[(notes["title"].str.contains("PHARM", case=False, na=False)) & (notes["encounterType"] == "PATIENT_OUTREACH")],
        "CHW": notes[notes["title"].str.contains("^CHW$|^CHW_LEAD$", case=False, na=False, regex=True)],
        "Care Coordinator": notes[notes["title"].str.contains("CARE_COORDINATOR|CARE_COORDINATOR_LEAD", case=False, na=False)],
    }
    ext = ClinicalExtractor(str(REPO / "data"))
    sym = SymbolicReasoner(str(WL / "notebooks/neurosymbolic/knowledge_graphs"), rules_path=str(REPO / "data/rules/clinical_rules_expanded.json"))
    rows = []
    for label, sub in subsets.items():
        if label != "Pharmacy Reviews" and len(sub) > 1500:
            sub = sub.sample(1500, random_state=SEED)
        for _, r in sub.iterrows():
            t = str(r["text"]); e = ext.extract(t)
            ctx = ClinicalContext(patient_id="x", conditions=e.conditions, medications=e.medications,
                                  recent_encounters=[], goals=[], demographics={}, risk_factors=[], current_state={})
            safe, _ = sym.check_contraindications(ctx, "medication_review")
            rows.append({"doc_type": label, "text_len": len(t), "y": 0 if safe else 1,
                         "n_meds": len(e.medications), "n_conds": len(e.conditions),
                         "both": int(len(e.medications) > 0 and len(e.conditions) > 0)})
    return pd.DataFrame(rows)


def main():
    df = build_all()
    df["log_len"] = np.log10(df["text_len"].clip(lower=1))
    df["is_pharmacy"] = (df["doc_type"] == "Pharmacy Reviews").astype(int)
    res = {"n_total": len(df), "n_events": int(df["y"].sum()), "events_by_type": df.groupby("doc_type")["y"].sum().to_dict()}

    # Model 1: length only (all notes)
    m1 = smf.logit("y ~ log_len", data=df).fit(disp=0)
    res["m1_length_only"] = {"or_log_len": float(np.exp(m1.params["log_len"])), "p": float(m1.pvalues["log_len"]), "pseudo_r2": float(m1.prsquared)}

    # Model 2: pharmacy indicator + length (L2-penalized to handle separation)
    X = sm.add_constant(df[["log_len", "is_pharmacy"]].astype(float))
    try:
        m2 = sm.Logit(df["y"], X).fit_regularized(alpha=1.0, disp=0)
        res["m2_pharm_plus_length_L2"] = {"params": m2.params.to_dict(),
                                          "or_log_len": float(np.exp(m2.params["log_len"])),
                                          "or_is_pharmacy": float(np.exp(m2.params["is_pharmacy"]))}
    except Exception as e:
        res["m2_pharm_plus_length_L2"] = {"error": str(e)}

    # Model 3: among notes that have BOTH a med and a condition (the co-mention precondition),
    # does pharmacy status still matter, adjusting for length? (mechanism test)
    sub = df[df["both"] == 1].copy()
    res["both_subset_n"] = len(sub); res["both_subset_events"] = int(sub["y"].sum())
    try:
        m3 = smf.logit("y ~ log_len + is_pharmacy", data=sub).fit(disp=0)
        conf = m3.conf_int()
        res["m3_within_comention"] = {
            "or_log_len": float(np.exp(m3.params["log_len"])), "p_log_len": float(m3.pvalues["log_len"]),
            "or_is_pharmacy": float(np.exp(m3.params["is_pharmacy"])), "p_is_pharmacy": float(m3.pvalues["is_pharmacy"]),
            "or_is_pharmacy_ci": [float(np.exp(conf.loc["is_pharmacy", 0])), float(np.exp(conf.loc["is_pharmacy", 1]))],
            "pseudo_r2": float(m3.prsquared)}
    except Exception as e:
        res["m3_within_comention"] = {"error": str(e)}

    # Model 4: within pharmacy reviews only, length effect
    pr = df[df["doc_type"] == "Pharmacy Reviews"]
    m4 = smf.logit("y ~ log_len", data=pr).fit(disp=0)
    res["m4_within_pharmacy_length"] = {"or_log_len": float(np.exp(m4.params["log_len"])), "p": float(m4.pvalues["log_len"])}

    # correlation of length and co-mention
    res["corr_loglen_both"] = float(df["log_len"].corr(df["both"]))
    res["median_len_by_type"] = df.groupby("doc_type")["text_len"].median().to_dict()

    (OUT / "stageD2_note_length_fixed.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
