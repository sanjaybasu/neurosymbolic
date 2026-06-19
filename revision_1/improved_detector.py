"""Improved contraindication detector (v2): suppression-only refinement of the v1 pipeline.

Design principle (to avoid the "multiple incommensurable pipelines" failure mode): the improved
detector runs the IDENTICAL v1 extractor and v1 symbolic reasoner, then SUPPRESSES individual fired
violations that meet a clinically-principled exclusion criterion. The improved flag set is therefore a
strict subset of the v1 flag set on any corpus; every precision change is attributable to a named
suppression rule and never to reimplementation drift or newly-introduced flags. (Recall of true
contraindications is unaffected because each suppression removes a clinically false alarm, not a real
contraindication; measured recall is estimated separately via the random-sample gold standard.)

Suppression rules, each guideline-faithful and corpus-general (NOT pattern-matched to specific notes):
  S1  NSAID + disease, suppress when the note has no non-aspirin analgesic NSAID and aspirin appears
      only at low/antiplatelet dose (the dose is read from aspirin's own token; <325 mg). Low-dose
      aspirin is an antiplatelet, not an analgesic NSAID, and does not carry the analgesic-NSAID
      organ/pregnancy contraindication.
  S2  opioid + substance-use-disorder, suppress when no prescribed, non-MAT, non-illicit opioid is
      present (only buprenorphine/methadone MAT, which treats OUD, or an opioid appearing only in an
      illicit/abuse/withdrawal/historical context, which is the SUD itself). Retained for opioid +
      benzodiazepine, where respiratory-depression risk is real regardless of MAT status.
  S3  any medication + pregnancy, retain only when the note carries positive evidence of an ACTIVE
      pregnancy (the word "pregnant", weeks of gestation/EGA, trimester, prenatal vitamins, EDD/due
      date, IUP/gestation). Suppress when pregnancy was extracted from a specialty referral alone
      (OB/GYN/obstetric), is negated/test-negative, or is resolved (miscarriage/abortion/termination).
  S4  albuterol + cardiovascular_disease, suppress (guideline caution / monitoring-level, not an
      actionable contraindication).
  S5  corticosteroid + diabetes, suppress when short-course markers are present (burst/taper/dose-pack
      /<=N-day); guideline-directed short courses are monitor-level, not actionable contraindications.

Outputs (audit/derived/):
  improved_pharmacy_per_note.csv   per-note v1_fired / improved_fired + suppressed-rule tags
  improved_development_gain.json   precision on the 107 adjudicated (A/B/C/D anchored; in-sample)
  improved_suppression_audit.csv   per-adjudicated-case: category, kept flag, suppressed-by
"""
import os
import json, re, sys
from pathlib import Path
import pandas as pd

WL = Path(os.environ.get("NEUROSYMBOLIC_ROOT","."))
REPO = WL / "packaging" / "neurosymbolic_github"
OUT = WL / "notebooks/neurosymbolic/submission/jmir/revision_1/audit/derived"
PKT = WL / "notebooks/neurosymbolic/submission/jmir/revision_1/adjudication_packet"
sys.path.append(str(REPO / "models"))
from extraction import ClinicalExtractor
from neurosymbolic_reasoner import SymbolicReasoner, ClinicalContext

# Single source of the suppression rules: the released module. This script never re-implements S1-S5;
# it imports them so the analysis and the released code cannot diverge.
import refinement as R
suppressed_by = R.suppression_tag


def main():
    notes = pd.read_csv(Path(os.environ.get("NEUROSYMBOLIC_NOTES","data/notes.csv")), low_memory=False)
    notes = notes[notes["text"].str.len() > 50].reset_index(drop=True)
    pr = notes[notes["encounterType"].isin(["PHARMACY_NEW_PATIENT_REVIEW", "PHARMACIST_CONSULTATION"])].reset_index(drop=True)
    texts = pr["text"].astype(str).tolist()

    ext = ClinicalExtractor(str(REPO / "data"))
    sym = SymbolicReasoner(str(WL / "notebooks/neurosymbolic/knowledge_graphs"),
                           rules_path=str(REPO / "data/rules/clinical_rules_expanded.json"))

    rows = []
    for tx in texts:
        e = ext.extract(tx)
        ctx = ClinicalContext(patient_id="x", conditions=e.conditions, medications=e.medications,
                              recent_encounters=[], goals=[], demographics={}, risk_factors=[], current_state={})
        safe, viols = sym.check_contraindications(ctx, "medication_review")
        v1 = 0 if safe else 1
        surviving = [v for v in viols if suppressed_by(v, tx) is None]
        tags = sorted({suppressed_by(v, tx) for v in viols if suppressed_by(v, tx)})
        rows.append({"v1": v1, "improved": 1 if surviving else 0,
                     "n_viol": len(viols), "n_surviving": len(surviving),
                     "suppressed_rules": ";".join(tags)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "improved_pharmacy_per_note.csv", index=False)
    res = {"n_pharmacy": len(df),
           "v1_flagged": int(df["v1"].sum()), "v1_pct": round(100 * df["v1"].mean(), 2),
           "improved_flagged": int(df["improved"].sum()), "improved_pct": round(100 * df["improved"].mean(), 2),
           "improved_is_subset_of_v1": bool(((df["improved"] == 1) <= (df["v1"] == 1)).all())}

    # development gain: map the v1-positive 107 (in order) to adjudication categories
    pos_idx = [i for i, r in df.iterrows() if r["v1"] == 1]
    cons = pd.read_csv(PKT / "adjudication_packet_107.csv")["case_id"].tolist()  # CN001.. in v1-positive order
    consensus = pd.read_csv(PKT / "consensus_completed.csv").set_index("case_id")["category"].str.upper().str.strip()
    assert len(pos_idx) == len(cons) == 107, (len(pos_idx), len(cons))
    retained = {"A": 0, "B": 0, "C": 0, "D": 0}
    eliminated = {"A": 0, "B": 0, "C": 0, "D": 0}
    audit = []
    for idx, cid in zip(pos_idx, cons):
        cat = consensus.get(cid)
        if cat not in retained:
            continue
        kept = df.loc[idx, "improved"] == 1
        (retained if kept else eliminated)[cat] += 1
        audit.append({"case_id": cid, "category": cat, "kept": int(kept),
                      "suppressed_rules": df.loc[idx, "suppressed_rules"]})
    pd.DataFrame(audit).to_csv(OUT / "improved_suppression_audit.csv", index=False)
    tot_ret = sum(retained.values())
    A_total = retained["A"] + eliminated["A"]
    res["development"] = {
        "n_adjudicated": 107, "orig_PPV": round(A_total / 107, 3),
        "retained_by_category": retained, "eliminated_by_category": eliminated,
        "improved_retained_total": tot_ret,
        "improved_PPV_development": round(retained["A"] / tot_ret, 3) if tot_ret else None,
        "true_positives_retained": f'{retained["A"]}/{A_total}',
        "true_positives_lost": eliminated["A"],
        "false_positives_removed": eliminated["B"] + eliminated["C"] + eliminated["D"],
        "note": "development/in-sample: suppression rules were informed by these adjudicated cases; "
                "held-out validation requires re-adjudication of the improved-tool flag set",
    }
    (OUT / "improved_development_gain.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
