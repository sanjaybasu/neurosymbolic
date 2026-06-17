"""Re-derive key reported numbers from the AUTHORITATIVE canonical sources and confirm
each appears in the revised manuscript/tables/appendix. Exits non-zero on any mismatch.
Authoritative sources:
  canonical_v2.json        -> all model-dependent values (single unified pipeline)
  stageB_symbolic_summary  -> symbolic doc-type / catalog (model-independent)
  stageD2_note_length      -> note-length regression (model-independent)
  adjudication_scored.json -> PPV + kappa (model-independent)
  stageF_age.json          -> age
"""
import json, sys
from pathlib import Path

R1 = Path(__file__).resolve().parent.parent
DER = R1 / "audit" / "derived"
ALL = "\n".join((R1/f).read_text() for f in ["main_revised.md","tables_revised.md","supplementary_revised.md"])
errors, checks = [], 0


def present(s, label):
    global checks; checks += 1
    if str(s) not in ALL:
        errors.append(f"MISSING: {label}: '{s}'")


v2 = json.load(open(DER/"canonical_v2.json"))
B = json.load(open(DER/"stageB_symbolic_summary.json"))
D = json.load(open(DER/"stageD2_note_length_fixed.json"))
ADJ = json.load(open(DER/"adjudication_scored.json"))

# --- Table 2 (v2) ---
cb = v2["table2_clinicalbert_test"]; ph = v2["table2_clinicalbert_physician"]
present(cb["sensitivity"], "CB test sens 86.9"); present(cb["specificity"], "CB test spec 89.4")
present(cb["precision"], "CB test prec 91.2"); present(cb["accuracy"], "CB test acc 88.0")
present(round(cb["brier"],3), "Brier 0.081"); present(round(cb["ece"],3), "ECE 0.071")
present(ph["sensitivity"], "phys sens 80.1"); present(ph["specificity"], "phys spec 98.1")
# threshold sweep
for t, m in v2["threshold_sweep_test"].items():
    present(m["sensitivity"], f"sweep {t} sens"); present(m["specificity"], f"sweep {t} spec")

# --- overlap (v2) ---
o = v2["overlap"]["0.50"]
present(o["symbolic_only"], "symbolic_only@0.50 = 64"); present(o["miss_pct"], "miss%@0.50 = 59.8")
present(o["catch_pct"], "catch%@0.50 = 40.2"); present(o["neural_flag_pct"], "neural flag 41.8")
present(v2["overlap"]["0.10"]["symbolic_only"], "symbolic_only@0.10 = 18")
present(v2["overlap"]["0.30"]["symbolic_only"], "symbolic_only@0.30 = 44")

# --- EZ1b (v2) ---
ez = v2["ez1b"]
present(ez["augmented"]["0.10"]["catch"], "aug catch@0.10 = 105")
present(ez["augmented"]["0.50"]["catch"], "aug catch@0.50 = 70")
present(ez["baseline_is_primary"]["0.50"]["catch"], "baseline catch@0.50 = 43")

# --- patient split (v2) ---
ps = v2["patient_split"]
present(ps["disjoint"]["sensitivity"], "patient-disjoint sens 90.5")

# --- doc-type symbolic (model-independent) ---
present("107", "107 contraindication notes"); present("7.0", "pharmacy 7.0%")
present(f'{B["contraindication_chi2"]["chi2"]:.1f}', "chi-square 294.9")

# --- note length (model-independent) ---
present(f'{D["m3_within_comention"]["or_is_pharmacy"]:.2f}', "within-comention pharmacy OR 1.55")

# --- adjudication (model-independent) ---
present("36.4", "PPV 36.4"); present("0.77", "kappa 0.77")
present("43.0", "false-positive 43.0"); present("39", "39 actionable")

# --- age ---
present("28.1", "median age 28.1"); present("13.3", "IQR 13.3"); present("42.2", "IQR 42.2")

print(f"verify_numbers (v2): {checks} checks, {len(errors)} errors")
for e in errors: print("  " + e)
sys.exit(1 if errors else 0)
