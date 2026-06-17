"""
Evaluate neurosymbolic system on real-world pharmacy documentation.

Compares symbolic knowledge-graph contraindication detection and ClinicalBERT
hazard classification across four documentation types:
  - Pharmacy reviews (PHARMACY_NEW_PATIENT_REVIEW, PHARMACIST_CONSULTATION)
  - Pharmacy outreach (PATIENT_OUTREACH by pharmacy staff)
  - CHW notes (by CHW/CHW_LEAD staff)
  - Care coordinator notes (by CARE_COORDINATOR/CARE_COORDINATOR_LEAD staff)

Outputs:
  - results/pharmacy_evaluation_results.json  (all metrics)
  - figures/figure2_symbolic_by_notetype.png   (bar chart)
  - figures/figure3_overlap_analysis.png       (overlap heatmap)
"""

import json
import sys
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "models"))
sys.path.append(str(REPO_ROOT / "scripts"))

from extraction import ClinicalExtractor
from neurosymbolic_reasoner import SymbolicReasoner, ClinicalContext
from train_clinicalbert_hybrid import ClinicalBERTClassifier, MODEL_NAME, MAX_LEN
from transformers import AutoTokenizer

# Paths
DATA_ROOT = REPO_ROOT.parent.parent / "data" / "real_inputs"
NOTES_PATH = DATA_ROOT / "notes" / "encounter notes.csv"
MEMBER_PATH = DATA_ROOT / "member_attributes.parquet"
ELIGIBILITY_PATH = DATA_ROOT / "eligibility.parquet"
KG_DIR = str(REPO_ROOT.parent.parent / "notebooks" / "neurosymbolic" / "knowledge_graphs")
RULES_PATH = str(REPO_ROOT / "data" / "rules" / "clinical_rules_expanded.json")
VOCAB_DIR = str(REPO_ROOT / "data")
MODEL_DIR = REPO_ROOT / "trained_models"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT.parent.parent / "notebooks" / "neurosymbolic" / "submission" / "jmir" / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_notes():
    """Load and categorize encounter notes by documentation type."""
    notes = pd.read_csv(NOTES_PATH, low_memory=False)
    notes = notes[notes["text"].str.len() > 50].reset_index(drop=True)

    pharmacy_reviews = notes[
        notes["encounterType"].isin(
            ["PHARMACY_NEW_PATIENT_REVIEW", "PHARMACIST_CONSULTATION"]
        )
    ].copy()

    pharmacy_outreach = notes[
        (notes["title"].str.contains("PHARM", case=False, na=False))
        & (notes["encounterType"] == "PATIENT_OUTREACH")
    ].copy()

    chw_notes = notes[
        notes["title"].str.contains("^CHW$|^CHW_LEAD$", case=False, na=False, regex=True)
    ].copy()

    cc_notes = notes[
        notes["title"].str.contains(
            "CARE_COORDINATOR|CARE_COORDINATOR_LEAD", case=False, na=False
        )
    ].copy()

    return {
        "pharmacy_reviews": pharmacy_reviews,
        "pharmacy_outreach": pharmacy_outreach,
        "chw_notes": chw_notes,
        "cc_notes": cc_notes,
        "all_notes": notes,
    }


def run_symbolic_analysis(subset, extractor, symbolic, label, max_n=None):
    """Run symbolic layer on a set of notes. Returns detailed results."""
    if max_n and len(subset) > max_n:
        subset = subset.sample(max_n, random_state=SEED).reset_index(drop=True)

    records = []
    for i, row in subset.iterrows():
        text = str(row["text"])
        extraction = extractor.extract(text)

        ctx = ClinicalContext(
            patient_id=row.get("WaymarkId", "unknown"),
            conditions=extraction.conditions,
            medications=extraction.medications,
            recent_encounters=[],
            goals=[],
            demographics={},
            risk_factors=[],
            current_state={},
        )

        is_safe, violations = symbolic.check_contraindications(
            ctx, "medication_review"
        )
        risk_scores = symbolic.compute_risk_cascade(ctx)
        max_risk = max(risk_scores.values()) if risk_scores else 0.0
        required = symbolic.find_required_interventions(ctx)

        records.append(
            {
                "WaymarkId": row.get("WaymarkId", "unknown"),
                "text_len": len(text),
                "n_meds": len(extraction.medications),
                "n_conds": len(extraction.conditions),
                "medications": extraction.medications,
                "conditions": extraction.conditions,
                "contraindication": not is_safe,
                "n_violations": len(violations) if violations else 0,
                "violations": violations if violations else [],
                "risk_score": max_risk,
                "n_required": len(required),
                "any_symbolic": (not is_safe) or max_risk > 0 or len(required) > 0,
            }
        )

    df = pd.DataFrame(records)
    n = len(df)

    summary = {
        "label": label,
        "n": n,
        "has_medication": int((df["n_meds"] > 0).sum()),
        "has_condition": int((df["n_conds"] > 0).sum()),
        "has_both": int(((df["n_meds"] > 0) & (df["n_conds"] > 0)).sum()),
        "contraindication_count": int(df["contraindication"].sum()),
        "risk_cascade_count": int((df["risk_score"] > 0).sum()),
        "required_intervention_count": int((df["n_required"] > 0).sum()),
        "any_symbolic_count": int(df["any_symbolic"].sum()),
        "has_medication_pct": float((df["n_meds"] > 0).mean()) * 100,
        "has_condition_pct": float((df["n_conds"] > 0).mean()) * 100,
        "has_both_pct": float(((df["n_meds"] > 0) & (df["n_conds"] > 0)).mean())
        * 100,
        "contraindication_pct": float(df["contraindication"].mean()) * 100,
        "risk_cascade_pct": float((df["risk_score"] > 0).mean()) * 100,
        "required_intervention_pct": float((df["n_required"] > 0).mean()) * 100,
        "any_symbolic_pct": float(df["any_symbolic"].mean()) * 100,
        "median_text_len": int(df["text_len"].median()),
        "mean_meds_per_note": float(df["n_meds"].mean()),
        "mean_conds_per_note": float(df["n_conds"].mean()),
    }

    return summary, df


def run_clinicalbert_on_notes(subset, tokenizer, model, device):
    """Run ClinicalBERT on a set of notes. Returns probabilities."""
    probs = []
    with torch.no_grad():
        for _, row in subset.iterrows():
            text = str(row["text"])[:512]
            enc = tokenizer(
                text,
                truncation=True,
                max_length=MAX_LEN,
                padding="max_length",
                return_tensors="pt",
            )
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            prob = torch.sigmoid(logits).cpu().numpy().item()
            probs.append(prob)
    return probs


def catalog_violations(df):
    """Catalog all contraindication types and their frequencies."""
    all_violations = []
    for violations in df["violations"]:
        if violations:
            all_violations.extend(violations)
    counts = Counter(all_violations)
    catalog = []
    for v, count in counts.most_common():
        catalog.append({"violation": v, "count": count})
    return catalog


def overlap_analysis(df_symbolic, bert_probs, thresholds=[0.10, 0.30, 0.50]):
    """Analyze overlap between symbolic and ClinicalBERT detection."""
    results = {}
    for t in thresholds:
        bert_flag = [p >= t for p in bert_probs]
        sym_flag = df_symbolic["contraindication"].tolist()

        both = sum(b and s for b, s in zip(bert_flag, sym_flag))
        bert_only = sum(b and not s for b, s in zip(bert_flag, sym_flag))
        sym_only = sum(not b and s for b, s in zip(bert_flag, sym_flag))
        neither = sum(not b and not s for b, s in zip(bert_flag, sym_flag))

        results[f"threshold_{t:.2f}"] = {
            "bert_and_symbolic": both,
            "bert_only": bert_only,
            "symbolic_only": sym_only,
            "neither": neither,
            "bert_total": both + bert_only,
            "symbolic_total": both + sym_only,
        }

        # BERT scores on contraindication notes
        contra_probs = [
            p for p, s in zip(bert_probs, sym_flag) if s
        ]
        if contra_probs:
            results[f"threshold_{t:.2f}"]["bert_on_contra_mean"] = float(
                np.mean(contra_probs)
            )
            results[f"threshold_{t:.2f}"]["bert_on_contra_median"] = float(
                np.median(contra_probs)
            )
            results[f"threshold_{t:.2f}"]["bert_catches_contra"] = sum(
                p >= t for p in contra_probs
            )
            results[f"threshold_{t:.2f}"]["bert_misses_contra"] = sum(
                p < t for p in contra_probs
            )

    return results


def generate_figures(symbolic_summaries, overlap_results):
    """Generate publication-quality figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "font.family": "serif"})

    # Figure 2: Symbolic firing rates by note type
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [s["label"] for s in symbolic_summaries]
    contra_rates = [s["contraindication_pct"] for s in symbolic_summaries]
    risk_rates = [s["risk_cascade_pct"] for s in symbolic_summaries]
    req_rates = [s["required_intervention_pct"] for s in symbolic_summaries]

    x = np.arange(len(labels))
    width = 0.25

    bars1 = ax.bar(x - width, contra_rates, width, label="Contraindication", color="#c0392b")
    bars2 = ax.bar(x, risk_rates, width, label="Risk Cascade", color="#e67e22")
    bars3 = ax.bar(x + width, req_rates, width, label="Required Intervention", color="#2980b9")

    ax.set_ylabel("Firing Rate (%)")
    ax.set_title("")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(max(contra_rates), max(risk_rates), max(req_rates)) * 1.15)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Add sample sizes
    for i, s in enumerate(symbolic_summaries):
        ax.annotate(
            f"n={s['n']}",
            xy=(i, -2),
            ha="center",
            fontsize=9,
            color="gray",
        )

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "Figure2_SymbolicFiring.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 2 to {FIGURES_DIR / 'Figure2_SymbolicFiring.png'}")

    # Figure 3: Overlap analysis (ClinicalBERT vs Symbolic at threshold 0.50)
    fig, ax = plt.subplots(figsize=(7, 5))
    ov = overlap_results["threshold_0.50"]

    data = [
        [ov["bert_and_symbolic"], ov["bert_only"]],
        [ov["symbolic_only"], ov["neither"]],
    ]
    data_arr = np.array(data)

    im = ax.imshow(data_arr, cmap="YlOrRd", aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Symbolic\nContraindication", "No Symbolic\nContraindication"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["ClinicalBERT\nHazard (>=0.50)", "ClinicalBERT\nNo Hazard (<0.50)"])

    for i in range(2):
        for j in range(2):
            text = ax.text(
                j,
                i,
                f"{data_arr[i, j]}",
                ha="center",
                va="center",
                color="white" if data_arr[i, j] > 200 else "black",
                fontsize=16,
                fontweight="bold",
            )

    ax.set_title("ClinicalBERT vs Symbolic Detection on Pharmacy Reviews (n=1,519)")
    fig.colorbar(im, ax=ax, label="Number of Notes")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "Figure3_Overlap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3 to {FIGURES_DIR / 'Figure3_Overlap.png'}")


def population_demographics():
    """Compute population demographics."""
    members = pd.read_parquet(MEMBER_PATH)
    eligibility = pd.read_parquet(ELIGIBILITY_PATH)

    # Age from birth_date
    ref_date = pd.Timestamp("2025-01-01")
    members["age"] = (ref_date - members["birth_date"]).dt.days / 365.25

    demo = {
        "n_members": len(members),
        "age_mean": float(members["age"].mean()),
        "age_std": float(members["age"].std()),
        "gender": members["gender"].value_counts().to_dict(),
        "ethnicity": members["ethnicity"].value_counts().to_dict() if "ethnicity" in members.columns else {},
    }

    # State from eligibility
    demo["state"] = eligibility["state"].value_counts().to_dict()
    demo["payer"] = eligibility["payer"].value_counts().to_dict()
    demo["race"] = eligibility["race"].value_counts().to_dict()

    return demo


def main():
    print("Loading notes...")
    note_sets = load_notes()

    print(f"Total notes (>50 chars): {len(note_sets['all_notes'])}")
    print(f"Pharmacy reviews: {len(note_sets['pharmacy_reviews'])}")
    print(f"Pharmacy outreach: {len(note_sets['pharmacy_outreach'])}")
    print(f"CHW notes: {len(note_sets['chw_notes'])}")
    print(f"CC notes: {len(note_sets['cc_notes'])}")

    # Note volume summary
    all_notes = note_sets["all_notes"]
    note_volumes = {
        "total": len(all_notes),
        "unique_patients": int(all_notes["WaymarkId"].nunique()),
        "by_encounter_type": all_notes["encounterType"].value_counts().head(15).to_dict(),
        "by_title": all_notes["title"].value_counts().head(15).to_dict(),
    }

    # Initialize components
    print("Initializing extractor and symbolic reasoner...")
    extractor = ClinicalExtractor(VOCAB_DIR)
    symbolic = SymbolicReasoner(KG_DIR, rules_path=RULES_PATH)

    # Run symbolic analysis on each note type
    print("\nRunning symbolic analysis on pharmacy reviews...")
    pharm_review_summary, pharm_review_df = run_symbolic_analysis(
        note_sets["pharmacy_reviews"], extractor, symbolic, "Pharmacy Reviews"
    )

    print("Running symbolic analysis on pharmacy outreach (sample 1500)...")
    pharm_outreach_summary, pharm_outreach_df = run_symbolic_analysis(
        note_sets["pharmacy_outreach"], extractor, symbolic, "Pharmacy Outreach", max_n=1500
    )

    print("Running symbolic analysis on CHW notes (sample 1500)...")
    chw_summary, chw_df = run_symbolic_analysis(
        note_sets["chw_notes"], extractor, symbolic, "CHW Notes", max_n=1500
    )

    print("Running symbolic analysis on CC notes (sample 1500)...")
    cc_summary, cc_df = run_symbolic_analysis(
        note_sets["cc_notes"], extractor, symbolic, "Care Coordinator Notes", max_n=1500
    )

    symbolic_summaries = [
        pharm_review_summary,
        pharm_outreach_summary,
        chw_summary,
        cc_summary,
    ]

    # Catalog contraindication types
    print("\nCataloging contraindication types...")
    violation_catalog = catalog_violations(pharm_review_df)

    # Run ClinicalBERT on pharmacy reviews
    print("\nRunning ClinicalBERT on pharmacy reviews...")
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = ClinicalBERTClassifier(MODEL_NAME).to(device)
    state = torch.load(MODEL_DIR / "clinicalbert_mixed.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    bert_probs = run_clinicalbert_on_notes(
        note_sets["pharmacy_reviews"].reset_index(drop=True),
        tokenizer,
        model,
        device,
    )

    bert_summary = {
        "n": len(bert_probs),
        "mean_prob": float(np.mean(bert_probs)),
        "median_prob": float(np.median(bert_probs)),
    }
    for t in [0.10, 0.20, 0.30, 0.50]:
        flagged = sum(1 for p in bert_probs if p >= t)
        bert_summary[f"flagged_at_{t:.2f}"] = flagged
        bert_summary[f"flagged_at_{t:.2f}_pct"] = flagged / len(bert_probs) * 100

    # Overlap analysis
    print("Running overlap analysis...")
    overlap = overlap_analysis(pharm_review_df, bert_probs)

    # Population demographics
    print("Computing demographics...")
    demographics = population_demographics()

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(symbolic_summaries, overlap)

    # Compile results
    results = {
        "population": demographics,
        "note_volumes": note_volumes,
        "symbolic_by_note_type": symbolic_summaries,
        "contraindication_catalog": violation_catalog,
        "clinicalbert_on_pharmacy_reviews": bert_summary,
        "overlap_analysis": overlap,
    }

    # Save
    out_path = RESULTS_DIR / "pharmacy_evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("PHARMACY EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nPopulation: {demographics['n_members']:,} Medicaid enrollees")
    print(f"Total notes analyzed: {note_volumes['total']:,}")
    print(
        f"\n{'Note Type':<25} {'n':>6} {'Med%':>7} {'Cond%':>7} {'Both%':>7} "
        f"{'Contra%':>8} {'Risk%':>7} {'Req%':>7} {'Any%':>7}"
    )
    print("-" * 85)
    for s in symbolic_summaries:
        print(
            f"{s['label']:<25} {s['n']:>6} {s['has_medication_pct']:>6.1f}% "
            f"{s['has_condition_pct']:>6.1f}% {s['has_both_pct']:>6.1f}% "
            f"{s['contraindication_pct']:>7.1f}% {s['risk_cascade_pct']:>6.1f}% "
            f"{s['required_intervention_pct']:>6.1f}% {s['any_symbolic_pct']:>6.1f}%"
        )

    print(f"\nContraindication catalog ({len(violation_catalog)} types):")
    for v in violation_catalog:
        print(f"  {v['count']:>3}x  {v['violation']}")

    print(f"\nClinicalBERT on pharmacy reviews:")
    print(f"  Mean prob: {bert_summary['mean_prob']:.3f}")
    print(f"  Flagged at 0.50: {bert_summary['flagged_at_0.50']} ({bert_summary['flagged_at_0.50_pct']:.1f}%)")

    ov50 = overlap["threshold_0.50"]
    print(f"\nOverlap (threshold 0.50):")
    print(f"  Both detect: {ov50['bert_and_symbolic']}")
    print(f"  BERT only: {ov50['bert_only']}")
    print(f"  Symbolic only: {ov50['symbolic_only']}")
    print(f"  Neither: {ov50['neither']}")


if __name__ == "__main__":
    main()
