# Revision 1 analysis code

Analysis scripts used for the first revision. They reproduce the reported tables
and numbers and run the additional analyses requested in review. They require local
protected health information that is not distributed with this repository; set the
data root for your environment before running. No patient data, model outputs, or
manuscript text are included here.

## Primary pipeline and requested analyses

- `01_canonical_analysis.py`: single end-to-end pipeline that trains one model (seed 42) and reproduces the model-dependent tables and the new analyses.
- `01b_note_length_fixed.py`: logistic regression of contraindication detection on note length and documentation type.
- `02_symbolic_augmented_neural.py`: neural classifier trained on symbolic-generated labels.
- `03_retraining_experiments.py`: OR-fusion ablation, symbolic-augmented neural, patient-level split, TF-IDF, and encoder comparison.
- `05_unified_canonical.py`: assembles the single canonical results object (`canonical_v2`) that every model-dependent value in the manuscript is read from.
- `04_figures.py`: figure generation from the derived outputs.

## Error analysis, refinement, and precision

- `../models/refinement.py`: the released suppression module (rules S1-S5). It operates only by suppressing individual fired detections, so the refined flag set is a strict subset of the original (v1 = 107, v2 = 67 on the pharmacy-review corpus). Single source of the suppression logic used below.
- `improved_detector.py`: applies `refinement.py` across the corpus and writes the per-detection suppression audit (which detections are kept versus suppressed, and by which rule).
- `stat_harness.py`: the refined positive predictive value, the hypergeometric enrichment test for the suppressed detections, the seeded bootstrap percentile interval for the change in positive predictive value (seed 20260617, 10,000 resamples), and the temporal-stability split.

## Verification-bias-corrected recall and held-out precision

- `gold_standard_sample.py`: draws the seeded random sample of unflagged pharmacy reviews and the census of suppressed notes for missed-contraindication adjudication.
- `held_out_packet.py`: builds the blinded re-adjudication packet of refined detections.
- `verification_bias_estimator.py`: the two-phase Horvitz-Thompson estimator (sampled unflagged stratum plus suppressed-note census, the census treated as zero-variance) that recovers recall, specificity, and prevalence with Wilson-propagated intervals. Reads the adjudication packets; with blank packets it emits a labeled placeholder and does not write the canonical recall file.
- `score_round2_final.py`: reads the completed rater and reconciliation files, computes the dual-rater consensus and agreement (Cohen kappa), runs the estimator, and writes the canonical recall, specificity, prevalence, and held-out precision values reported in the manuscript.
- `held_out_eval.py`: scores the completed blinded re-adjudication (held-out positive predictive value).

## Audits

- `verify_numbers.py`: re-derives every registered reportable value from the canonical outputs and fails on any mismatch.
- `consistency_audit.py`: scans for stale values, internal contradictions, citation order, and required cross-references.
- `editor_pattern_audit.py`: checks the manuscript against the recurring failure patterns from prior review.

Inputs not included: encounter notes, member attributes, eligibility, and trained
model weights.
