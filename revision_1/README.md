# Revision 1 analysis code

Analysis scripts used for the first revision. These reproduce the reported figures
and run the additional analyses requested in review. They require local protected
health information that is not distributed with this repository; set the data root
for your environment before running. No patient data, model outputs, or manuscript
text are included here.

Scripts:
- 01_canonical_analysis.py: single pipeline that reproduces the reported tables and runs the new analyses.
- 01b_note_length_fixed.py: logistic regression of contraindication detection on note length and documentation type.
- 02_symbolic_augmented_neural.py: neural classifier trained on symbolic-generated labels.
- 03_retraining_experiments.py: OR-fusion ablation, symbolic-augmented neural, patient-level split, TF-IDF, encoder comparison.
- 04_figures.py: figure generation from derived outputs.
- verify_numbers.py / consistency_audit.py: checks that reported values match derived outputs and that the text is internally consistent.

Inputs not included: encounter notes, member attributes, eligibility, and trained model weights.
