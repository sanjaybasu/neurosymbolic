# Hybrid neural and knowledge-graph contraindication detection in clinical pharmacy notes

Code for a hybrid medication-safety system that combines a neural clinical text
classifier (ClinicalBERT) with a complementary symbolic knowledge-graph layer to
detect drug-disease and drug-drug contraindications in clinical pharmacy
documentation from a Medicaid care management program. This repository accompanies
a manuscript under review at JMIR Medical Informatics.

## What is and is not included

Included: model and pipeline code, the clinical rules knowledge graph, the
entity-extraction vocabularies, and the synthetic physician-authored scenario library.

Not included: patient data of any kind, model outputs or results, manuscript text,
tables, and figures. The system requires protected health information that is not
distributed here. Reported numbers live in the manuscript, not in this repository.

## Method

Each note is processed by two independent components combined at the output by OR logic:

1. Neural component: a Bio_ClinicalBERT encoder with a linear classification head and
   sigmoid activation for binary hazard classification, fine-tuned on physician-authored
   scenarios and operational notes.
2. Symbolic component: a clinical rules knowledge graph of 62 directed edges
   (30 contraindication/interaction, 19 risk-amplification, 13 required-intervention) derived from
   US Food and Drug Administration labeling and warnings, the American Geriatrics Society Beers
   Criteria, and additional society guidelines, with a rule-based entity extractor and
   sentence-scoped negation.
3. OR fusion: any symbolic safety signal can raise a note to hazard status; symbolic
   evidence does not lower a neural hazard classification. The two components are
   complementary detectors rather than an integrated model.

## Repository structure

```
models/            model and reasoning definitions
scripts/           training and evaluation pipeline
revision_1/        analysis code for the first revision (see revision_1/README.md)
knowledge_graphs/  clinical rules and care-pathway graphs (JSON)
data/              extraction vocabularies, rules, and the synthetic scenario library
```

## Reproducing

The pipeline expects local source data that is not included. Configure the data root
for your environment, then run the scripts in `scripts/` and `revision_1/`. See
`revision_1/README.md` for the revision analyses.

## License

See LICENSE.
