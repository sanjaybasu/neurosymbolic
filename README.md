# Neurosymbolic Clinical Decision Support System

This repository contains the implementation of a neurosymbolic learning system for clinical decision support in Medicaid population health management. The system integrates supervised learning (TF-IDF + Logistic Regression) with symbolic knowledge graph reasoning to identify safety hazards in clinical text with high sensitivity and interpretability.

## System Architecture

The system uses a hybrid architecture:
1.  **Supervised Classifier**: A statistical model trained on physician-labeled scenarios to detect hazards.
2.  **Symbolic Reasoner**: A knowledge graph-based engine that checks for explicit contraindications and risk cascades using clinical guidelines (e.g., CPIC, Beers Criteria).

## Repository Structure

- **`scripts/`**: Python scripts for data processing, training, and evaluation.
- **`models/`**: Definitions of the `HazardDetector` and `NeurosymbolicReasoner` classes.
- **`knowledge_graphs/`**: Logic for constructing the patient and clinical rules knowledge graphs.
- **`results/`**: Directory for storing evaluation metrics and outputs.
- **`paper/`**: Manuscript drafts and related documentation.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/sanjaybasu/neurosymbolic.git
    cd neurosymbolic
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train the model and run the evaluation:

```bash
python3 scripts/04_train_hybrid_detector.py
```

## Citation

If you use this code in your research, please cite:

> Basu S, Patel S. Neurosymbolic Learning for Clinical Decision Support in Medicaid Population Health Management. *Scientific Reports* (Under Review). 2025.

## License

MIT License
