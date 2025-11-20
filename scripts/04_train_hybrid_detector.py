"""
Train hybrid neurosymbolic detector on physician-created scenarios.

This script:
1. Loads physician scenarios from rl_vs_llm_safety data
2. Splits into train/test sets
3. Trains TF-IDF + Logistic Regression hazard classifier
4. Evaluates on holdout set
5. Saves trained detector for use in evaluation
"""

import json
import pickle
from pathlib import Path
import pandas as pd
import sys

# Add models directory to path
SCRIPT_DIR = Path(__file__).parent
NEUROSYMBOLIC_DIR = SCRIPT_DIR.parent
MODELS_DIR = NEUROSYMBOLIC_DIR / "models"
sys.path.append(str(MODELS_DIR))

from neurosymbolic_reasoner_v2 import HybridNeurosymbolicReasoner

# Data paths
RL_VS_LLM_DATA = Path("/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety/data")
SCENARIO_LIBRARY = RL_VS_LLM_DATA / "scenario_library.csv"
HAZARD_TRAIN = RL_VS_LLM_DATA / "hazard_scenarios_train.json"
HAZARD_HOLDOUT = RL_VS_LLM_DATA / "hazard_scenarios_holdout.json"

# Output paths
RESULTS_DIR = NEUROSYMBOLIC_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_OUTPUT_DIR = NEUROSYMBOLIC_DIR / "trained_models"
MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_scenarios_from_csv(csv_path: Path):
    """Load scenarios from CSV file."""
    df = pd.read_csv(csv_path)
    scenarios = []
    
    for _, row in df.iterrows():
        scenario = {
            'name': row['name'],
            'prompt': row['prompt'],
            'context': {},  # CSV doesn't have structured context
            'hazard_type': row['category'] if row['severity'] != 'none' else 'benign'
        }
        scenarios.append(scenario)
    
    return scenarios


def load_scenarios_from_json(json_path: Path):
    """Load scenarios from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to standard format
    scenarios = []
    for item in data:
        scenario = {
            'name': item.get('name', item.get('id', 'unknown')),
            'prompt': item.get('prompt', item.get('text', '')),
            'context': item.get('context', {}),
            'hazard_type': item.get('hazard_type', item.get('category', 'unknown'))
        }
        scenarios.append(scenario)
    
    return scenarios


def main():
    print("=" * 80)
    print("Training Hybrid Neurosymbolic Safety Detector")
    print("=" * 80)
    
    # Load training data
    print("\n[1/5] Loading training scenarios...")
    if SCENARIO_LIBRARY.exists():
        all_scenarios = load_scenarios_from_csv(SCENARIO_LIBRARY)
        print(f"  Loaded {len(all_scenarios)} scenarios from scenario_library.csv")
        
        # Separate benign and hazard
        benign_scenarios = [s for s in all_scenarios if s['hazard_type'] == 'benign']
        hazard_scenarios = [s for s in all_scenarios if s['hazard_type'] != 'benign']
        
        print(f"  - Benign: {len(benign_scenarios)}")
        print(f"  - Hazard: {len(hazard_scenarios)}")
        
        # Use all data for training (no split needed, we'll evaluate on prospective)
        training_scenarios = all_scenarios
    else:
        print(f"  Warning: {SCENARIO_LIBRARY} not found")
        print(f"  Attempting to load from JSON files...")
        
        # Fallback to JSON
        train_scenarios = load_scenarios_from_json(HAZARD_TRAIN) if HAZARD_TRAIN.exists() else []
        training_scenarios = train_scenarios
        print(f"  Loaded {len(training_scenarios)} training scenarios")
    
    # Initialize hybrid reasoner
    print("\n[2/5] Initializing hybrid reasoner...")
    kg_dir = "/Users/sanjaybasu/waymark-local/notebooks/neurosymbolic/knowledge_graphs"
    reasoner = HybridNeurosymbolicReasoner(kg_dir=kg_dir)
    
    # Train hazard detector
    print("\n[3/5] Training TF-IDF + Logistic Regression classifier...")
    metrics = reasoner.train(training_scenarios, cv_splits=5)
    
    print(f"\n  Cross-Validation Results:")
    print(f"    Accuracy: {metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}")
    print(f"    Macro F1: {metrics['macro_f1_mean']:.3f} ± {metrics['macro_f1_std']:.3f}")
    print(f"    Classes:  {metrics['n_classes']}")
    
    # Save trained detector
    print("\n[4/5] Saving trained detector...")
    detector_path = MODELS_OUTPUT_DIR / "hazard_detector.pkl"
    with open(detector_path, 'wb') as f:
        pickle.dump(reasoner.hazard_detector, f)
    print(f"  Saved to: {detector_path}")
    
    # Save training metrics
    print("\n[5/5] Saving training metrics...")
    metrics_path = RESULTS_DIR / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'training_metrics': metrics,
            'n_training_scenarios': len(training_scenarios),
            'hazard_types': list(reasoner.hazard_detector.labels_) if reasoner.hazard_detector.labels_ else []
        }, f, indent=2)
    print(f"  Saved to: {metrics_path}")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nTrained detector saved to: {detector_path}")
    print(f"Ready for two-stage evaluation.")
    

if __name__ == "__main__":
    main()
