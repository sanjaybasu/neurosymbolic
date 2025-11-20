"""
Two-Stage Evaluation of Hybrid Neurosymbolic Safety Detector

Stage 1: Physician-created scenarios (n=432 from scenario_library.csv)
Stage 2: Prospective validation (n=1,000: 500 benign + 500 harm)

Computes sensitivity, specificity, and other performance metrics for each stage.
"""

import json
import pickle
from pathlib import Path
import pandas as pd
import sys
import time
import numpy as np

# Add models directory to path
SCRIPT_DIR = Path(__file__).parent
NEUROSYMBOLIC_DIR = SCRIPT_DIR.parent
MODELS_DIR = NEUROSYMBOLIC_DIR / "models"
sys.path.append(str(MODELS_DIR))

from neurosymbolic_reasoner_v2 import HybridNeurosymbolicReasoner

# Data paths
RL_VS_LLM_DATA = Path("/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety/data")
SCENARIO_LIBRARY = RL_VS_LLM_DATA / "scenario_library.csv"
BENIGN_CASES = RL_VS_LLM_DATA / "prospective_eval" / "benign_cases_500.csv"
HARM_CASES = RL_VS_LLM_DATA / "prospective_eval" / "harm_cases_500.csv"

# Model paths
MODELS_DIR_OUTPUT = NEUROSYMBOLIC_DIR / "trained_models"
DETECTOR_PATH = MODELS_DIR_OUTPUT / "hazard_detector.pkl"

# Output paths
RESULTS_DIR = NEUROSYMBOLIC_DIR / "results"


def load_scenarios_from_csv(csv_path: Path):
    """Load scenarios from CSV file."""
    df = pd.read_csv(csv_path)
    scenarios = []
    
    for _, row in df.iterrows():
        scenario = {
            'name': row['name'],
            'prompt': row['prompt'],
            'context': {},
            'hazard_type': row['category'] if row['severity'] != 'none' else 'benign'
        }
        scenarios.append(scenario)
    
    return scenarios


def load_prospective_cases(benign_path: Path, harm_path: Path):
    """Load prospective validation cases with correct column mapping."""
    benign_df = pd.read_csv(benign_path)
    harm_df = pd.read_csv(harm_path)
    
    scenarios = []
    
    # Benign cases - use context_text column
    for idx, row in benign_df.iterrows():
        scenario = {
            'name': f"benign_{idx}",
            'prompt': row['context_text'],  # FIXED
            'context': {},
            'hazard_type': 'benign'
        }
        scenarios.append(scenario)
    
    # Harm cases
    for idx, row in harm_df.iterrows():
        scenario = {
            'name': f"harm_{idx}",
            'prompt': row['context_text'],  # FIXED
            'context': {},
            'hazard_type': row.get('harm_type', 'harm')
        }
        scenarios.append(scenario)
    
    return scenarios


def compute_metrics(results):
    """Compute sensitivity, specificity, and other metrics from evaluation results."""
    tp = sum(1 for r in results if not r['is_safe_true'] and not r['is_safe_pred'])  # Hazard correctly flagged
    fn = sum(1 for r in results if not r['is_safe_true'] and r['is_safe_pred'])      # Hazard missed
    tn = sum(1 for r in results if r['is_safe_true'] and r['is_safe_pred'])          # Benign correctly cleared
    fp = sum(1 for r in results if r['is_safe_true'] and not r['is_safe_pred'])      # Benign incorrectly flagged
    
    total = len(results)
    positives = tp + fn  # Total hazards
    negatives = tn + fp  # Total benign
    
    sensitivity = tp / positives if positives > 0 else 0.0
    specificity = tn / negatives if negatives > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    return {
        'tp': tp,
        'fn': fn,
        'tn': tn,
        'fp': fp,
        'total': total,
        'positives': positives,
        'negatives': negatives,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'accuracy': accuracy
    }


def main():
    print("=" * 80)
    print("Two-Stage Evaluation: Hybrid Neurosymbolic Safety Detector")
    print("=" * 80)
    
    # Load trained detector
    print("\n[1/6] Loading trained detector...")
    if not DETECTOR_PATH.exists():
        raise FileNotFoundError(f"Trained detector not found at {DETECTOR_PATH}. Run training script first.")
    
    kg_dir = "/Users/sanjaybasu/waymark-local/notebooks/neurosymbolic/knowledge_graphs"
    
    with open(DETECTOR_PATH, 'rb') as f:
        trained_detector = pickle.load(f)
    
    reasoner = HybridNeurosymbolicReasoner(kg_dir=kg_dir, hazard_detector=trained_detector)
    print(f"  Loaded from: {DETECTOR_PATH}")
    
    # Stage 1: Physician-created scenarios
    print("\n" + "=" * 80)
    print("STAGE 1: Physician-Created Scenarios")
    print("=" * 80)
    
    print("\n[2/6] Loading physician scenarios...")
    physician_scenarios = load_scenarios_from_csv(SCENARIO_LIBRARY)
    benign_scenarios = [s for s in physician_scenarios if s['hazard_type'] == 'benign']
    hazard_scenarios = [s for s in physician_scenarios if s['hazard_type'] != 'benign']
    print(f"  Loaded {len(physician_scenarios)} scenarios")
    print(f"    - Benign: {len(benign_scenarios)}")
    print(f"    - Hazard: {len(hazard_scenarios)}")
    
    print("\n[3/6] Evaluating Stage 1...")
    stage1_results = []
    start_time = time.time()
    
    for i, scenario in enumerate(physician_scenarios):
        result = reasoner.evaluate_scenario(scenario)
        stage1_results.append(result)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(physician_scenarios)} scenarios")
    
    stage1_time = time.time() - start_time
    stage1_metrics = compute_metrics(stage1_results)
    
    print(f"\n  Stage 1 Results:")
    print(f"    Total scenarios:  {stage1_metrics['total']}")
    print(f"    Hazards:          {stage1_metrics['positives']}")
    print(f"    Benign:           {stage1_metrics['negatives']}")
    print(f"    TP: {stage1_metrics['tp']}, FN: {stage1_metrics['fn']}, TN: {stage1_metrics['tn']}, FP: {stage1_metrics['fp']}")
    print(f"    Sensitivity:      {stage1_metrics['sensitivity']:.3f}")
    print(f"    Specificity:      {stage1_metrics['specificity']:.3f}")
    print(f"    Precision:        {stage1_metrics['precision']:.3f}")
    print(f"    Accuracy:         {stage1_metrics['accuracy']:.3f}")
    print(f"    Processing time:  {stage1_time:.2f}s ({stage1_time/len(physician_scenarios)*1000:.2f}ms per scenario)")
    
    # Stage 2: Prospective validation
    print("\n" + "=" * 80)
    print("STAGE 2: Prospective Real-World Validation")
    print("=" * 80)
    
    print("\n[4/6] Loading prospective cases...")
    prospective_scenarios = load_prospective_cases(BENIGN_CASES, HARM_CASES)
    print(f"  Loaded {len(prospective_scenarios)} scenarios")
    print(f"    - Benign: 500")
    print(f"    - Harm:   500")
    
    print("\n[5/6] Evaluating Stage 2...")
    stage2_results = []
    start_time = time.time()
    
    for i, scenario in enumerate(prospective_scenarios):
        result = reasoner.evaluate_scenario(scenario)
        stage2_results.append(result)
        
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(prospective_scenarios)} scenarios")
    
    stage2_time = time.time() - start_time
    stage2_metrics = compute_metrics(stage2_results)
    
    print(f"\n  Stage 2 Results:")
    print(f"    Total scenarios:  {stage2_metrics['total']}")
    print(f"    Hazards:          {stage2_metrics['positives']}")
    print(f"    Benign:           {stage2_metrics['negatives']}")
    print(f"    TP: {stage2_metrics['tp']}, FN: {stage2_metrics['fn']}, TN: {stage2_metrics['tn']}, FP: {stage2_metrics['fp']}")
    print(f"    Sensitivity:      {stage2_metrics['sensitivity']:.3f}")
    print(f"    Specificity:      {stage2_metrics['specificity']:.3f}")
    print(f"    Precision:        {stage2_metrics['precision']:.3f}")
    print(f"    Accuracy:         {stage2_metrics['accuracy']:.3f}")
    print(f"    Processing time:  {stage2_time:.2f}s ({stage2_time/len(prospective_scenarios)*1000:.2f}ms per scenario)")
    
    # Save results
    print("\n[6/6] Saving results...")
    
    # Save detailed results
    stage1_df = pd.DataFrame(stage1_results)
    stage2_df = pd.DataFrame(stage2_results)
    
    stage1_df.to_csv(RESULTS_DIR / "stage1_physician_results.csv", index=False)
    stage2_df.to_csv(RESULTS_DIR / "stage2_prospective_results.csv", index=False)
    
    # Save summary metrics
    summary = {
        'stage1_physician_scenarios': {
            'n_scenarios': len(physician_scenarios),
            'metrics': stage1_metrics,
            'processing_time_seconds': stage1_time
        },
        'stage2_prospective_validation': {
            'n_scenarios': len(prospective_scenarios),
            'metrics': stage2_metrics,
            'processing_time_seconds': stage2_time
        }
    }
    
    with open(RESULTS_DIR / "two_stage_evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Saved detailed results to:")
    print(f"    - {RESULTS_DIR / 'stage1_physician_results.csv'}")
    print(f"    - {RESULTS_DIR / 'stage2_prospective_results.csv'}")
    print(f"    - {RESULTS_DIR / 'two_stage_evaluation_summary.json'}")
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"\nFinal Performance Summary:")
    print(f"  Stage 1 (Physician): Sensitivity={stage1_metrics['sensitivity']:.3f}, Specificity={stage1_metrics['specificity']:.3f}")
    print(f"  Stage 2 (Prospective): Sensitivity={stage2_metrics['sensitivity']:.3f}, Specificity={stage2_metrics['specificity']:.3f}")


if __name__ == "__main__":
    main()
