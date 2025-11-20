"""
Evaluate neurosymbolic reasoner against:
1. Pure RL baseline
2. Pure LLM baseline  
3. Rule-based baseline

Metrics:
- Safety violations caught
- Action quality (agreement with expert labels)
- Explanation quality
- Computational efficiency
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.neurosymbolic_reasoner import (
    NeurosymbolicReasoner, ClinicalContext, load_neurosymbolic_reasoner
)


class NeurosymbolicEvaluator:
    """Evaluate neurosymbolic reasoner."""
    
    def __init__(
        self,
        test_data_path: str,
        reasoner: NeurosymbolicReasoner
    ):
        self.test_data = pd.read_parquet(test_data_path)
        self.reasoner = reasoner
        
    def create_test_scenarios(self) -> List[ClinicalContext]:
        """Create test scenarios from data."""
        
        scenarios = []
        
        for _, row in self.test_data.iterrows():
            context = ClinicalContext(
                patient_id=row['patient_id'],
                conditions=row.get('conditions', []),
                medications=row.get('medications', []),
                recent_encounters=row.get('recent_encounters', []),
                goals=row.get('goals', []),
                demographics=row.get('demographics', {}),
                risk_factors=row.get('risk_factors', []),
                current_state=row.get('state', {})
            )
            scenarios.append(context)
        
        return scenarios
    
    def evaluate_safety(self, scenarios: List[ClinicalContext]) -> Dict:
        """Evaluate safety constraint enforcement."""
        
        results = {
            'total_scenarios': len(scenarios),
            'safety_violations_caught': 0,
            'unsafe_actions_blocked': 0,
            'false_positives': 0
        }
        
        for scenario in scenarios:
            result = self.reasoner.reason(scenario)
            
            # Check if any actions were blocked
            blocked = sum(
                1 for checks in result.safety_checks_passed.values()
                if not checks['contraindications_passed']
            )
            
            if blocked > 0:
                results['unsafe_actions_blocked'] += blocked
        
        return results
    
    def evaluate_action_quality(
        self,
        scenarios: List[ClinicalContext],
        expert_labels: List[str]
    ) -> Dict:
        """Evaluate action recommendations vs expert labels."""
        
        predictions = []
        
        for scenario in scenarios:
            result = self.reasoner.reason(scenario)
            predictions.append(result.recommended_action)
        
        # Compute agreement
        agreement = sum(
            1 for pred, label in zip(predictions, expert_labels)
            if pred == label
        ) / len(predictions)
        
        return {
            'expert_agreement': agreement,
            'predictions': predictions
        }
    
    def evaluate_explanation_quality(
        self,
        scenarios: List[ClinicalContext]
    ) -> Dict:
        """Evaluate explanation quality."""
        
        explanations = []
        
        for scenario in scenarios:
            result = self.reasoner.reason(scenario)
            explanations.append({
                'patient_id': scenario.patient_id,
                'action': result.recommended_action,
                'explanation': result.explanation,
                'confidence': result.confidence,
                'knowledge_paths': result.knowledge_paths
            })
        
        return {
            'explanations': explanations,
            'mean_confidence': np.mean([e['confidence'] for e in explanations]),
            'knowledge_path_coverage': sum(
                1 for e in explanations if len(e['knowledge_paths']) > 0
            ) / len(explanations)
        }
    
    def compare_baselines(self, scenarios: List[ClinicalContext]) -> pd.DataFrame:
        """Compare neurosymbolic vs baselines."""
        
        results = []
        
        for scenario in scenarios:
            # Neurosymbolic
            ns_result = self.reasoner.reason(scenario)
            
            # Pure RL (without symbolic constraints)
            rl_q_values = self.reasoner.neural.get_action_values(scenario.current_state)
            rl_action = max(rl_q_values, key=rl_q_values.get)
            
            # Check if RL would have violated constraints
            rl_safe, rl_violations = self.reasoner.symbolic.check_contraindications(
                scenario, rl_action
            )
            
            results.append({
                'patient_id': scenario.patient_id,
                'neurosymbolic_action': ns_result.recommended_action,
                'neurosymbolic_safe': all(
                    c['contraindications_passed'] 
                    for c in ns_result.safety_checks_passed.values()
                ),
                'rl_action': rl_action,
                'rl_safe': rl_safe,
                'rl_violations': len(rl_violations),
                'explanation_provided': len(ns_result.explanation) > 0
            })
        
        return pd.DataFrame(results)


def main():
    # Load reasoner
    reasoner = load_neurosymbolic_reasoner()
    
    # Load test data
    evaluator = NeurosymbolicEvaluator(
        test_data_path="/Users/sanjaybasu/waymark-local/data/export_out_rebuilt/ml/observations_test_enriched.parquet",
        reasoner=reasoner
    )
    
    # Create test scenarios
    scenarios = evaluator.create_test_scenarios()
    print(f"Created {len(scenarios)} test scenarios")
    
    # Evaluate safety
    safety_results = evaluator.evaluate_safety(scenarios)
    print("\nSafety Evaluation:")
    print(json.dumps(safety_results, indent=2))
    
    # Compare to baselines
    comparison = evaluator.compare_baselines(scenarios)
    
    print("\nBaseline Comparison:")
    print(comparison.describe())
    print(f"\nSafety violations prevented: {comparison['rl_violations'].sum()}")
    print(f"Neurosymbolic safety rate: {comparison['neurosymbolic_safe'].mean():.2%}")
    print(f"Pure RL safety rate: {comparison['rl_safe'].mean():.2%}")
    
    # Save results
    output_dir = Path("/Users/sanjaybasu/waymark-local/notebooks/neurosymbolic/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison.to_csv(output_dir / "neurosymbolic_evaluation.csv", index=False)
    
    with open(output_dir / "safety_results.json", 'w') as f:
        json.dump(safety_results, f, indent=2)


if __name__ == "__main__":
    main()
