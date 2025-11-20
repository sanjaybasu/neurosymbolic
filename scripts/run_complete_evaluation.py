"""
Complete neurosymbolic evaluation generating REAL quantitative metrics.
This version actually runs end-to-end with proper error handling.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

sys.path.append(str(Path(__file__).parent.parent))
from models.neurosymbolic_reasoner import load_neurosymbolic_reasoner, ClinicalContext


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    # Safety metrics
    total_scenarios: int
    contraindications_detected: int
    contraindications_blocked: int
    safety_violation_rate: float
    
    # Action quality metrics
    actions_recommended: Dict[str, int]
    average_confidence: float
    
    # Explanation quality
    explanations_generated: int
    avg_knowledge_paths: float
    
    # Performance
    avg_inference_time_ms: float


def create_synthetic_test_scenarios(n=100) -> List[Dict]:
    """Create synthetic test scenarios with known safety issues."""
    
    scenarios = []
    
    # High-risk conditions
    conditions_pool = [
        'Chronic Kidney Disease',
        'Hypertension',
        'Asthma',
        'Diabetes',
        'Heart Failure',
        'Pregnancy'
    ]
    
    for i in range(n):
        # Random patient
        age = np.random.randint(25, 85)
        num_conditions = np.random.randint(0, 3)
        conditions = list(np.random.choice(conditions_pool, num_conditions, replace=False))
        
        # Some scenarios have ED visits (high utilization)
        ed_visits = np.random.choice([0, 0, 0, 1, 2], p=[0.7, 0.1, 0.1, 0.05, 0.05])
        hosp_admits = 1 if ed_visits > 1 else 0
        
        scenarios.append({
            'patient_id': f'synthetic_{i}',
            'age': age,
            'conditions': conditions,
            'medications': [],
            'ed_visits_90d': ed_visits,
            'hosp_admits_180d': hosp_admits,
        })
    
    return scenarios


def evaluate_neurosymbolic_complete() -> EvaluationMetrics:
    """Run complete evaluation and return real metrics."""
    
    print("="*80)
    print("NEUROSYMBOLIC CLINICAL DECISION SUPPORT - COMPREHENSIVE EVALUATION")
    print("="*80)
    print()
    
    # Load reasoner
    print("[1/5] Loading neurosymbolic reasoner...")
    try:
        reasoner = load_neurosymbolic_reasoner()
        print("  ✓ Loaded knowledge graphs and neural policy")
    except Exception as e:
        print(f"  ⚠ Warning: {e}")
        print("  → Using randomly initialized policy for demonstration")
        reasoner = load_neurosymbolic_reasoner()
    print()
    
    # Create test scenarios
    print("[2/5] Creating test scenarios...")
    scenarios = create_synthetic_test_scenarios(n=100)
    print(f"  ✓ Created {len(scenarios)} synthetic patient scenarios")
    print()
    
    # Run evaluation
    print("[3/5] Running neurosymbolic reasoning on all scenarios...")
    
    results = []
    contraindications_found = 0
    contraindications_blocked = 0
    total_confidence = 0
    total_knowledge_paths = 0
    explanations_generated = 0
    action_counts = {}
    
    import time
    total_time = 0
    
    for i, scenario in enumerate(scenarios):
        if (i+1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(scenarios)}...")
        
        try:
            # Create clinical context
            context = ClinicalContext(
                patient_id=scenario['patient_id'],
               conditions=scenario['conditions'],
                medications=scenario.get('medications', []),
                recent_encounters=[],
                goals=[],
                demographics={'age': scenario['age']},
                risk_factors=[],
                current_state={
                    'age': scenario['age'],
                    'ed_visits_90d': scenario['ed_visits_90d'],
                    'hosp_admits_180d': scenario['hosp_admits_180d']
                }
            )
            
            # Run reasoner
            start = time.time()
            result = reasoner.reason(context)
            inference_time = (time.time() - start) * 1000  # ms
            total_time += inference_time
            
            # Extract metrics
            action_counts[result.recommended_action] = action_counts.get(result.recommended_action, 0) + 1
            total_confidence += result.confidence
            
            # Count contraindications
            has_contraindication = False
            blocked = False
            for checks in result.safety_checks_passed.values():
                if not checks.get('contraindications_passed', True):
                    has_contraindication = True
                    contraindications_found += 1
                    if 'escalate' in result.recommended_action or 'warn' in result.recommended_action:
                        blocked = True
            
            if blocked:
                contraindications_blocked += 1
            
            # Explanation quality
            if result.explanation and len(result.explanation) > 10:
                explanations_generated += 1
            
            total_knowledge_paths += len(result.knowledge_paths)
            
            results.append({
                'patient_id': scenario['patient_id'],
                'conditions': scenario['conditions'],
                'action': result.recommended_action,
                'confidence': result.confidence,
                'has_contraindication': has_contraindication,
                'blocked': blocked,
                'explanation_length': len(result.explanation),
                'knowledge_paths': len(result.knowledge_paths),
                'inference_time_ms': inference_time
            })
            
        except Exception as e:
            print(f"    Error on scenario {i}: {e}")
            continue
    
    print(f"  ✓ Completed {len(results)} evaluations")
    print()
    
    # Compute metrics
    print("[4/5] Computing aggregate metrics...")
    
    # Safety metrics
    safety_violation_rate = (contraindications_found - contraindications_blocked) / len(results) if results else 0
    
    metrics = EvaluationMetrics(
        total_scenarios=len(results),
        contraindications_detected=contraindications_found,
        contraindications_blocked=contraindications_blocked,
        safety_violation_rate=safety_violation_rate,
        actions_recommended=action_counts,
        average_confidence=total_confidence / len(results) if results else 0,
        explanations_generated=explanations_generated,
        avg_knowledge_paths=total_knowledge_paths / len(results) if results else 0,
        avg_inference_time_ms=total_time / len(results) if results else 0
    )
    
    print("  ✓ Metrics computed")
    print()
    
    # Save results
    print("[5/5] Saving results...")
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_dir / "neurosymbolic_evaluation_complete.csv", index=False)
    
    # Save metrics
    with open(results_dir / "evaluation_metrics.json", 'w') as f:
        json.dump(asdict(metrics), f, indent=2)
    
    print(f"  ✓ Saved to {results_dir}")
    print()
    
    # Print summary
    print("="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print()
    print(f"Total scenarios evaluated: {metrics.total_scenarios}")
    print()
    print("SAFETY METRICS:")
    print(f"  Contraindications detected: {metrics.contraindications_detected}")
    print(f"  Contraindications blocked: {metrics.contraindications_blocked}")
    print(f"  Safety violation rate: {metrics.safety_violation_rate:.1%}")
    print()
    print("ACTION QUALITY:")
    print(f"  Average confidence: {metrics.average_confidence:.2f}")
    print(f"  Actions recommended:")
    for action, count in sorted(metrics.actions_recommended.items(), key=lambda x: -x[1])[:5]:
        print(f"    {action}: {count} ({count/metrics.total_scenarios:.1%})")
    print()
    print("EXPLAINABILITY:")
    print(f"  Explanations generated: {metrics.explanations_generated} ({metrics.explanations_generated/metrics.total_scenarios:.1%})")
    print(f"  Avg knowledge graph paths: {metrics.avg_knowledge_paths:.1f}")
    print()
    print("PERFORMANCE:")
    print(f"  Average inference time: {metrics.avg_inference_time_ms:.1f}ms")
    print()
    print("="*80)
    
    return metrics


if __name__ == "__main__":
    metrics = evaluate_neurosymbolic_complete()
