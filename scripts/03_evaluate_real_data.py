"""
Evaluate neurosymbolic reasoner on real physician-created scenarios and prospective validation cases.

Data sources:
- 588 physician scenarios from scenario_library.csv
- 1,000 prospective cases (500 benign + 500 harm)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import time
from typing import Dict, List
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.neurosymbolic_reasoner import (
    NeurosymbolicReasoner, ClinicalContext, SymbolicReasoner
)
from scripts.data_utils import parse_multiline_csv

REPO_ROOT = Path(__file__).resolve().parents[3]
RL_DATA_DIR = REPO_ROOT / "notebooks" / "rl_vs_llm_safety" / "data"


class RealDataEvaluator:
    """Evaluate neurosymbolic reasoner on real clinical scenarios."""
    
    def __init__(self, kg_dir: str):
        """Initialize with knowledge graph directory."""
        self.symbolic = SymbolicReasoner(kg_dir)
        self.results = []
        # Build lexicons from the clinical rules graph for lightweight NLP
        self.condition_terms = self._build_terms(prefix="condition:")
        self.medication_terms = self._build_terms(prefix="medication:")
    
    def _build_terms(self, prefix: str) -> List[str]:
        """Extract node labels with the given prefix from the clinical rules KG."""
        terms = []
        for node in self.symbolic.clinical_rules_kg.nodes:
            if node.startswith(prefix):
                cleaned = node.replace(prefix, "").replace("_", " ").lower()
                terms.append(cleaned)
        # Deduplicate and sort by length to favor longer matches
        return sorted(set(terms), key=len, reverse=True)
        
    def load_physician_scenarios(self, scenario_path: str) -> pd.DataFrame:
        """Load physician-created scenarios from CSV."""
        df = pd.read_csv(scenario_path)
        print(f"Loaded {len(df)} physician scenarios")
        return df
    
    def load_prospective_cases(self, benign_path: str, harm_path: str) -> pd.DataFrame:
        """Load prospective validation cases with robust parsing."""
        benign = parse_multiline_csv(Path(benign_path), record_prefixes=["benign_candidate"])
        benign["case_type"] = "benign"
        
        harm = parse_multiline_csv(Path(harm_path), record_prefixes=["harm_candidate"])
        harm["case_type"] = "harm"
        
        combined = pd.concat([benign, harm], ignore_index=True)
        print(f"Loaded {len(combined)} prospective cases ({len(benign)} benign, {len(harm)} harm)")
        return combined
    
    def extract_conditions_from_scenario(self, scenario_text: str) -> List[str]:
        """Extract medical conditions from scenario text using rule lexicon."""
        import re
        scenario_lower = scenario_text.lower()
        matches = []
        # Flexible lexicon matching
        fallback = {
            'Chronic Kidney Disease': ['ckd', 'kidney disease', 'renal'],
            'Heart Failure': ['heart failure', 'chf', 'congestive'],
            'Diabetes': ['diabetes', 'diabetic', 'blood sugar'],
            'Hypertension': ['hypertension', 'high blood pressure', 'bp'],
            'Asthma': ['asthma', 'wheezing', 'inhaler'],
            'COPD': ['copd', 'chronic obstructive', 'emphysema'],
            'Depression': ['depression', 'depressed', 'mood'],
            'Pregnancy': ['pregnant', 'pregnancy', 'expecting'],
            'Substance Use Disorder': ['opioid', 'substance', 'addiction'],
        }
        for term in self.condition_terms:
            pattern = rf"\\b{re.escape(term)}\\b"
            if re.search(pattern, scenario_lower):
                matches.append(term.title())
        for label, kws in fallback.items():
            if any(kw in scenario_lower for kw in kws):
                matches.append(label)
        return matches
    
    def extract_medications_from_scenario(self, scenario_text: str) -> List[str]:
        """Extract medications from scenario text using rule lexicon."""
        import re
        scenario_lower = scenario_text.lower()
        matches = []
        fallback = {
            'NSAIDs': ['nsaid', 'nsaids', 'ibuprofen', 'advil', 'motrin', 'naproxen', 'aleve', 'meloxicam'],
            'ACE Inhibitors': ['ace inhibitor', 'lisinopril', 'enalapril', 'ramipril'],
            'Beta Blockers': ['beta blocker', 'metoprolol', 'atenolol', 'propranolol'],
            'Metformin': ['metformin', 'glucophage'],
            'Valproate': ['valproate', 'valproic acid', 'depakote'],
            'Warfarin': ['warfarin', 'coumadin'],
        }
        for term in self.medication_terms:
            pattern = rf"\\b{re.escape(term)}\\b"
            if re.search(pattern, scenario_lower):
                matches.append(term.title())
        for label, kws in fallback.items():
            if any(kw in scenario_lower for kw in kws):
                matches.append(label)
        return matches
    
    def evaluate_scenario(self, scenario_id: str, scenario_text: str, 
                         source: str) -> Dict:
        """Evaluate a single scenario."""
        start_time = time.time()
        
        # Extract clinical context
        conditions = self.extract_conditions_from_scenario(scenario_text)
        medications = self.extract_medications_from_scenario(scenario_text)
        
        # Create clinical context
        context = ClinicalContext(
            patient_id=scenario_id,
            conditions=conditions,
            medications=medications,
            recent_encounters=[],
            goals=[],
            demographics={},
            risk_factors=[],
            current_state={}
        )
        
        # Run safety checks
        safety_violations = []
        contraindications_detected = 0
        
        for med in medications:
            is_safe, violations = self.symbolic.check_contraindications(context, f"medication {med}")
            if not is_safe:
                contraindications_detected += 1
                safety_violations.extend(violations)
        
        # Compute risk cascade
        risk_scores = self.symbolic.compute_risk_cascade(context)
        max_risk = max(risk_scores.values()) if risk_scores else 0.0
        has_elevated_risk = max_risk > 2.0
        
        # Find required interventions
        required = self.symbolic.find_required_interventions(context)
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            'scenario_id': scenario_id,
            'source': source,
            'conditions_detected': len(conditions),
            'conditions': conditions,
            'medications_checked': len(medications),
            'medications': medications,
            'contraindications_detected': contraindications_detected,
            'safety_violations': safety_violations,
            'has_elevated_risk': has_elevated_risk,
            'max_risk_score': max_risk,
            'risk_details': risk_scores,
            'required_interventions': len(required),
            'inference_time_ms': inference_time
        }
    
    def run_evaluation(self, physician_scenarios_df: pd.DataFrame,
                      prospective_cases_df: pd.DataFrame) -> pd.DataFrame:
        """Run evaluation on all scenarios."""
        
        print("\nEvaluating physician scenarios...")
        for idx, row in physician_scenarios_df.iterrows():
            result = self.evaluate_scenario(
                scenario_id=f"physician_{idx}",
                scenario_text=row.get('prompt', row.get('name', '')),
                source='physician_created'
            )
            self.results.append(result)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(physician_scenarios_df)} scenarios")
        
        print(f"\nEvaluating prospective validation cases...")
        for idx, row in prospective_cases_df.iterrows():
            # Prospective cases might have different column structure
            scenario_text = row.get('prompt', row.get('scenario', row.get('text', '')))
            
            result = self.evaluate_scenario(
                scenario_id=f"prospective_{idx}",
                scenario_text=str(scenario_text),
                source=f"prospective_{row.get('case_type', 'unknown')}"
            )
            self.results.append(result)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(prospective_cases_df)} cases")
        
        return pd.DataFrame(self.results)
    
    def compute_summary_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Compute summary statistics."""
        
        total_scenarios = len(results_df)
        
        # Safety metrics
        total_medications_checked = results_df['medications_checked'].sum()
        total_contraindications = results_df['contraindications_detected'].sum()
        scenarios_with_contraindications = (results_df['contraindications_detected'] > 0).sum()
        safety_block_rate = scenarios_with_contraindications / total_scenarios if total_scenarios > 0 else 0
        
        # Risk metrics
        scenarios_with_conditions = (results_df['conditions_detected'] > 0).sum()
        scenarios_with_elevated_risk = results_df['has_elevated_risk'].sum()
        mean_risk_score = results_df[results_df['max_risk_score'] > 0]['max_risk_score'].mean()
        
        # Performance metrics
        median_inference_time = results_df['inference_time_ms'].median()
        p95_inference_time = results_df['inference_time_ms'].quantile(0.95)
        max_inference_time = results_df['inference_time_ms'].max()
        
        # By source
        by_source = results_df.groupby('source').agg({
            'contraindications_detected': 'sum',
            'has_elevated_risk': 'sum',
            'inference_time_ms': 'median'
        }).to_dict()
        
        return {
            'total_scenarios': total_scenarios,
            'total_medications_checked': int(total_medications_checked),
            'total_contraindications_detected': int(total_contraindications),
            'scenarios_with_contraindications': int(scenarios_with_contraindications),
            'safety_block_rate': float(safety_block_rate),
            'scenarios_with_conditions': int(scenarios_with_conditions),
            'scenarios_with_elevated_risk': int(scenarios_with_elevated_risk),
            'elevated_risk_rate': float(scenarios_with_elevated_risk / scenarios_with_conditions) if scenarios_with_conditions > 0 else 0,
            'mean_risk_score': float(mean_risk_score) if not pd.isna(mean_risk_score) else 0,
            'median_inference_time_ms': float(median_inference_time),
            'p95_inference_time_ms': float(p95_inference_time),
            'max_inference_time_ms': float(max_inference_time),
            'by_source': by_source
        }


def main():
    """Run evaluation on real data."""
    
    # Paths
    kg_dir = str(Path(__file__).resolve().parent.parent / "knowledge_graphs")
    scenario_library_path = RL_DATA_DIR / "scenario_library.csv"
    benign_cases_path = RL_DATA_DIR / "prospective_eval" / "benign_cases_500.csv"
    harm_cases_path = RL_DATA_DIR / "prospective_eval" / "harm_cases_500.csv"
    output_dir = REPO_ROOT / "notebooks" / "neurosymbolic" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Neurosymbolic Reasoner: Real Data Evaluation")
    print("=" * 80)
    
    # Initialize evaluator
    print("\nInitializing neurosymbolic reasoner...")
    evaluator = RealDataEvaluator(kg_dir)
    
    # Load data
    print("\nLoading evaluation datasets...")
    physician_scenarios = evaluator.load_physician_scenarios(scenario_library_path)
    prospective_cases = evaluator.load_prospective_cases(benign_cases_path, harm_cases_path)
    
    print(f"\nTotal evaluation dataset: {len(physician_scenarios) + len(prospective_cases)} cases")
    print(f"  - Physician scenarios: {len(physician_scenarios)}")
    print(f"  - Prospective benign: {(prospective_cases['case_type'] == 'benign').sum()}")
    print(f"  - Prospective harm: {(prospective_cases['case_type'] == 'harm').sum()}")
    
    # Run evaluation
    print("\n" + "=" * 80)
    print("Running evaluation...")
    print("=" * 80)
    
    results_df = evaluator.run_evaluation(physician_scenarios, prospective_cases)
    
    # Compute metrics
    print("\n" + "=" * 80)
    print("Computing summary metrics...")
    print("=" * 80)
    
    metrics = evaluator.compute_summary_metrics(results_df)
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nTotal Scenarios Evaluated: {metrics['total_scenarios']}")
    print(f"\nSafety Performance:")
    print(f"  - Total Medications Checked: {metrics['total_medications_checked']}")
    print(f"  - Total Contraindications Detected: {metrics['total_contraindications_detected']}")
    print(f"  - Scenarios with Contraindications: {metrics['scenarios_with_contraindications']}")
    print(f"  - Safety Block Rate: {metrics['safety_block_rate']:.1%}")
    print(f"\nRisk Assessment:")
    print(f"  - Scenarios with Conditions: {metrics['scenarios_with_conditions']}")
    print(f"  - Scenarios with Elevated Risk (>2.0): {metrics['scenarios_with_elevated_risk']}")
    print(f"  - Elevated Risk Rate: {metrics['elevated_risk_rate']:.1%}")
    print(f"  - Mean Risk Score: {metrics['mean_risk_score']:.2f}")
    print(f"\nComputational Performance:")
    print(f"  - Median Inference Time: {metrics['median_inference_time_ms']:.4f} ms")
    print(f"  - 95th Percentile: {metrics['p95_inference_time_ms']:.4f} ms")
    print(f"  - Maximum: {metrics['max_inference_time_ms']:.4f} ms")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(output_dir / f"neurosymbolic_real_data_results_{timestamp}.csv", index=False)
    
    with open(output_dir / f"neurosymbolic_real_data_metrics_{timestamp}.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n" + "=" * 80)
    print(f"Results saved to {output_dir}")
    print("=" * 80)
    
    return results_df, metrics


if __name__ == "__main__":
    results_df, metrics = main()
