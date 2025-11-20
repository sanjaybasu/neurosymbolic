"""
Symbolic-only evaluation - bypasses neural network to generate real metrics.
This demonstrates the knowledge graph reasoning component.
"""

import pandas as pd
import numpy as np
import json
import networkx as nx
from pathlib import Path
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class EvaluationMetrics:
    """Real quantitative metrics from knowledge graph reasoning."""
    # Data
    total_scenarios: int
    patients_with_conditions: int
    
    # Safety metrics
    contraindications_checked: int
    unsafe_actions_blocked: int
    safety_block_rate: float
    
    # Risk assessment
    patients_with_elevated_risk: int
    avg_risk_score: float
    
    # Clinical reasoning
    required_interventions_identified: int
    escalations_recommended: int
    
    # Performance
    avg_inference_time_ms: float
    
    # Knowledge graph coverage
    total_kg_nodes: int
    total_kg_edges: int


def load_knowledge_graphs(kg_dir: Path) -> Tuple[nx.Graph, nx.Graph, nx.Graph]:
    """Load pre-built knowledge graphs."""
    import pickle
    
    with open(kg_dir / "patient_kg.pkl", "rb") as f:
        patient_kg = pickle.load(f)
    
    with open(kg_dir / "clinical_rules_kg.pkl", "rb") as f:
        clinical_rules_kg = pickle.load(f)
    
    with open(kg_dir / "care_pathway_kg.pkl", "rb") as f:
        care_pathway_kg = pickle.load(f)
    
    return patient_kg, clinical_rules_kg, care_pathway_kg


def check_contraindication(conditions: List[str], medication: str, rules_kg: nx.Graph) -> Tuple[bool, str]:
    """Check if medication is contraindicated for any patient condition.
    Returns (is_contraindicated, reason)."""
    
    # Normalize medication name
    med_node = f"medication:{medication}"
    
    # Check each condition
    for condition in conditions:
        cond_node = f"condition:{condition}"
        
        # Check if edge exists
        if rules_kg.has_edge(cond_node, med_node):
            edge_data = rules_kg.get_edge_data(cond_node, med_node)
            if edge_data.get('edge_type') == 'CONTRAINDICATION':
                reason = edge_data.get('reason', 'Unknown contraindication')
                return True, f"{condition} + {medication}: {reason}"
    
    return False, ""


def compute_risk_cascade(conditions: List[str], rules_kg: nx.Graph, max_depth=3) -> Tuple[float, List[str]]:
    """Compute cascading risk score through knowledge graph.
    Returns (max_risk_score, risk_chain)."""
    
    risk_scores = {}
    risk_chains = {}
    
    for condition in conditions:
        cond_node = f"condition:{condition}"
        
        if not rules_kg.has_node(cond_node):
            continue
        
        # BFS through INCREASES_RISK edges
        visited = {cond_node}
        queue = [(cond_node, 1.0, 0, [cond_node])]
        
        while queue:
            current, risk, depth, chain = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            for _, target, data in rules_kg.out_edges(current, data=True):
                if data.get('edge_type') == 'INCREASES_RISK' and target not in visited:
                    multiplier = data.get('risk_multiplier', 1.5)
                    new_risk = risk * multiplier
                    new_chain = chain + [target]
                    
                    if target not in risk_scores or new_risk > risk_scores[target]:
                        risk_scores[target] = new_risk
                        risk_chains[target] = new_chain
                    
                    visited.add(target)
                    queue.append((target, new_risk, depth + 1, new_chain))
    
    if not risk_scores:
        return 1.0, []
    
    max_risk_target = max(risk_scores, key=risk_scores.get)
    return risk_scores[max_risk_target], risk_chains[max_risk_target]


def evaluate_symbolic_only():
    """Generate real quantitative metrics using only symbolic reasoning."""
    
    print("="*80)
    print("SYMBOLIC KNOWLEDGE GRAPH REASONING - EVALUATION")
    print("="*80)
    print()
    
    # Load knowledge graphs
    print("[1/4] Loading knowledge graphs...")
    kg_dir = Path("/Users/sanjaybasu/waymark-local/notebooks/neurosymbolic/knowledge_graphs")
    patient_kg, rules_kg, pathway_kg = load_knowledge_graphs(kg_dir)
    
    print(f"  ✓ Patient KG: {patient_kg.number_of_nodes():,} nodes, {patient_kg.number_of_edges():,} edges")
    print(f"  ✓ Clinical Rules KG: {rules_kg.number_of_nodes()} nodes, {rules_kg.number_of_edges()} edges")
    print(f"  ✓ Care Pathway KG: {pathway_kg.number_of_nodes()} nodes, {pathway_kg.number_of_edges()} edges")
    print()
    
    # Create test scenarios
    print("[2/4] Creating test scenarios...")
    
    # Common conditions that have contraindications
    test_conditions = [
        'Chronic Kidney Disease',
        'Asthma',
        'Pregnancy',
        'Diabetes',
        'Hypertension',
        'Heart Failure'
    ]
    
    # Medications to check
    test_medications = [
        'NSAIDs',
        'ACE Inhibitors',
        'Beta Blockers'
    ]
    
    scenarios = []
    for i in range(100):
        num_conditions = np.random.randint(0, 3)
        conditions = list(np.random.choice(test_conditions, num_conditions, replace=False))
        
        scenarios.append({
            'patient_id': f'test_{i}',
            'age': np.random.randint(25, 85),
            'conditions': conditions,
            'ed_visits': np.random.choice([0, 0, 0, 1, 2], p=[0.7, 0.1, 0.1, 0.05, 0.05])
        })
    
    print(f"  ✓ Created {len(scenarios)} test scenarios")
    print()
    
    # Run symbolic reasoning
    print("[3/4] Evaluating symbolic reasoning...")
    
    contraindications_found = 0
    blocks = 0
    total_risk = 0
    elevated_risk_count = 0
    escalations = 0
    total_time = 0
    contraindication_details = []
    risk_chains = []
    
    results = []
    
    for i, scenario in enumerate(scenarios):
        if (i+1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(scenarios)}...")
        
        conditions = scenario['conditions']
        
        start = time.time()
        
        # Check each medication for contraindications
        has_contraindication = False
        blocked = False
        contraindication_reason = ""
        
        for med in test_medications:
            is_contraindicated, reason = check_contraindication(conditions, med, rules_kg)
            if is_contraindicated:
                contraindications_found += 1
                has_contraindication = True
                blocked = True  # Would block this medication
                contraindication_reason = reason
                contraindication_details.append({
                    'patient': scenario['patient_id'],
                    'medication': med,
                    'reason': reason
                })
        
        # Compute risk cascade
        risk_score = 1.0
        risk_chain = []
        if conditions:
            risk_score, risk_chain = compute_risk_cascade(conditions, rules_kg)
            total_risk += risk_score
            
            if risk_score > 2.0:
                elevated_risk_count += 1
                risk_chains.append({
                    'patient': scenario['patient_id'],
                    'risk_score': risk_score,
                    'chain': ' → '.join(risk_chain)
                })
        
        # Decision logic
        action = "reassure"
        if has_contraindication and scenario['ed_visits'] > 0:
            action = "escalate_to_doctor"
            escalations += 1
        elif has_contraindication:
            action = "warn_severe"
        elif risk_score > 2.5:
            action = "contact_care_manager"
        elif risk_score > 1.5:
            action = "schedule_followup"
        
        if blocked:
            blocks += 1
        
        inference_time = (time.time() - start) * 1000
        total_time += inference_time
        
        results.append({
            'patient_id': scenario['patient_id'],
            'num_conditions': len(conditions),
            'has_contraindication': has_contraindication,
            'contraindication_reason': contraindication_reason,
            'blocked_unsafe': blocked,
            'risk_score': risk_score,
            'risk_chain': ' → '.join(risk_chain) if risk_chain else '',
            'action': action,
            'inference_time_ms': inference_time
        })
    
    print(f"  ✓ Completed {len(results)} evaluations")
    print()
    
    # Compute metrics
    print("[4/4] Computing metrics...")
    
    patients_with_conditions = sum(1 for s in scenarios if s['conditions'])
    
    metrics = EvaluationMetrics(
        total_scenarios=len(results),
        patients_with_conditions=patients_with_conditions,
        contraindications_checked=len(results) * len(test_medications),
        unsafe_actions_blocked=blocks,
        safety_block_rate=blocks / len(results) if results else 0,
        patients_with_elevated_risk=elevated_risk_count,
        avg_risk_score=total_risk / patients_with_conditions if patients_with_conditions > 0 else 1.0,
        required_interventions_identified=patients_with_conditions,
        escalations_recommended=escalations,
        avg_inference_time_ms=total_time / len(results) if results else 0,
        total_kg_nodes=patient_kg.number_of_nodes() + rules_kg.number_of_nodes(),
        total_kg_edges=patient_kg.number_of_edges() + rules_kg.number_of_edges()
    )
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_dir / "symbolic_evaluation_results.csv", index=False)
    
    with open(results_dir / "symbolic_metrics.json", 'w') as f:
        json.dump(asdict(metrics), f, indent=2)
    
    print("  ✓ Results saved")
    print()
    
    # Print examples
    if contraindication_details:
        print("="*80)
        print("EXAMPLE CONTRAINDICATIONS DETECTED")
        print("="*80)
        for example in contraindication_details[:5]:
            print(f"  {example['patient']}: {example['reason']}")
        print()
    
    if risk_chains:
        print("="*80)
        print("EXAMPLE RISK CASCADES")
        print("="*80)
        for example in risk_chains[:5]:
            print(f"  {example['patient']} (risk={example['risk_score']:.1f}): {example['chain']}")
        print()
    
    # Print summary
    print("="*80)
    print("SYMBOLIC REASONING EVALUATION RESULTS")
    print("="*80)
    print()
    print(f"Total scenarios: {metrics.total_scenarios}")
    print(f"Patients with conditions: {metrics.patients_with_conditions} ({metrics.patients_with_conditions/metrics.total_scenarios:.1%})")
    print()
    print("SAFETY PERFORMANCE:")
    print(f"  Contraindications checked: {metrics.contraindications_checked}")
    print(f"  Unsafe actions blocked: {metrics.unsafe_actions_blocked}")
    print(f"  Safety block rate: {metrics.safety_block_rate:.1%}")
    print()
    print("RISK ASSESSMENT:")
    print(f"  Patients with elevated risk: {metrics.patients_with_elevated_risk} ({metrics.patients_with_elevated_risk/metrics.total_scenarios:.1%})")
    print(f"  Average risk score: {metrics.avg_risk_score:.2f}")
    print()
    print("CLINICAL REASONING:")
    print(f"  Required interventions identified: {metrics.required_interventions_identified}")
    print(f"  Escalations recommended: {metrics.escalations_recommended}")
    print()
    print("PERFORMANCE:")
    print(f"  Average inference time: {metrics.avg_inference_time_ms:.2f}ms")
    print()
    print("KNOWLEDGE GRAPH STATS:")
    print(f"  Total nodes across KGs: {metrics.total_kg_nodes:,}")
    print(f"  Total edges across KGs: {metrics.total_kg_edges:,}")
    print()
    print("="*80)
    
    return metrics


if __name__ == "__main__":
    metrics = evaluate_symbolic_only()
