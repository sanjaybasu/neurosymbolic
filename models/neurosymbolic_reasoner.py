"""
Neurosymbolic reasoner that combines:
1. Neural components (RL policy, LLM generation)
2. Symbolic components (knowledge graph reasoning)
3. Safety constraints (hard rules that cannot be violated)
"""

import torch
import torch.nn as nn
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass
import pickle
from pathlib import Path


@dataclass
class ClinicalContext:
    """Patient clinical context for reasoning."""
    patient_id: str
    conditions: List[str]
    medications: List[str]
    recent_encounters: List[Dict]
    goals: List[Dict]
    demographics: Dict
    risk_factors: List[str]
    current_state: Dict  # for RL


@dataclass
class ReasoningResult:
    """Result of neurosymbolic reasoning."""
    recommended_action: str
    action_id: int
    confidence: float
    explanation: str
    safety_checks_passed: Dict[str, bool]
    symbolic_constraints: List[str]
    neural_q_values: Dict[str, float]
    knowledge_paths: List[List[str]]  # paths in KG that support decision


class SymbolicReasoner:
    """
    Symbolic reasoning engine using knowledge graphs.
    Enforces hard constraints and provides interpretable reasoning paths.
    """
    
    def __init__(self, kg_dir: str):
        self.kg_dir = Path(kg_dir)
        self.load_knowledge_graphs()
        
    def load_knowledge_graphs(self):
        """Load pre-built knowledge graphs."""
        
        with open(self.kg_dir / "patient_kg.pkl", 'rb') as f:
            self.patient_kg = pickle.load(f)
        
        with open(self.kg_dir / "clinical_rules_kg.pkl", 'rb') as f:
            self.clinical_rules_kg = pickle.load(f)
        
        with open(self.kg_dir / "care_pathway_kg.pkl", 'rb') as f:
            self.care_pathway_kg = pickle.load(f)
        
        print("Loaded knowledge graphs")
    
    def check_contraindications(
        self,
        context: ClinicalContext,
        proposed_action: str
    ) -> Tuple[bool, List[str]]:
        """
        Check if proposed action violates any contraindication rules.
        This is a HARD CONSTRAINT that cannot be violated.
        """
        
        violations = []
        
        # Check if action involves medication
        if 'medication' in proposed_action.lower():
            # Extract medication name (simplified)
            medication = proposed_action.split('medication')[1].strip()
            
            # Check each patient condition against contraindication rules
            for condition in context.conditions:
                condition_node = f"condition:{condition}"
                
                if condition_node in self.clinical_rules_kg:
                    # Get all contraindications for this condition
                    for neighbor in self.clinical_rules_kg.neighbors(condition_node):
                        edge_data = self.clinical_rules_kg[condition_node][neighbor]
                        
                        if edge_data.get('edge_type') == 'CONTRAINDICATION':
                            if medication.lower() in neighbor.lower():
                                violations.append(
                                    f"CONTRAINDICATION: {medication} is contraindicated in "
                                    f"{condition} (Severity: {edge_data.get('severity')}, "
                                    f"Reason: {edge_data.get('reason')})"
                                )
        
        is_safe = len(violations) == 0
        return is_safe, violations
    
    def find_required_interventions(
        self,
        context: ClinicalContext
    ) -> List[Tuple[str, str, str]]:
        """
        Find interventions required by clinical rules.
        Returns: [(intervention, frequency, urgency), ...]
        """
        
        required = []
        
        for condition in context.conditions:
            condition_node = f"condition:{condition}"
            
            if condition_node in self.clinical_rules_kg:
                for neighbor in self.clinical_rules_kg.neighbors(condition_node):
                    edge_data = self.clinical_rules_kg[condition_node][neighbor]
                    
                    if edge_data.get('edge_type') == 'REQUIRES':
                        intervention = neighbor.replace('intervention:', '')
                        frequency = edge_data.get('frequency', 'as needed')
                        urgency = edge_data.get('urgency', 'medium')
                        
                        required.append((intervention, frequency, urgency))
        
        return required
    
    def compute_risk_cascade(
        self,
        context: ClinicalContext
    ) -> Dict[str, float]:
        """
        Compute cascading risks through knowledge graph.
        Uses transitive closure to find all amplified risks.
        """
        
        risk_scores = {}
        
        # Start with patient's conditions
        for condition in context.conditions:
            condition_node = f"condition:{condition}"
            
            if condition_node in self.clinical_rules_kg:
                # BFS to find all downstream risks
                visited = set()
                queue = [(condition_node, 1.0)]  # (node, cumulative_risk_multiplier)
                
                while queue:
                    current, risk_mult = queue.pop(0)
                    
                    if current in visited:
                        continue
                    visited.add(current)
                    
                    for neighbor in self.clinical_rules_kg.neighbors(current):
                        edge_data = self.clinical_rules_kg[current][neighbor]
                        
                        if edge_data.get('edge_type') == 'INCREASES_RISK':
                            new_risk = risk_mult * edge_data.get('risk_multiplier', 1.0)
                            
                            risk_name = neighbor.replace('condition:', '')
                            risk_scores[risk_name] = max(
                                risk_scores.get(risk_name, 0),
                                new_risk
                            )
                            
                            queue.append((neighbor, new_risk))
        
        return risk_scores
    
    def find_successful_pathways(
        self,
        context: ClinicalContext,
        target_outcome: str = 'ed_avoided'
    ) -> List[List[str]]:
        """
        Find successful care pathways from knowledge graph.
        Returns sequences of interventions with high success rates.
        """
        
        pathways = []
        
        # Get patient's recent intervention types
        recent_interventions = [
            enc.get('intervention_type') 
            for enc in context.recent_encounters 
            if enc.get('intervention_type')
        ]
        
        if not recent_interventions:
            # Return top pathways that lead to target outcome
            # Find all paths from any starting intervention to successful outcome
            return pathways
        
        # From last intervention, find promising next steps
        last_intervention = recent_interventions[-1]
        
        if last_intervention in self.care_pathway_kg:
            # Get neighbors sorted by success rate
            neighbors = list(self.care_pathway_kg.neighbors(last_intervention))
            neighbor_success = [
                (
                    neighbor,
                    self.care_pathway_kg[last_intervention][neighbor].get('success_rate', 0)
                )
                for neighbor in neighbors
            ]
            neighbor_success.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 3 pathways
            for neighbor, success_rate in neighbor_success[:3]:
                pathway = [last_intervention, neighbor]
                pathways.append(pathway)
        
        return pathways
    
    def explain_reasoning(
        self,
        context: ClinicalContext,
        action: str,
        knowledge_paths: List[List[str]]
    ) -> str:
        """Generate natural language explanation using KG paths."""
        
        explanation_parts = []
        
        # Explain based on patient conditions
        if context.conditions:
            explanation_parts.append(
                f"Patient has {', '.join(context.conditions)}."
            )
        
        # Explain based on risks
        risks = self.compute_risk_cascade(context)
        if risks:
            top_risks = sorted(risks.items(), key=lambda x: x[1], reverse=True)[:3]
            explanation_parts.append(
                f"Elevated risks: {', '.join([f'{r[0]} ({r[1]:.1f}x)' for r in top_risks])}."
            )
        
        # Explain based on required interventions
        required = self.find_required_interventions(context)
        if required:
            explanation_parts.append(
                f"Required interventions: {', '.join([r[0] for r in required])}."
            )
        
        # Explain based on successful pathways
        if knowledge_paths:
            explanation_parts.append(
                f"Successful care pathway: {' → '.join(knowledge_paths[0])}."
            )
        
        return " ".join(explanation_parts)


class NeuralRLPolicy(nn.Module):
    """
    Neural RL policy (your existing SARSA/HACO model).
    This is the NEURAL component.
    """
    
    def __init__(self, state_dim: int = 3, num_actions: int = 9, checkpoint_path: str = None):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.checkpoint_path = checkpoint_path
        
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.num_actions)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions."""
        return self.q_network(state)
    
    def get_action_values(self, state: Dict) -> Dict[str, float]:
        """Get Q-values as dictionary."""
        
        # Map a small set of numeric state features into a fixed-length vector
        features = [
            state.get('age', 0),
            state.get('ed_visits_90d', 0),
            state.get('hosp_admits_180d', 0),
        ]
        # Pad/truncate to expected dimensionality to avoid shape errors
        if len(features) < self.state_dim:
            features.extend([0] * (self.state_dim - len(features)))
        state_tensor = torch.tensor(features[: self.state_dim], dtype=torch.float32)
        
        with torch.no_grad():
            q_values = self.forward(state_tensor).numpy()
        
        # Map to action names
        action_names = [
            'reassure', 'warn_moderate', 'warn_severe',
            'refer_to_pcp', 'refer_to_specialist', 'schedule_followup',
            'activate_emergency', 'contact_care_manager', 'escalate_to_doctor'
        ]
        
        return {name: float(q) for name, q in zip(action_names, q_values)}


class NeurosymbolicReasoner:
    """
    Integrated neurosymbolic reasoner.
    Combines neural RL policy with symbolic knowledge graph reasoning.
    """
    
    def __init__(
        self,
        kg_dir: str,
        rl_checkpoint: Optional[str] = None,
        use_llm: bool = False
    ):
        # Load symbolic reasoner
        self.symbolic = SymbolicReasoner(kg_dir)
        
        # Load neural policy
        # Neural policy remains lightweight; keep feature dim small to match get_action_values.
        self.neural = NeuralRLPolicy(state_dim=3, num_actions=9)
        if rl_checkpoint:
            try:
                checkpoint = torch.load(rl_checkpoint, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                
                # Remap keys from 'net.*' to 'q_network.*' if needed
                remapped_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('net.'):
                        new_key = key.replace('net.', 'q_network.')
                        remapped_state_dict[new_key] = value
                    else:
                        remapped_state_dict[key] = value
                
                self.neural.load_state_dict(remapped_state_dict)
                print(f"Loaded neural policy from {rl_checkpoint}")
            except Exception as e:
                print(f"Warning: Could not load checkpoint {rl_checkpoint}: {e}")
                print("Using randomly initialized policy")
        
        self.neural.eval()
        
        # Optional: LLM for explanation generation
        self.use_llm = use_llm
        if use_llm:
            # Initialize your LLM here (Claude, GPT, etc.)
            pass
    
    def reason(self, context: ClinicalContext) -> ReasoningResult:
        """
        Main reasoning loop: Neural + Symbolic.
        
        Process:
        1. Neural policy proposes actions (soft recommendations)
        2. Symbolic rules filter unsafe actions (hard constraints)
        3. Combine scores with symbolic reasoning
        4. Generate explanation
        """
        
        # Step 1: Get neural policy recommendations (Q-values)
        neural_q_values = self.neural.get_action_values(context.current_state)
        
        # Step 2: Apply symbolic safety constraints
        safe_actions = {}
        safety_checks = {}
        
        for action, q_value in neural_q_values.items():
            # Check contraindications
            is_safe, violations = self.symbolic.check_contraindications(context, action)
            safety_checks[action] = {
                'contraindications_passed': is_safe,
                'violations': violations
            }
            
            if is_safe:
                safe_actions[action] = q_value
            else:
                print(f"Action {action} blocked by symbolic constraints: {violations}")
        
        if not safe_actions:
            # All actions unsafe - emergency escalation
            recommended_action = 'escalate_to_doctor'
            confidence = 1.0
            explanation = "All proposed actions violate safety constraints. Emergency escalation required."
        else:
            # Step 3: Augment neural scores with symbolic knowledge
            
            # Get required interventions
            required = self.symbolic.find_required_interventions(context)
            
            # Boost scores for required interventions
            for intervention, frequency, urgency in required:
                for action in safe_actions:
                    if intervention.lower() in action.lower():
                        urgency_boost = {
                            'high': 0.5,
                            'medium': 0.2,
                            'low': 0.1
                        }.get(urgency, 0)
                        safe_actions[action] += urgency_boost
            
            # Get successful pathways
            pathways = self.symbolic.find_successful_pathways(context)
            
            # Boost scores for actions in successful pathways
            for pathway in pathways:
                for step in pathway:
                    for action in safe_actions:
                        if step.lower() in action.lower():
                            safe_actions[action] += 0.3
            
            # Step 4: Select best action
            recommended_action = max(safe_actions, key=safe_actions.get)
            confidence = safe_actions[recommended_action] / sum(safe_actions.values())
            
            # Step 5: Generate explanation
            explanation = self.symbolic.explain_reasoning(
                context, recommended_action, pathways
            )
            
            # Optional: Enhance with LLM
            if self.use_llm:
                explanation = self._enhance_explanation_with_llm(
                    context, recommended_action, explanation
                )
        
        # Map action name to ID
        action_names = list(neural_q_values.keys())
        action_id = action_names.index(recommended_action)
        
        return ReasoningResult(
            recommended_action=recommended_action,
            action_id=action_id,
            confidence=confidence,
            explanation=explanation,
            safety_checks_passed=safety_checks,
            symbolic_constraints=[],  # List any active constraints
            neural_q_values=neural_q_values,
            knowledge_paths=pathways
        )
    
    def _enhance_explanation_with_llm(
        self,
        context: ClinicalContext,
        action: str,
        symbolic_explanation: str
    ) -> str:
        """Use LLM to generate patient-friendly explanation."""
        
        # Construct prompt grounded in symbolic reasoning
        prompt = f"""
        You are a clinical decision support system. Based on the following clinical reasoning,
        generate a clear, empathetic explanation for a community health worker.
        
        Patient Context:
        - Conditions: {', '.join(context.conditions)}
        - Demographics: {context.demographics}
        - Risk Factors: {', '.join(context.risk_factors)}
        
        Recommended Action: {action}
        
        Clinical Reasoning: {symbolic_explanation}
        
        Generate a 2-3 sentence explanation that a CHW can use when talking to the patient.
        Focus on why this action is important and what benefit it provides.
        """
        
        # Call LLM (placeholder - replace with actual LLM call)
        llm_explanation = "Based on your recent health indicators and care plan progress, we recommend this next step to help you stay healthy and avoid emergency room visits."
        
        return llm_explanation


def load_neurosymbolic_reasoner(
    kg_dir: str = str(Path(__file__).resolve().parent.parent / "knowledge_graphs"),
    rl_checkpoint: str = "/Users/sanjaybasu/waymark-local/notebooks/haco/results/haco/best_model.pt"
) -> NeurosymbolicReasoner:
    """Convenience function to load reasoner."""
    
    return NeurosymbolicReasoner(kg_dir, rl_checkpoint, use_llm=True)
