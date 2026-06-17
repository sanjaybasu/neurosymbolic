"""
Hybrid Neurosymbolic Safety Detector (Version 2)

Integrates ML-based hazard classification with symbolic knowledge graph reasoning:
1. TF-IDF + Logistic Regression for hazard type detection
2. Knowledge graph contraindication checking
3. Risk cascade computation via graph traversal
4. Unified action selection

This combines the high sensitivity of learned classifiers with the interpretability
and safety guarantees of symbolic reasoning.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Add models directory to path
sys.path.append(str(Path(__file__).parent))

from hazard_detection import HazardDetector, DetectionResult
from neurosymbolic_reasoner import SymbolicReasoner, ClinicalContext
from extraction import ClinicalExtractor


# Map hazard types to knowledge graph conditions
HAZARD_TO_CONDITION_MAP = {
    'otc_toxicity': ['substance_toxicity'],
    'contraindicated_otc': ['medication_contraindication'],
    'pregnancy_medication': ['pregnancy'],
    'renal_contraindication': ['chronic_kidney_disease'],
    'drug_interaction': ['polypharmacy'],
    'rx_adherence': ['medication_nonadherence'],
    'suicide_risk': ['mental_health_crisis'],
    'metabolic_emergency': ['diabetes', 'metabolic_disorder'],
    'cardiac_emergency': ['cardiovascular_disease', 'heart_failure'],
    'neuro_emergency': ['stroke', 'neurological_emergency'],
    'obstetric_emergency': ['pregnancy', 'obstetric_emergency'],
    'anaphylaxis': ['allergic_reaction'],
    'pediatric_emergency': ['pediatric_emergency'],
    'privacy': [],  # No clinical condition
    'privacy_proxy': [],  # No clinical condition
    'benign': [],
    'unknown': [],
}

# Map detection results to clinical actions
HAZARD_TO_ACTION_MAP = {
    'otc_toxicity': 'warn_and_escalate',
    'contraindicated_otc': 'warn_with_alternative',
    'pregnancy_medication': 'stop_and_contact_ob',
    'renal_contraindication': 'warn_and_escalate',
    'drug_interaction': 'warn_and_escalate',
    'rx_adherence': 'warn_with_alternative',
    'suicide_risk': 'activate_emergency',
    'metabolic_emergency': 'activate_emergency',
    'cardiac_emergency': 'activate_emergency',
    'neuro_emergency': 'activate_emergency',
    'obstetric_emergency': 'activate_emergency',
    'anaphylaxis': 'activate_emergency',
    'pediatric_emergency': 'activate_emergency',
    'misuse_escalation': 'activate_emergency',
    'privacy': 'refuse_and_redirect',
    'privacy_proxy': 'refuse_and_request_consent',
    'med_reconciliation': 'warn_and_escalate',
    'benign': 'reassure',
    'unknown': 'reassure',
}


class HybridNeurosymbolicReasoner:
    """
    Hybrid safety detection combining ML hazard classification with symbolic KG reasoning.
    """
    
    def __init__(
        self,
        kg_dir: str,
        hazard_detector: Optional[HazardDetector] = None,
        confidence_threshold: float = 0.15,
        rules_path: Optional[str] = None,
        vocab_dir: Optional[str] = None,
        class_thresholds: Optional[Dict[str, float]] = None,
        or_fusion: bool = False,
    ):
        """
        Initialize hybrid reasoner.
        
        Args:
            kg_dir: Directory containing knowledge graphs
            hazard_detector: Pre-trained HazardDetector (if None, must call train() first)
            confidence_threshold: Minimum confidence for hazard predictions
            rules_path: Optional path to expanded rules JSON
            vocab_dir: Optional path to synonym vocabularies for extraction
        """
        base_dir = Path(__file__).resolve().parent.parent
        rules_file = Path(rules_path) if rules_path else base_dir / "data" / "rules" / "clinical_rules_expanded.json"
        vocab_root = Path(vocab_dir) if vocab_dir else base_dir / "data"

        self.symbolic = SymbolicReasoner(kg_dir, rules_path=str(rules_file))
        self.extractor = ClinicalExtractor(str(vocab_root))
        self.hazard_detector = hazard_detector or HazardDetector(threshold=confidence_threshold)
        self.is_trained = hazard_detector is not None
        self.confidence_threshold = confidence_threshold
        self.class_thresholds = class_thresholds or {}
        self.or_fusion = or_fusion
        
    def train(self, scenarios, cv_splits: int = 5) -> Dict[str, float]:
        """
        Train the hazard detection classifier.
        
        Args:
            scenarios: List of scenarios with prompt, context, hazard_type attributes
            cv_splits: Number of cross-validation splits
            
        Returns:
            Dictionary with training metrics (accuracy_mean, macro_f1_mean, etc.)
        """
        metrics = self.hazard_detector.fit(scenarios, cv_splits=cv_splits)
        self.is_trained = True
        return metrics
    
    def detect_and_reason(
        self,
        prompt: str,
        context: Dict,
        patient_id: str = "unknown"
    ) -> Dict:
        """
        Perform hybrid detection: ML classification + symbolic reasoning.
        
        Args:
            prompt: Clinical text/question
            context: Structured context dictionary
            patient_id: Patient identifier
            
        Returns:
            Dictionary with:
                - hazard_detection: DetectionResult from ML classifier
                - clinical_context: Extracted clinical context
                - contraindications: List of detected contraindications
                - risk_score: Computed risk cascade score
                - risk_details: Risk relationship details
                - recommended_action: Safety action to take
                - reasoning: Explanation of decision
        """
        if not self.is_trained:
            raise RuntimeError("HazardDetector must be trained before calling detect_and_reason()")
        
        # Step 1: ML-based hazard detection
        detection = self.hazard_detector.predict(prompt, context)
        
        # Step 2: Extract medications/conditions with negation + temporality filtering
        extraction = self.extractor.extract(prompt)

        # Step 3: Map hazard to clinical conditions for KG reasoning
        conditions = set(HAZARD_TO_CONDITION_MAP.get(detection.label, []))
        conditions.update(extraction.conditions)
        conditions.update(context.get("conditions", []))
        medications = set(extraction.medications)
        medications.update(context.get("medications", []))
        
        # Step 4: Create clinical context for symbolic reasoning
        clinical_context = ClinicalContext(
            patient_id=patient_id,
            conditions=list(conditions),
            medications=list(medications),
            recent_encounters=[],
            goals=[],
            demographics=context.get('demographics', {}),
            risk_factors=[],
            current_state={}
        )
        
        # Step 5: Symbolic reasoning - contraindication checking
        contraindications = []
        is_safe_symbolic, violations = self.symbolic.check_contraindications(
            clinical_context,
            "medication_review"
        )
        if not is_safe_symbolic:
            contraindications.extend(violations)
        
        # Step 6: Symbolic reasoning - risk cascade computation
        risk_scores = self.symbolic.compute_risk_cascade(clinical_context)
        max_risk = max(risk_scores.values()) if risk_scores else 0.0
        required = self.symbolic.find_required_interventions(clinical_context)
        
        # Step 7: Action selection (ML hazard type overrides if high confidence)
        class_thresh = self.class_thresholds.get(detection.label, self.confidence_threshold)
        if detection.confidence >= class_thresh and detection.label not in {"benign", "unknown"}:
            # High-confidence ML prediction drives action
            recommended_action = HAZARD_TO_ACTION_MAP.get(detection.label, 'reassure')
            primary_reasoning = f"ML hazard detection: {detection.label} (confidence: {detection.confidence:.2f})"
        else:
            # Symbolic reasoning detects contraindications or elevated risks
            if contraindications:
                recommended_action = 'warn_and_escalate'
                primary_reasoning = f"Knowledge graph contraindication detected: {contraindications[0]}"
            elif max_risk > 2.5:
                recommended_action = 'warn_with_alternative'
                primary_reasoning = f"Elevated risk score: {max_risk:.2f}"
            elif required:
                recommended_action = 'warn_with_alternative'
                primary_reasoning = f"Required interventions missing: {', '.join([r[0] for r in required])}"
            else:
                # Default to reassure
                recommended_action = 'reassure'
                primary_reasoning = "No safety concerns detected"

        # OR fusion: if symbolic detects contraindication or high risk, upgrade classification
        if self.or_fusion and detection.label in {"benign", "unknown"}:
            if contraindications or max_risk > 2.5 or required:
                recommended_action = 'warn_and_escalate'
                primary_reasoning = "Symbolic safety signal triggered (OR fusion)"
        
        # Compile comprehensive reasoning
        reasoning_parts = [primary_reasoning]
        if contraindications:
            reasoning_parts.append(f"Contraindications: {', '.join(contraindications)}")
        if risk_scores:
            reasoning_parts.append(f"Risk scores: {risk_scores}")
        if required:
            reasoning_parts.append(f"Required: {', '.join([r[0] for r in required])}")
        
        return {
            'hazard_detection': detection,
            'clinical_context': clinical_context,
            'contraindications': contraindications,
            'risk_score': max_risk,
            'risk_details': risk_scores,
            'recommended_action': recommended_action,
            'reasoning': ' | '.join(reasoning_parts),
            'is_safe': recommended_action == 'reassure',
            'extraction': {
                'conditions': extraction.conditions,
                'medications': extraction.medications,
                'negated': extraction.negated_mentions,
                'historical': extraction.historical_mentions,
                'raw_mentions': extraction.debug.get("raw_mentions", []),
            },
        }
    
    def _extract_medications(self, text: str) -> List[str]:
        """Extract medication mentions from text using keyword matching."""
        medications = []
        text_lower = str(text).lower()
        
        # Common medication keywords
        med_keywords = {
            'NSAIDs': ['nsaid', 'ibuprofen', 'advil', 'motrin', 'aleve', 'naproxen'],
            'ACE Inhibitors': ['ace inhibitor', 'lisinopril', 'enalapril', 'ramipril'],
            'Beta Blockers': ['beta blocker', 'metoprolol', 'atenolol', 'propranolol'],
        }
        
        for medication, keywords in med_keywords.items():
            if any(kw in text_lower for kw in keywords):
                medications.append(medication)
        
        return medications
    
    def evaluate_scenario(self, scenario) -> Dict:
        """
        Evaluate a single scenario and return detailed results.
        
        Args:
            scenario: Scenario object or dict with prompt, context, hazard_type
            
        Returns:
            Dictionary with evaluation results including ground truth comparison
        """
        if hasattr(scenario, 'prompt'):
            prompt = scenario.prompt
            context = scenario.context if hasattr(scenario, 'context') else {}
            hazard_true = scenario.hazard_type
            scenario_name = scenario.name if hasattr(scenario, 'name') else "unknown"
        else:
            prompt = scenario.get('prompt', '')
            context = scenario.get('context', {})
            hazard_true = scenario.get('hazard_type', 'unknown')
            scenario_name = scenario.get('name', 'unknown')
        
        # Run hybrid detection
        result = self.detect_and_reason(prompt, context, patient_id=scenario_name)
        
        # Compare to ground truth
        hazard_pred = result['hazard_detection'].label
        is_safe_pred = result['is_safe']
        
        # Ground truth: hazard scenarios require action, benign scenarios don't
        is_safe_true = (hazard_true == 'benign')
        
        return {
            'scenario': scenario_name,
            'prompt': str(prompt)[:100],  # First 100 chars for reference
            'hazard_true': hazard_true,
            'hazard_pred': hazard_pred,
            'detection_confidence': result['hazard_detection'].confidence,
            'detection_probabilities': result['hazard_detection'].probabilities,
            'contraindications_detected': len(result['contraindications']),
            'risk_score': result['risk_score'],
            'recommended_action': result['recommended_action'],
            'is_safe_pred': is_safe_pred,
            'is_safe_true': is_safe_true,
            'correct': (is_safe_pred == is_safe_true),
            'reasoning': result['reasoning']
        }
