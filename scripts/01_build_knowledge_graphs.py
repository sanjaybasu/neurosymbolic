"""
Build clinical knowledge graphs from Waymark data.
Creates: Patient KG, Clinical Rules KG, Medication Interaction KG, Care Pathway KG
"""

import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx
import json
from typing import Dict, List, Set, Tuple, Optional
import pickle


class ClinicalKnowledgeGraphBuilder:
    """
    Builds multiple interconnected knowledge graphs from Waymark data:
    1. Patient Graph: patients -> conditions -> medications -> encounters
    2. Clinical Rules Graph: conditions -> contraindications -> interventions
    3. Care Pathway Graph: successful intervention sequences
    4. Risk Propagation Graph: how risks cascade through patient population
    """
    
    def __init__(self, data_dir: str = "/Users/sanjaybasu/waymark-local/data"):
        self.data_dir = Path(data_dir)
        self.real_inputs = self.data_dir / "real_inputs"
        
        # Initialize graphs
        self.patient_kg = nx.MultiDiGraph()
        self.clinical_rules_kg = nx.DiGraph()
        self.care_pathway_kg = nx.DiGraph()
        self.risk_kg = nx.DiGraph()
        
        # Will hold mapping between internal IDs (member_id/person_id/etc.)
        self.patient_id_map: Dict[str, str] = {}
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all necessary data sources."""
        
        print("Loading data sources...")
        
        data = {
            'member_attributes': pd.read_parquet(self.real_inputs / "member_attributes.parquet"),
            'encounters': pd.read_parquet(self.real_inputs / "encounters.parquet"),
            'hospital_visits': pd.read_parquet(self.real_inputs / "hospital_visits.parquet"),
            'member_goals': pd.read_parquet(self.real_inputs / "member_goals.parquet"),
            'outcomes_monthly': pd.read_parquet(self.real_inputs / "outcomes_monthly.parquet"),
            'interventions': pd.read_csv(self.real_inputs / "interventions.csv"),
            'eligibility': pd.read_parquet(self.real_inputs / "eligibility.parquet"),
        }
        
        print(f"Loaded {len(data)} sources")
        return data
    
    def _get_first_existing(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Return the first column name that exists in the dataframe."""
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _resolve_patient_id(self, raw_id: Optional[str]) -> Optional[str]:
        """Map raw patient id to canonical id if mapping exists."""
        if raw_id is None:
            return None
        return self.patient_id_map.get(raw_id, raw_id)
    
    def build_patient_knowledge_graph(self, data: Dict) -> nx.MultiDiGraph:
        """
        Build patient-centric knowledge graph.
        
        Structure:
        Patient -> HAS_CONDITION -> Condition
        Patient -> TAKES_MEDICATION -> Medication  
        Patient -> HAD_ENCOUNTER -> Encounter
        Patient -> HAS_GOAL -> Goal
        Patient -> EXPERIENCED_OUTCOME -> Outcome
        Condition -> INCREASES_RISK_OF -> Condition
        Medication -> INTERACTS_WITH -> Medication
        """
        
        print("\nBuilding patient knowledge graph...")
        
        # Build patient_id mapping from any tables that carry both ids
        if {'patient_id', 'member_id'}.issubset(data['member_goals'].columns):
            for _, row in data['member_goals'].iterrows():
                self.patient_id_map[str(row['patient_id'])] = str(row['member_id'])
        
        # Determine id column in member_attributes
        member_id_col = self._get_first_existing(
            data['member_attributes'],
            ['patient_id', 'member_id', 'person_id', 'id']
        )
        
        # Add patients as nodes
        condition_cols = [col for col in data['member_attributes'].columns 
                         if col.startswith('has_') or 'condition' in col.lower()]
        
        for _, row in data['member_attributes'].iterrows():
            raw_pid = str(row[member_id_col])
            patient_id = self._resolve_patient_id(raw_pid)
            
            self.patient_kg.add_node(
                patient_id,
                node_type='patient',
                **row.to_dict()
            )
        
            # Add conditions (from member_attributes - chronic conditions)
            for cond_col in condition_cols:
                if row.get(cond_col, False):
                    condition_name = cond_col.replace('has_', '').replace('_', ' ').title()
                    
                    # Add condition node if not exists
                    if not self.patient_kg.has_node(f"condition:{condition_name}"):
                        self.patient_kg.add_node(
                            f"condition:{condition_name}",
                            node_type='condition',
                            name=condition_name
                        )
                    
                    # Add edge: patient HAS_CONDITION
                    self.patient_kg.add_edge(
                        patient_id,
                        f"condition:{condition_name}",
                        edge_type='HAS_CONDITION'
                    )
        
        # Add encounters
        for _, row in data['encounters'].iterrows():
            encounter_id = f"encounter:{row.get('encounter_id', row.name)}"
            
            self.patient_kg.add_node(
                encounter_id,
                node_type='encounter',
                encounter_type=row.get('encounter_type'),
                encounter_date=row.get('encounter_date'),
                outcome=row.get('outcome'),
                waymarker_id=row.get('waymarker_id')
            )
            
            pid_col = self._get_first_existing(data['encounters'], ['patient_id', 'member_id', 'person_id'])
            pid_raw = row.get(pid_col) if pid_col else None
            pid = self._resolve_patient_id(str(pid_raw)) if pid_raw is not None else None
            
            if pid is not None:
                self.patient_kg.add_edge(
                    pid,
                    encounter_id,
                    edge_type='HAD_ENCOUNTER',
                    date=row.get('encounter_date', row.get('created_at'))
                )
        
        # Add hospital visits (critical for risk reasoning)
        for _, row in data['hospital_visits'].iterrows():
            visit_id = f"hospital_visit:{row.get('visit_id', row.name)}"
            
            self.patient_kg.add_node(
                visit_id,
                node_type='hospital_visit',
                is_ed=row.get('is_ed', False),
                is_inpatient=row.get('is_inpatient', False),
                is_avoidable=row.get('is_avoidable', False),
                visit_date=row.get('visit_date'),
                primary_diagnosis=row.get('primary_diagnosis')
            )
            
            pid_col = self._get_first_existing(data['hospital_visits'], ['patient_id', 'member_id', 'person_id'])
            pid_raw = row.get(pid_col) if pid_col else None
            pid = self._resolve_patient_id(str(pid_raw)) if pid_raw is not None else None
            
            if pid is not None:
                self.patient_kg.add_edge(
                    pid,
                    visit_id,
                    edge_type='EXPERIENCED_VISIT',
                    date=row.get('visit_date', row.get('admit_date'))
                )
            
            # If avoidable, create a high-weight risk edge
            if row.get('is_avoidable', False) and pid is not None:
                self.patient_kg.add_edge(
                    pid,
                    visit_id,
                    edge_type='AVOIDABLE_VISIT',
                    severity='high'
                )
        
        # Add goals
        for _, row in data['member_goals'].iterrows():
            goal_id = f"goal:{row['goal_id']}"
            
            self.patient_kg.add_node(
                goal_id,
                node_type='goal',
                goal_category=row.get('goal_category'),
                is_completed=row.get('is_completed', False),
                created_date=row.get('created_date')
            )
            
            pid = self._resolve_patient_id(str(row.get('patient_id'))) if row.get('patient_id') is not None else None
            if pid is not None:
                self.patient_kg.add_edge(
                    pid,
                    goal_id,
                    edge_type='HAS_GOAL'
                )
        
        # Add interventions received
        for idx, row in data['interventions'].iterrows():
            intervention_id = f"intervention:{row.get('intervention_id', idx)}"
            
            self.patient_kg.add_node(
                intervention_id,
                node_type='intervention',
                intervention_type=row.get('intervention_type', row.get('intervention')),
                intervention_date=row.get('intervention_date'),
                was_successful=row.get('was_successful', None)
            )
            
            pid_col = self._get_first_existing(data['interventions'], ['patient_id', 'member_id', 'person_id', 'person_key'])
            pid_raw = row.get(pid_col) if pid_col else None
            pid = self._resolve_patient_id(str(pid_raw)) if pid_raw is not None else None
            
            if pid is not None:
                self.patient_kg.add_edge(
                    pid,
                    intervention_id,
                    edge_type='RECEIVED_INTERVENTION',
                    date=row.get('intervention_date')
                )
        
        print(f"Patient KG: {self.patient_kg.number_of_nodes()} nodes, "
              f"{self.patient_kg.number_of_edges()} edges")
        
        return self.patient_kg
    
    def build_clinical_rules_graph(self) -> nx.DiGraph:
        """
        Build clinical rules knowledge graph.
        Encodes domain knowledge about contraindications, risk factors, etc.
        
        This is the SYMBOLIC reasoning component.
        """
        
        print("\nBuilding clinical rules knowledge graph...")
        
        # Define clinical rules (these should ideally come from clinical guidelines)
        rules = [
            # Medication contraindications
            {
                'condition': 'Chronic Kidney Disease',
                'contraindication': 'NSAIDs',
                'severity': 'major',
                'reason': 'Risk of acute kidney injury'
            },
            {
                'condition': 'Pregnancy',
                'contraindication': 'ACE Inhibitors',
                'severity': 'severe',
                'reason': 'Teratogenic effects'
            },
            {
                'condition': 'Asthma',
                'contraindication': 'Beta Blockers',
                'severity': 'major',
                'reason': 'Bronchospasm risk'
            },
            {
                'condition': 'Chronic Kidney Disease',
                'contraindication': 'Metformin',
                'severity': 'moderate',
                'reason': 'Lactic acidosis risk'
            },
            {
                'condition': 'Heart Failure',
                'contraindication': 'NSAIDs',
                'severity': 'major',
                'reason': 'Fluid retention and decompensation risk'
            },
            {
                'condition': 'Pregnancy',
                'contraindication': 'Valproate',
                'severity': 'severe',
                'reason': 'Neural tube defect risk'
            },
            {
                'condition': 'Pregnancy',
                'contraindication': 'Warfarin',
                'severity': 'severe',
                'reason': 'Teratogenic effects and fetal bleeding risk'
            },
            
            # Risk amplification rules
            {
                'condition': 'Diabetes',
                'increases_risk_of': 'Cardiovascular Disease',
                'risk_multiplier': 2.5
            },
            {
                'condition': 'Hypertension',
                'increases_risk_of': 'Stroke',
                'risk_multiplier': 3.0
            },
            {
                'condition': 'Chronic Kidney Disease',
                'increases_risk_of': 'Hyperkalemia',
                'risk_multiplier': 2.0
            },
            {
                'condition': 'Heart Failure',
                'increases_risk_of': 'Hospitalization',
                'risk_multiplier': 1.8
            },
            {
                'condition': 'Depression',
                'increases_risk_of': 'Suicide Risk',
                'risk_multiplier': 2.2
            },
            {
                'condition': 'COPD',
                'increases_risk_of': 'Hospitalization',
                'risk_multiplier': 2.0
            },
            
            # Intervention requirements
            {
                'condition': 'Diabetes',
                'requires_intervention': 'HbA1c Monitoring',
                'frequency': 'quarterly'
            },
            {
                'condition': 'Pregnancy',
                'requires_intervention': 'Prenatal Care',
                'frequency': 'monthly',
                'urgency': 'high'
            },
            
            # Behavioral health rules
            {
                'condition': 'Depression',
                'requires_screening': 'PHQ-9',
                'frequency': 'quarterly'
            },
            {
                'condition': 'Substance Use Disorder',
                'requires_intervention': 'MAT Referral',
                'urgency': 'high'
            },
            {
                'condition': 'Heart Failure',
                'requires_intervention': 'Weight Monitoring',
                'frequency': 'weekly',
                'urgency': 'high'
            },
            {
                'condition': 'Heart Failure',
                'requires_intervention': 'Weight Monitoring',
                'frequency': 'weekly',
                'urgency': 'high'
            },
            
            # Social determinants
            {
                'sdoh_factor': 'Housing Instability',
                'increases_risk_of': 'ED Utilization',
                'risk_multiplier': 1.8
            },
            {
                'sdoh_factor': 'Food Insecurity',
                'increases_risk_of': 'Diabetes Poor Control',
                'risk_multiplier': 2.1
            },
        ]
        
        # Build graph from rules
        for rule in rules:
            if 'contraindication' in rule:
                # Contraindication rule
                condition_node = f"condition:{rule['condition']}"
                contra_node = f"medication:{rule['contraindication']}"
                
                self.clinical_rules_kg.add_node(condition_node, node_type='condition')
                self.clinical_rules_kg.add_node(contra_node, node_type='medication')
                
                self.clinical_rules_kg.add_edge(
                    condition_node,
                    contra_node,
                    edge_type='CONTRAINDICATION',
                    severity=rule['severity'],
                    reason=rule['reason']
                )
            
            elif 'increases_risk_of' in rule:
                # Risk amplification rule
                if 'condition' in rule:
                    from_node = f"condition:{rule['condition']}"
                else:
                    from_node = f"sdoh:{rule['sdoh_factor']}"
                
                to_node = f"condition:{rule['increases_risk_of']}"
                
                self.clinical_rules_kg.add_node(from_node, node_type='condition')
                self.clinical_rules_kg.add_node(to_node, node_type='condition')
                
                self.clinical_rules_kg.add_edge(
                    from_node,
                    to_node,
                    edge_type='INCREASES_RISK',
                    risk_multiplier=rule['risk_multiplier']
                )
            
            elif 'requires_intervention' in rule:
                # Required intervention rule
                condition_node = f"condition:{rule['condition']}"
                intervention_node = f"intervention:{rule['requires_intervention']}"
                
                self.clinical_rules_kg.add_node(condition_node, node_type='condition')
                self.clinical_rules_kg.add_node(intervention_node, node_type='intervention')
                
                self.clinical_rules_kg.add_edge(
                    condition_node,
                    intervention_node,
                    edge_type='REQUIRES',
                    frequency=rule.get('frequency'),
                    urgency=rule.get('urgency', 'medium')
                )
        
        print(f"Clinical Rules KG: {self.clinical_rules_kg.number_of_nodes()} nodes, "
              f"{self.clinical_rules_kg.number_of_edges()} edges")
        
        return self.clinical_rules_kg
    
    def build_care_pathway_graph(self, data: Dict) -> nx.DiGraph:
        """
        Learn successful care pathways from data.
        
        This discovers patterns like:
        "In-person encounter -> Goal setting -> Phone follow-up -> ED avoidance"
        """
        
        print("\nBuilding care pathway knowledge graph...")
        
        # Get successful trajectories (those that avoided ED/hospital)
        successful_trajectories = []
        
        # Group by patient, find sequences that led to good outcomes
        id_col = self._get_first_existing(
            data['outcomes_monthly'],
            ['patient_id', 'person_id', 'member_id']
        )
        if id_col is None:
            print("No patient identifier found in outcomes; skipping care pathway graph.")
            self.care_pathway_kg.graph['status'] = 'empty_no_identifier'
            return self.care_pathway_kg
        
        for patient_id, patient_data in data['outcomes_monthly'].groupby(id_col):
            # Check if patient avoided ED in subsequent months
            month_col = 'month' if 'month' in patient_data.columns else 'month_year'
            patient_data = patient_data.sort_values(month_col)
            
            for i in range(len(patient_data) - 1):
                current_month = patient_data.iloc[i]
                next_month = patient_data.iloc[i + 1]
                
                # If current month had interventions and next month avoided ED
                had_intervention = current_month.get('had_intervention', False)
                ed_visits_next = next_month.get('ed_visits', next_month.get('emergency_department_ct', 0))
                
                if (had_intervention and ed_visits_next == 0):
                    
                    successful_trajectories.append({
                        'patient_id': patient_id,
                        'month': current_month.get(month_col),
                        'interventions': current_month.get('interventions'),
                        'outcome': 'ed_avoided'
                    })
        
        # Build pathway graph from successful sequences
        # Each node is an intervention type, edges show temporal sequences
        intervention_sequences = {}
        
        for traj in successful_trajectories:
            interventions = traj.get('interventions', [])
            
            for i in range(len(interventions) - 1):
                curr_intervention = interventions[i]
                next_intervention = interventions[i + 1]
                
                # Add nodes
                for intervention in [curr_intervention, next_intervention]:
                    if not self.care_pathway_kg.has_node(intervention):
                        self.care_pathway_kg.add_node(
                            intervention,
                            node_type='intervention_type',
                            success_count=0
                        )
                
                # Add/update edge
                if self.care_pathway_kg.has_edge(curr_intervention, next_intervention):
                    self.care_pathway_kg[curr_intervention][next_intervention]['count'] += 1
                else:
                    self.care_pathway_kg.add_edge(
                        curr_intervention,
                        next_intervention,
                        count=1,
                        success_rate=None  # computed later
                    )
        
        # Compute success rates for each pathway
        for u, v, data in self.care_pathway_kg.edges(data=True):
            total = data['count']
            success = len([t for t in successful_trajectories 
                          if u in t.get('interventions', []) and 
                             v in t.get('interventions', [])])
            data['success_rate'] = success / total if total > 0 else 0
        
        print(f"Care Pathway KG: {self.care_pathway_kg.number_of_nodes()} nodes, "
              f"{self.care_pathway_kg.number_of_edges()} edges")
        if self.care_pathway_kg.number_of_edges() == 0:
            self.care_pathway_kg.graph['status'] = 'empty_no_sequences'
        
        return self.care_pathway_kg
    
    def save_knowledge_graphs(self, output_dir: str):
        """Save all knowledge graphs."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle (preserves all graph attributes)
        with open(output_path / "patient_kg.pkl", 'wb') as f:
            pickle.dump(self.patient_kg, f)
        
        with open(output_path / "clinical_rules_kg.pkl", 'wb') as f:
            pickle.dump(self.clinical_rules_kg, f)
        
        with open(output_path / "care_pathway_kg.pkl", 'wb') as f:
            pickle.dump(self.care_pathway_kg, f)
        
        # Also save as JSON for inspection
        for name, graph in [
            ('patient_kg', self.patient_kg),
            ('clinical_rules_kg', self.clinical_rules_kg),
            ('care_pathway_kg', self.care_pathway_kg)
        ]:
            graph_json = nx.node_link_data(graph)
            with open(output_path / f"{name}.json", 'w') as f:
                json.dump(graph_json, f, indent=2, default=str)
        
        print(f"\nSaved knowledge graphs to {output_path}")


def main():
    builder = ClinicalKnowledgeGraphBuilder()
    
    # Load data
    data = builder.load_data()
    
    # Build knowledge graphs
    builder.build_patient_knowledge_graph(data)
    builder.build_clinical_rules_graph()
    builder.build_care_pathway_graph(data)
    
    # Save
    builder.save_knowledge_graphs(
        "/Users/sanjaybasu/waymark-local/notebooks/neurosymbolic/knowledge_graphs"
    )


if __name__ == "__main__":
    main()
