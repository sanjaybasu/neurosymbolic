def load_prospective_cases(benign_path: Path, harm_path: Path):
    """Load prospective validation cases with correct column mapping."""
    benign_df =pd.read_csv(benign_path)
    harm_df = pd.read_csv(harm_path)
    
    scenarios = []
    
    # Benign cases - use context_text column
    for idx, row in benign_df.iterrows():
        scenario = {
            'name': f"benign_{idx}",
            'prompt': row['context_text'],  # FIXED: Use context_text column
            'context': {},
            'hazard_type': 'benign'
        }
        scenarios.append(scenario)
    
    # Harm cases - use context_text and harm_type columns
    for idx, row in harm_df.iterrows():
        scenario = {
            'name': f"harm_{idx}",
            'prompt': row['context_text'],  # FIXED: Use context_text column
            'context': {},
            'hazard_type': row.get('harm_type', 'harm')  # Use actual harm type from CSV
        }
        scenarios.append(scenario)
    
    return scenarios
