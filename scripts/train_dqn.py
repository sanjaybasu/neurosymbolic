"""
Train a simple Deep Q-Network (DQN) on tabular state features.

This is a lightweight offline trainer intended as a starting point. It:
- Loads a parquet file with state features (age, ed_visits_90d, hosp_admits_180d, optional extras)
- Expects an action label column (default: 'action') mapping to the neurosymbolic action set
- Expects a reward column (default: 'reward'); if missing and --synthetic-actions is set,
  it will create synthetic actions/rewards for smoke testing.
- Performs offline DQN updates with a replay buffer and target network.

NOTE: Replace the reward shaping with your task-specific logic. The provided synthetic mode is
only for wiring/tests and will not yield clinically meaningful policies.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from notebooks.neurosymbolic.models.dqn_agent import DQNAgent, DQNConfig


ACTION_NAMES: List[str] = [
    "reassure",
    "warn_moderate",
    "warn_severe",
    "start_medication_nsaids",
    "start_medication_ace_inhibitors",
    "refer_to_pcp",
    "schedule_followup",
    "contact_care_manager",
    "escalate_to_doctor",
]


def build_state(row: pd.Series, feature_names: List[str]) -> np.ndarray:
    """Extract a fixed-length state vector from a pandas row."""
    return np.array([row.get(f, 0.0) for f in feature_names], dtype=np.float32)


def synthetic_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Attach synthetic actions/rewards for smoke testing."""
    rng = np.random.default_rng(seed=42)
    df = df.copy()
    df["action"] = rng.integers(low=0, high=len(ACTION_NAMES), size=len(df))
    # Simple reward: small negative for all steps
    df["reward"] = -0.1
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("/Users/sanjaybasu/waymark-local/data/export_out_rebuilt/ml/observations_train.parquet"),
        help="Path to training parquet.",
    )
    parser.add_argument(
        "--action-col",
        type=str,
        default="action",
        help="Column name for discrete action labels.",
    )
    parser.add_argument(
        "--reward-col",
        type=str,
        default="reward",
        help="Column name for rewards.",
    )
    parser.add_argument(
        "--feature-cols",
        type=str,
        default="age,ed_visits_90d,hosp_admits_180d",
        help="Comma-separated list of feature columns to use as state.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of passes over the dataset.",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=10_000,
        help="Max training steps per epoch (offline sampling).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for DQN updates.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )
    parser.add_argument(
        "--synthetic-actions",
        action="store_true",
        help="If set, synthesize actions/rewards when missing (for smoke tests only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/sanjaybasu/waymark-local/notebooks/haco/results/haco/best_model.pt"),
        help="Path to save trained checkpoint.",
    )
    args = parser.parse_args()

    feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]

    df = pd.read_parquet(args.train_path)
    if args.action_col not in df.columns or args.reward_col not in df.columns:
        if args.synthetic_actions:
            print("Action/reward columns missing; generating synthetic labels for smoke test.")
            df = synthetic_labels(df)
        else:
            raise ValueError(f"Dataset must contain '{args.action_col}' and '{args.reward_col}' columns.")

    # Map actions to indices; if string labels, map via ACTION_NAMES
    if df[args.action_col].dtype == object:
        action_to_idx = {name: i for i, name in enumerate(ACTION_NAMES)}
        if not set(df[args.action_col].unique()).issubset(action_to_idx.keys()):
            missing = set(df[args.action_col].unique()) - set(action_to_idx.keys())
            raise ValueError(f"Unrecognized actions in data: {missing}")
        actions = df[args.action_col].map(action_to_idx).to_numpy()
    else:
        actions = df[args.action_col].astype(int).to_numpy()

    rewards = df[args.reward_col].astype(float).to_numpy()
    states = np.stack([build_state(row, feature_cols) for _, row in df.iterrows()])

    # For offline, we set next_state = state and done=True (no trajectories).
    # Replace with actual next_state/done if available.
    next_states = states.copy()
    dones = np.ones(len(df), dtype=bool)

    cfg = DQNConfig(
        state_dim=len(feature_cols),
        n_actions=len(ACTION_NAMES),
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        device=args.device,
    )
    agent = DQNAgent(cfg)

    # Load offline data into buffer
    for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
        agent.push((s, a, r, ns, d))

    total_steps = args.epochs * args.steps_per_epoch
    print(f"Starting offline training for {total_steps} steps on {len(df)} samples.")
    losses = []
    for step in range(total_steps):
        loss = agent.update()
        if loss is not None:
            losses.append(loss)
        if (step + 1) % 10_000 == 0:
            avg_loss = np.mean(losses[-1000:]) if losses else None
            print(f"Step {step+1}/{total_steps} avg_loss={avg_loss}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(args.output))
    print(f"Saved DQN checkpoint to {args.output}")


if __name__ == "__main__":
    main()
