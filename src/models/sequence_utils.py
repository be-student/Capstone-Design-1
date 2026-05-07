"""
PyTorch Dataset and Sequence Preparation Utilities.

Converts customer feature data into sequential format suitable for
LSTM/Transformer input. Supports configurable window sizes, zero-padding
for short sequences, chronological ordering, and a safe pseudo-sequence
fallback for legacy tabular training paths.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union


def create_sequences(
    data: pd.DataFrame,
    labels: pd.DataFrame,
    window_size: int,
    time_col: str = "month",
    customer_col: str = "customer_id",
    label_col: str = "churn_label",
) -> Dict[str, np.ndarray]:
    """Convert tabular customer data into fixed-length sequences.

    For each customer, extracts the last `window_size` timesteps of feature
    data. If a customer has fewer timesteps, the sequence is left-padded
    with zeros.

    Args:
        data: DataFrame with columns [customer_col, time_col, feature_cols...].
        labels: DataFrame with columns [customer_col, label_col].
        window_size: Number of timesteps per sequence.
        time_col: Column name for the time dimension.
        customer_col: Column name for customer identifier.
        label_col: Column name for the churn label.

    Returns:
        Dictionary with keys:
            - "sequences": np.ndarray of shape (n_customers, window_size, n_features)
            - "labels": np.ndarray of shape (n_customers,)
            - "customer_ids": list of customer identifiers
    """
    feature_cols = [
        c for c in data.columns if c not in (customer_col, time_col)
    ]
    n_features = len(feature_cols)

    # Build label lookup
    label_map = dict(zip(labels[customer_col], labels[label_col]))

    # Get customers that exist in both data and labels
    data_customers = set(data[customer_col].unique())
    label_customers = set(labels[customer_col].unique())
    common_customers = sorted(data_customers & label_customers)

    sequences = []
    seq_labels = []
    customer_ids = []

    for cid in common_customers:
        # Get this customer's data sorted by time
        cust_data = data[data[customer_col] == cid].sort_values(time_col)
        feat_values = cust_data[feature_cols].values  # (n_timesteps, n_features)

        n_timesteps = len(feat_values)

        # Create fixed-length sequence with zero-padding
        seq = np.zeros((window_size, n_features), dtype=np.float32)

        if n_timesteps >= window_size:
            # Take the last window_size timesteps
            seq = feat_values[-window_size:].astype(np.float32)
        else:
            # Left-pad with zeros, place data at the end
            seq[-n_timesteps:] = feat_values.astype(np.float32)

        sequences.append(seq)
        seq_labels.append(label_map[cid])
        customer_ids.append(cid)

    return {
        "sequences": np.array(sequences, dtype=np.float32),
        "labels": np.array(seq_labels, dtype=np.float32),
        "customer_ids": customer_ids,
    }


def scale_sequences(
    sequences: np.ndarray,
    feature_mean: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Standardize 3D sequence tensors feature-wise across all timesteps."""
    seq = np.asarray(sequences, dtype=np.float32)
    if seq.ndim != 3:
        raise ValueError("Expected sequences with shape (n, t, f).")

    flattened = seq.reshape(-1, seq.shape[-1])
    if feature_mean is None:
        feature_mean = flattened.mean(axis=0)
    if feature_std is None:
        feature_std = flattened.std(axis=0) + 1e-8

    scaled = ((seq - feature_mean.reshape(1, 1, -1))
              / feature_std.reshape(1, 1, -1)).astype(np.float32)
    return {
        "sequences": scaled,
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "sequence_source": "event_sequence",
    }


def tabular_to_pseudo_sequences(
    data: pd.DataFrame,
    window_size: int,
    feature_mean: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Convert tabular rows to deterministic pseudo-sequences.

    This keeps the legacy DL path working, but marks the source explicitly so
    callers can distinguish it from real event-sequence inputs.
    """
    values = data.values.astype(np.float32)

    if feature_mean is None:
        feature_mean = values.mean(axis=0)
    if feature_std is None:
        feature_std = values.std(axis=0) + 1e-8

    normalized = (values - feature_mean) / feature_std
    sequences = np.tile(
        normalized[:, np.newaxis, :], (1, window_size, 1)
    ).astype(np.float32)

    for t in range(window_size):
        scale = (t + 1) / window_size
        sequences[:, t, :] *= scale

    return {
        "sequences": sequences,
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "sequence_source": "pseudo_sequence",
    }


def prepare_sequence_training_data(
    data: Union[pd.DataFrame, Dict[str, np.ndarray]],
    labels: Optional[pd.DataFrame] = None,
    window_size: int = 6,
    time_col: str = "month",
    customer_col: str = "customer_id",
    label_col: str = "churn_label",
    feature_mean: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Normalize either real sequence input or a prebuilt sequence payload."""
    if isinstance(data, dict):
        payload = dict(data)
    else:
        if labels is None:
            raise ValueError("labels must be provided for sequential DataFrame input.")
        payload = create_sequences(
            data=data,
            labels=labels,
            window_size=window_size,
            time_col=time_col,
            customer_col=customer_col,
            label_col=label_col,
        )

    scaled = scale_sequences(
        payload["sequences"],
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    payload.update(scaled)
    return payload


class ChurnSequenceDataset(Dataset):
    """PyTorch Dataset for churn prediction sequences.

    Wraps numpy arrays of sequences and labels into a Dataset compatible
    with torch DataLoader.

    Args:
        sequences: Array of shape (n_samples, window_size, n_features).
        labels: Array of shape (n_samples,) with binary churn labels.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Initialize dataset with sequences and labels."""
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.sequences)

    def __getitem__(self, idx: int):
        """Return (sequence, label) tuple as tensors.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (sequence_tensor, label_tensor) where:
                - sequence_tensor: shape (window_size, n_features), float32
                - label_tensor: scalar float32
        """
        return self.sequences[idx], self.labels[idx]
