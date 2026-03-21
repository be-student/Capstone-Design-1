"""
PyTorch Dataset and Sequence Preparation Utilities.

Converts customer feature data into sequential format suitable for
LSTM/Transformer input. Supports configurable window sizes, zero-padding
for short sequences, and proper chronological ordering.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional


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
