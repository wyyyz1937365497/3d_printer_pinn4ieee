"""
Data processing utilities
"""

import random
import pickle
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(file_path: str) -> Any:
    """
    Load data from pickle file

    Args:
        file_path: Path to the data file

    Returns:
        Loaded data object
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_data(data: Any, file_path: str):
    """
    Save data to pickle file

    Args:
        data: Data to save
        file_path: Path to save the data
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def normalize_data(data: np.ndarray,
                   scaler: Optional[StandardScaler] = None,
                   fit: bool = True) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize data using StandardScaler

    Args:
        data: Input data array of shape (n_samples, n_features)
        scaler: Existing scaler to use (if None, creates new one)
        fit: Whether to fit the scaler on the data

    Returns:
        Normalized data and the fitted scaler
    """
    if scaler is None:
        scaler = StandardScaler()

    if fit:
        normalized_data = scaler.fit_transform(data)
    else:
        normalized_data = scaler.transform(data)

    return normalized_data, scaler


def create_sequences(data: np.ndarray,
                    seq_len: int,
                    pred_len: int,
                    stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences and prediction targets from time series data

    Args:
        data: Time series data of shape (n_timesteps, n_features)
        seq_len: Length of input sequences
        pred_len: Length of prediction targets
        stride: Stride between sequences

    Returns:
        Input sequences (X) and prediction targets (y)
    """
    X, y = [], []

    for i in range(0, len(data) - seq_len - pred_len + 1, stride):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + pred_len])

    return np.array(X), np.array(y)


def split_data(X: np.ndarray,
               y: np.ndarray,
               train_ratio: float = 0.7,
               val_ratio: float = 0.15,
               test_ratio: float = 0.15,
               shuffle: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split data into train, validation, and test sets

    Args:
        X: Input features
        y: Targets
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        shuffle: Whether to shuffle the data

    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing (X, y) tuple
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    splits = {
        'train': (X[:train_end], y[:train_end]),
        'val': (X[train_end:val_end], y[train_end:val_end]),
        'test': (X[val_end:], y[val_end:]),
    }

    return splits


def compute_rul(labels: np.ndarray, fault_indices: np.ndarray) -> np.ndarray:
    """
    Compute Remaining Useful Life (RUL) for each time step

    Args:
        labels: Fault labels (0 for normal, 1 for fault)
        fault_indices: Indices where faults occur

    Returns:
        RUL values for each time step
    """
    rul = np.zeros(len(labels))

    for fault_idx in fault_indices:
        # Assign RUL values for time steps before fault
        rul[:fault_idx] = np.maximum(rul[:fault_idx], fault_idx - np.arange(fault_idx))

    return rul


def denormalize_data(normalized_data: np.ndarray,
                    scaler: StandardScaler) -> np.ndarray:
    """
    Denormalize data using fitted scaler

    Args:
        normalized_data: Normalized data
        scaler: Fitted StandardScaler

    Returns:
        Denormalized data
    """
    return scaler.inverse_transform(normalized_data)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array

    Args:
        tensor: PyTorch tensor

    Returns:
        Numpy array
    """
    return tensor.detach().cpu().numpy()


def numpy_to_tensor(array: np.ndarray,
                   device: str = 'cpu',
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor

    Args:
        array: Numpy array
        device: Device to place tensor on
        dtype: Data type of tensor

    Returns:
        PyTorch tensor
    """
    return torch.from_numpy(array).to(device=device, dtype=dtype)
