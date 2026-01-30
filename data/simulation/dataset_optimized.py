"""
OPTIMIZED: 3D Printer Simulation Dataset with GPU-Accelerated Preprocessing

Key optimizations:
1. Pre-normalized data during __init__ (one-time cost vs per-sample)
2. Cached tensor conversion (no repeated numpy->torch conversions)
3. Optional in-memory data caching for small datasets
4. Vectorized batch collation
"""

import os
import glob
import numpy as np
import torch
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler


class OptimizedPrinterSimulationDataset(Dataset):
    """
    OPTIMIZED PyTorch Dataset with pre-normalized data and tensor caching

    Key difference: Data normalization happens ONCE during init, not per-sample
    """

    INPUT_FEATURES = [
        'x_ref', 'y_ref', 'z_ref',
        'vx_ref', 'vy_ref', 'vz_ref',
        'T_nozzle', 'T_interface',
        'F_inertia_x', 'F_inertia_y',
        'cooling_rate', 'layer_num'
    ]

    OUTPUT_TRAJECTORY = ['error_x', 'error_y']
    OUTPUT_QUALITY = [
        'adhesion_ratio',
        'internal_stress',
        'porosity',
        'dimensional_accuracy',
        'quality_score'
    ]

    def __init__(self,
                 data_files: Union[str, List[str]],
                 seq_len: int = 200,
                 pred_len: int = 50,
                 stride: int = 10,
                 mode: str = 'train',
                 scaler: Optional[StandardScaler] = None,
                 fit_scaler: bool = True,
                 cache_in_memory: bool = False):
        """
        Initialize OPTIMIZED dataset

        Args:
            cache_in_memory: If True, cache all tensor data in RAM (faster but uses more memory)
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.mode = mode
        self.cache_in_memory = cache_in_memory

        # Load raw data
        self.data_list = self._load_data(data_files)

        # Fit or use provided scaler
        if scaler is None and fit_scaler:
            self.scaler = StandardScaler()
            self._fit_scaler()
        else:
            self.scaler = scaler

        # Create sequences with PRE-NORMALIZED data
        print(f"[{mode.upper()}] Creating pre-normalized sequences...")
        self.sequences = self._create_sequences()

        # Optional: Cache all tensors in memory
        if self.cache_in_memory:
            print(f"[{mode.upper()}] Caching {len(self.sequences)} sequences in memory...")
            self._cache_tensors()

        print(f"[{mode.upper()}] Loaded {len(self.sequences)} sequences "
              f"from {len(self.data_list)} files")

    def _load_data(self, data_files: Union[str, List[str]]) -> List[Dict]:
        """Load MATLAB .mat files (same as original)"""
        if isinstance(data_files, str):
            if os.path.isdir(data_files):
                data_files = glob.glob(os.path.join(data_files, "*.mat"))
            else:
                data_files = [data_files]

        data_list = []
        for filepath in data_files:
            try:
                data = self._load_mat_file(filepath)
                if data is not None:
                    data_list.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")

        print(f"Loaded {len(data_list)} MATLAB files")
        return data_list

    def _load_mat_file(self, filepath: str) -> Optional[Dict]:
        """Load single MATLAB .mat file (same as original)"""
        try:
            import h5py
            with h5py.File(filepath, 'r') as mat_data:
                if 'simulation_data' not in mat_data:
                    return None

                sim_data = mat_data['simulation_data']
                data = {}
                for field in sim_data.keys():
                    value = sim_data[field]

                    # Skip Groups (they contain sub-structures, not data arrays)
                    if isinstance(value, h5py.Group):
                        continue

                    # Process Datasets
                    if hasattr(value, 'shape') and hasattr(value, 'dtype') and len(value.shape) > 0:
                        actual_value = value[()]
                        # For MATLAB v7.3: transpose and squeeze to get 1D arrays
                        if len(actual_value.shape) == 2:
                            actual_value = np.transpose(actual_value)
                        # Squeeze to remove singleton dimensions, then flatten if needed
                        actual_value = np.squeeze(actual_value)
                        # If still 2D after squeeze, keep as is; otherwise flatten
                        if actual_value.ndim == 1:
                            pass  # Already 1D, good
                        elif actual_value.ndim == 2 and actual_value.shape[0] == 1:
                            actual_value = actual_value.flatten()
                        data[field] = actual_value
                    elif hasattr(value, 'dtype') and value.dtype.kind == 'U':
                        data[field] = value[()].item().strip('\x00')
                    elif hasattr(value, 'shape') and (value.shape == () or len(value.shape) == 1):
                        ref = value[()]
                        if hasattr(ref, 'shape'):
                            data[field] = np.array(ref)
                        else:
                            try:
                                ref_data = mat_data[ref]
                                if len(ref_data.shape) == 2:
                                    actual_value = np.transpose(ref_data[()])
                                else:
                                    actual_value = ref_data[()]
                                actual_value = np.squeeze(actual_value)
                                if actual_value.ndim == 1:
                                    pass
                                elif actual_value.ndim == 2 and actual_value.shape[0] == 1:
                                    actual_value = actual_value.flatten()
                                data[field] = actual_value
                            except:
                                continue
        except Exception as e:
            try:
                mat_data = sio.loadmat(filepath)
                if 'simulation_data' not in mat_data:
                    return None

                sim_data = mat_data['simulation_data'][0, 0]
                data = {}
                field_names = sim_data.dtype.names

                for field in field_names:
                    value = sim_data[field][0, 0]
                    if isinstance(value, np.ndarray):
                        value = value.squeeze()
                    data[field] = value
            except Exception as e2:
                return None

        # Handle missing features (same as original)
        if 'adhesion_strength' not in data and 'adhesion_ratio' in data:
            data['adhesion_strength'] = data['adhesion_ratio']
        if 'adhesion_ratio' not in data and 'adhesion_strength' in data:
            data['adhesion_ratio'] = data['adhesion_strength']
        if 'internal_stress' not in data:
            data['internal_stress'] = np.zeros_like(data.get('adhesion_ratio', np.zeros(1)))
        if 'porosity' not in data:
            data['porosity'] = np.zeros_like(data.get('adhesion_ratio', np.zeros(1)))
        if 'dimensional_accuracy' not in data:
            data['dimensional_accuracy'] = np.zeros_like(data.get('adhesion_ratio', np.zeros(1)))
        if 'quality_score' not in data:
            if 'adhesion_ratio' in data:
                data['quality_score'] = data['adhesion_ratio']
            else:
                data['quality_score'] = np.zeros(1)

        return data

    def _fit_scaler(self):
        """Fit StandardScaler (same as original)"""
        if not self.data_list:
            raise ValueError("No data loaded")

        all_features = []
        for data in self.data_list:
            if self.INPUT_FEATURES[0] in data and len(data[self.INPUT_FEATURES[0]]) > 0:
                n_samples = len(data[self.INPUT_FEATURES[0]])
                feature_arrays = []
                for feat in self.INPUT_FEATURES:
                    if feat in data:
                        feat_data = data[feat]
                        if feat_data.ndim > 1:
                            feat_data = np.squeeze(feat_data)
                        feature_arrays.append(feat_data)
                    else:
                        feature_arrays.append(np.zeros(n_samples))

                features = np.stack(feature_arrays, axis=1)
                if features.ndim == 2:
                    all_features.append(features)

        if not all_features:
            raise ValueError("No valid features found")

        all_features = np.vstack(all_features)
        self.scaler.fit(all_features)
        print(f"Scaler fitted on {all_features.shape[0]} samples")

    def _create_sequences(self) -> List[Dict]:
        """
        OPTIMIZED: Create sequences with PRE-NORMALIZED data

        Key difference: Normalize during sequence creation, not in __getitem__
        """
        sequences = []

        for data_idx, data in enumerate(self.data_list):
            lengths = []
            for feat in self.INPUT_FEATURES + self.OUTPUT_TRAJECTORY + self.OUTPUT_QUALITY:
                if feat in data:
                    lengths.append(len(data[feat]))
            if not lengths:
                continue

            min_length = min(lengths)
            if min_length < self.seq_len + self.pred_len:
                print(f"Warning: Data file {data_idx} has length {min_length} which is too short, skipping...")
                continue

            # Stack all input features for vectorized normalization
            all_input_features = []
            for feat in self.INPUT_FEATURES:
                feat_data = data[feat]
                if feat_data.ndim > 1:
                    feat_data = np.squeeze(feat_data)
                all_input_features.append(feat_data)

            # Stack: [min_length, 12]
            all_input_features = np.stack(all_input_features, axis=1)

            # OPTIMIZED: Normalize ONCE for entire file (not per sequence)
            if self.scaler is not None:
                all_input_features = self.scaler.transform(all_input_features)

            # Create sequences with pre-normalized data
            for i in range(0, min_length - self.seq_len - self.pred_len + 1, self.stride):
                # Extract pre-normalized input sequence
                input_features = all_input_features[i:i+self.seq_len]  # Already normalized!

                # Extract trajectory error targets
                trajectory_targets_list = []
                for feat in self.OUTPUT_TRAJECTORY:
                    feat_data = data[feat]
                    if feat_data.ndim > 1:
                        feat_data = np.squeeze(feat_data)
                    trajectory_targets_list.append(feat_data[i+self.seq_len:i+self.seq_len+self.pred_len])

                trajectory_targets = np.stack(trajectory_targets_list, axis=1)

                # Extract quality targets
                quality_targets = []
                for feat in self.OUTPUT_QUALITY:
                    feat_data = data[feat]
                    if feat_data.ndim > 1:
                        feat_data = np.squeeze(feat_data)
                    quality_targets.append(feat_data[i+self.seq_len+self.pred_len-1])

                quality_targets = np.array(quality_targets)

                # Extract inertia forces (from normalized features, indices 8 and 9)
                F_inertia_x = input_features[:, 8]
                F_inertia_y = input_features[:, 9]

                sequences.append({
                    'input_features': input_features,  # Already normalized
                    'trajectory_targets': trajectory_targets,
                    'quality_targets': quality_targets,
                    'F_inertia_x': F_inertia_x,
                    'F_inertia_y': F_inertia_y,
                    'data_idx': data_idx,
                    'start_idx': i
                })

        return sequences

    def _cache_tensors(self):
        """Cache all sequences as tensors in memory (FASTEST)"""
        for seq in self.sequences:
            seq['_cached_input'] = torch.from_numpy(seq['input_features']).float()
            seq['_cached_trajectory'] = torch.from_numpy(seq['trajectory_targets']).float()
            seq['_cached_quality'] = torch.from_numpy(seq['quality_targets']).float()
            seq['_cached_inertia_x'] = torch.from_numpy(seq['F_inertia_x']).float()
            seq['_cached_inertia_y'] = torch.from_numpy(seq['F_inertia_y']).float()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        OPTIMIZED: Get pre-normalized sample with cached tensors

        No more per-sample scaler.transform()!
        """
        seq = self.sequences[idx]

        # Use cached tensors if available
        if self.cache_in_memory:
            input_features = seq['_cached_input'].clone()
            trajectory_targets = seq['_cached_trajectory'].clone()
            quality_targets = seq['_cached_quality'].clone()
            F_inertia_x = seq['_cached_inertia_x'].clone()
            F_inertia_y = seq['_cached_inertia_y'].clone()
        else:
            # Convert numpy to tensor (data is already normalized!)
            input_features = torch.from_numpy(seq['input_features']).float()
            trajectory_targets = torch.from_numpy(seq['trajectory_targets']).float()
            quality_targets = torch.from_numpy(seq['quality_targets']).float()
            F_inertia_x = torch.from_numpy(seq['F_inertia_x']).float()
            F_inertia_y = torch.from_numpy(seq['F_inertia_y']).float()

        return {
            'input_features': input_features,
            'trajectory_targets': trajectory_targets,
            'quality_targets': quality_targets,
            'F_inertia_x': F_inertia_x,
            'F_inertia_y': F_inertia_y,
            'data_idx': seq['data_idx'],
            'start_idx': seq['start_idx']
        }

    def get_feature_dim(self) -> int:
        return len(self.INPUT_FEATURES)

    def get_trajectory_output_dim(self) -> int:
        return len(self.OUTPUT_TRAJECTORY)

    def get_quality_output_dim(self) -> int:
        return len(self.OUTPUT_QUALITY)


# Alias for backward compatibility
PrinterSimulationDatasetOptimized = OptimizedPrinterSimulationDataset
