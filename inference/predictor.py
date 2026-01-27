"""
Unified predictor for real-time inference
"""

import torch
import numpy as np
from typing import Dict, Union, Optional
from pathlib import Path

from ..models import UnifiedPINNSeq3D
from ..config.base_config import BaseConfig


class UnifiedPredictor:
    """
    Real-time predictor for 3D printer quality and trajectory optimization

    Usage:
        predictor = UnifiedPredictor.load_from_checkpoint('checkpoints/best_model.pth')
        results = predictor.predict(sensor_data)
    """

    def __init__(self, model: UnifiedPINNSeq3D, config: BaseConfig, device: str = 'cpu'):
        """
        Initialize predictor

        Args:
            model: Trained model
            config: Configuration object
            device: Device to run inference on
        """
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: str = 'cpu') -> 'UnifiedPredictor':
        """
        Load predictor from checkpoint

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on

        Returns:
            UnifiedPredictor instance
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Recreate config
        config = BaseConfig()
        if 'model_config' in checkpoint:
            config_dict = checkpoint['model_config']
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Create model
        model = UnifiedPINNSeq3D(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        return cls(model, config, device)

    @torch.no_grad()
    def predict(self,
               sensor_data: Union[np.ndarray, torch.Tensor],
               return_dict: bool = True) -> Dict[str, np.ndarray]:
        """
        Make predictions on sensor data

        Args:
            sensor_data: Input sensor data
                        - Shape: [batch, seq_len, num_features] or [seq_len, num_features]
                        - Type: numpy array or torch tensor
            return_dict: Whether to return results as dictionary

        Returns:
            Dictionary with predictions:
                - quality: Quality metrics (RUL, temperature, vibration, quality_score)
                - fault: Fault classification (probs, predicted class)
                - trajectory: Trajectory corrections (dx, dy, dz)
        """
        # Convert to tensor if needed
        if isinstance(sensor_data, np.ndarray):
            sensor_data = torch.from_numpy(sensor_data).float()

        # Add batch dimension if needed
        if sensor_data.dim() == 2:
            sensor_data = sensor_data.unsqueeze(0)

        # Move to device
        sensor_data = sensor_data.to(self.device)

        # Forward pass
        outputs = self.model(sensor_data)

        # Convert to numpy and format results
        results = {
            'quality': {
                'rul': outputs['rul'].cpu().numpy(),
                'temperature': outputs['temperature'].cpu().numpy(),
                'vibration_x': outputs['vibration_x'].cpu().numpy(),
                'vibration_y': outputs['vibration_y'].cpu().numpy(),
                'quality_score': outputs['quality_score'].cpu().numpy(),
            },
            'fault': {
                'probabilities': outputs['fault_probs'].cpu().numpy(),
                'predicted_class': outputs['fault_pred'].cpu().numpy(),
            },
            'trajectory': {
                'dx': outputs['displacement_x'].cpu().numpy(),
                'dy': outputs['displacement_y'].cpu().numpy(),
                'dz': outputs['displacement_z'].cpu().numpy(),
            },
        }

        if not return_dict:
            return results

        return results

    def predict_quality_only(self, sensor_data: Union[np.ndarray, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Predict quality metrics only (faster)

        Args:
            sensor_data: Input sensor data

        Returns:
            Quality predictions dictionary
        """
        if isinstance(sensor_data, np.ndarray):
            sensor_data = torch.from_numpy(sensor_data).float()

        if sensor_data.dim() == 2:
            sensor_data = sensor_data.unsqueeze(0)

        sensor_data = sensor_data.to(self.device)

        with torch.no_grad():
            outputs = self.model.predict_quality(sensor_data)

        return {
            'rul': outputs['rul'].cpu().numpy(),
            'temperature': outputs['temperature'].cpu().numpy(),
            'vibration_x': outputs['vibration_x'].cpu().numpy(),
            'vibration_y': outputs['vibration_y'].cpu().numpy(),
            'quality_score': outputs['quality_score'].cpu().numpy(),
        }

    def predict_fault_only(self, sensor_data: Union[np.ndarray, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Predict fault classification only (faster)

        Args:
            sensor_data: Input sensor data

        Returns:
            Fault predictions dictionary
        """
        if isinstance(sensor_data, np.ndarray):
            sensor_data = torch.from_numpy(sensor_data).float()

        if sensor_data.dim() == 2:
            sensor_data = sensor_data.unsqueeze(0)

        sensor_data = sensor_data.to(self.device)

        with torch.no_grad():
            outputs = self.model.predict_fault(sensor_data)

        return {
            'probabilities': outputs['fault_probs'].cpu().numpy(),
            'predicted_class': outputs['fault_pred'].cpu().numpy(),
        }

    def predict_trajectory_only(self, sensor_data: Union[np.ndarray, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Predict trajectory corrections only (faster)

        Args:
            sensor_data: Input sensor data

        Returns:
            Trajectory corrections dictionary
        """
        if isinstance(sensor_data, np.ndarray):
            sensor_data = torch.from_numpy(sensor_data).float()

        if sensor_data.dim() == 2:
            sensor_data = sensor_data.unsqueeze(0)

        sensor_data = sensor_data.to(self.device)

        with torch.no_grad():
            outputs = self.model.predict_trajectory(sensor_data)

        return {
            'dx': outputs['displacement_x'].cpu().numpy(),
            'dy': outputs['displacement_y'].cpu().numpy(),
            'dz': outputs['displacement_z'].cpu().numpy(),
        }

    def should_stop_printing(self, sensor_data: Union[np.ndarray, torch.Tensor],
                           threshold: float = 0.5) -> bool:
        """
        Decide whether to stop printing based on quality prediction

        Args:
            sensor_data: Input sensor data
            threshold: Quality score threshold (below this, stop printing)

        Returns:
            True if printing should stop, False otherwise
        """
        quality = self.predict_quality_only(sensor_data)
        quality_score = quality['quality_score'].item()

        return quality_score < threshold

    def get_trajectory_correction(self, sensor_data: Union[np.ndarray, torch.Tensor],
                                 position: np.ndarray) -> np.ndarray:
        """
        Get corrected trajectory position

        Args:
            sensor_data: Input sensor data
            position: Current [x, y, z] position

        Returns:
            Corrected [x, y, z] position
        """
        corrections = self.predict_trajectory_only(sensor_data)
        dx = corrections['dx'].item()
        dy = corrections['dy'].item()
        dz = corrections['dz'].item()

        corrected_position = position + np.array([dx, dy, dz])
        return corrected_position
