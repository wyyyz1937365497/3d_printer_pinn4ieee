"""
Base model class for all models in the framework
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseModel(nn.Module, ABC):
    """
    Base class for all models

    Provides common interface and utilities
    """

    def __init__(self, config: Any):
        """
        Initialize base model

        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass - must be implemented by subclasses

        Returns:
            Model outputs
        """
        pass

    def get_num_params(self) -> int:
        """
        Get total number of parameters

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """
        Get number of trainable parameters

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self):
        """Freeze all model parameters"""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all model parameters"""
        for param in self.parameters():
            param.requires_grad = True

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None,
                       epoch: Optional[int] = None, loss: Optional[float] = None):
        """
        Save model checkpoint

        Args:
            path: Path to save checkpoint
            optimizer: Optimizer state (optional)
            epoch: Current epoch (optional)
            loss: Current loss (optional)
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config,
            'num_params': self.get_num_params(),
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None,
                       device: str = 'cpu') -> Dict[str, Any]:
        """
        Load model checkpoint

        Args:
            path: Path to checkpoint
            optimizer: Optimizer to load state into (optional)
            device: Device to load model on

        Returns:
            Checkpoint dictionary with additional info
        """
        checkpoint = torch.load(path, map_location=device)

        self.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint

    @classmethod
    def from_checkpoint(cls, path: str, config: Any, device: str = 'cpu'):
        """
        Create model from checkpoint

        Args:
            path: Path to checkpoint
            config: Configuration object
            device: Device to load model on

        Returns:
            Loaded model instance
        """
        model = cls(config)
        model.load_checkpoint(path, device=device)
        return model

    def to_device(self, device: str):
        """
        Move model to device

        Args:
            device: Device to move to ('cpu' or 'cuda')

        Returns:
            Self (for method chaining)
        """
        self.to(device)
        return self
