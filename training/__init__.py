"""
Training module for 3D Printer PINN-Seq3D Framework
"""

from .losses import MultiTaskLoss
from .trainer import Trainer

__all__ = ['MultiTaskLoss', 'Trainer']
