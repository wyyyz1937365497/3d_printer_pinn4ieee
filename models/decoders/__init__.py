"""
Decoder modules for the PINN-Seq3D framework
"""

from .quality_decoder import QualityPredictionHead
from .trajectory_decoder import TrajectoryCorrectionHead

__all__ = ['QualityPredictionHead', 'TrajectoryCorrectionHead']
