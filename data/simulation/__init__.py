"""
Physics-based 3D printing simulation package

This package implements realistic physics models for generating
synthetic sensor data and quality metrics.
"""

from .dataset import (
    PrinterSimulationDataset,
    create_dataloaders,
)

__all__ = [
    # Dataset loading
    'PrinterSimulationDataset',
    'create_dataloaders',
]
