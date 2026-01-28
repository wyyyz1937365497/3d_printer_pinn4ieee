"""
Physics-based 3D printing simulation package

This package implements realistic physics models for generating
synthetic sensor data and quality metrics.
"""

from .dataset import (
    PrinterSimulationDataset,
    create_dataloaders,
)

from .thermal_model import (
    SimulationParameters,
    ThermalSimulationModel,
    VibrationSimulationModel,
    MotorCurrentSimulationModel,
    TrajectorySimulationModel,
)

from .quality_formation_model import (
    InterlayerAdhesionModel,
    InternalStressModel,
    PorosityFormationModel,
    DimensionalAccuracyModel,
    QualityScoreModel,
)

from .simulation_pipeline import (
    PrintConfiguration,
    CompletePrintSimulation,
    SimulationDatasetGenerator,
)

__all__ = [
    # Dataset loading
    'PrinterSimulationDataset',
    'create_dataloaders',

    # Parameters and basic models
    'SimulationParameters',
    'ThermalSimulationModel',
    'VibrationSimulationModel',
    'MotorCurrentSimulationModel',
    'TrajectorySimulationModel',

    # Quality models
    'InterlayerAdhesionModel',
    'InternalStressModel',
    'PorosityFormationModel',
    'DimensionalAccuracyModel',
    'QualityScoreModel',

    # Pipeline
    'PrintConfiguration',
    'CompletePrintSimulation',
    'SimulationDatasetGenerator',
]
