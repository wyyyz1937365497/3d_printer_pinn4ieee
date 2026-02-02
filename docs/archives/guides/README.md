# Temporary Guides Archive

**Purpose**: Collection of temporary documentation and guides created during project development.

---

## Overview

This directory contains guides and documentation that were created for specific purposes during the project's evolution. While these documents are no longer actively maintained, they may contain useful information for understanding the development process or for reference purposes.

---

## Available Guides

### 1. DATA_REGENERATION_GUIDE.md
**Purpose**: Guide for regenerating simulation data with updated physics parameters

**Key Content**:
- Updated physics parameters (stiffness, mass, damping)
- Expected error ranges (±50-100 μm)
- Paper narrative logic for error correction

**Use When**:
- Need to regenerate training data
- Understanding parameter impact on error magnitude

---

### 2. PARALLEL_INSTRUCTIONS.md
**Purpose**: Instructions for parallel data collection across multiple test models

**Key Content**:
- Current progress status (3DBenchy, bearing5, Nautilus, Boat)
- Three parallel collection scripts
- Step-by-step execution guide

**Use When**:
- Running large-scale data generation
- Optimizing data collection time

---

### 3. PERFORMANCE_OPTIMIZATION.md
**Purpose**: Performance optimization guide for training pipeline

**Key Content**:
- Identified bottlenecks (data preprocessing, GPU utilization, DataLoader)
- Solutions for each bottleneck
- Expected performance improvements

**Use When**:
- Training is slower than expected
- Need to optimize training throughput

---

### 4. PHYSICS_IMPROVEMENTS.md
**Purpose**: Summary of physics constraint improvements based on Ender-3 V2 parameters

**Key Content**:
- Updated `PhysicsConfig` in `config/base_config.py`
- Ender-3 V2 actual parameters
- Comparison with original parameters

**Use When**:
- Understanding physics parameter choices
- Validating parameter sources

---

### 5. README_REALTIME.md
**Purpose**: Detailed technical documentation for real-time trajectory correction system

**Key Content**:
- Complete system architecture
- File structure
- Performance metrics
- Design decisions
- Configuration parameters
- Git commit guidelines

**Use When**:
- Need detailed implementation reference
- Understanding system design rationale

---

## Historical Context

These guides were created during the project's transition from multi-physics modeling (trajectory + thermal + adhesion) to focused real-time trajectory correction. They document:
- Parameter tuning experiments
- Performance optimization efforts
- System rebuild decisions
- Data generation strategies

---

## Maintenance Status

**Status**: Not actively maintained

These documents are preserved for historical reference but may contain outdated information. For current system documentation, please refer to:
- [Main Project README](../../README.md)
- [docs/README.md](../../README.md)
- Specific method documentation in [docs/methods/](../../methods/)

---

## Related Documentation

- **Project History**: See [../history/](../history/) for rebuild and reorganization records
- **Quick Reference**: See [../quick_ref/](../quick_ref/) for setup guides
- **Chinese Notes**: See [../chinese_notes/](../chinese_notes/) for Chinese documentation

---

**Last Updated**: 2026-02-02
