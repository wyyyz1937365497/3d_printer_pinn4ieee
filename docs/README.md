# Documentation Index

**Purpose**: Central navigation hub for all project documentation related to paper writing and system understanding.

---

## Quick Navigation for Paper Writing ⭐

### Essential Resources
- **[Formulas & Symbols Library](theory/formulas.md)** ⭐ - Complete physics equations and LaTeX code
- **[Paper Structure Template](writing/structure_template.md)** ⭐ - IEEE paper template with section outlines
- **[Figure Templates](figures/templates/trajectory_plots.py)** ⭐ - Python scripts for publication-quality figures
- **[Bibliography](writing/latex/bibliography.bib)** ⭐ - Reference papers in BibTeX format

### Quick Start
1. Start with **[Structure Template](writing/structure_template.md)** for paper outline
2. Find formulas in **[Formula Library](theory/formulas.md)** for equations
3. Use **[Figure Templates](figures/templates/)** for consistent visualizations
4. Reference papers from **[Bibliography](writing/latex/bibliography.bib)**

---

## Documentation by Type

### 📘 Theory (Physics & Mathematical Models)

#### Core Documents
- **[Formulas Library](theory/formulas.md)** ⭐ - All equations, symbols, and LaTeX code
- [Overview](theory/overview.md) - Research background and theoretical framework
- [Trajectory Dynamics](theory/trajectory_dynamics.md) - Second-order system modeling

**Purpose**: Understand the physics behind FDM printing errors and the mathematical models used in simulation.

---

### 📗 Methods (Implementation Details)

#### Core Documents
- [Simulation System](methods/simulation_system.md) - MATLAB simulation architecture
- [Firmware Effects](methods/firmware_effects.md) - Marlin firmware error sources
- [Data Generation](methods/data_generation.md) - Training data generation strategy
- [Neural Network](methods/neural_network.md) - Model architecture and design
- [Training Pipeline](methods/training_pipeline.md) - End-to-end training workflow

**Purpose**: Learn how the system works and how to implement/use it.

---

### 📙 Experiments (Experimental Setup)

#### Core Documents
- [Setup](experiments/setup.md) - Printer configuration and test models
- [Datasets](experiments/datasets.md) - Data collection and statistics
- [Metrics](experiments/metrics.md) - Evaluation metrics and benchmarks
- [Ablation Studies](experiments/ablation_studies.md) - Component analysis

**Purpose**: Understand the experimental design and how to evaluate performance.

---

### 📕 Results (Performance Analysis)

#### Core Documents
- [Trajectory Error](results/trajectory_error.md) - Error statistics and analysis
- [Quality Prediction](results/quality_prediction.md) - Print quality metrics
- [Correction Performance](results/correction_performance.md) - Correction effectiveness
- [Comparisons](results/comparisons.md) - Comparison with other methods

**Purpose**: Review experimental results and validate the approach.

---

### 🎨 Figures (Visualization Resources)

#### Templates
- [trajectory_plots.py](figures/templates/trajectory_plots.py) - 2D trajectory, time series, distribution plots
- [heatmaps.py](figures/templates/heatmaps.py) - Error heat map visualization
- [comparison_charts.py](figures/templates/comparison_charts.py) - Before/after comparison plots

#### Examples
- [examples/](figures/examples/) - Sample figures generated from templates

**Usage**:
```bash
cd docs/figures/templates
python trajectory_plots.py --test  # Generate test plots
```

---

### ✍️ Writing Aids (Paper Writing Resources)

#### Core Documents
- **[Structure Template](writing/structure_template.md)** ⭐ - Complete paper outline
- [Section Templates](writing/section_templates/) - Individual chapter templates
  - [introduction.md](writing/section_templates/introduction.md)
  - [related_work.md](writing/section_templates/related_work.md)
  - [methodology.md](writing/section_templates/methodology.md)
  - [experiments.md](writing/section_templates/experiments.md)
  - [conclusions.md](writing/section_templates/conclusions.md)

#### LaTeX Resources
- [bibliography.bib](writing/latex/bibliography.bib) - References in BibTeX format
- [commands.tex](writing/latex/commands.tex) - Custom LaTeX commands for physics symbols
- [preamble.tex](writing/latex/preamble.tex) - Standard LaTeX preamble

#### Phrase Banks
- [introduction.txt](writing/phrase_bank/introduction.txt) - Opening sentences and transitions
- [methodology.txt](writing/phrase_bank/methodology.txt) - Methods section phrases
- [results.txt](writing/phrase_bank/results.txt) - Results reporting phrases

---

### 📦 Archives (Historical Documents)

#### Quick Reference
- [QUICK_START_ENHANCED.md](archives/quick_ref/QUICK_START_ENHANCED.md) - 5-minute setup guide
- [快速开始.md](archives/quick_ref/快速开始.md) - Chinese quick start
- [THESIS_WRITING_QUICK_REF.md](archives/quick_ref/THESIS_WRITING_QUICK_REF.md) - Thesis writing tips

#### Chinese Notes
- [论文方法论_轨迹误差仿真.md](archives/chinese_notes/论文方法论_轨迹误差仿真.md) - Methodology (中文)
- [固件增强仿真系统_技术文档.md](archives/chinese_notes/固件增强仿真系统_技术文档.md) - Firmware effects (中文)
- [数据收集与训练指南.md](archives/chinese_notes/数据收集与训练指南.md) - Data guide (中文)

#### Guides (Temporary Documentation)
- **[README](archives/guides/README.md)** - Index of temporary guides
- [README_REALTIME.md](archives/guides/README_REALTIME.md) - Detailed real-time system documentation
- [DATA_REGENERATION_GUIDE.md](archives/guides/DATA_REGENERATION_GUIDE.md) - Data regeneration with updated physics parameters
- [PARALLEL_INSTRUCTIONS.md](archives/guides/PARALLEL_INSTRUCTIONS.md) - Parallel data collection instructions
- [PERFORMANCE_OPTIMIZATION.md](archives/guides/PERFORMANCE_OPTIMIZATION.md) - Performance optimization guide
- [PHYSICS_IMPROVEMENTS.md](archives/guides/PHYSICS_IMPROVEMENTS.md) - Physics constraint improvements

#### History (Project Evolution)
- **[README](archives/history/README.md)** - Index of historical documentation
- [REBUILD_SUMMARY.md](archives/history/REBUILD_SUMMARY.md) - Real-time correction system rebuild summary
- [REORGANIZATION_NOTES.md](archives/history/REORGANIZATION_NOTES.md) - Project reorganization notes

#### Old Documents
- [old_docs.md](archives/old_docs.md) - Index of deprecated documentation (including archived thermal/adhesion models)

---

## Document Statistics

- **Theory**: 3 documents (overview, dynamics, formulas)
- **Methods**: 5 documents
- **Experiments**: 4 documents
- **Results**: 4 documents
- **Figures**: 3 template scripts + examples
- **Writing**: Complete template system (structure, sections, LaTeX, phrases)
- **Archives**:
  - Quick reference: 3 documents
  - Chinese notes: 3 documents
  - Guides: 4 documents (data regeneration, parallel collection, performance, physics)
  - History: 2 documents (rebuild summary, reorganization notes)
  - Old docs: Index of deprecated documentation
- **Total**: 25+ core documents + 12+ archived documents

---

## Usage Scenarios

### 🎓 Writing a Paper

1. **Start**: Review [Structure Template](writing/structure_template.md) for outline
2. **Theory**: Extract formulas from [Formula Library](theory/formulas.md)
3. **Methods**: Adapt from [Section Templates](writing/section_templates/)
4. **Figures**: Generate using [Plot Templates](figures/templates/)
5. **References**: Import from [Bibliography](writing/latex/bibliography.bib)
6. **Polish**: Use [Phrase Bank](writing/phrase_bank/) for language

### 🔬 Understanding the System

1. **Big Picture**: Read [Theory Overview](theory/overview.md)
2. **Simulation**: Study [Simulation System](methods/simulation_system.md)
3. **Data**: Review [Data Generation](methods/data_generation.md)
4. **Training**: Check [Training Pipeline](methods/training_pipeline.md)

### 📊 Analyzing Results

1. **Metrics**: Review [Evaluation Metrics](experiments/metrics.md)
2. **Results**: Study [Trajectory Error Results](results/trajectory_error.md)
3. **Figures**: Generate with [Plot Templates](figures/templates/)
4. **Comparisons**: Check [vs Other Methods](results/comparisons.md)

### 🚀 Quick System Setup

1. **Setup**: [Quick Start Guide](archives/quick_ref/QUICK_START_ENHANCED.md)
2. **Simulation**: [Simulation Guide](methods/simulation_system.md)
3. **Training**: [Training Pipeline](methods/training_pipeline.md)
4. **Reference**: [Formula Library](theory/formulas.md)

---

## Legend

- ⭐ = Highly recommended for paper writing
- 📘 = Theory and background
- 📗 = Implementation and methods
- 📙 = Experiments and setup
- 📕 = Results and analysis
- 🎨 = Figures and visualization
- ✍️ = Writing aids

---

## Search Tips

### Finding Specific Information

**For equations**: Start in [theory/formulas.md](theory/formulas.md)
- Search for "dynamics", "firmware", "LSTM"
- Use equation anchors: {#eq:second-order}

**For implementation**: Check methods/ directory
- [Simulation System](methods/simulation_system.md) for MATLAB
- [Neural Network](methods/neural_network.md) for model

**For experimental details**: Go to experiments/ directory
- [Setup](experiments/setup.md) for configuration
- [Metrics](experiments/metrics.md) for evaluation

**For results**: See results/ directory
- [Trajectory Error](results/trajectory_error.md) for statistics
- [Correction Performance](results/correction_performance.md) for effectiveness

---

## Cross-Reference System

All documents include related document links at the bottom:

```markdown
---
## Related Documents
- **Next**: [Next Document](path/to/next.md)
- **Previous**: [Previous Document](path/to/prev.md)
- **See Also**: [Related Document](path/to/related.md)
- **Formulas**: [Formula Library](theory/formulas.md#section)
```

---

## Document Maintenance

**Last Updated**: 2026-02-01
**Version**: 1.0
**Maintainer**: 3D Printer PINN Project Team

### Adding New Documents

1. Place in appropriate category directory (theory/, methods/, etc.)
2. Add cross-references to related documents
3. Update this index with link and description
4. Run LaTeX compilation if adding .bib entries

### Updating Existing Documents

1. Check cross-references and update as needed
2. Maintain consistent formatting and style
3. Update "Last Updated" date
4. Increment version number for major changes

---

**For questions or suggestions**, please refer to the project README or contact the maintainers.
