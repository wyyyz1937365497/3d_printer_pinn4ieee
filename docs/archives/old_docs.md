# Archived Documentation

**Purpose**: Index of legacy documentation kept for reference.

---

## Overview

This directory contains documentation that has been superseded by the reorganized documentation structure. These files are kept for historical reference but should not be used for active development or paper writing.

---

## Archives Structure

```
archives/
├── old_docs.md           # This file
├── chinese_notes/        # Chinese documentation (中文文档)
└── quick_ref/            # Quick reference guides
```

---

## Chinese Notes (中文笔记)

### 固件增强仿真系统_技术文档.md

**Content**: Complete technical documentation of the firmware-enhanced simulation system in Chinese.

**Why archived**: Content has been extracted and reorganized into:
- `../theory/trajectory_dynamics.md` - Dynamics model
- `../theory/thermal_model.md` - Thermal model
- `../methods/simulation_system.md` - System architecture
- `../methods/firmware_effects.md` - Firmware effects

**When to reference**: If you need to understand the original Chinese documentation or compare with the new structure.

---

### 快速开始.md

**Content**: Quick start guide for the project in Chinese.

**Why archived**: Superseded by `../README.md` and reorganized experiments documentation.

**See instead**: `../experiments/setup.md`, `../methods/data_generation.md`

---

### 数据收集与训练指南.md

**Content**: Guide for data collection and model training in Chinese.

**Why archived**: Content has been reorganized into:
- `../methods/data_generation.md` - Data generation strategy
- `../methods/training_pipeline.md` - Training workflow
- `../experiments/datasets.md` - Dataset documentation

**See instead**: The new methods and experiments documentation.

---

### 论文方法论_轨迹误差仿真.md

**Content**: Thesis methodology for trajectory error simulation in Chinese.

**Why archived**: Comprehensive content has been reorganized into:
- `../theory/` - Theoretical foundations
- `../methods/` - Implementation details
- `../results/` - Results and analysis

**See instead**: The new documentation structure for complete coverage.

---

## Quick Reference (速查文档)

### QUICK_START_ENHANCED.md

**Content**: Enhanced quick start guide.

**Why archived**: Outdated quick start guide.

**See instead**: `../README.md` for current project overview.

---

### THESIS_WRITING_QUICK_REF.md

**Content**: Quick reference for thesis writing.

**Why archived**: Superseded by:
- `../writing/structure_template.md` - Paper structure
- `../theory/formulas.md` - Formula library
- `../writing/latex/` - LaTeX resources

**See instead**: New writing resources with more comprehensive content.

---

### THESIS_DOCUMENTATION.md

**Content**: Complete thesis documentation including literature review and system design.

**Why archived**: Content has been extensively reorganized:
- Literature review → scattered across theory/ and methods/
- System design → `../methods/simulation_system.md`
- Physics models → `../theory/*.md`

**Note**: This was a 32KB comprehensive document. Most content is now in the new structure.

**See instead**: Individual theory and methods documents for focused content.

---

### TECHNICAL_DOCUMENTATION.md

**Content**: Technical documentation with physics models and algorithms (Chinese).

**Why archived**: Similar to 固件增强仿真系统_技术文档.md, content has been reorganized.

**See instead**: `../theory/trajectory_dynamics.md`, `../theory/thermal_model.md`

---

### DATA_GUIDE.md

**Content**: Guide to simulation data format and structure.

**Why archived**: Superseded by `../experiments/datasets.md`.

**See instead**: `../experiments/datasets.md` for current data format documentation.

---

### SIMULATION_DATA_GUIDE.md

**Content**: Guide to simulation data generation.

**Why archived**: Content integrated into:
- `../methods/data_generation.md` - Data generation strategy
- `../methods/simulation_system.md` - Simulation system
- `../experiments/datasets.md` - Dataset documentation

**See instead**: The new methods and experiments documentation.

---

## Other Archives

### DIVERSITY_RECOMMENDATIONS.md

**Content**: Recommendations for ensuring data diversity in training.

**Why archived**: Implicit state inference related content. Not needed for trajectory correction focus.

**Note**: These recommendations may still be useful if you later work on multi-task learning.

---

### evaluation_script_fix.md

**Content**: Notes on fixing evaluation scripts.

**Why archived**: Historical troubleshooting notes, no longer relevant.

**Note**: May contain useful debugging insights for evaluation issues.

---

### physics_constraint_fix.md

**Content**: Notes on fixing physics constraints in the model.

**Why archived**: Historical bug fix documentation.

**Note**: Useful for understanding how physics constraints were debugged and validated.

---

## Migration Notes

### What Was Improved

The new documentation structure improves upon the archived documents by:

1. **Better Organization**:
   - Separated by type (theory, methods, experiments, results)
   - Easier to find specific information
   - Clear cross-references between documents

2. **More Detail**:
   - Complete mathematical derivations
   - Algorithm pseudocode
   - Implementation examples
   - Validation against literature

3. **Paper-Focused**:
   - LaTeX code examples
   - IEEE formatting guidance
   - Reference management
   - Writing templates

4. **English & Chinese**:
   - English for formal papers
   - Chinese preserved in archives for reference
   - Bilingual approach maintained

### What to Do If You Need Archived Content

1. **For Reference**: Browse the archives to understand the original structure
2. **For Content**: Most archived content has been reorganized - check the new structure first
3. **For Comparison**: Archives help track how documentation evolved

---

## Recommendations

### When to Use Archives

✅ **Use archives when**:
- Understanding the original project structure
- Looking for historical context
- Comparing old vs new documentation
- Needing Chinese language explanations
- Debugging based on historical notes

❌ **Don't use archives for**:
- Active development (use new structure)
- Paper writing (use writing/ resources)
- Learning the system (use README.md)
- Implementation (use methods/ and experiments/)

---

## Archive Maintenance

### Adding to Archives

When a document becomes obsolete:
1. Move it to the appropriate archive subdirectory
2. Update this index with the reason for archiving
3. Add a "See instead" reference to the replacement
4. Note any unique value in the archived document

### Cleaning Archives

Consider deleting archived documents if:
- The content is completely duplicated in new structure
- No historical value remains
- Six months have passed without any references

---

## Related Documentation

**Active Documentation**:
- [Main README](../README.md) - Project overview
- [Theory](../theory/) - Theoretical foundations
- [Methods](../methods/) - Implementation details
- [Experiments](../experiments/) - Experimental setup
- [Results](../results/) - Results and analysis
- [Writing](../writing/) - Paper writing resources

---

**Last Updated**: 2026-02-02
**Archive Version**: 1.0
