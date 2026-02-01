# 项目重组说明

## 日期
2026-02-01

## 主要变更

### 1. 目录重组
- `matlab_simulation/` → `simulation/` (更简洁的命名)
- `data_simulation_*` → `data/simulation/*` (统一管理，移除前缀)

### 2. 删除的文件

#### 根目录临时脚本
- `analyze_data_sufficiency.py` - 数据充足性分析
- `analyze_performance.py` - 性能分析
- `diagnose_data.py` - 数据诊断
- `estimate_model_params.py` - 参数估算
- `quick_evaluate.py` - 快速评估
- `check_data_stats.py` - 数据统计
- `collect_*.m` - 数据收集脚本（已移至simulation/）

#### 未使用的模块目录
- `evaluation/` - 旧的评估模块（已被experiments/evaluate_realtime.py替代）
- `training/` - 旧的训练模块（已被experiments/train_realtime.py替代）
- `models/decoders/` - 未使用的解码器
- `models/encoders/` - 未使用的编码器
- `data/simulation/` - 旧的数据集（已被data/realtime_dataset.py替代）

#### 未使用的实验脚本
- `experiments/train_realtime_corrector.py` - 旧版训练脚本
- `experiments/diagnose_model_prediction.py` - 诊断脚本

#### 旧的配置文件
- `config/base_config.py` - 旧系统配置
- `config/model_config.py` - 旧模型配置

### 3. 更新的路径

所有数据路径从 `data_simulation_*` 更新为 `data/simulation/*`：

- `config/realtime_config.py`
- `experiments/train_realtime.py`
- `experiments/evaluate_realtime.py`
- `experiments/visualize_realtime.py`
- `experiments/visualize_trajectory_heatmap.py`
- `scripts/test_realtime.py`

### 4. 更新的模块导入

`config/__init__.py`:
- 移除: `from .base_config import BaseConfig`
- 移除: `from .model_config import ModelConfig, get_config`
- 添加: `from .realtime_config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG`

## 新的项目结构

```
3d_printer_pinn4ieee/
├── config/                    # 配置文件
│   ├── __init__.py
│   └── realtime_config.py
├── data/                      # 数据处理
│   ├── realtime_dataset.py
│   └── simulation/            # 仿真数据目录
│       ├── 3DBenchy_PLA_1h28m_sampled_48layers/
│       ├── bearing5_PLA_2h27m_sampled_15layers/
│       ├── Nautilus_Gears_Plate_PLA_3h36m_sampled_12layers/
│       └── simple_boat5_PLA_4h4m_sampled_74layers/
├── models/                    # 模型定义
│   ├── __init__.py
│   ├── base_model.py
│   └── realtime_corrector.py
├── experiments/               # 实验脚本
│   ├── train_realtime.py
│   ├── evaluate_realtime.py
│   ├── visualize_realtime.py
│   └── visualize_trajectory_heatmap.py
├── scripts/                   # 工具脚本
│   └── test_realtime.py
├── simulation/                # MATLAB仿真（原matlab_simulation/）
│   ├── physics_parameters.m
│   ├── run_full_simulation.m
│   ├── collect_*.m
│   ├── +planner/
│   ├── +stepper/
│   └── *.py（转换工具）
├── utils/                     # 工具函数
│   ├── __init__.py
│   ├── data_utils.py
│   ├── logger.py
│   └── physics_utils.py
├── checkpoints/
│   └── realtime_corrector/
├── results/
│   ├── figures/
│   ├── realtime_evaluation/
│   └── realtime_visualization/
└── docs/
```

## 统计

### 删除内容
- 文件: 15+ 个临时分析/诊断脚本
- 目录: 5个未使用的模块目录
- 代码行数: ~3000+ 行

### 重组内容
- 数据目录: 4个数据集整合到 `data/simulation/`
- MATLAB脚本: 重命名为 `simulation/`
- 数据文件: 149个.mat文件（~12GB）

## 回滚

如需回滚到重组前状态：

```bash
# 查看清理前标签
git tag -l

# 回滚到清理前状态
git checkout pre-cleanup-reorg-20260201
```

## 验证

重组成功后，以下命令应该正常工作：

```bash
# 测试配置
python -c "from config.realtime_config import DATA_CONFIG; print(DATA_CONFIG['data_pattern'])"
# 输出: data/simulation/*

# 测试数据加载
python -c "import glob; print(len(glob.glob('data/simulation/*/*.mat')))"
# 输出: 149

# 运行系统测试
python scripts/test_realtime.py
# 输出: ✓ 所有测试通过
```
