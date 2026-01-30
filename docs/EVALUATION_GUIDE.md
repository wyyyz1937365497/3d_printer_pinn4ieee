# 模型评估指南

## 可用的评估脚本

项目提供了两种评估方式：

| 评估脚本 | 功能 | 输出 |
|----------|------|------|
| `evaluate_implicit_state_tcn.py` | 基础评估 | CSV指标、预测vs目标散点图、误差直方图 |
| `evaluate_implicit_state_tcn_comprehensive.py` | 综合评估 | JSON+HTML报告、多种可视化、详细指标 |

## 快速开始

### 方法1：基础评估（推荐）

评估优化训练的模型：

```bash
python experiments/evaluate_implicit_state_tcn.py \
  --model_path checkpoints/implicit_state_tcn_optimized/best_model.pth \
  --data_path "data_simulation_*/" \
  --batch_size 512 \
  --device cuda
```

**输出位置：**
- `results/implicit_state_tcn_metrics.csv` - 评估指标
- `results/figures/implicit_state_tcn_pred_vs_target.png` - 预测vs目标散点图
- `results/figures/implicit_state_tcn_error_hist.png` - 误差分布直方图

### 方法2：综合评估（更详细）

```bash
python experiments/evaluate_implicit_state_tcn_comprehensive.py \
  --model_path checkpoints/implicit_state_tcn_optimized/best_model.pth \
  --data_dir "data_simulation_*/" \
  --batch_size 512 \
  --save_dir evaluation_results \
  --device cuda
```

**输出位置：**
- `evaluation_results/metrics_report.json` - 完整指标报告
- `evaluation_results/evaluation_report.html` - HTML可视化报告
- `evaluation_results/figures/` - 所有可视化图表

## 评估指标说明

### 1. 基础指标

| 指标 | 说明 | 理想值 |
|------|------|--------|
| **MAE** (Mean Absolute Error) | 平均绝对误差 | 越小越好 |
| **RMSE** (Root Mean Squared Error) | 均方根误差 | 越小越好 |
| **R²** (R-squared) | 决定系数 | 接近1 |

### 2. 综合指标（仅在综合评估中）

| 指标 | 说明 | 理想值 |
|------|------|--------|
| **MAPE** | 平均绝对百分比误差 | <10% |
| **Max Error** | 最大误差 | 越小越好 |
| **Median Error** | 误差中位数 | 越小越好 |
| **Correlation** | 预测与目标的相关系数 | 接近1 |
| **Within Tolerance** | 在10%容差内的比例 | >90% |

### 3. 任务特定指标

模型预测5个质量指标：

1. **adhesion_strength** (粘合力强度) - [0, 1] ratio
2. **internal_stress** (内应力) - MPa
3. **porosity** (孔隙率) - %
4. **dimensional_accuracy** (尺寸精度) - mm
5. **quality_score** (质量分数) - [0, 1]

## 评估结果解读

### 预测vs目标散点图

理想情况下，点应该沿对角线（红色虚线）分布：
- 点越接近对角线，预测越准确
- 散布越大，模型误差越大

### 误差分布直方图

显示误差的分布情况：
- 中心化在0表示无偏估计
- 窄峰表示预测稳定
- 宽峰或多个峰值表示存在问题

### 典型性能基准

| 任务 | MAE (好) | MAE (一般) | MAE (差) |
|------|----------|-----------|---------|
| Adhesion | <0.05 | 0.05-0.15 | >0.15 |
| Stress | <2 MPa | 2-5 MPa | >5 MPa |
| Porosity | <2% | 2-5% | >5% |
| Accuracy | <0.1 mm | 0.1-0.3 mm | >0.3 mm |
| Quality | <0.05 | 0.05-0.15 | >0.15 |

## 高级用法

### 1. 评估单个数据集

```bash
python experiments/evaluate_implicit_state_tcn.py \
  --model_path checkpoints/implicit_state_tcn_optimized/best_model.pth \
  --data_path data_simulation_3DBenchy_PLA_1h28m_sampled_48layers/ \
  --batch_size 256
```

### 2. 使用CPU评估（如果没有GPU）

```bash
python experiments/evaluate_implicit_state_tcn.py \
  --model_path checkpoints/implicit_state_tcn_optimized/best_model.pth \
  --data_path "data_simulation_*/" \
  --device cpu
```

### 3. 比较不同模型

```bash
# 模型1
python experiments/evaluate_implicit_state_tcn.py \
  --model_path checkpoints/model1/best_model.pth \
  --data_path "data_simulation_*/"

# 模型2
python experiments/evaluate_implicit_state_tcn.py \
  --model_path checkpoints/model2/best_model.pth \
  --data_path "data_simulation_*/"

# 比较CSV结果
diff results/implicit_state_tcn_metrics.csv results/model2_metrics.csv
```

### 4. 生成HTML报告（综合评估）

```bash
python experiments/evaluate_implicit_state_tcn_comprehensive.py \
  --model_path checkpoints/implicit_state_tcn_optimized/best_model.pth \
  --data_dir "data_simulation_*/" \
  --generate_html

# 在浏览器中打开
firefox evaluation_results/evaluation_report.html
# 或
google-chrome evaluation_results/evaluation_report.html
```

## 查看评估结果

### 命令行查看指标

```bash
# 基础评估
cat results/implicit_state_tcn_metrics.csv

# 综合评估
cat evaluation_results/metrics_report.json | python -m json.tool
```

### 可视化图表

```bash
# Linux
xdg-open results/figures/implicit_state_tcn_pred_vs_target.png

# macOS
open results/figures/implicit_state_tcn_pred_vs_target.png

# Windows
start results/figures\implicit_state_tcn_pred_vs_target.png
```

## 常见问题

### Q1: 评估时内存不足

**解决方案：** 减小batch_size
```bash
--batch_size 128  # 或更小
```

### Q2: 评估速度慢

**解决方案：**
- 使用GPU：`--device cuda`
- 增大batch_size：`--batch_size 1024`
- 使用优化数据集（修改评估脚本使用`OptimizedPrinterSimulationDataset`）

### Q3: 找不到模型检查点

**解决方案：**
```bash
# 查找所有检查点
find checkpoints -name "*.pth"

# 使用绝对路径
--model_path /full/path/to/checkpoint/best_model.pth
```

### Q4: 中文显示问题

如果图表中中文显示为方框：

```bash
# 安装中文字体
sudo apt-get install fonts-wqy-microhei  # Linux
# 或在evaluate脚本中修改字体设置
```

## 性能优化建议

如果需要频繁评估，可以修改评估脚本使用优化数据集：

```python
# 在 evaluate_implicit_state_tcn.py 中
from data.simulation.dataset_optimized import OptimizedPrinterSimulationDataset

# 替换 PrinterSimulationDataset 为 OptimizedPrinterSimulationDataset
# 记得传递训练集的scaler
```

这样可以加速数据加载10倍以上。

## 自动化评估流程

创建评估脚本：

```bash
#!/bin/bash
# evaluate_all_models.sh

MODELS=(
    "checkpoints/implicit_state_tcn/best_model.pth"
    "checkpoints/implicit_state_tcn_enhanced/best_model.pth"
    "checkpoints/implicit_state_tcn_optimized/best_model.pth"
)

for model in "${MODELS[@]}"; do
    name=$(basename $(dirname $model))
    echo "Evaluating $name..."

    python experiments/evaluate_implicit_state_tcn_comprehensive.py \
        --model_path $model \
        --data_dir "data_simulation_*/" \
        --save_dir evaluation_results/$name \
        --batch_size 512

    echo "Done: $name"
done

echo "All evaluations complete!"
```

使用：
```bash
chmod +x evaluate_all_models.sh
./evaluate_all_models.sh
```
