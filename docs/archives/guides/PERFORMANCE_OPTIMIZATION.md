# 性能优化指南

## 已识别的性能瓶颈

根据训练日志分析（吞吐量从2052降到539 samples/s），主要问题包括：

### 1. **数据预处理瓶颈**（最关键）
- **问题**：`dataset.py:366` 中每个样本都调用 `scaler.transform()`
- **影响**：CPU密集型操作，每个batch都需要等待CPU完成数据标准化
- **解决方案**：使用优化版数据集，在初始化时一次性完成数据标准化

### 2. **GPU利用不充分**
- **问题**：没有使用 `torch.compile`
- **影响**：无法利用PyTorch 2.0+的JIT编译加速
- **解决方案**：启用 `torch.compile` 和 'reduce-overhead' 模式

### 3. **DataLoader配置不够优化**
- **问题**：`prefetch_factor=2` 太小
- **影响**：数据预取不足，GPU经常等待数据
- **解决方案**：增加 `prefetch_factor` 至 `max(2, num_workers)`

### 4. **CPU-GPU数据传输开销**
- **问题**：每个样本都需要numpy->torch转换并传输到GPU
- **影响**：CPU-GPU传输成为瓶颈
- **解决方案**：使用 `non_blocking=True` 和内存缓存

## 优化后的训练命令

### 基础优化（推荐）

```bash
python experiments/train_implicit_state_tcn_optimized.py \
  --data_dir "data_simulation_*" \
  --epochs 100 \
  --batch_size 512 \
  --lr 1e-3 \
  --lambda_physics 0.5 \
  --num_workers 8 \
  --prefetch_factor 8 \
  --use_amp
```

### 高性能优化（需要足够内存）

```bash
python experiments/train_implicit_state_tcn_optimized.py \
  --data_dir "data_simulation_*" \
  --epochs 100 \
  --batch_size 512 \
  --lr 1e-3 \
  --lambda_physics 0.5 \
  --num_workers 8 \
  --prefetch_factor 8 \
  --cache_data \
  --use_amp \
  --use_torch_compile
```

## 新增文件

### 1. 优化数据集：`data/simulation/dataset_optimized.py`

关键改进：
- **预先归一化**：在 `_create_sequences()` 时一次性完成标准化，而不是在 `__getitem__()` 中每次调用
- **内存缓存**：可选地将所有数据预加载为PyTorch张量并缓存在内存中
- **减少CPU-GPU同步**：使用 `non_blocking=True`

性能提升：**预计 2-5x** 数据加载速度

### 2. 性能分析工具：`analyze_performance.py`

使用方法：
```bash
# 分析标准数据集性能
python analyze_performance.py \
  --data_dir "data_simulation_*" \
  --batch_size 512 \
  --num_workers 8 \
  --num_iterations 100

# 分析优化数据集性能
python analyze_performance.py \
  --data_dir "data_simulation_*" \
  --batch_size 512 \
  --num_workers 8 \
  --use_optimized_dataset \
  --cache_data \
  --num_iterations 100
```

输出：
- 各阶段耗时（数据加载、CPU->GPU传输、前向传播、反向传播、优化器更新）
- 吞吐量估算
- 瓶颈识别
- 优化建议

### 3. 优化的训练脚本：`experiments/train_implicit_state_tcn_optimized.py`

新增参数：
- `--prefetch_factor`: DataLoader预取因子（默认：max(2, num_workers)）
- `--cache_data`: 将数据集缓存在内存中
- `--use_torch_compile`: 使用PyTorch 2.0+编译优化（默认：启用）

## 优化技术详解

### 1. 预归一化数据

**原版：**
```python
def __getitem__(self, idx):
    seq = self.sequences[idx]
    # 每个样本都要标准化 - CPU密集型
    if self.scaler is not None:
        input_features = self.scaler.transform(seq['input_features'])
    return {'input_features': torch.FloatTensor(input_features), ...}
```

**优化版：**
```python
def _create_sequences(self):
    # 在初始化时一次性完成所有数据的标准化
    all_input_features = self.scaler.transform(all_input_features)

    # 序列已经是标准化的
    sequences.append({
        'input_features': all_input_features[i:i+self.seq_len],
        ...
    })

def __getitem__(self, idx):
    # 直接返回，无需重复标准化
    seq = self.sequences[idx]
    return {'input_features': torch.from_numpy(seq['input_features']), ...}
```

### 2. torch.compile 优化

```python
# PyTorch 2.0+的JIT编译
model = torch.compile(model, mode='reduce-overhead')
```

**编译模式：**
- `reduce-overhead`：减少Python开销，适合小批量
- `max-autotune`：最佳性能，但编译时间长

**性能提升：** 20-40%

### 3. DataLoader 优化配置

```python
dataloader_kwargs = {
    'batch_size': batch_size,
    'num_workers': num_workers,           # 多进程加载
    'pin_memory': True,                    # CPU-GPU传输优化
    'persistent_workers': True,            # 保持worker进程
    'prefetch_factor': 8,                  # 预取8个batch
}
```

## 性能对比

| 配置 | 预计吞吐量 | GPU利用率 | CPU利用率 |
|------|-----------|----------|----------|
| 原版 | ~500 samples/s | ~30% | ~100% (单核) |
| 优化版（基础） | ~2000 samples/s | ~60% | ~60% (多核) |
| 优化版（内存缓存） | ~4000 samples/s | ~85% | ~40% (多核) |

## 其他建议

### 调整 Batch Size

如果GPU内存允许：
```bash
--batch_size 1024  # 更大的batch可以更好地利用GPU
```

### 调整 Workers

根据CPU核心数：
```bash
--num_workers 16  # 通常是CPU核心数
```

### 监控工具

使用 `nvidia-smi` 监控GPU：
```bash
watch -n 1 nvidia-smi
```

使用 `htop` 监控CPU：
```bash
htop
```

## 故障排除

### 内存不足

如果使用 `--cache_data` 出现内存不足：
- 移除 `--cache_data` 参数
- 减少 `--num_workers`
- 减少 `--prefetch_factor`

### torch.compile 失败

如果编译失败：
- 检查PyTorch版本：`python -c "import torch; print(torch.__version__)"`
- 需要 PyTorch 2.0+
- 使用 `--use_torch_compile=False` 禁用

### 数据加载仍然慢

1. 先运行性能分析器：
   ```bash
   python analyze_performance.py --data_dir "data_simulation_*"
   ```

2. 查看瓶颈报告并应用建议

3. 检查磁盘I/O（如果数据在HDD上，考虑移到SSD）
