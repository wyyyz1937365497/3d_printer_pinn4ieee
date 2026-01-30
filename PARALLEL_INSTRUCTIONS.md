# 并行数据收集说明

## 📊 当前状态

- ✅ 3DBenchy: 48/48层完成
- 🔄 bearing5: 45/75层（还剩30层，当前MATLAB正在运行）
- ⏳ Nautilus: 0/56层待处理
- ⏳ simple_boat5: 0/74层待处理

## ⚡ 并行方案

创建了3个独立脚本：

1. **collect_bearing5_remaining.m** - 完成bearing5剩余30层（层46-75）
2. **collect_nautilus_all.m** - 处理Nautilus全部56层
3. **collect_boat_all.m** - 处理simple_boat5采样74层

## 🚀 执行步骤

### 方案A: 当前bearing5完成后启动并行（推荐）

**步骤1**: 等待当前bearing5完成（约30分钟）

**步骤2**: 并行启动3个MATLAB实例

```bash
# 终端1: Nautilus (56层，预计28分钟)
matlab -batch "collect_nautilus_all" 2>&1 | tee nautilus_collection.log

# 终端2: simple_boat5 (74层，预计37分钟)
matlab -batch "collect_boat_all" 2>&1 | tee boat_collection.log

# 终端3: bearing5剩余层（如果还没完成）
matlab -batch "collect_bearing5_remaining" 2>&1 | tee bearing5_remaining.log
```

### 方案B: 立即启动（需要先停止当前MATLAB）

**步骤1**: 停止当前的MATLAB进程

**步骤2**: 并行启动

```bash
# 终端1
matlab -batch "collect_bearing5_remaining" 2>&1 | tee bearing5_remaining.log

# 终端2
matlab -batch "collect_nautilus_all" 2>&1 | tee nautilus_collection.log

# 终端3
matlab -batch "collect_boat_all" 2>&1 | tee boat_collection.log
```

## ⏱️ 时间估算

| 脚本 | 层数 | 预计时间 |
|------|------|----------|
| bearing5_remaining | 30层 | 15分钟 |
| nautilus_all | 56层 | 28分钟 |
| boat_all | 74层 | 37分钟 |

**串行总时间**: 80分钟
**3实例并行时间**: ~40分钟（节省50%）

## 📈 总进度

- 已完成: 93/253层 (36.8%)
- 剩余: 160层
- 并行处理时间: 约60-70分钟

## ⚠️ 注意事项

1. GPU会被3个实例共享，每个实例速度会略降
2. 确保有足够的磁盘空间（~500MB）
3. 每个脚本会自动创建输出目录
4. 脚本会跳过已完成的层

## ✅ 完成后验证

```bash
# 统计总文件数
find data_simulation_* -name "*.mat" | wc -l

# 验证数据加载
python -c "from data.simulation import PrinterSimulationDataset; import glob; files = glob.glob('data_simulation_*/*.mat'); print(f'找到 {len(files)} 个.mat文件'); ds = PrinterSimulationDataset(files, seq_len=200, pred_len=50, stride=5, mode='train', fit_scaler=True); print(f'训练样本: {len(ds)}')"
```

预期结果: ~253个.mat文件, ~66,000训练样本
