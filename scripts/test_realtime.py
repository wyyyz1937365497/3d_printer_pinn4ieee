"""
端到端测试脚本 - 验证实时轨迹修正系统

测试流程:
1. 测试数据加载
2. 测试模型推理
3. 测试训练循环 (1个epoch)
4. 验证性能指标
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from data.realtime_dataset import RealTimeTrajectoryDataset, load_mat_file, create_sequences
from models.realtime_corrector import RealTimeCorrector


def test_data_loading():
    """测试1: 数据加载"""
    print("\n" + "="*80)
    print("测试1: 数据加载")
    print("="*80)

    import glob
    data_pattern = "data_simulation_*"
    all_dirs = glob.glob(data_pattern)

    if not all_dirs:
        print("  ✗ 未找到数据目录")
        return False

    # 获取第一个.mat文件
    mat_files = []
    for d in all_dirs:
        mat_files.extend(glob.glob(os.path.join(d, "*.mat")))

    if not mat_files:
        print("  ✗ 未找到.mat文件")
        return False

    test_file = mat_files[0]
    print(f"  测试文件: {test_file}")

    # 加载数据
    features, labels = load_mat_file(test_file)
    print(f"  ✓ 特征形状: {features.shape}")
    print(f"  ✓ 标签形状: {labels.shape}")

    # 验证维度
    if features.shape[1] != 4:
        print(f"  ✗ 特征维度错误 (应为4, 实际{features.shape[1]})")
        return False

    if labels.shape[1] != 2:
        print(f"  ✗ 标签维度错误 (应为2, 实际{labels.shape[1]})")
        return False

    # 创建序列
    sequences, targets = create_sequences(features, labels, seq_len=20, pred_len=1)
    print(f"  ✓ 序列形状: {sequences.shape}")
    print(f"  ✓ 目标形状: {targets.shape}")

    # 测试数据集
    dataset = RealTimeTrajectoryDataset([test_file], seq_len=20, pred_len=1)
    sample = dataset[0]
    print(f"  ✓ 样本输入形状: {sample['input'].shape}")
    print(f"  ✓ 样本目标形状: {sample['target'].shape}")

    # 验证形状
    if sample['input'].shape != (20, 4):
        print(f"  ✗ 输入形状错误 (应为(20, 4), 实际{sample['input'].shape})")
        return False

    if sample['target'].shape != (1, 2):
        print(f"  ✗ 目标形状错误 (应为(1, 2), 实际{sample['target'].shape})")
        return False

    print("  ✓ 数据加载测试通过")
    return True


def test_model_inference():
    """测试2: 模型推理"""
    print("\n" + "="*80)
    print("测试2: 模型推理")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")

    # 创建模型
    model = RealTimeCorrector(input_size=4, hidden_size=56, num_layers=2, dropout=0.1)
    model.to(device)
    model.eval()

    # 获取模型信息
    info = model.get_model_info()
    print(f"  参数量: {info['total_parameters']:,}")

    # 验证参数量
    if info['total_parameters'] > 50000:
        print(f"  ✗ 参数量超标 (应<50K)")
        return False
    print(f"  ✓ 参数量满足要求")

    # 测试前向传播
    batch_size = 4
    seq_len = 20
    dummy_input = torch.randn(batch_size, seq_len, 4).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出形状: {output.shape}")

    # 验证输出形状
    if output.shape != (batch_size, 2):
        print(f"  ✗ 输出形状错误 (应为({batch_size}, 2), 实际{output.shape})")
        return False
    print(f"  ✓ 输出形状正确")

    # 测试推理速度
    import time

    dummy_single = torch.randn(1, seq_len, 4).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_single)

    # 计时
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_single)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    end = time.perf_counter()

    avg_time_ms = (end - start) / 100 * 1000
    print(f"  平均推理时间: {avg_time_ms:.3f} ms")

    # 验证推理时间
    if avg_time_ms > 1.0:
        print(f"  ✗ 推理时间超标 (应<1ms)")
        return False
    print(f"  ✓ 推理时间满足要求")

    print("  ✓ 模型推理测试通过")
    return True


def test_training_loop():
    """测试3: 训练循环"""
    print("\n" + "="*80)
    print("测试3: 训练循环 (1个epoch)")
    print("="*80)

    import glob
    import random

    # 准备数据
    data_pattern = "data_simulation_*"
    all_dirs = glob.glob(data_pattern)
    mat_files = []
    for d in all_dirs:
        mat_files.extend(glob.glob(os.path.join(d, "*.mat")))

    random.shuffle(mat_files)

    # 只用少量文件测试
    test_files = mat_files[:min(5, len(mat_files))]
    print(f"  使用文件数: {len(test_files)}")

    # 创建数据集
    train_dataset = RealTimeTrajectoryDataset(test_files, seq_len=20, pred_len=1)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    print(f"  训练样本数: {len(train_dataset)}")

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealTimeCorrector(input_size=4, hidden_size=56, num_layers=2, dropout=0.1)
    model.to(device)

    # 损失和优化器
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 训练1个batch
    model.train()
    for batch in train_loader:
        inputs = batch['input'].to(device)
        targets = batch['target'].squeeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        print(f"  训练损失: {loss.item():.6f}")
        break

    print("  ✓ 训练循环测试通过")
    return True


def test_evaluation():
    """测试4: 评估"""
    print("\n" + "="*80)
    print("测试4: 评估")
    print("="*80)

    import glob
    import random
    from sklearn.metrics import r2_score, mean_absolute_error

    # 准备数据
    data_pattern = "data_simulation_*"
    all_dirs = glob.glob(data_pattern)
    mat_files = []
    for d in all_dirs:
        mat_files.extend(glob.glob(os.path.join(d, "*.mat")))

    random.shuffle(mat_files)

    # 划分数据
    n_train = int(0.7 * len(mat_files))
    n_val = int(0.15 * len(mat_files))
    test_files = mat_files[n_train + n_val:]

    # 只用少量文件测试
    test_files = test_files[:min(3, len(test_files))]
    train_files = mat_files[:min(3, len(mat_files))]

    print(f"  测试文件数: {len(test_files)}")

    # 创建数据集
    train_dataset = RealTimeTrajectoryDataset(train_files, seq_len=20, pred_len=1)
    test_dataset = RealTimeTrajectoryDataset(
        test_files, seq_len=20, pred_len=1, scaler=train_dataset.scaler
    )

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealTimeCorrector(input_size=4, hidden_size=56, num_layers=2, dropout=0.1)
    model.to(device)
    model.eval()

    # 评估
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].squeeze(1)

            outputs = model(inputs)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # 计算指标
    r2_x = r2_score(targets[:, 0], preds[:, 0])
    r2_y = r2_score(targets[:, 1], preds[:, 1])
    r2_avg = (r2_x + r2_y) / 2

    mae = mean_absolute_error(targets, preds)

    print(f"  X轴 R²: {r2_x:.4f}")
    print(f"  Y轴 R²: {r2_y:.4f}")
    print(f"  平均 R²: {r2_avg:.4f}")
    print(f"  MAE: {mae:.6f} mm")

    # 注意: 未训练的模型指标会很低,这里只测试流程
    print("  ✓ 评估流程测试通过")
    return True


def main():
    print("\n" + "="*80)
    print("实时轨迹修正系统 - 端到端测试")
    print("="*80)

    results = []

    # 运行所有测试
    results.append(("数据加载", test_data_loading()))
    results.append(("模型推理", test_model_inference()))
    results.append(("训练循环", test_training_loop()))
    results.append(("评估流程", test_evaluation()))

    # 打印总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)

    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "="*80)
    if all_passed:
        print("✓ 所有测试通过!")
        print("\n系统已就绪,可以开始训练:")
        print("  python experiments/train_realtime.py --data_dir \"data_simulation_*\"")
    else:
        print("✗ 部分测试失败,请检查错误信息")
    print("="*80)


if __name__ == '__main__':
    main()
