"""
诊断模型预测行为 - 为什么loss很低但R²很差？

分析：
1. 预测方差 vs 目标方差
2. 预测均值 vs 目标均值
3. 与baseline的比较
"""

import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy import stats

from config import get_config
from models import TrajectoryErrorTransformer
from data.simulation import PrinterSimulationDataset


def diagnose_model_predictions(model, test_loader, device):
    """
    诊断模型预测行为
    """
    model.eval()

    all_preds_x = []
    all_preds_y = []
    all_targets_x = []
    all_targets_y = []

    with torch.no_grad():
        for batch in test_loader:
            input_features = batch['input_features'].to(device)
            trajectory_targets = batch['trajectory_targets']

            # 前向传播
            outputs = model(input_features)

            # 提取序列预测并取平均（与评估一致）
            if 'displacement_x_seq' in outputs:
                pred_x = outputs['displacement_x_seq'].mean(dim=1).cpu().numpy()  # [batch]
                pred_y = outputs['displacement_y_seq'].mean(dim=1).cpu().numpy()
            else:
                pred_x = outputs['error_x'][:, -1, :].cpu().numpy()
                pred_y = outputs['error_y'][:, -1, :].cpu().numpy()

            # 目标也取平均
            target_x = trajectory_targets[:, :, 0].mean(dim=1).cpu().numpy()
            target_y = trajectory_targets[:, :, 1].mean(dim=1).cpu().numpy()

            all_preds_x.extend(pred_x.flatten())
            all_preds_y.extend(pred_y.flatten())
            all_targets_x.extend(target_x.flatten())
            all_targets_y.extend(target_y.flatten())

    # 转为numpy数组
    pred_x = np.array(all_preds_x)
    pred_y = np.array(all_preds_y)
    target_x = np.array(all_targets_x)
    target_y = np.array(all_targets_y)

    print("=" * 80)
    print("模型预测行为诊断")
    print("=" * 80)

    # 1. X轴分析
    print("\n【X轴分析】")
    print(f"目标:  均值={target_x.mean():.6f}, 标准差={target_x.std():.6f}, 方差={target_x.var():.6f}")
    print(f"预测:  均值={pred_x.mean():.6f}, 标准差={pred_x.std():.6f}, 方差={pred_x.var():.6f}")

    # 方差比
    variance_ratio_x = pred_x.var() / target_x.var()
    print(f"\n方差比 (预测/目标): {variance_ratio_x:.4f}")
    if variance_ratio_x < 0.1:
        print("  ⚠️  预测方差太小！模型预测几乎是常数")
    elif variance_ratio_x < 0.5:
        print("  ⚠️  预测方差偏小，模型预测变化不足")
    elif variance_ratio_x > 2.0:
        print("  ⚠️  预测方差过大，模型预测过于剧烈")
    else:
        print("  ✓ 预测方差合理")

    # R²分析
    r2_x = stats.linregress(target_x, pred_x).rvalue ** 2
    print(f"\nR² = {r2_x:.4f}")
    if r2_x < 0.1:
        print("  ⚠️  R²很低，模型几乎没有学到真实模式")
    elif r2_x < 0.5:
        print("  ⚠️  R²偏低，模型学到部分模式")
    else:
        print("  ✓ R²良好")

    # 2. Y轴分析
    print("\n【Y轴分析】")
    print(f"目标:  均值={target_y.mean():.6f}, 标准差={target_y.std():.6f}, 方差={target_y.var():.6f}")
    print(f"预测:  均值={pred_y.mean():.6f}, 标准差={pred_y.std():.6f}, 方差={pred_y.var():.6f}")

    variance_ratio_y = pred_y.var() / target_y.var()
    print(f"\n方差比 (预测/目标): {variance_ratio_y:.4f}")
    if variance_ratio_y < 0.1:
        print("  ⚠️  预测方差太小！模型预测几乎是常数")
    elif variance_ratio_y < 0.5:
        print("  ⚠️  预测方差偏小，模型预测变化不足")
    elif variance_ratio_y > 2.0:
        print("  ⚠️  预测方差过大，模型预测过于剧烈")
    else:
        print("  ✓ 预测方差合理")

    r2_y = stats.linregress(target_y, pred_y).rvalue ** 2
    print(f"\nR² = {r2_y:.4f}")
    if r2_y < 0.1:
        print("  ⚠️  R²很低，模型几乎没有学到真实模式")
    elif r2_y < 0.5:
        print("  ⚠️  R²偏低，模型学到部分模式")
    else:
        print("  ✓ R²良好")

    # 3. 与baseline比较
    print("\n" + "=" * 80)
    print("与Baseline比较")
    print("=" * 80)

    # Baseline 1: 预测常数（均值）
    baseline_mse_x_mean = ((target_x - target_x.mean()) ** 2).mean()
    model_mse_x = ((target_x - pred_x) ** 2).mean()
    print(f"\nX轴:")
    print(f"  预测均值 (baseline): MSE = {baseline_mse_x_mean:.6f}")
    print(f"  模型预测:           MSE = {model_mse_x:.6f}")
    if model_mse_x < baseline_mse_x_mean:
        improvement = (1 - model_mse_x / baseline_mse_x_mean) * 100
        print(f"  ✓ 模型优于baseline {improvement:.1f}%")
    else:
        print(f"  ⚠️  模型不如预测均值！")

    baseline_mse_y_mean = ((target_y - target_y.mean()) ** 2).mean()
    model_mse_y = ((target_y - pred_y) ** 2).mean()
    print(f"\nY轴:")
    print(f"  预测均值 (baseline): MSE = {baseline_mse_y_mean:.6f}")
    print(f"  模型预测:           MSE = {model_mse_y:.6f}")
    if model_mse_y < baseline_mse_y_mean:
        improvement = (1 - model_mse_y / baseline_mse_y_mean) * 100
        print(f"  ✓ 模型优于baseline {improvement:.1f}%")
    else:
        print(f"  ⚠️  模型不如预测均值！")

    # Baseline 2: 预测0
    baseline_mse_x_zero = (target_x ** 2).mean()
    baseline_mse_y_zero = (target_y ** 2).mean()
    print(f"\n预测0 (baseline):")
    print(f"  X轴 MSE = {baseline_mse_x_zero:.6f}")
    print(f"  Y轴 MSE = {baseline_mse_y_zero:.6f}")

    # 4. 为什么loss低但R²低？
    print("\n" + "=" * 80)
    print("问题分析")
    print("=" * 80)

    # 计算训练loss和R²的关系
    print("\n为什么训练loss很低但R²很差？")

    # 分解MSE
    mse_x = ((target_x - pred_x) ** 2).mean()
    bias_x = (pred_x.mean() - target_x.mean()) ** 2
    variance_x = ((pred_x - pred_x.mean()).var())
    target_variance_x = target_x.var()

    print(f"\nMSE分解 (X轴):")
    print(f"  总MSE = {mse_x:.6f}")
    print(f"  偏差² (bias²) = {bias_x:.6f} ({bias_x/mse_x*100:.1f}%)")
    print(f"  目标方差 = {target_variance_x:.6f}")
    print(f"  预测方差 = {variance_x:.6f}")

    if variance_x < 0.1 * target_variance_x:
        print("\n  ⚠️  问题诊断：")
        print("  【预测方差太小】模型预测几乎是常数（预测均值）")
        print("  虽然MSE低（因为目标均值接近0），但R²低（因为没有预测变化）")
        print("\n  可能原因：")
        print("  1. 正则化太强")
        print("  2. 物理约束限制了预测范围")
        print("  3. 模型容量不足")
        print("  4. 训练策略问题（如学习率太低）")

    print("\n" + "=" * 80)
    return {
        'pred_x': pred_x,
        'pred_y': pred_y,
        'target_x': target_x,
        'target_y': target_y,
        'r2_x': r2_x,
        'r2_y': r2_y,
        'variance_ratio_x': variance_ratio_x,
        'variance_ratio_y': variance_ratio_y,
    }


def main():
    parser = argparse.ArgumentParser(description='诊断模型预测行为')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/trajectory_correction/best_model.pth')
    parser.add_argument('--data_dir', type=str, default='data_simulation_*')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print("=" * 80)
    print("模型预测诊断工具")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_dir}")
    print()

    # 加载配置
    config = get_config(preset='unified')
    device = torch.device(args.device)

    # 加载数据
    import glob
    import random

    mat_files = []
    expanded_paths = glob.glob(args.data_dir)
    for path in expanded_paths:
        if os.path.isfile(path) and path.endswith('.mat'):
            mat_files.append(path)
        elif os.path.isdir(path):
            mat_files.extend(glob.glob(os.path.join(path, "*.mat")))

    if not mat_files:
        raise ValueError(f"No .mat files found in {args.data_dir}")

    random.shuffle(mat_files)
    n_train = int(0.7 * len(mat_files))
    n_val = int(0.15 * len(mat_files))
    test_files = mat_files[n_train + n_val:]

    print(f"加载测试数据: {len(test_files)} 个文件")

    # 创建测试集（使用训练的scaler）
    train_dataset_temp = PrinterSimulationDataset(
        mat_files[:n_train],
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        stride=config.data.stride,
        mode='train'
    )

    test_dataset = PrinterSimulationDataset(
        test_files,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        stride=config.data.stride,
        mode='test',
        scaler=train_dataset_temp.scaler
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"测试样本: {len(test_dataset)}")

    # 加载模型
    print(f"\n加载模型...")
    model = torch.load(args.checkpoint, map_location=device)

    # 处理DataParallel checkpoint
    if isinstance(model, dict) and 'model_state_dict' in model:
        state = model['model_state_dict']
        if any(k.startswith('module.') for k in state.keys()):
            print("  移除DataParallel前缀...")
            new_state = {}
            for k, v in state.items():
                if k.startswith('module.'):
                    new_state[k[len('module.'):]] = v
                else:
                    new_state[k] = v
            state = new_state

        # 创建模型并加载权重
        from models import TrajectoryErrorTransformer
        model_instance = TrajectoryErrorTransformer(config).to(device)
        model_instance.load_state_dict(state)
        model = model_instance

    model.eval()

    # 诊断
    results = diagnose_model_predictions(model, test_loader, device)

    print("\n诊断完成！")
    print()


if __name__ == '__main__':
    main()
