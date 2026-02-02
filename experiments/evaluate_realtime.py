"""
评估实时轨迹修正器

计算指标:
- R² Score
- MAE / RMSE
- Correlation
- 推理时间
"""

import os
import sys
import argparse
from pathlib import Path
import json
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from data.realtime_dataset import RealTimeTrajectoryDataset
from models.realtime_corrector import RealTimeCorrector


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    model = RealTimeCorrector(
        input_size=4,
        hidden_size=56,
        num_layers=2,
        dropout=0.1
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def build_test_loader(data_pattern, seq_len=20, batch_size=256, num_workers=2):
    """构建测试数据加载器"""
    import glob
    import random

    # 查找所有.mat文件
    all_dirs = glob.glob(data_pattern)
    mat_files = []
    for d in all_dirs:
        mat_files.extend(glob.glob(os.path.join(d, "*.mat")))

    if not mat_files:
        raise ValueError(f"未找到.mat文件: {data_pattern}")

    # 随机打乱
    random.shuffle(mat_files)

    # 划分数据集 (只取测试集)
    n_train = int(0.7 * len(mat_files))
    n_val = int(0.15 * len(mat_files))
    test_files = mat_files[n_train + n_val:]

    print(f"\n测试数据集:")
    print(f"  测试集文件数: {len(test_files)}")

    # 需要先拟合scaler (使用训练集)
    train_files = mat_files[:n_train]
    temp_train_dataset = RealTimeTrajectoryDataset(
        train_files,
        seq_len=seq_len,
        pred_len=1,
        scaler=None,
        mode='train'
    )

    # 创建测试数据集
    test_dataset = RealTimeTrajectoryDataset(
        test_files,
        seq_len=seq_len,
        pred_len=1,
        scaler=temp_train_dataset.scaler,
        mode='test'
    )

    print(f"  测试样本数: {len(test_dataset):,}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader


def evaluate_model(model, test_loader, device):
    """评估模型性能"""
    all_preds = []
    all_targets = []

    print("\n评估中...")
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].squeeze(1)  # [batch, 2]

            outputs = model(inputs)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # 拼接所有结果
    preds = np.concatenate(all_preds, axis=0)  # [N, 2]
    targets = np.concatenate(all_targets, axis=0)  # [N, 2]

    # 计算指标
    r2_x = r2_score(targets[:, 0], preds[:, 0])
    r2_y = r2_score(targets[:, 1], preds[:, 1])
    r2_avg = (r2_x + r2_y) / 2

    mae_x = mean_absolute_error(targets[:, 0], preds[:, 0])
    mae_y = mean_absolute_error(targets[:, 1], preds[:, 1])
    mae_avg = (mae_x + mae_y) / 2

    rmse_x = np.sqrt(mean_squared_error(targets[:, 0], preds[:, 0]))
    rmse_y = np.sqrt(mean_squared_error(targets[:, 1], preds[:, 1]))
    rmse_avg = (rmse_x + rmse_y) / 2

    # 相关系数
    corr_x = np.corrcoef(targets[:, 0], preds[:, 0])[0, 1]
    corr_y = np.corrcoef(targets[:, 1], preds[:, 1])[0, 1]
    corr_avg = (corr_x + corr_y) / 2

    metrics = {
        'r2_x': float(r2_x),
        'r2_y': float(r2_y),
        'r2_avg': float(r2_avg),
        'mae_x': float(mae_x),
        'mae_y': float(mae_y),
        'mae_avg': float(mae_avg),
        'rmse_x': float(rmse_x),
        'rmse_y': float(rmse_y),
        'rmse_avg': float(rmse_avg),
        'corr_x': float(corr_x),
        'corr_y': float(corr_y),
        'corr_avg': float(corr_avg),
    }

    return metrics, preds, targets


def measure_inference_time(model, device, num_iterations=1000):
    """测量推理时间"""
    model.eval()

    batch_size = 1
    seq_len = 20
    input_size = 4

    dummy_input = torch.randn(batch_size, seq_len, input_size).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 计时
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    end = time.perf_counter()

    avg_time_ms = (end - start) / num_iterations * 1000
    throughput = num_iterations / (end - start)

    return {
        'avg_time_ms': float(avg_time_ms),
        'throughput': float(throughput),
    }


def print_metrics(metrics, inference_perf):
    """打印评估结果"""
    print("\n" + "="*80)
    print("评估结果")
    print("="*80)

    print("\n【预测准确性】")
    print("-"*80)
    print(f"  X轴 R²:      {metrics['r2_x']:.4f}")
    print(f"  Y轴 R²:      {metrics['r2_y']:.4f}")
    print(f"  平均 R²:      {metrics['r2_avg']:.4f}")
    print()
    print(f"  X轴 MAE:     {metrics['mae_x']:.6f} mm")
    print(f"  Y轴 MAE:     {metrics['mae_y']:.6f} mm")
    print(f"  平均 MAE:    {metrics['mae_avg']:.6f} mm")
    print()
    print(f"  X轴 RMSE:    {metrics['rmse_x']:.6f} mm")
    print(f"  Y轴 RMSE:    {metrics['rmse_y']:.6f} mm")
    print(f"  平均 RMSE:   {metrics['rmse_avg']:.6f} mm")
    print()
    print(f"  X轴相关系数: {metrics['corr_x']:.4f}")
    print(f"  Y轴相关系数: {metrics['corr_y']:.4f}")
    print(f"  平均相关系数: {metrics['corr_avg']:.4f}")

    print("\n【推理性能】")
    print("-"*80)
    print(f"  平均推理时间: {inference_perf['avg_time_ms']:.3f} ms")
    print(f"  吞吐量:       {inference_perf['throughput']:.0f} inferences/sec")

    print("\n【目标达成情况】")
    print("-"*80)
    target_mae = 0.05
    target_r2 = 0.8
    target_time = 1.0

    if metrics['mae_avg'] < target_mae:
        print(f"  ✓ MAE目标达成 ({metrics['mae_avg']:.4f} < {target_mae})")
    else:
        print(f"  ✗ MAE未达标 ({metrics['mae_avg']:.4f} >= {target_mae})")

    if metrics['r2_avg'] > target_r2:
        print(f"  ✓ R²目标达成 ({metrics['r2_avg']:.4f} > {target_r2})")
    else:
        print(f"  ✗ R²未达标 ({metrics['r2_avg']:.4f} <= {target_r2})")

    if inference_perf['avg_time_ms'] < target_time:
        print(f"  ✓ 推理时间目标达成 ({inference_perf['avg_time_ms']:.3f} ms < {target_time} ms)")
    else:
        print(f"  ✗ 推理时间未达标 ({inference_perf['avg_time_ms']:.3f} ms >= {target_time} ms)")

    print("="*80)


def save_results(metrics, inference_perf, checkpoint_path):
    """保存评估结果"""
    results = {
        'metrics': metrics,
        'inference_performance': inference_perf,
        'checkpoint': str(checkpoint_path),
    }

    output_dir = Path('results/realtime_evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n评估结果已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate real-time trajectory corrector')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/simulation/*',
                       help='Data directory pattern')
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')

    args = parser.parse_args()

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 加载模型
    print(f"\n加载模型: {args.checkpoint}")
    model, checkpoint = load_model(args.checkpoint, device)
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  验证损失: {checkpoint['val_loss']:.6f}")

    # 构建测试数据加载器
    test_loader = build_test_loader(
        args.data_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 评估模型
    metrics, preds, targets = evaluate_model(model, test_loader, device)

    # 测量推理时间
    print("\n测量推理性能...")
    inference_perf = measure_inference_time(model, device, num_iterations=1000)

    # 打印结果
    print_metrics(metrics, inference_perf)

    # 保存结果
    save_results(metrics, inference_perf, args.checkpoint)


if __name__ == '__main__':
    main()
