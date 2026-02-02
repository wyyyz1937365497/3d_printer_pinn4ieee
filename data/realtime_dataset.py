"""
实时轨迹修正数据集

使用4维输入特征 [x_ref, y_ref, vx_ref, vy_ref]
预测2维轨迹误差 [error_x, error_y]

设计原则:
- 最小化输入，满足实时性要求
- 短序列 (seq_len=20)，快速推理
- 单步预测 (pred_len=1)，10ms提前量
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import glob
import random


def load_mat_file(mat_path):
    """
    从.mat文件加载4维特征和2维标签

    Args:
        mat_path: .mat文件路径

    Returns:
        features: (N, 4) numpy数组 - [x_ref, y_ref, vx_ref, vy_ref]
        labels: (N, 2) numpy数组 - [error_x, error_y]
    """
    try:
        with h5py.File(mat_path, 'r') as f:
            data = f['simulation_data']

            # 提取4维特征
            x_ref = data['x_ref'][:].flatten()
            y_ref = data['y_ref'][:].flatten()
            vx_ref = data['vx_ref'][:].flatten()
            vy_ref = data['vy_ref'][:].flatten()

            features = np.stack([x_ref, y_ref, vx_ref, vy_ref], axis=1)  # (N, 4)

            # 提取2维标签
            error_x = data['error_x'][:].flatten()
            error_y = data['error_y'][:].flatten()

            labels = np.stack([error_x, error_y], axis=1)  # (N, 2)

        return features, labels

    except Exception as e:
        print(f"错误: 加载 {mat_path} 失败: {e}")
        return None, None


def create_sequences(features, labels, seq_len=20, pred_len=1):
    """
    从时序数据创建训练样本

    Args:
        features: (N, 4) 输入特征
        labels: (N, 2) 标签
        seq_len: 输入序列长度
        pred_len: 预测序列长度

    Returns:
        sequences: (M, seq_len, 4) 输入序列
        targets: (M, pred_len, 2) 预测目标
    """
    N = features.shape[0]

    # 计算可以创建的序列数量
    num_sequences = (N - seq_len - pred_len + 1) // seq_len

    sequences = []
    targets = []

    for i in range(num_sequences):
        start_idx = i * seq_len
        end_idx = start_idx + seq_len
        target_idx = end_idx  # 预测下一个时间步

        # 输入序列
        seq = features[start_idx:end_idx]
        sequences.append(seq)

        # 目标 (预测下一个时间步的误差)
        target = labels[target_idx:target_idx + pred_len]
        targets.append(target)

    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    return sequences, targets


class RealTimeTrajectoryDataset(Dataset):
    """
    实时轨迹修正数据集

    输入: [batch, seq_len=20, 4] - x_ref, y_ref, vx_ref, vy_ref
    输出: [batch, pred_len=1, 2] - error_x, error_y
    """

    def __init__(
        self,
        files,
        seq_len=20,
        pred_len=1,
        scaler=None,
        mode='train'
    ):
        """
        Args:
            files: .mat文件列表
            seq_len: 输入序列长度
            pred_len: 预测序列长度
            scaler: StandardScaler对象 (可选，如果不提供则创建新的)
            mode: 'train', 'val', 或 'test'
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mode = mode

        # 加载所有数据
        print(f"加载 {len(files)} 个文件 ({mode})...")
        all_features = []
        all_labels = []

        for file_path in files:
            features, labels = load_mat_file(file_path)
            if features is not None:
                all_features.append(features)
                all_labels.append(labels)

        # 合并所有数据
        self.all_features = np.concatenate(all_features, axis=0)  # (Total_N, 4)
        self.all_labels = np.concatenate(all_labels, axis=0)        # (Total_N, 2)

        print(f"  总数据点: {self.all_features.shape[0]:,}")

        # 创建序列
        print(f"  创建序列 (seq_len={seq_len}, pred_len={pred_len})...")
        self.sequences, self.targets = create_sequences(
            self.all_features, self.all_labels, seq_len, pred_len
        )

        print(f"  创建了 {self.sequences.shape[0]:,} 个序列")

        # 归一化输入特征（输出误差不归一化，因为其范围不确定）
        if scaler is None:
            self.scaler = StandardScaler()
            # 总是拟合scaler
            all_features_for_fit = self.all_features.reshape(-1, 4)
            self.scaler.fit(all_features_for_fit)
            print(f"  输入Scaler已拟合")
            print(f"  注意: 输出误差不归一化，模型将学习原始误差值")
        else:
            self.scaler = scaler
            print(f"  使用提供的scaler")

        # 应用scaler到输入
        original_shape = self.sequences.shape
        self.sequences = self.scaler.transform(
            self.sequences.reshape(-1, 4)
        ).reshape(original_shape)

        # 输出误差保持原始尺度（mm单位）
        print(f"  数据形状: sequences={self.sequences.shape}, targets={self.targets.shape}")
        print(f"  输出误差范围: X=[{self.all_labels[:,0].min()*1000:.2f}, {self.all_labels[:,0].max()*1000:.2f}] μm")
        print(f"                Y=[{self.all_labels[:,1].min()*1000:.2f}, {self.all_labels[:,1].max()*1000:.2f}] μm")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        返回一个样本

        Returns:
            input_features: [seq_len, 4]
            targets: [pred_len, 2]
        """
        return {
            'input': torch.tensor(self.sequences[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32)
        }


def build_loaders(data_dir, seq_len=20, batch_size=256, num_workers=2):
    """
    构建训练、验证和测试数据加载器

    Args:
        data_dir: 数据目录模式 (e.g., "data_simulation_*")
        seq_len: 序列长度
        batch_size: 批次大小
        num_workers: DataLoader工作进程数

    Returns:
        train_loader, val_loader, test_loader
    """
    # 查找所有.mat文件
    mat_files = glob.glob(f"{data_dir}/*.mat")
    if not mat_files:
        raise ValueError(f"在 {data_dir} 中未找到.mat文件")

    print(f"\n找到 {len(mat_files)} 个.mat文件")

    # 随机打乱
    random.shuffle(mat_files)

    # 划分数据集
    n_train = int(0.7 * len(mat_files))
    n_val = int(0.15 * len(mat_files))

    train_files = mat_files[:n_train]
    val_files = mat_files[n_train:n_train + n_val]
    test_files = mat_files[n_train + n_val:]

    print(f"  训练集: {len(train_files)} 个文件")
    print(f"  验证集: {len(val_files)} 个文件")
    print(f"  测试集: {len(test_files)} 个文件")

    # 创建数据集
    train_dataset = RealTimeTrajectoryDataset(
        train_files,
        seq_len=seq_len,
        scaler=None,
        fit_scaler=True,  # 训练集拟合scaler
        mode='train'
    )

    val_dataset = RealTimeTrajectoryDataset(
        val_files,
        seq_len=seq_len,
        scaler=train_dataset.scaler,  # 使用训练集的scaler
        fit_scaler=False,
        mode='val'
    )

    test_dataset = RealTimeTrajectoryDataset(
        test_files,
        seq_len=seq_len,
        scaler=train_dataset.scaler,
        fit_scaler=False,
        mode='test'
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False
    )

    return train_loader, val_loader, test_loader, train_dataset.scaler


# 测试代码
if __name__ == '__main__':
    print("="*80)
    print("测试实时数据集")
    print("="*80)

    # 测试加载单个文件
    mat_file = 'data_simulation_3DBenchy_PLA_1h28m_sampled_48layers/layer01_ender3v2.mat'
    features, labels = load_mat_file(mat_file)

    print(f"\n单文件测试:")
    print(f"  特征形状: {features.shape}")
    print(f"  标签形状: {labels.shape}")
    print(f"  X_ref范围: [{features[:, 0].min():.2f}, {features[:, 0].max():.2f}]")
    print(f"  VX_ref范围: [{features[:, 2].min():.2f}, {features[:, 2].max():.2f}]")
    print(f"  Error X范围: [{labels[:, 0].min():.4f}, {labels[:, 0].max():.4f}]")

    # 测试序列创建
    sequences, targets = create_sequences(features, labels, seq_len=20, pred_len=1)
    print(f"\n序列创建:")
    print(f"  序列形状: {sequences.shape}")
    print(f"  目标形状: {targets.shape}")
    print(f"  示例输入[0]: {sequences[0, 0]}")
    print(f"  示例目标[0]: {targets[0]}")

    # 测试数据集
    print(f"\n测试完整数据集:")
    dataset = RealTimeTrajectoryDataset([mat_file], seq_len=20, pred_len=1)
    sample = dataset[0]
    print(f"  输入形状: {sample['input'].shape}")
    print(f"  目标形状: {sample['target'].shape}")

    print("\n✓ 数据集测试通过")
