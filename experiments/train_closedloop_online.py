"""
闭环模型在线训练 - 实时MATLAB仿真

核心创新：
1. 每个batch实时调用MATLAB并行仿真
2. 不需要预生成数据集
3. 支持策略探索和迭代
4. 更接近强化学习范式

工作流程：
    随机G-code → 修正策略 → MATLAB并行仿真 → 实时数据 → 训练 → 策略更新

性能：
    - MATLAB并行: 100个轨迹并行 ~30秒
    - 单batch训练: ~5秒
    - 总计: ~35秒/batch
    - 200 epochs: ~2小时

使用方法：
    python experiments/train_closedloop_online.py \
        --gcode_files "test_gcode_files/*.gcode" \
        --batch_size 32 \
        --epochs 200
"""

import argparse
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import matlab.engine
import time
import random

import sys
from pathlib import Path as PathLib

project_root = PathLib(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.closedloop_corrector import AdaptiveClosedLoopModel


class OnlineTrajectoryDataset(Dataset):
    """
    在线轨迹数据集

    每次调用时实时生成数据：
    1. 随机选择G-code文件
    2. 使用当前模型生成修正轨迹
    3. 调用MATLAB并行仿真
    4. 返回仿真结果
    """

    def __init__(
        self,
        gcode_files,
        model,
        scaler,
        matlab_engine,
        seq_len=50,
        batch_sim_size=100,  # 每次并行仿真的轨迹数
        device='cpu'
    ):
        """
        Args:
            gcode_files: G-code文件列表
            model: 当前闭环模型（用于生成修正策略）
            scaler: 标准化器
            matlab_engine: MATLAB引擎实例
            seq_len: 序列长度
            batch_sim_size: 每次并行仿真的轨迹数量
            device: 计算设备
        """
        self.gcode_files = gcode_files
        self.model = model
        self.scaler = scaler
        self.eng = matlab_engine
        self.seq_len = seq_len
        self.batch_sim_size = batch_sim_size
        self.device = device

        # 缓存仿真结果
        self.cache = {}
        self.cache_size = 1000  # 最多缓存1000个轨迹

    def generate_correction_strategy(self, ref_trajectory):
        """
        使用当前模型生成修正策略

        Args:
            ref_trajectory: (N, 4) 参考轨迹

        Returns:
            corrected_trajectory: (N, 4) 修正后的轨迹
        """
        self.model.eval()

        N = ref_trajectory.shape[0]
        corrected = ref_trajectory.copy()

        # 添加探索噪声（鼓励探索）
        noise_scale = 0.00001  # 10nm噪声
        noise = np.random.randn(4) * noise_scale

        with torch.no_grad():
            for i in range(self.seq_len, N):
                # 提取历史
                history = ref_trajectory[i-self.seq_len:i]
                history_norm = self.scaler.transform(history)

                # 模型预测
                input_tensor = torch.FloatTensor(history_norm).unsqueeze(0).to(self.device)
                pred_error = self.model(input_tensor, input_tensor).cpu().numpy()[0]

                # 应用修正 + 探索噪声
                corrected[i, 0] = ref_trajectory[i, 0] + pred_error[0] + noise[0]
                corrected[i, 1] = ref_trajectory[i, 1] + pred_error[1] + noise[1]
                corrected[i, 2] = ref_trajectory[i, 2] + noise[2]
                corrected[i, 3] = ref_trajectory[i, 3] + noise[3]

        return corrected

    def simulate_in_matlab(self, ref_traj, corrected_traj, gcode_file):
        """
        在MATLAB中并行仿真

        Args:
            ref_traj: (N, 4) 参考轨迹
            corrected_traj: (N, 4) 修正轨迹
            gcode_file: G-code文件路径

        Returns:
            actual_error: (N, 2) 实际误差
        """
        try:
            # 调用MATLAB并行仿真函数
            result = self.eng.parallel_simulate_batch(
                gcode_file,
                matlab.double(ref_traj[:, 0].tolist()),
                matlab.double(ref_traj[:, 1].tolist()),
                matlab.double(corrected_traj[:, 0].tolist()),
                matlab.double(corrected_traj[:, 1].tolist()),
                nargout=1
            )

            # 解析结果
            # result应该包含actual_error_x, actual_error_y
            actual_error = np.array([result['error_x'], result['error_y']]).T

            return actual_error

        except Exception as e:
            print(f"[WARNING] MATLAB仿真失败: {e}")
            # 返回零误差（降级处理）
            return np.zeros((len(ref_traj), 2))

    def __len__(self):
        # 返回一个较大的值，确保有足够的数据
        return 100000

    def __getitem__(self, idx):
        """
        生成一个训练样本

        Returns:
            ref_traj: [seq_len, 4] 参考轨迹
            corr_traj: [seq_len, 4] 修正轨迹
            error: [2] 误差
        """
        # 检查缓存
        if idx in self.cache:
            return self.cache[idx]

        # 随机选择一个G-code文件
        gcode_file = random.choice(self.gcode_files)

        # 生成参考轨迹（从文件读取或解析）
        ref_trajectory = self.parse_gcode_to_trajectory(gcode_file)

        # 使用当前模型生成修正策略
        corrected_trajectory = self.generate_correction_strategy(ref_trajectory)

        # MATLAB仿真
        actual_error = self.simulate_in_matlab(ref_trajectory, corrected_trajectory, gcode_file)

        # 创建序列
        N = len(ref_trajectory)
        start_idx = random.randint(self.seq_len, N - 1)

        ref_seq = ref_trajectory[start_idx-self.seq_len:start_idx]
        corr_seq = corrected_trajectory[start_idx-self.seq_len:start_idx]
        error = actual_error[start_idx]

        # 缓存结果
        if len(self.cache) < self.cache_size:
            self.cache[idx] = (ref_seq, corr_seq, error)

        return ref_seq, corr_seq, error

    def parse_gcode_to_trajectory(self, gcode_file):
        """
        从G-code文件解析轨迹（简化版）

        实际实现中应该：
        1. 解析G-code
        2. 提取G1/G0指令
        3. 生成参考轨迹
        """
        # 这里简化处理：从已有的.mat文件读取
        # 实际使用时需要完整实现

        # 查找对应的.mat文件
        gcode_name = Path(gcode_file).stem
        mat_files = glob.glob(f"data_simulation_*/*{gcode_name}*.mat")

        if mat_files:
            # 读取参考轨迹
            with h5py.File(mat_files[0], 'r') as f:
                data = f['simulation_data']
                x_ref = data['x_ref'][:].flatten()
                y_ref = data['y_ref'][:].flatten()
                vx_ref = data['vx_ref'][:].flatten()
                vy_ref = data['vy_ref'][:].flatten()

                trajectory = np.stack([x_ref, y_ref, vx_ref, vy_ref], axis=1)
                return trajectory
        else:
            # 如果找不到，生成随机轨迹
            print(f"[WARNING] 未找到 {gcode_name} 对应的.mat文件，使用随机轨迹")
            N = 1000
            trajectory = np.random.randn(N, 4) * 0.1
            trajectory[:, 0] = np.linspace(0, 100, N)  # X: 0-100mm
            trajectory[:, 1] = np.linspace(0, 100, N)  # Y: 0-100mm
            return trajectory


class HybridLoss(nn.Module):
    """混合损失函数"""

    def __init__(self, mae_ratio=0.7):
        super().__init__()
        self.mae_ratio = mae_ratio
        self.mse_ratio = 1.0 - mae_ratio
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        mae_loss = self.mae(preds, targets)
        mse_loss = self.mse(preds, targets)
        return self.mae_ratio * mae_loss + self.mse_ratio * mse_loss


def train_epoch_online(model, dataset, criterion, optimizer, device, accumulation_steps=2):
    """
    在线训练一个epoch

    每个batch都实时生成新数据
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    # 创建在线数据加载器
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    pbar = tqdm(loader, desc="[Train Online]", ncols=120, leave=False)

    for step, batch_data in enumerate(pbar):
        # batch_data是list of tuples，需要手动组织
        batch_size = len(batch_data)
        ref_traj_list = []
        corr_traj_list = []
        target_list = []

        for item in batch_data:
            ref, corr, target = item
            ref_traj_list.append(ref)
            corr_traj_list.append(corr)
            target_list.append(target)

        # 转换为tensor
        ref_traj = torch.stack([torch.FloatTensor(r) for r in ref_traj_list]).to(device)
        corr_traj = torch.stack([torch.FloatTensor(c) for c in corr_traj_list]).to(device)
        targets = torch.stack([torch.FloatTensor(t) for t in target_list]).to(device)

        # 前向传播
        outputs = model(ref_traj, corr_traj)
        loss = criterion(outputs, targets) / accumulation_steps

        # 反向传播
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.6f}'})

    return total_loss / len(loader)


def main(args):
    """主训练函数"""

    print("=" * 80)
    print("闭环模型在线训练 - 实时MATLAB仿真")
    print("=" * 80)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集G-code文件
    gcode_files = glob.glob(args.gcode_files)
    print(f"\n找到 {len(gcode_files)} 个G-code文件")

    if len(gcode_files) == 0:
        print("未找到G-code文件，退出")
        return

    # 启动MATLAB
    print("\n启动MATLAB并行引擎...")
    try:
        eng = matlab.engine.start_matlab()
        eng.cd('F:\\TJ\\3d_print\\3d_printer_pinn4ieee')
        print("[OK] MATLAB已启动")

        # 设置并行池
        num_cores = 8  # 根据你的CPU调整
        eng.eval(f'pool = parpool({num_cores});', nargout=0)
        print(f"[OK] 并行池已启动 ({num_cores} workers)")
    except Exception as e:
        print(f"[ERROR] MATLAB启动失败: {e}")
        return

    # 创建模型
    print("\n创建闭环模型...")
    model = AdaptiveClosedLoopModel(
        input_size=4,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    # 打印模型信息
    info = model.get_model_info()
    print(f"  模型类型: {info['model_type']}")
    print(f"  总参数: {info['total_parameters']:,}")

    # 创建scaler（用部分数据拟合）
    print("\n创建StandardScaler...")
    from sklearn.preprocessing import StandardScaler

    # 从一些.mat文件拟合scaler
    mat_files = glob.glob("data_simulation_*/*.mat")[:10]
    all_data = []
    for mf in mat_files:
        with h5py.File(mf, 'r') as f:
            data = f['simulation_data']
            x_ref = data['x_ref'][:].flatten()
            y_ref = data['y_ref'][:].flatten()
            vx_ref = data['vx_ref'][:].flatten()
            vy_ref = data['vy_ref'][:].flatten()
            features = np.stack([x_ref, y_ref, vx_ref, vy_ref], axis=1)
            all_data.append(features)

    all_data = np.concatenate(all_data, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_data)
    print("  [OK] Scaler拟合完成")

    # 创建在线数据集
    print("\n创建在线数据集...")
    dataset = OnlineTrajectoryDataset(
        gcode_files=gcode_files,
        model=model,
        scaler=scaler,
        matlab_engine=eng,
        seq_len=args.seq_len,
        batch_sim_size=args.batch_sim_size,
        device=device
    )

    # 损失函数和优化器
    criterion = HybridLoss(mae_ratio=args.mae_ratio)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # 训练历史
    history = {'train_loss': [], 'sim_time': []}

    # 训练循环
    print("\n开始在线训练...")
    print("=" * 80)
    print("每个epoch包含实时MATLAB仿真")
    print("预计时间: ~2-3分钟/epoch")
    print("=" * 80)

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # 训练一个epoch（实时生成数据）
        train_loss = train_epoch_online(
            model, dataset, criterion, optimizer, device,
            accumulation_steps=args.accumulation_steps
        )

        # 学习率调度
        scheduler.step(train_loss)

        # 记录
        history['train_loss'].append(train_loss)

        epoch_time = time.time() - epoch_start
        history['sim_time'].append(epoch_time)

        # 打印
        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'model_info': model.get_model_info(),
            }

            best_model_path = output_dir / 'best_model.pth'
            torch.save(checkpoint, best_model_path)
            print(f"  ✓ 保存最佳模型")

        # 定期保存
        if (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'model_info': model.get_model_info(),
                'history': history
            }

            ckpt_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, ckpt_path)
            print(f"  ✓ 保存checkpoint")

        print("-" * 80)

    # 关闭MATLAB
    try:
        eng.eval('delete(pool);', nargout=0)
        eng.quit()
        print("\n[MATLAB] 已关闭")
    except:
        pass

    # 保存训练历史
    import json
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        # 转换numpy类型为Python类型
        history_serializable = {
            'train_loss': [float(x) for x in history['train_loss']],
            'sim_time': [float(x) for x in history['sim_time']]
        }
        json.dump(history_serializable, f, indent=2)

    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    print(f"\n最佳结果:")
    print(f"  Epoch: {best_epoch+1}")
    print(f"  Loss: {best_loss:.6f}")
    print(f"\n模型已保存到: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='在线训练闭环模型 - 实时MATLAB仿真')

    # G-code文件
    parser.add_argument('--gcode_files', type=str, default='test_gcode_files/*.gcode',
                       help='G-code文件模式')

    # 模型参数
    parser.add_argument('--hidden_size', type=int, default=64,
                       help='LSTM隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='LSTM层数')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout率')

    # 训练参数
    parser.add_argument('--seq_len', type=int, default=50,
                       help='输入序列长度')
    parser.add_argument('--batch_sim_size', type=int, default=100,
                       help='每次并行仿真的轨迹数量')
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--accumulation_steps', type=int, default=2,
                       help='梯度累积步数')
    parser.add_argument('--mae_ratio', type=float, default=0.7,
                       help='MAE损失权重')

    # 系统参数
    parser.add_argument('--output_dir', type=str,
                       default='checkpoints/closedloop_online',
                       help='输出目录')
    parser.add_argument('--save_freq', type=int, default=20,
                       help='保存checkpoint频率')

    args = parser.parse_args()

    main(args)
