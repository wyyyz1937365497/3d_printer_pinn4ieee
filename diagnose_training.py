"""
快速诊断脚本 - 检查训练循环中的数据流
"""

import torch
from data.simulation import PrinterSimulationDataset
from models import UnifiedPINNSeq3D
from config import get_config
import glob

print("=" * 60)
print("训练数据流诊断")
print("=" * 60)

# 加载数据
mat_files = glob.glob("data_simulation_3DBenchy_PLA_1h28m_layers_*/layer*.mat")[:5]
dataset = PrinterSimulationDataset(
    mat_files,
    seq_len=200,
    pred_len=50,
    stride=10,
    mode='train',
    fit_scaler=True
)

sample = dataset[0]

print("\n1. 数据集输出:")
print(f"   Input features shape: {sample['input_features'].shape}")
print(f"   Trajectory targets shape: {sample['trajectory_targets'].shape}")
print(f"   Quality targets shape: {sample['quality_targets'].shape}")
print(f"   F_inertia_x shape: {sample['F_inertia_x'].shape}")
print(f"   F_inertia_y shape: {sample['F_inertia_y'].shape}")

# 模拟训练循环中的数据处理
print("\n2. 训练循环中的数据处理:")
batch_data = {k: v.unsqueeze(0) for k, v in sample.items() if isinstance(v, torch.Tensor)}  # 添加batch维
print(f"   Batch input_features: {batch_data['input_features'].shape}")

# 提取轨迹目标
traj_targets = batch_data['trajectory_targets']
print(f"   Raw trajectory_targets: {traj_targets.shape}")

targets = {}
targets['displacement_x'] = traj_targets[:, :, 0:1].mean(dim=1, keepdim=True)
targets['displacement_y'] = traj_targets[:, :, 1:2].mean(dim=1, keepdim=True)
print(f"   Processed displacement_x: {targets['displacement_x'].shape}")
print(f"   Processed displacement_y: {targets['displacement_y'].shape}")

# 模型输出
print("\n3. 模型输出:")
config = get_config(preset='unified')
model = UnifiedPINNSeq3D(config)
model.eval()

with torch.no_grad():
    outputs = model(batch_data['input_features'])

print(f"   Output keys: {list(outputs.keys())}")
if 'displacement_x_seq' in outputs:
    print(f"   displacement_x_seq shape: {outputs['displacement_x_seq'].shape}")
if 'error_x' in outputs:
    print(f"   error_x shape: {outputs['error_x'].shape}")

# 调整输出形状（与训练逻辑一致）
if 'displacement_x_seq' in outputs:
    outputs['displacement_x'] = outputs['displacement_x_seq'].mean(dim=1, keepdim=True)
elif 'error_x' in outputs:
    outputs['displacement_x'] = outputs['error_x'].unsqueeze(-1)

if 'displacement_y_seq' in outputs:
    outputs['displacement_y'] = outputs['displacement_y_seq'].mean(dim=1, keepdim=True)
elif 'error_y' in outputs:
    outputs['displacement_y'] = outputs['error_y'].unsqueeze(-1)

if 'displacement_x' in outputs:
    print(f"   Processed displacement_x shape: {outputs['displacement_x'].shape}")

# 检查inputs
print("\n4. 物理损失inputs:")
inputs = {
    'F_inertia_x': batch_data.get('F_inertia_x'),
    'F_inertia_y': batch_data.get('F_inertia_y'),
}
if inputs['F_inertia_x'] is not None:
    print(f"   F_inertia_x shape: {inputs['F_inertia_x'].shape}")
    print(f"   F_inertia_x range: [{inputs['F_inertia_x'].min():.3f}, {inputs['F_inertia_x'].max():.3f}]")

# 测试损失计算
print("\n5. 损失计算测试:")
from training.losses import MultiTaskLoss

criterion = MultiTaskLoss(
    lambda_quality=config.lambda_quality,
    lambda_fault=config.lambda_fault,
    lambda_trajectory=config.lambda_trajectory,
    lambda_physics=config.lambda_physics,
)

physics_params = {
    'mass_x': config.physics.mass_x,
    'mass_y': config.physics.mass_y,
    'stiffness': config.physics.stiffness,
    'damping': config.physics.damping,
}

# 添加质量目标
targets['quality_score'] = batch_data['quality_targets'][:, -1:]

losses = criterion(outputs, targets, physics_params, inputs)

print(f"   Total loss: {losses['total'].item():.6f}")
print(f"   Quality: {losses['quality'].item():.6f}")
print(f"   Trajectory: {losses['trajectory'].item():.6f}")
print(f"   Physics: {losses['physics'].item():.6f}")
print(f"   Fault: {losses['fault'].item():.6f}")

if losses['trajectory'].item() > 0:
    print("\n✓ 轨迹损失正常！")
else:
    print("\n✗ 轨迹损失为0 - 检查模型输出和targets")

if losses['physics'].item() > 0:
    print("✓ 物理损失正常！")
else:
    print("✗ 物理损失为0 - 检查inputs")

print("\n" + "=" * 60)
