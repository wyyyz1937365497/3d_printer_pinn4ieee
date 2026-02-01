"""
Train trajectory error correction model

Refactored from previous_code/3D_printer_loss_perdict for paper-ready training.
"""

import os
import sys
import argparse
import random
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from config import get_config
from models import TrajectoryErrorTransformer
from data.simulation import PrinterSimulationDataset
from training.losses import MultiTaskLoss
from utils import set_seed


def build_loaders(data_dir, config, batch_size):
    import glob

    mat_files = glob.glob(os.path.join(data_dir, "*.mat"))
    if not mat_files:
        raise ValueError(f"No .mat files found in {data_dir}")

    random.shuffle(mat_files)
    n_train = int(0.7 * len(mat_files))
    n_val = int(0.15 * len(mat_files))

    train_files = mat_files[:n_train]
    val_files = mat_files[n_train:n_train + n_val]

    train_dataset = PrinterSimulationDataset(
        train_files,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        stride=config.data.stride,
        mode='train',
        scaler=None,
        fit_scaler=True
    )

    val_dataset = PrinterSimulationDataset(
        val_files,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        stride=config.data.stride,
        mode='val',
        scaler=train_dataset.scaler,
        fit_scaler=False
    )

    # 创建DistributedSampler（支持DDP）
    train_sampler = None
    val_sampler = None
    # 延迟导入，只在需要时
    try:
        from torch.utils.data.distributed import DistributedSampler
        # 检查DDP是否已初始化
        if torch.distributed.is_initialized():
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
    except:
        pass

    # 使用sampler或default的shuffle
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train trajectory correction model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .mat files')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch (for resuming)')

    # 多GPU训练选项
    parser.add_argument('--ddp', action='store_true', help='Use DistributedDataParallel (DDP) instead of DataParallel')
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='Number of GPUs for DDP')

    args = parser.parse_args()
    set_seed(args.seed)

    # DDP初始化
    if args.ddp:
        # 设置环境变量（如果未设置）
        if 'RANK' not in os.environ:
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = str(args.world_size)
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'

        # 初始化进程组
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if rank == 0:
            print(f"\n使用DDP进行多GPU训练: {world_size} GPUs")
    else:
        device = torch.device(args.device)
        rank = 0
        world_size = 1

    config = get_config(preset='unified')
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr

    # DDP: 每个进程的batch_size是总batch_size除以world_size
    if args.ddp:
        effective_batch_size = config.training.batch_size // world_size
    else:
        effective_batch_size = config.training.batch_size

    train_loader, val_loader = build_loaders(args.data_dir, config, effective_batch_size)

    model = TrajectoryErrorTransformer(config).to(device)

    # 完全关闭物理约束损失（让模型自由学习数据模式）
    # 即使参数修正了，物理约束仍然限制了模型的学习能力
    criterion = MultiTaskLoss(
        lambda_quality=0.0,
        lambda_fault=0.0,
        lambda_trajectory=1.0,
        lambda_physics=0.0,  # 完全关闭，让模型从数据中学习
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.training.weight_decay)

    # Resume from checkpoint if specified (BEFORE DDP wrapping)
    start_epoch = 0
    best_val = float('inf')

    if args.resume and rank == 0:  # 只有rank 0加载checkpoint
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)

        # Handle DataParallel/DDP prefix
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('module.') for k in state_dict.keys()) or any(k.startswith('module.') for k in state_dict.keys()):
            print("  Removing DataParallel/DDP prefix...")
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[len('module.'):]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        print("  Model weights loaded")

        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("  Optimizer state restored")

        # Restore training state
        resume_epoch = checkpoint.get('epoch', -1)
        start_epoch = resume_epoch + 1
        best_val = checkpoint.get('val_loss', float('inf'))

        print(f"  Checkpoint was saved at epoch {resume_epoch}")
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Previous best validation loss: {best_val:.6f}")

    # Multi-GPU support (AFTER loading checkpoint)
    if args.ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if rank == 0:
            print(f"  Model wrapped with DDP on {world_size} GPUs")
    elif torch.cuda.device_count() > 1:
        print(f"\nUsing {torch.cuda.device_count()} GPUs for training!")
        model = torch.nn.DataParallel(model)
    else:
        print("\nUsing single GPU for training")

    for epoch in range(start_epoch, args.epochs):
        # DDP: 设置epoch用于DistributedSampler
        if args.ddp and hasattr(train_loader, 'sampler') and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0

        # 训练循环 with tqdm (只在rank 0显示)
        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]",
                       ncols=120, leave=False)
            iterator = pbar
        else:
            iterator = train_loader

        for batch in iterator:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(batch['input_features'])

            traj_targets = batch['trajectory_targets']
            targets = {
                'displacement_x_seq': traj_targets[:, :, 0:1],
                'displacement_y_seq': traj_targets[:, :, 1:2],
                'displacement_x': traj_targets[:, :, 0:1].mean(dim=1, keepdim=True),
                'displacement_y': traj_targets[:, :, 1:2].mean(dim=1, keepdim=True),
                'displacement_z': torch.zeros_like(traj_targets[:, :, 0:1].mean(dim=1, keepdim=True)),
            }

            # Add mean displacement for physics loss
            if 'displacement_x_seq' in outputs:
                outputs['displacement_x'] = outputs['displacement_x_seq'].mean(dim=1, keepdim=True)
            if 'displacement_y_seq' in outputs:
                outputs['displacement_y'] = outputs['displacement_y_seq'].mean(dim=1, keepdim=True)

            inputs = {
                'F_inertia_x': batch.get('F_inertia_x'),
                'F_inertia_y': batch.get('F_inertia_y'),
            }

            losses = criterion(outputs, targets, config.physics.__dict__, inputs)
            loss = losses['total']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            optimizer.step()

            total_loss += loss.item()

            # 更新进度条显示当前loss
            if rank == 0:
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        # DDP: 同步所有进程的loss
        if args.ddp:
            loss_tensor = torch.tensor([total_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            total_loss = loss_tensor.item()

        avg_loss = total_loss / max(1, len(train_loader))

        # 验证循环
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            if rank == 0:
                pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]",
                              ncols=120, leave=False)
                iterator_val = pbar_val
            else:
                iterator_val = val_loader

            for batch in iterator_val:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(batch['input_features'])

                traj_targets = batch['trajectory_targets']
                targets = {
                    'displacement_x_seq': traj_targets[:, :, 0:1],
                    'displacement_y_seq': traj_targets[:, :, 1:2],
                    'displacement_x': traj_targets[:, :, 0:1].mean(dim=1, keepdim=True),
                    'displacement_y': traj_targets[:, :, 1:2].mean(dim=1, keepdim=True),
                    'displacement_z': torch.zeros_like(traj_targets[:, :, 0:1].mean(dim=1, keepdim=True)),
                }

                if 'displacement_x_seq' in outputs:
                    outputs['displacement_x'] = outputs['displacement_x_seq'].mean(dim=1, keepdim=True)
                if 'displacement_y_seq' in outputs:
                    outputs['displacement_y'] = outputs['displacement_y_seq'].mean(dim=1, keepdim=True)

                inputs = {
                    'F_inertia_x': batch.get('F_inertia_x'),
                    'F_inertia_y': batch.get('F_inertia_y'),
                }

                losses = criterion(outputs, targets, config.physics.__dict__, inputs)
                val_loss += losses['total'].item()

                # 更新进度条
                if rank == 0:
                    pbar_val.set_postfix({'val_loss': f'{losses["total"].item():.6f}'})

        # DDP: 同步所有进程的val_loss
        if args.ddp:
            val_loss_tensor = torch.tensor([val_loss], device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_loss_tensor.item()

        val_loss = val_loss / max(1, len(val_loader))

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Train: {avg_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            if rank == 0:  # 只有rank 0保存checkpoint
                ckpt_dir = Path('checkpoints/trajectory_correction')
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                # 保存前移除DDP/DataParallel前缀
                model_state = model.module.state_dict() if (args.ddp or hasattr(model, 'module')) else model.state_dict()

                # Save complete checkpoint for resuming
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, ckpt_dir / 'best_model.pth')

                # Also save last checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, ckpt_dir / 'last_model.pth')

    # 清理DDP
    if args.ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
