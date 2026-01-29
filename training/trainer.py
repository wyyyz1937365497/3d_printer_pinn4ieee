"""
用于训练和评估模型的训练器类
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from contextlib import nullcontext
from typing import Dict, Optional, Any, List
from tqdm import tqdm

from utils.logger import Logger
from training.losses import MultiTaskLoss


class Trainer:
    """
    PINN-Seq3D模型的统一训练器

    处理训练、验证、检查点保存和日志记录
    """

    def __init__(self,
                 model: nn.Module,
                 config: Any,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 test_loader: Optional[DataLoader] = None):
        """
        初始化训练器

        参数:
            model: 要训练的模型
            config: 配置对象
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            test_loader: 测试数据加载器（可选）
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 设置设备
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 设置损失函数
        self.criterion = MultiTaskLoss(
            lambda_quality=config.lambda_quality,
            lambda_fault=config.lambda_fault,
            lambda_trajectory=config.lambda_trajectory,
            lambda_physics=config.lambda_physics,
        )

        # 设置优化器
        self.optimizer = self._create_optimizer()

        # 设置调度器
        self.scheduler = self._create_scheduler()

        # 为混合精度设置梯度缩放器
        self.scaler = GradScaler('cuda') if config.training.mixed_precision else None

        # 设置日志记录器
        self.logger = Logger(
            name=config.experiment_name,
            log_dir=config.training.log_dir
        )

        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # 历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
        }

        self.logger.log_model_summary(model)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        if self.config.training.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=self.config.training.betas,
                eps=self.config.training.eps,
            )
        elif self.config.training.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=self.config.training.betas,
                eps=self.config.training.eps,
            )
        else:
            raise ValueError(f"未知优化器: {self.config.training.optimizer}")

        return optimizer

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        if self.config.training.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.min_lr,
            )
        elif self.config.training.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        elif self.config.training.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
            )
        else:
            scheduler = None

        return scheduler

    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个周期

        返回:
            包含训练指标的字典
        """
        self.model.train()
        
        total_loss = 0.0
        loss_components = {
            'quality': 0.0,
            'fault': 0.0,
            'trajectory': 0.0,
            'physics': 0.0,
        }

        num_batches = len(self.train_loader)
        progress_bar = tqdm(self.train_loader, desc=f"周期 {self.current_epoch + 1}")
        accum_step = 0

        for batch_idx, batch in enumerate(progress_bar):
            # 在累积开始时清零梯度
            if accum_step == 0:
                self.optimizer.zero_grad()
            
            # 将批次移到设备上
            batch_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # 构建目标字典 - 将数据集的格式转换为损失函数期望的格式
            targets = {}
            
            # 处理轨迹目标 trajectory_targets: [batch, pred_len, 2] -> displacement_x/y
            if 'trajectory_targets' in batch_data and batch_data['trajectory_targets'].numel() > 0:
                traj_targets = batch_data['trajectory_targets']  # [batch, pred_len, 2]
                # 计算平均误差作为目标（更稳定）
                targets['displacement_x'] = traj_targets[:, :, 0:1].mean(dim=1, keepdim=True)  # [batch, 1, 1]
                targets['displacement_y'] = traj_targets[:, :, 1:2].mean(dim=1, keepdim=True)  # [batch, 1, 1]
                targets['displacement_z'] = torch.zeros_like(targets['displacement_x'])  # No z in data
            
            # 处理质量目标 quality_targets: [batch, 5] 
            if 'quality_targets' in batch_data and batch_data['quality_targets'].numel() > 0:
                quality_targets = batch_data['quality_targets']  # [batch, 5]
                # 质量指标的顺序: adhesion_ratio, internal_stress, porosity, dimensional_accuracy, quality_score
                targets['quality_score'] = quality_targets[:, -1:]  # [batch, 1]

            # 前向传播和损失计算
            outputs = self.model(batch_data['input_features'])
            
            # 调整模型输出的形状
            # 优先使用序列输出，否则回退到error_x/y
            if 'displacement_x_seq' in outputs:
                outputs['displacement_x'] = outputs['displacement_x_seq'].mean(dim=1, keepdim=True)  # [batch, 1, 1]
            elif 'error_x' in outputs:
                outputs['displacement_x'] = outputs['error_x'].unsqueeze(-1)  # [batch, 1] -> [batch, 1, 1]

            if 'displacement_y_seq' in outputs:
                outputs['displacement_y'] = outputs['displacement_y_seq'].mean(dim=1, keepdim=True)  # [batch, 1, 1]
            elif 'error_y' in outputs:
                outputs['displacement_y'] = outputs['error_y'].unsqueeze(-1)  # [batch, 1] -> [batch, 1, 1]

            if 'displacement_z_seq' in outputs:
                outputs['displacement_z'] = outputs['displacement_z_seq'].mean(dim=1, keepdim=True)  # [batch, 1, 1]
            
            # 准备物理配置参数
            physics_params = {
                'mass_x': self.config.physics.mass_x,
                'mass_y': self.config.physics.mass_y,
                'stiffness': self.config.physics.stiffness,
                'damping': self.config.physics.damping,
                'thermal_diffusivity': self.config.physics.thermal_diffusivity,
            }
            
            # 准备输入特征（包含惯性力）用于物理损失计算
            inputs = {
                'F_inertia_x': batch_data.get('F_inertia_x'),
                'F_inertia_y': batch_data.get('F_inertia_y'),
            }
            
            # Debug: 在第一个batch时打印信息
            if batch_idx == 0 and accum_step == 0 and self.current_epoch == 0:
                print(f"\n[DEBUG Epoch {self.current_epoch+1}] Batch data keys: {list(batch_data.keys())}")
                print(f"[DEBUG] Targets keys: {list(targets.keys())}")
                print(f"[DEBUG] Outputs keys: {list(outputs.keys())}")
                print(f"[DEBUG] Has trajectory targets: {'displacement_x' in targets}")
                if 'displacement_x' in targets:
                    print(f"[DEBUG]   displacement_x target: shape={targets['displacement_x'].shape}, range=[{targets['displacement_x'].min():.6f}, {targets['displacement_x'].max():.6f}]")
                if 'displacement_x' in outputs:
                    print(f"[DEBUG]   displacement_x output: shape={outputs['displacement_x'].shape}, range=[{outputs['displacement_x'].min():.6f}, {outputs['displacement_x'].max():.6f}]")
                if 'displacement_x_seq' in outputs:
                    print(f"[DEBUG]   displacement_x_seq: shape={outputs['displacement_x_seq'].shape}")
                if 'F_inertia_x' in inputs and inputs['F_inertia_x'] is not None:
                    print(f"[DEBUG] F_inertia_x shape: {inputs['F_inertia_x'].shape}, range: [{inputs['F_inertia_x'].min():.3f}, {inputs['F_inertia_x'].max():.3f}]")
                else:
                    print(f"[DEBUG] F_inertia_x is None or missing!")
            
            losses = self.criterion(
                outputs,
                targets,
                physics_params,
                inputs
            )
            loss = losses['total'] / self.config.training.accumulation_steps

            # 反向传播（每次都要做，用于梯度累积）
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accum_step += 1

            # 梯度累积 - 只在累积完成或达到最后一个批次时更新
            if accum_step == self.config.training.accumulation_steps or (batch_idx + 1) == num_batches:
                # 梯度裁剪
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )

                # 优化器步骤
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                accum_step = 0

            # 累积损失
            total_loss += losses['total'].item()
            for key in loss_components:
                if key in losses:
                    loss_components[key] += losses[key].item()

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
            })

        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}

        return {'total': avg_loss, **avg_components}

    @torch.no_grad()
    def validate(self, data_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        验证模型

        参数:
            data_loader: 要使用的数据加载器（默认：val_loader）

        返回:
            包含验证指标的字典
        """
        if data_loader is None:
            data_loader = self.val_loader

        if data_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        loss_components = {
            'quality': 0.0,
            'fault': 0.0,
            'trajectory': 0.0,
            'physics': 0.0,
        }

        num_batches = len(data_loader)

        for batch in data_loader:
            # 将批次移到设备上
            batch_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # 构建目标字典 - 将数据集的格式转换为损失函数期望的格式
            targets = {}
            
            # 处理轨迹目标 trajectory_targets: [batch, pred_len, 2] -> displacement_x/y
            if 'trajectory_targets' in batch_data and batch_data['trajectory_targets'].numel() > 0:
                traj_targets = batch_data['trajectory_targets']  # [batch, pred_len, 2]
                # 计算平均误差作为目标（更稳定）
                targets['displacement_x'] = traj_targets[:, :, 0:1].mean(dim=1, keepdim=True)  # [batch, 1, 1]
                targets['displacement_y'] = traj_targets[:, :, 1:2].mean(dim=1, keepdim=True)  # [batch, 1, 1]
                targets['displacement_z'] = torch.zeros_like(targets['displacement_x'])  # No z in data
            
            # 处理质量目标 quality_targets: [batch, 5] 
            if 'quality_targets' in batch_data and batch_data['quality_targets'].numel() > 0:
                quality_targets = batch_data['quality_targets']  # [batch, 5]
                # 质量指标的顺序: adhesion_ratio, internal_stress, porosity, dimensional_accuracy, quality_score
                targets['quality_score'] = quality_targets[:, -1:]  # [batch, 1]

            # 前向传播
            outputs = self.model(batch_data['input_features'])
            
            # 调整模型输出的形状，将error_x/y映射到displacement_x/y
            # 模型输出 error_x/y: [batch, 1] -> 需要扩展为 [batch, 1, 1]
            if 'error_x' in outputs:
                outputs['displacement_x'] = outputs['error_x'].unsqueeze(-1)  # [batch, 1] -> [batch, 1, 1]
            if 'error_y' in outputs:
                outputs['displacement_y'] = outputs['error_y'].unsqueeze(-1)  # [batch, 1] -> [batch, 1, 1]
            if 'displacement_z' not in outputs:
                outputs['displacement_z'] = torch.zeros(outputs.get('displacement_x', targets.get('displacement_x')).shape, device=self.device)
            
            # 准备物理配置参数
            physics_params = {
                'mass_x': self.config.physics.mass_x,
                'mass_y': self.config.physics.mass_y,
                'stiffness': self.config.physics.stiffness,
                'damping': self.config.physics.damping,
                'thermal_diffusivity': self.config.physics.thermal_diffusivity,
            }
            
            # 准备输入特征（包含惯性力）用于物理损失计算
            inputs = {
                'F_inertia_x': batch_data.get('F_inertia_x'),
                'F_inertia_y': batch_data.get('F_inertia_y'),
            }
            
            losses = self.criterion(
                outputs,
                targets,
                physics_params,
                inputs
            )

            # 累积损失
            total_loss += losses['total'].item()
            for key in loss_components:
                if key in losses:
                    loss_components[key] += losses[key].item()

        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}

        return {'total': avg_loss, **avg_components}

    def train(self):
        """
        主训练循环
        """
        self.logger.info(f"开始在 {self.device} 上训练")
        self.logger.info(f"总周期数: {self.config.training.num_epochs}")
        self.logger.info(f"训练样本数: {len(self.train_loader.dataset)}")

        if self.val_loader:
            self.logger.info(f"验证样本数: {len(self.val_loader.dataset)}")

        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # 训练
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = {}
            if self.val_loader:
                val_metrics = self.validate()

            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('total', train_metrics['total']))
                else:
                    self.scheduler.step()

            # 记录指标
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start_time

            self.logger.log_metrics(train_metrics, epoch + 1, prefix="Train")
            if val_metrics:
                self.logger.log_metrics(val_metrics, epoch + 1, prefix="Val")

            self.logger.info(
                f"周期 {epoch + 1}/{self.config.training.num_epochs} - "
                f"时间: {epoch_time:.2f}s - 学习率: {current_lr:.6f}"
            )

            # 保存历史
            self.history['train_loss'].append(train_metrics['total'])
            if val_metrics:
                self.history['val_loss'].append(val_metrics['total'])

            # 保存检查点
            if (epoch + 1) % self.config.training.save_every == 0:
                self.save_checkpoint(epoch + 1, train_metrics['total'])

            # 保存最佳模型
            if val_metrics:
                if val_metrics['total'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total']
                    self.patience_counter = 0
                    self.save_checkpoint(epoch + 1, val_metrics['total'], is_best=True)
                    self.logger.info(f"新的最佳模型已保存，验证损失: {val_metrics['total']:.4f}")
                else:
                    self.patience_counter += 1

                # 提前停止
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    self.logger.info(f"在周期 {epoch + 1} 提前停止")
                    break

        self.logger.info("训练完成")

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """
        保存模型检查点

        参数:
            epoch: 当前周期
            loss: 当前损失
            is_best: 是否是迄今为止的最佳模型
        """
        checkpoint_dir = os.path.join(
            self.config.training.checkpoint_dir,
            self.config.experiment_name
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_epoch_{epoch}.pth"
        )

        self.model.save_checkpoint(
            checkpoint_path,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=loss,
        )

        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            self.model.save_checkpoint(
                best_path,
                optimizer=self.optimizer,
                epoch=epoch,
                loss=loss,
            )
            self.logger.info(f"最佳模型已保存至 {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载模型检查点

        参数:
            checkpoint_path: 检查点路径
        """
        checkpoint = self.model.load_checkpoint(
            checkpoint_path,
            optimizer=self.optimizer,
            device=self.device,
        )

        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('loss', float('inf'))

        self.logger.info(f"检查点已从 {checkpoint_path} 加载")
        self.logger.info(f"从周期 {self.current_epoch} 恢复训练")

    def test(self) -> Dict[str, float]:
        """
        测试模型

        返回:
            包含测试指标的字典
        """
        if self.test_loader is None:
            self.logger.warning("未提供测试加载器")
            return {}

        self.logger.info("开始测试...")
        test_metrics = self.validate(self.test_loader)
        self.logger.log_metrics(test_metrics, 0, prefix="Test")

        return test_metrics