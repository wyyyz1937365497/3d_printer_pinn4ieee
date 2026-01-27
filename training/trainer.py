"""
Trainer class for training and evaluating models
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional, Any, List
from tqdm import tqdm

from ..utils.logger import Logger
from .losses import MultiTaskLoss


class Trainer:
    """
    Unified trainer for PINN-Seq3D models

    Handles training, validation, checkpointing, and logging
    """

    def __init__(self,
                 model: nn.Module,
                 config: Any,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 test_loader: Optional[DataLoader] = None):
        """
        Initialize trainer

        Args:
            model: Model to train
            config: Configuration object
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            test_loader: Test data loader (optional)
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Setup loss function
        self.criterion = MultiTaskLoss(
            lambda_quality=config.lambda_quality,
            lambda_fault=config.lambda_fault,
            lambda_trajectory=config.lambda_trajectory,
            lambda_physics=config.lambda_physics,
        )

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Setup gradient scaler for mixed precision
        self.scaler = GradScaler() if config.training.mixed_precision else None

        # Setup logger
        self.logger = Logger(
            name=config.experiment_name,
            log_dir=config.training.log_dir
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
        }

        self.logger.log_model_summary(model)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer"""
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
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")

        return optimizer

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
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
        Train for one epoch

        Returns:
            Dictionary with training metrics
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
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            if self.config.training.mixed_precision:
                with autocast():
                    outputs = self.model(inputs['features'])
                    losses = self.criterion(
                        outputs,
                        inputs,
                        inputs.get('physics_config', None)
                    )
                    loss = losses['total'] / self.config.training.accumulation_steps
            else:
                outputs = self.model(inputs['features'])
                losses = self.criterion(
                    outputs,
                    inputs,
                    inputs.get('physics_config', None)
                )
                loss = losses['total'] / self.config.training.accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.training.accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Accumulate losses
            total_loss += losses['total'].item()
            for key in loss_components:
                if key in losses:
                    loss_components[key] += losses[key].item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
            })

        # Compute average losses
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}

        return {'total': avg_loss, **avg_components}

    @torch.no_grad()
    def validate(self, data_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Validate the model

        Args:
            data_loader: Data loader to use (default: val_loader)

        Returns:
            Dictionary with validation metrics
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
            # Move batch to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            outputs = self.model(inputs['features'])
            losses = self.criterion(
                outputs,
                inputs,
                inputs.get('physics_config', None)
            )

            # Accumulate losses
            total_loss += losses['total'].item()
            for key in loss_components:
                if key in losses:
                    loss_components[key] += losses[key].item()

        # Compute average losses
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}

        return {'total': avg_loss, **avg_components}

    def train(self):
        """
        Main training loop
        """
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Total epochs: {self.config.training.num_epochs}")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")

        if self.val_loader:
            self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = {}
            if self.val_loader:
                val_metrics = self.validate()

            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('total', train_metrics['total']))
                else:
                    self.scheduler.step()

            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start_time

            self.logger.log_metrics(train_metrics, epoch + 1, prefix="Train")
            if val_metrics:
                self.logger.log_metrics(val_metrics, epoch + 1, prefix="Val")

            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.training.num_epochs} - "
                f"Time: {epoch_time:.2f}s - LR: {current_lr:.6f}"
            )

            # Save history
            self.history['train_loss'].append(train_metrics['total'])
            if val_metrics:
                self.history['val_loss'].append(val_metrics['total'])

            # Save checkpoint
            if (epoch + 1) % self.config.training.save_every == 0:
                self.save_checkpoint(epoch + 1, train_metrics['total'])

            # Save best model
            if val_metrics:
                if val_metrics['total'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total']
                    self.patience_counter = 0
                    self.save_checkpoint(epoch + 1, val_metrics['total'], is_best=True)
                    self.logger.info(f"New best model saved with val_loss: {val_metrics['total']:.4f}")
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        self.logger.info("Training completed")

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """
        Save model checkpoint

        Args:
            epoch: Current epoch
            loss: Current loss
            is_best: Whether this is the best model so far
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
            self.logger.info(f"Best model saved to {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint

        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = self.model.load_checkpoint(
            checkpoint_path,
            optimizer=self.optimizer,
            device=self.device,
        )

        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('loss', float('inf'))

        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}")

    def test(self) -> Dict[str, float]:
        """
        Test the model

        Returns:
            Dictionary with test metrics
        """
        if self.test_loader is None:
            self.logger.warning("No test loader provided")
            return {}

        self.logger.info("Starting testing...")
        test_metrics = self.validate(self.test_loader)
        self.logger.log_metrics(test_metrics, 0, prefix="Test")

        return test_metrics
