"""
Performance Profiler for Training Pipeline

This script helps identify performance bottlenecks in the training process:
1. Data loading time
2. Data transfer (CPU->GPU) time
3. Forward pass time
4. Backward pass time
5. Optimizer step time
"""

import os
import sys
import time
import argparse
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from config import get_config
from models.implicit import ImplicitStateTCN, AdaptiveMultiTaskLoss
from data.simulation import PrinterSimulationDataset
from data.simulation.dataset_optimized import OptimizedPrinterSimulationDataset


class PerformanceProfiler:
    """Profile training pipeline performance"""

    def __init__(self, model, dataloader, criterion, device, use_amp=True):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # Timing statistics
        self.timings = {
            'data_loading': [],
            'cpu_to_gpu': [],
            'forward': [],
            'backward': [],
            'optimizer_step': [],
            'total_step': []
        }

        # Enable CUDA timing if available
        if torch.cuda.is_available():
            self.cuda_enabled = True
            torch.cuda.synchronize()
        else:
            self.cuda_enabled = False

    def _synchronize(self):
        """Synchronize CUDA if available"""
        if self.cuda_enabled:
            torch.cuda.synchronize()

    def profile_batch(self, batch, optimizer):
        """Profile a single training batch"""

        # Warm up
        self._synchronize()

        # 1. Data loading (already done by dataloader)
        # We measure the time it took to get the batch
        # This is approximated by the time between iterations

        # 2. CPU to GPU transfer
        start = time.perf_counter()
        input_features = batch['input_features'].to(self.device, non_blocking=True)
        quality_targets = batch['quality_targets'].to(self.device, non_blocking=True)
        targets = {
            'adhesion_strength': quality_targets[:, 0:1],
            'internal_stress': quality_targets[:, 1:2],
            'porosity': quality_targets[:, 2:3],
            'dimensional_accuracy': quality_targets[:, 3:4],
            'quality_score': quality_targets[:, 4:5],
        }
        self._synchronize()
        transfer_time = time.perf_counter() - start
        self.timings['cpu_to_gpu'].append(transfer_time)

        # 3. Forward pass
        start = time.perf_counter()
        if self.use_amp:
            with autocast('cuda'):
                outputs = self.model(input_features)
                losses = self.criterion(outputs, targets, inputs=None)
                loss = losses['total']
        else:
            outputs = self.model(input_features)
            losses = self.criterion(outputs, targets, inputs=None)
            loss = losses['total']
        self._synchronize()
        forward_time = time.perf_counter() - start
        self.timings['forward'].append(forward_time)

        # 4. Backward pass
        start = time.perf_counter()
        optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        self._synchronize()
        backward_time = time.perf_counter() - start
        self.timings['backward'].append(backward_time)

        # 5. Optimizer step
        start = time.perf_counter()
        if self.use_amp:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
        self._synchronize()
        optimizer_time = time.perf_counter() - start
        self.timings['optimizer_step'].append(optimizer_time)

    def profile_iterations(self, num_iterations=100, optimizer=None):
        """Profile multiple training iterations"""
        print(f"\nProfiling {num_iterations} iterations...")
        print("=" * 80)

        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        total_start = time.perf_counter()

        for i, batch in enumerate(self.dataloader):
            if i >= num_iterations:
                break

            self.profile_batch(batch, optimizer)

            if (i + 1) % 20 == 0:
                print(f"  Profiled {i + 1}/{num_iterations} iterations...")

        total_time = time.perf_counter() - total_start
        avg_total = total_time / num_iterations
        self.timings['total_step'] = [avg_total] * num_iterations

        print(f"  Completed {num_iterations} iterations in {total_time:.2f}s")
        print()

    def print_summary(self):
        """Print profiling summary"""
        print("=" * 80)
        print("PERFORMANCE PROFILING SUMMARY")
        print("=" * 80)

        for name, timings in self.timings.items():
            if not timings:
                continue
            avg = sum(timings) / len(timings)
            total = sum(timings)
            percentage = (total / sum(sum(t) for t in self.timings.values() if t)) * 100

            print(f"\n{name.replace('_', ' ').title()}:")
            print(f"  Average: {avg*1000:.2f} ms")
            print(f"  Total:   {total:.2f} s")
            print(f"  % of total: {percentage:.1f}%")

        # Estimate throughput
        if self.timings['total_step']:
            avg_step = sum(self.timings['total_step']) / len(self.timings['total_step'])
            batch_size = next(iter(self.dataloader))['input_features'].size(0)
            throughput = batch_size / avg_step

            print("\n" + "=" * 80)
            print(f"Estimated Throughput: {throughput:.0f} samples/s")
            print(f"Estimated Steps/sec: {1/avg_step:.2f}")
            print("=" * 80)

        # Bottleneck analysis
        print("\nBOTTLENECK ANALYSIS:")
        print("-" * 80)

        component_times = {
            name: sum(timings) for name, timings in self.timings.items() if timings
        }
        total_time = sum(component_times.values())
        sorted_components = sorted(component_times.items(), key=lambda x: x[1], reverse=True)

        for i, (name, time) in enumerate(sorted_components[:3]):
            percentage = (time / total_time) * 100
            print(f"  {i+1}. {name.replace('_', ' ').title()}: {percentage:.1f}% of total time")

        print()

        # Recommendations
        print("OPTIMIZATION RECOMMENDATIONS:")
        print("-" * 80)

        total_transfer = sum(self.timings.get('cpu_to_gpu', []))
        total_forward = sum(self.timings.get('forward', []))
        total_backward = sum(self.timings.get('backward', []))
        total_data = total_transfer + total_forward + total_backward
        total_all = sum(component_times.values())

        # Data loading bottleneck
        if 'data_loading' in self.timings and self.timings['data_loading']:
            data_loading_time = sum(self.timings['data_loading'])
            if data_loading_time / total_all > 0.3:
                print("  [DATA LOADING] - >30% of time:")
                print("    - Increase num_workers")
                print("    - Use optimized dataset with pre-normalization")
                print("    - Enable cache_in_memory if RAM is sufficient")

        # Transfer bottleneck
        if total_transfer / total_all > 0.2:
            print("  [CPU->GPU TRANSFER] - >20% of time:")
            print("    - Use pin_memory=True (already enabled)")
            print("    - Use non_blocking=True (already enabled)")
            print("    - Reduce data preprocessing on CPU")

        # GPU compute bottleneck
        if total_data / total_all > 0.5:
            gpu_util = (total_forward + total_backward) / total_data
            if gpu_util > 0.7:
                print("  [GPU COMPUTE] - High GPU utilization (good!):")
                print("    - Consider increasing batch size")
                print("    - Use torch.compile (already available)")
                print("    - Enable mixed precision (already enabled)")
            else:
                print("  [GPU UNDERUTILIZED]:")
                print("    - Increase batch size")
                print("    - Check for CPU bottlenecks")
                print("    - Profile with PyTorch profiler for details")

        print()


def main():
    parser = argparse.ArgumentParser(description='Profile training performance')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_optimized_dataset', action='store_true',
                       help='Use optimized dataset with pre-normalization')
    parser.add_argument('--cache_data', action='store_true',
                       help='Cache dataset in memory')
    parser.add_argument('--num_iterations', type=int, default=100,
                       help='Number of iterations to profile')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    print("=" * 80)
    print("PERFORMANCE PROFILING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num workers: {args.num_workers}")
    print(f"  Optimized dataset: {args.use_optimized_dataset}")
    print(f"  Cache data: {args.cache_data}")
    print(f"  Device: {args.device}")
    print()

    device = torch.device(args.device)

    # Load config
    config = get_config(preset='unified')
    config.training.batch_size = args.batch_size

    # Build dataloader
    print("Building dataloader...")
    import glob
    import random

    mat_files = glob.glob(os.path.join(args.data_dir, "*.mat"))
    random.shuffle(mat_files)
    train_files = mat_files[:int(0.7 * len(mat_files))]

    if args.use_optimized_dataset:
        print("Using OPTIMIZED dataset...")
        dataset = OptimizedPrinterSimulationDataset(
            train_files,
            seq_len=config.data.seq_len,
            pred_len=config.data.pred_len,
            stride=config.data.stride,
            mode='train',
            scaler=None,
            fit_scaler=True,
            cache_in_memory=args.cache_data
        )
    else:
        print("Using STANDARD dataset...")
        dataset = PrinterSimulationDataset(
            train_files,
            seq_len=config.data.seq_len,
            pred_len=config.data.pred_len,
            stride=config.data.stride,
            mode='train',
            scaler=None,
            fit_scaler=True
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=max(2, args.num_workers) if args.num_workers > 0 else None
    )

    print(f"  Samples: {len(dataset)}")
    print(f"  Batches: {len(dataloader)}")
    print()

    # Create model
    print("Creating model...")
    model = ImplicitStateTCN(config).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Create loss
    criterion = AdaptiveMultiTaskLoss(
        lambda_adhesion=1.0,
        lambda_stress=1.0,
        lambda_porosity=1.0,
        lambda_accuracy=1.0,
        lambda_quality=1.0,
        lambda_physics=0.5,
        use_adaptive_weights=False
    )

    # Create profiler
    profiler = PerformanceProfiler(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        use_amp=True
    )

    # Profile
    profiler.profile_iterations(num_iterations=args.num_iterations)

    # Print summary
    profiler.print_summary()


if __name__ == '__main__':
    main()
