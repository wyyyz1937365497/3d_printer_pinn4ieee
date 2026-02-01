#!/usr/bin/env python3
"""快速评估当前模型性能"""
import sys
sys.path.append('experiments')

# 简单导入并评估
import torch
from pathlib import Path

checkpoint_path = Path('checkpoints/trajectory_correction/best_model.pth')
if checkpoint_path.exists():
    print(f"✓ 找到模型: {checkpoint_path}")
    print(f"  文件大小: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  修改时间: {checkpoint_path.stat().st_mtime}")

    # 尝试加载checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("\n模型信息:")
        for key in sorted(checkpoint.keys()):
            if key == 'metrics':
                print(f"  {key}:")
                for m_key, m_val in checkpoint[key].items():
                    print(f"    {m_key}: {m_val}")
            else:
                print(f"  {key}: {checkpoint[key]}")
    except Exception as e:
        print(f"  加载错误: {e}")
else:
    print("✗ 未找到模型文件")
