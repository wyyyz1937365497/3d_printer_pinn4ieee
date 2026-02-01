"""
实时轨迹修正器 - 轻量级版本

设计目标:
- 输入: 4维 [x_ref, y_ref, vx_ref, vy_ref]
- 输出: 2维 [error_x, error_y]
- 推理时间: < 1ms
- 参数量: ~38K
"""

import torch
import torch.nn as nn
from typing import Dict


class RealTimeCorrector(nn.Module):
    """
    轻量级实时轨迹误差预测器

    架构:
        4维输入 → 32维编码 → 2层LSTM(56) → 2维输出
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 56,  # 优化: 减少以保持总参数 < 50K
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        初始化实时修正器

        Args:
            input_size: 输入特征维度 (默认4: x_ref, y_ref, vx_ref, vy_ref)
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout率
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # === 特征编码器 ===
        # 将4维输入映射到32维空间
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # === 时序建模器 (LSTM) ===
        # 捕获轨迹的动态模式
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # === 误差预测器 ===
        # 基于LSTM隐状态预测误差
        self.decoder = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [batch, seq_len, input_size] 输入特征序列

        Returns:
            output: [batch, 2] 预测的x,y误差
        """
        batch_size, seq_len, _ = x.shape

        # 1. 特征编码
        x = self.encoder(x)  # [batch, seq_len, 32]

        # 2. LSTM提取时序特征
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: [batch, seq_len, hidden_size]
        # h_n: [num_layers, batch, hidden_size]

        # 3. 取最后时间步的隐状态
        # 代表当前系统状态
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_size]

        # 4. 预测误差
        output = self.decoder(last_hidden)  # [batch, 2]

        return output

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        单步预测 (用于实时推理)

        Args:
            x: [batch, seq_len, input_size] 输入特征序列

        Returns:
            output: [batch, 2] 预测的x,y误差
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
        return output

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # 计算每层的参数量
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        lstm_params = sum(p.numel() for p in self.lstm.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())

        return {
            'model_type': 'RealTimeCorrector',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'encoder_params': encoder_params,
            'lstm_params': lstm_params,
            'decoder_params': decoder_params,
        }


# === 测试代码 ===
if __name__ == '__main__':
    import time

    print("=" * 80)
    print("测试实时轨迹修正器")
    print("=" * 80)

    # 创建模型
    model = RealTimeCorrector(
        input_size=4,
        hidden_size=56,  # 优化: 减少以保持总参数 < 50K
        num_layers=2,
        dropout=0.1
    )

    # 打印模型信息
    info = model.get_model_info()
    print("\n模型配置:")
    print("-" * 80)
    print(f"  输入维度: {info['input_size']}")
    print(f"  隐藏层大小: {info['hidden_size']}")
    print(f"  LSTM层数: {info['num_layers']}")
    print(f"\n参数量:")
    print(f"  编码器: {info['encoder_params']:,}")
    print(f"  LSTM: {info['lstm_params']:,}")
    print(f"  解码器: {info['decoder_params']:,}")
    print(f"  总计: {info['total_parameters']:,}")

    # 验证参数量目标
    if info['total_parameters'] < 50000:
        print(f"  ✓ 参数量满足要求 (< 50K)")
    else:
        print(f"  ✗ 参数量超标 (需要 < 50K)")

    # 测试前向传播
    print("\n前向传播测试:")
    print("-" * 80)
    batch_size = 4
    seq_len = 20
    input_size = 4

    x = torch.randn(batch_size, seq_len, input_size)
    print(f"  输入shape: {x.shape}")

    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"  输出shape: {output.shape}")
    print(f"  ✓ 前向传播成功")

    # 测试推理速度
    print("\n推理性能测试:")
    print("-" * 80)

    # 预热
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    # 测试单样本推理时间 (batch_size=1)
    x_single = torch.randn(1, seq_len, input_size)

    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = model(x_single)

        # 计时
        if torch.cuda.is_available():
            model = model.cuda()
            x_single = x_single.cuda()
            torch.cuda.synchronize()

        start = time.perf_counter()
        num_iterations = 1000
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x_single)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()

    avg_time_ms = (end - start) / num_iterations * 1000  # ms
    throughput = num_iterations / (end - start)

    print(f"  平均推理时间: {avg_time_ms:.3f} ms")
    print(f"  吞吐量: {throughput:.0f} inferences/sec")

    # 验证实时性目标
    if avg_time_ms < 1.0:
        print(f"  ✓ 满足实时性要求 (< 1ms)")
    else:
        print(f"  ✗ 不满足实时性要求 (需要 < 1ms)")

    # 测试批量推理
    print("\n批量推理测试:")
    print("-" * 80)
    batch_sizes = [1, 4, 16, 64, 256]

    for bs in batch_sizes:
        x_batch = torch.randn(bs, seq_len, input_size)

        with torch.no_grad():
            if torch.cuda.is_available():
                x_batch = x_batch.cuda()

            start = time.perf_counter()
            for _ in range(100):
                _ = model(x_batch)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()

        avg_time = (end - start) / 100 * 1000  # ms
        throughput_batch = bs / (end - start) * 1000

        print(f"  batch_size={bs:3d}: {avg_time:6.3f} ms ({throughput_batch:6.0f} inf/s)")

    print("\n" + "=" * 80)
    print("✓ 所有测试完成")
    print("=" * 80)
