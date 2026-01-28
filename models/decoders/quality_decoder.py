"""
质量预测解码器头

预测在打印过程中无法直接测量的隐含质量参数：
- 层间粘合强度 (层间粘合力)
- 内应力 (内应力)
- 孔隙率 (孔隙率)
- 尺寸精度 (尺寸精度)
- 综合质量评分 (综合质量评分)

这些参数是从可观测的传感器数据（温度、振动等）中使用物理信息神经网络推断出来的。
"""

import torch
import torch.nn as nn


class QualityPredictionHead(nn.Module):
    """
    用于推断隐含质量参数的质量预测头

    该模块预测在打印过程中无法直接测量的关键质量指标。
    相反，它使用可观测的传感器数据通过物理信息学习来推断这些隐藏参数。

    输出（隐含质量参数）：
    - 层间粘合强度 (MPa)：层间粘合强度
    - 内应力 (MPa)：打印过程中累积的残余应力
    - 孔隙率 (%)：打印件中的空隙比例
    - 尺寸精度 (mm)：与预期尺寸的偏差
    - 综合质量评分 [0, 1]：综合质量指标

    主要创新：
    - 弥补了可观测（温度、振动）和不可观测（粘附、应力）质量参数之间的差距
    - 使用PINN嵌入关联这些量的物理约束
    - 通过从早期阶段数据预测最终质量来实现提前停止
    """

    def __init__(self,
                 d_model: int,
                 hidden_dims: list = [256, 128],
                 dropout: float = 0.2,
                 num_outputs: int = 5):
        """
        初始化质量预测头

        参数:
            d_model: 输入维度（来自编码器）
            hidden_dims: 隐藏层维度
            dropout: Dropout率
            num_outputs: 质量输出数量（默认：5）
        """
        super().__init__()

        self.d_model = d_model
        self.num_outputs = num_outputs

        # 构建MLP层
        layers = []
        input_dim = d_model

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(input_dim, num_outputs))

        self.mlp = nn.Sequential(*layers)

        # 初始化输出层权重
        nn.init.xavier_uniform_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, encoder_output: torch.Tensor) -> dict:
        """
        前向传播

        参数:
            encoder_output: 编码张量 [batch, seq_len, d_model]

        返回:
            包含预测质量指标的字典
        """
        # 对序列长度进行全局平均池化
        pooled = encoder_output.mean(dim=1)  # [batch, d_model]

        # 通过MLP传递
        predictions = self.mlp(pooled)  # [batch, num_outputs]

        # 分割成各个输出（隐含质量参数）
        outputs = {
            'adhesion_strength': predictions[:, 0:1],  # 层间粘合强度 (MPa)
            'internal_stress': predictions[:, 1:2],    # 内/残余应力 (MPa)
            'porosity': torch.sigmoid(predictions[:, 2:3]) * 100,  # 孔隙率 (%), [0, 100]
            'dimensional_accuracy': predictions[:, 3:4],  # 尺寸精度误差 (mm)
            'quality_score': torch.sigmoid(predictions[:, 4:5]),  # 综合质量评分 [0, 1]
        }

        return outputs

    def get_output_dim(self) -> int:
        """
        获取总输出维度

        返回:
            输出数量
        """
        return self.num_outputs


class FaultClassificationHead(nn.Module):
    """
    故障分类头

    输出:
    - 4类故障的概率:
        0: 正常
        1: 喷嘴堵塞
        2: 机械松动
        3: 电机故障
    """

    def __init__(self,
                 d_model: int,
                 hidden_dims: list = [128],
                 dropout: float = 0.3,
                 num_classes: int = 4):
        """
        初始化故障分类头

        参数:
            d_model: 输入维度（来自编码器）
            hidden_dims: 隐藏层维度
            dropout: Dropout率
            num_classes: 故障类别数量
        """
        super().__init__()

        self.d_model = d_model
        self.num_classes = num_classes

        # 构建MLP层
        layers = []
        input_dim = d_model

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim

        # 输出层 (logits)
        layers.append(nn.Linear(input_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

        # 初始化输出层权重
        nn.init.xavier_uniform_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, encoder_output: torch.Tensor) -> dict:
        """
        前向传播

        参数:
            encoder_output: 编码张量 [batch, seq_len, d_model]

        返回:
            包含故障概率的字典
        """
        # 对序列长度进行全局平均池化
        pooled = encoder_output.mean(dim=1)  # [batch, d_model]

        # 通过MLP传递得到logits
        logits = self.mlp(pooled)  # [batch, num_classes]

        # 应用softmax得到概率
        probs = torch.softmax(logits, dim=-1)

        # 获取预测类别
        preds = torch.argmax(probs, dim=-1)

        return {
            'fault_logits': logits,
            'fault_probs': probs,
            'fault_pred': preds,
        }

    def get_output_dim(self) -> int:
        """
        获取输出维度

        返回:
            类别数量
        """
        return self.num_classes