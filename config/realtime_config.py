"""
实时轨迹修正系统配置
"""

# 数据配置
DATA_CONFIG = {
    'seq_len': 20,          # 序列长度 (0.2秒 @ 100Hz)
    'pred_len': 1,          # 预测长度 (单步, 10ms)
    'data_pattern': 'data_simulation_*',  # 数据目录模式
    'train_split': 0.7,     # 训练集比例
    'val_split': 0.15,      # 验证集比例
}

# 模型配置
MODEL_CONFIG = {
    'input_size': 4,        # 输入维度 [x_ref, y_ref, vx_ref, vy_ref]
    'output_size': 2,       # 输出维度 [error_x, error_y]
    'hidden_size': 56,      # LSTM隐藏层大小
    'num_layers': 2,        # LSTM层数
    'dropout': 0.1,         # Dropout率
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 256,           # 批次大小
    'epochs': 100,               # 训练轮数
    'lr': 1e-3,                  # 学习率
    'weight_decay': 1e-4,        # 权重衰减
    'accumulation_steps': 2,     # 梯度累积步数
    'mixed_precision': True,     # 混合精度训练
    'num_workers': 2,            # DataLoader工作进程数
    'patience': 15,              # 早停耐心值
    'seed': 42,                  # 随机种子
}

# 学习率调度器配置
SCHEDULER_CONFIG = {
    'type': 'CosineAnnealingWarmRestarts',
    'T_0': 10,          # 第一次重启的周期长度
    'T_mult': 2,        # 每次重启后周期长度的倍数
    'eta_min': 1e-6,    # 最小学习率
}

# 评估配置
EVALUATION_CONFIG = {
    'metrics': ['r2', 'mae', 'rmse', 'correlation'],
    'inference_iterations': 1000,  # 推理性能测试迭代次数
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'output_dir': 'results/realtime_visualization',
    'dpi': 150,
    'figsize': (12, 5),
}

# 目标性能指标
TARGET_METRICS = {
    'mae': 0.05,         # mm
    'r2': 0.8,           # R² score
    'inference_time': 1.0,  # ms
    'parameters': 50000,     # 参数量上限
}
