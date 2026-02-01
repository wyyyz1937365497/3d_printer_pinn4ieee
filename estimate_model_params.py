"""
估算轨迹误差Transformer模型参数量
"""

def estimate_transformer_params(d_model, num_heads, num_layers, d_ff, num_features, num_outputs):
    '''估算Transformer参数量'''

    # Embedding层
    embedding_params = num_features * d_model

    # 每层Transformer
    params_per_layer = (
        # Multi-head attention
        4 * d_model * d_model +  # Q, K, V, O投影
        4 * d_model              # 偏置
    ) + (
        # Feed-forward
        2 * d_model * d_ff +      # 两个线性层
        d_model + d_ff           # 偏置
    ) + (
        # Layer norms (2个)
        4 * d_model               # 2层 * (weight + bias)
    )

    transformer_params = num_layers * params_per_layer

    # LSTM解码器 (轨迹)
    lstm_hidden = 128
    lstm_layers = 2
    bidirectional = 2  # 双向LSTM

    # LSTM参数计算
    # Input gate, Forget gate, Output gate, Cell gate (4 gates)
    # Each gate has: (input_hidden + hidden_hidden) + biases
    input_to_hidden = d_model * (lstm_hidden * bidirectional)
    hidden_to_hidden = (lstm_hidden * bidirectional) * (lstm_hidden * bidirectional)
    biases = 4 * (lstm_hidden * bidirectional)

    lstm_params = lstm_layers * (input_to_hidden + hidden_to_hidden + biases)

    # 输出投影
    output_params = lstm_hidden * bidirectional * num_outputs

    total_params = int(embedding_params + transformer_params + lstm_params + output_params)

    return {
        'embedding': int(embedding_params),
        'transformer': int(transformer_params),
        'lstm': int(lstm_params),
        'output': int(output_params),
        'total': total_params
    }

# 使用实际配置
d_model = 256
num_heads = 8
num_layers = 6
d_ff = 1024
num_features = 12
num_outputs = 2  # error_x, error_y

params = estimate_transformer_params(d_model, num_heads, num_layers, d_ff, num_features, num_outputs)

print('模型参数估算 (Trajectory Error Transformer)')
print('=' * 60)
for key, value in params.items():
    print(f'{key:15s}: {value:>12,} ({value/1e6:.2f}M)')

print()
print(f'总参数: {params["total"]:>12,}')
print(f'参数量(M): {params["total"]/1e6:.2f}M')
print()

# 数据需求估算
print('=' * 60)
print('数据需求估算')
print('=' * 60)
print()

# 参数/样本比标准
standards = [
    ('优秀 (论文级)', 10, 1.0),
    ('良好 (工业级)', 20, 1.5),
    ('可接受 (最低)', 50, 2.0),
    ('不足', 100, 2.5),
]

for level, ratio, error_magnitude in standards:
    samples_needed = params['total'] / ratio
    print(f'{level:20s}: {ratio:3d}:1 = {samples_needed:>10,.0f} 样本 (误差±{error_magnitude}μm)')

print()
print('=' * 60)
print('每个gcode文件的数据贡献')
print('=' * 60)
print()

# 假设每个gcode文件的参数
gcode_configs = [
    ('3DBenchy (240层, 采样5×)', 48, 1200, 200),  # 48层 * 1200点/层 * 200样本/1000点
    ('bearing5 (75层, 全部)', 75, 1000, 200),
    ('Nautilus (56层, 全部)', 56, 2000, 200),
    ('simple_boat5 (369层, 采样5×)', 74, 1200, 200),
]

for name, layers, points_per_layer, samples_per_1k_points in gcode_configs:
    total_points = layers * points_per_layer
    total_samples = int(total_points / 1000 * samples_per_1k_points)
    print(f'{name:40s}: {total_points:>8,} 点 → {total_samples:>8,} 样本')

print()
print('=' * 60)
print('gcode文件需求估算')
print('=' * 60)
print()

# 不同质量级别需要多少个文件
avg_samples_per_file = 15000  # 平均每个文件贡献的样本数

print(f'假设每个gcode文件平均贡献: {avg_samples_per_file:,} 样本')
print()

for level, ratio, _ in standards:
    samples_needed = int(params['total'] / ratio)
    files_needed = max(1, round(samples_needed / avg_samples_per_file))
    print(f'{level:20s} ({ratio:3d}:1): 需要 {files_needed:2d} 个文件 = {samples_needed:>10,.0f} 样本')

print()
print('Current Data Collection Progress:')
print('  Existing: 4 files (3DBenchy, bearing5, Nautilus, simple_boat5)')
print('  Estimated samples: ~60,000')
ratio = params['total'] / 60000
print(f'  Param/Sample ratio: {ratio:.1f}:1 (Good)')
print()
