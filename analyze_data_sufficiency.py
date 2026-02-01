import numpy as np

# Current data statistics
train_samples = 307096
val_samples = 64275
total_samples = train_samples + val_samples
model_params = 6_186_524

# Calculate ratios
samples_per_param = total_samples / model_params
params_per_sample = model_params / total_samples

print("=" * 70)
print("数据集规模分析")
print("=" * 70)
print(f"\n模型参数: {model_params:,}")
print(f"训练样本: {train_samples:,}")
print(f"验证样本: {val_samples:,}")
print(f"总样本数: {total_samples:,}")
print(f"\n样本/参数比: {samples_per_param:.2f}")
print(f"参数/样本比: {params_per_sample:.1f}")

# Rules of thumb
print("\n" + "=" * 70)
print("经验法则评估")
print("=" * 70)

print(f"\n1. 经典规则 (每参数10-20样本):")
required_samples_conservative = model_params * 10
required_samples_ideal = model_params * 20
print(f"   保守估计: 需要 {required_samples_conservative:,} 样本")
print(f"   理想情况: 需要 {required_samples_ideal:,} 样本")
print(f"   当前样本: {total_samples:,} 样本")
print(f"   状态: {'✗ 样本不足' if total_samples < required_samples_conservative else '✓ 样本足够'}")

print(f"\n2. 现代深度学习规则:")
print(f"   对于正则化模型，每参数1-5样本即可")
print(f"   需要样本: {model_params * 1:,} - {model_params * 5:,}")
print(f"   当前样本: {total_samples:,}")
print(f"   状态: {'✓ 足够' if total_samples > model_params else '⚠ 偏少'}")

print(f"\n3. 过拟合风险评估:")
risk_ratio = params_per_sample
if risk_ratio > 50:
    risk = "❌ 高风险 - 必须增加数据或减小模型"
elif risk_ratio > 20:
    risk = "⚠️ 中等风险 - 需要强正则化"
elif risk_ratio > 10:
    risk = "✓ 低风险 - 正则化足够"
else:
    risk = "✓✓ 极低风险"
print(f"   参数/样本比: {risk_ratio:.1f}")
print(f"   风险评估: {risk}")

print(f"\n4. 训练建议:")
print(f"   • 总epoch数: {100}")
print(f"   • 每epoch训练批次: 74")
print(f"   • 总训练步数: {74 * 100}")
print(f"   • 每步样本数: 512")
print(f"   • 总梯度更新: {(307096 / 4096) * 100:.0f}")

print("\n" + "=" * 70)
print("对比参考")
print("=" * 70)

examples = [
    ("ImageNet (ResNet50)", 25.6e6, 1.28e6, 0.05),
    ("BERT Base", 110e6, 2.5e8, 2.27),
    ("GPT-2 Small", 124e6, 40e6, 0.32),
    ("本项目 (原模型)", 896e3, 371e3, 0.41),
    ("本项目 (新模型)", 6.2e6, 371e3, 0.06),
]

print(f"\n{'模型':<20} {'参数量':<12} {'样本量':<12} {'样本/参数':<12}")
print("-" * 70)
for name, params, samples, ratio in examples:
    print(f"{name:<20} {params:>11,.0f} {samples:>11,.0f} {ratio:>11.2f}")

print("\n" + "=" * 70)
print("结论")
print("=" * 70)
if samples_per_param < 0.1:
    print("\n⚠️ 数据量相对模型规模较小，建议:")
    print("  1. 使用强正则化 (dropout=0.2-0.3, weight_decay=0.01)")
    print("  2. 数据增强 (时序增强、噪声注入)")
    print("  3. 早停 (验证loss不下降即停止)")
    print("  4. 减小模型规模 (或增加数据)")
elif samples_per_param < 0.2:
    print("\n✓ 数据量适中，需要:")
    print("  1. 适度正则化")
    print("  2. 监控验证集性能")
    print("  3. 使用早停防止过拟合")
else:
    print("\n✓✓ 数据量充足，模型训练应该稳定")

print("\n当前配置评估:")
print(f"  • Dropout: 0.1-0.2 ✓")
print(f"  • Weight decay: {1e-3} ✓")
print(f"  • 早停: 建议添加")
print(f"  • 数据增强: 可选")
print()
