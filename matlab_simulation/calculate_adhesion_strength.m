function adhesion_strength = calculate_adhesion_strength(thermal_field, params)
%% 计算层间粘结强度
% 物理模型：基于分子扩散理论
% 粘结强度与扩散时间和扩散系数相关
%
% 参考：简单模型，实际粘结强度受多种因素影响
%
% 输入：
%   thermal_field - 温度场数据
%   params - 仿真参数
% 输出：
%   adhesion_strength - 粘结强度数据

fprintf('    计算层间粘结强度...\n');

%% 1. 提取温度数据
T_interface = thermal_field.T_interface;  % 层间温度
cooling_rate = thermal_field.cooling_rate;  % 冷却速率
time_above_Tm = thermal_field.time_above_melting;  % 时间高于熔点

n_points = length(T_interface);

%% 2. 分子扩散系数（Arrhenius方程）
% D = D0 * exp(-Ea / (R * T))
% D0: 预扩散因子
% Ea: 活化能
% R: 气体常数
% T: 绝对温度

% PLA典型参数（近似）
D0 = 1e-4;  % 预扩散因子 (m^2/s)
Ea = 50e3;  % 活化能 (J/mol)
R = 8.314;  % 气体常数 (J/mol·K)

% 计算扩散系数（转换为m^2/s）
T_absolute_K = T_interface + 273.15;
D = D0 * exp(-Ea / (R * T_absolute_K));

%% 3. 粘结强度模型
% 基于分子扩散理论的简化模型
% 粘结强度与扩散深度的平方根成正比
% σ_adhesion ∝ sqrt(D * t)

% 最大粘结强度（完全熔融时的强度）
sigma_max = 30;  % MPa（PLA典型值）

% 计算扩散深度
diffusion_depth = sqrt(D .* time_above_Tm);  % m

% 归一化扩散深度（相对于层高）
normalized_diffusion = diffusion_depth / (params.layer_height / 1000);  % 无量纲

% 粘结强度（与扩散深度相关）
strength = sigma_max * (1 - exp(-3 * normalized_diffusion));

% 考虑冷却速率的影响
% 快速冷却会降低分子扩散
cooling_factor = 1 ./ (1 + 0.1 * cooling_rate);
strength = strength .* cooling_factor;

% 确保强度在合理范围内
strength = max(0, min(sigma_max, strength));

%% 4. 温度历史影响
% 层间温度越高，粘结越强
% 简化模型：线性插值
T_adhesion_min = params.T_bed + 20;  % 最小粘结温度
T_adhesion_optimal = params.melting_point - 10;  % 最佳粘结温度

temperature_factor = zeros(size(T_interface));
for i = 1:n_points
    if T_interface(i) < T_adhesion_min
        temperature_factor(i) = 0.2;  % 最低强度
    elseif T_interface(i) < T_adhesion_optimal
        % 线性增加
        temperature_factor(i) = 0.2 + 0.8 * (T_interface(i) - T_adhesion_min) / ...
                                         (T_adhesion_optimal - T_adhesion_min);
    else
        temperature_factor(i) = 1.0;  % 最大强度
    end
end

strength = strength .* temperature_factor;

%% 5. 局部粘结强度变化
% 计算粘结强度的空间梯度
strength_gradient = gradient(strength);

%% 6. 粘结强度分布统计
% 粘结强度分布（直方图数据）
strength_bins = 0:1:sigma_max;
strength_histogram = histcounts(strength, strength_bins);

%% 7. 质量指标
% 弱粘结区域（强度低于阈值的比例）
weak_bond_threshold = sigma_max * 0.5;
weak_bond_ratio = sum(strength < weak_bond_threshold) / length(strength);

% 平均粘结强度
mean_strength = mean(strength);

% 粘结强度标准差（均匀性指标）
std_strength = std(strength);

% 粘结强度变异系数
cv_strength = std_strength / mean_strength;

%% 8. 各向异性指标
% 简化：假设X方向和Y方向的粘结强度不同
% 实际应该根据打印路径方向计算
anisotropy_factor = 1.0;  % 初始假设各向同性

% 可以根据打印路径方向调整
% （这里需要G-code路径信息，暂时简化）

%% 9. 构建输出数据结构
adhesion_strength = struct();

% 主要输出：粘结强度
adhesion_strength.strength = strength;  % MPa
adhesion_strength.strength_gradient = strength_gradient;

% 扩散参数
adhesion_strength.diffusion_coefficient = D;
adhesion_strength.diffusion_depth = diffusion_depth;
adhesion_strength.time_above_melting = time_above_Tm;

% 影响因素
adhesion_strength.temperature_factor = temperature_factor;
adhesion_strength.cooling_factor = cooling_factor;
adhesion_strength.T_interface = T_interface;

% 统计信息
adhesion_strength.mean = mean_strength;
adhesion_strength.std = std_strength;
adhesion_strength.min = min(strength);
adhesion_strength.max = max(strength);
adhesion_strength.cv = cv_strength;

% 质量指标
adhesion_strength.weak_bond_ratio = weak_bond_ratio;
adhesion_strength.anisotropy_factor = anisotropy_factor;

% 分布数据
adhesion_strength.strength_bins = strength_bins;
adhesion_strength.strength_histogram = strength_histogram;

% 综合粘结质量评分 (0-1)
adhesion_quality_score = 1 - weak_bond_ratio - 0.5 * cv_strength;
adhesion_quality_score = max(0, min(1, adhesion_quality_score));
adhesion_strength.quality_score = adhesion_quality_score;

fprintf('    粘结强度计算完成\n');
fprintf('    平均强度: %.2f MPa, 标准差: %.2f MPa\n', ...
    mean_strength, std_strength);
fprintf('    弱粘结比例: %.1f%%, 质量评分: %.2f\n', ...
    weak_bond_ratio * 100, adhesion_quality_score);

end
