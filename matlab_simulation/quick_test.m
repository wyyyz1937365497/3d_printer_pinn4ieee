%% 快速测试脚本 - 3D打印仿真系统
% 用途：快速验证仿真系统是否正常工作
% 时间：约2-3分钟
% 样本数：5个

clear; clc; close all;

fprintf('========================================\n');
fprintf('3D打印仿真系统 - 快速测试\n');
fprintf('========================================\n\n');

%% 1. 最小参数设置（快速测试）
fprintf('设置测试参数...\n');

params = struct();
params.simulation_name = 'quick_test';
params.num_samples = 5;  % 仅5个样本
params.num_corners = 5;  % 仅5个转角

params.trajectory_type = 'random_rectangles';
params.bed_size = [100, 100];  % 小打印床
params.layer_height = 0.2;
params.num_layers = 10;

params.print_speed = 50;
params.travel_speed = 150;
params.acceleration = 1500;
params.jerk = 10;

% 传动系统（基于GT2皮带实验数据）
% 参考文献: Wang et al. (2018) MDPI Machines 7(4):75
params.mass_x = 0.5;
params.mass_y = 0.5;
params.stiffness_x = 2000000;  % GT2皮带刚度 (N/m)
params.stiffness_y = 2000000;
params.damping_x = 40;  % 阻尼 (N·s/m), ζ≈0.02
params.damping_y = 40;

% 热学参数
params.T_nozzle = 220;
params.T_bed = 60;
params.T_ambient = 25;
params.fan_speed = 255;

% 材料参数（PLA）
params.material_density = 1240;
params.material_specific_heat = 1800;
params.material_thermal_conductivity = 0.13;
params.melting_point = 150;

% 挤出参数
params.nozzle_diameter = 0.4;
params.extrusion_width = 0.45;
params.extrusion_multiplier = 1.0;

fprintf('  样本数: %d\n', params.num_samples);
fprintf('  轨迹类型: %s\n', params.trajectory_type);
fprintf('  打印床: %d x %d mm\n', params.bed_size);
fprintf('  参数设置完成\n\n');

%% 2. 运行单个样本的完整流程
fprintf('运行测试样本...\n');
fprintf('----------------------------------------\n');

% 固定随机种子（可重复）
rng(42);

%% 2.1 生成G-code轨迹
fprintf('[步骤 1/5] 生成G-code轨迹...\n');
tic;
[gcode_data, trajectory] = generate_or_parse_gcode(params);
t1 = toc;
fprintf('  完成: %.2f 秒\n', t1);
fprintf('  轨迹点数: %d\n', length(trajectory.time));
fprintf('  转角数量: %d\n\n', sum(gcode_data.is_corner));

%% 2.2 仿真轨迹误差
fprintf('[步骤 2/5] 仿真轨迹误差...\n');
tic;
trajectory_error = simulate_trajectory_error(trajectory, params);
t2 = toc;
fprintf('  完成: %.2f 秒\n', t2);
fprintf('  最大误差: %.4f mm\n', trajectory_error.max_error);
fprintf('  RMS误差: %.4f mm\n', trajectory_error.rms_error);
fprintf('  系统固有频率: %.2f rad/s\n', trajectory_error.omega_n_x);
fprintf('  阻尼比: %.3f\n\n', trajectory_error.zeta_x);

%% 2.3 仿真温度场
fprintf('[步骤 3/5] 仿真温度场...\n');
tic;
thermal_field = simulate_thermal_field(trajectory, params);
t3 = toc;
fprintf('  完成: %.2f 秒\n', t3);
fprintf('  温度范围: %.1f - %.1f °C\n', thermal_field.T_min, thermal_field.T_max);
fprintf('  平均温度: %.1f °C\n', thermal_field.T_mean);
fprintf('  平均冷却速率: %.2f °C/s\n\n', mean(thermal_field.cooling_rate));

%% 2.4 计算粘结强度
fprintf('[步骤 4/5] 计算粘结强度...\n');
tic;
adhesion_strength = calculate_adhesion_strength(thermal_field, params);
t4 = toc;
fprintf('  完成: %.2f 秒\n', t4);
fprintf('  平均粘结强度: %.2f MPa\n', adhesion_strength.mean);
fprintf('  标准差: %.2f MPa\n', adhesion_strength.std);
fprintf('  弱粘结比例: %.1f%%\n', adhesion_strength.weak_bond_ratio * 100);
fprintf('  质量评分: %.2f\n\n', adhesion_strength.quality_score);

%% 2.5 保存数据
fprintf('[步骤 5/5] 保存测试数据...\n');
tic;

% 整合数据
sample_data = struct();
sample_data.params = params;
sample_data.gcode_data = gcode_data;
sample_data.trajectory = trajectory;
sample_data.trajectory_error = trajectory_error;
sample_data.thermal_field = thermal_field;
sample_data.adhesion_strength = adhesion_strength;

% 保存
if ~exist('./output', 'dir')
    mkdir('./output');
end
test_filename = './output/quick_test_data.mat';
save(test_filename, 'sample_data', 'params', '-v7.3');

t5 = toc;
fprintf('  完成: %.2f 秒\n', t5);
fprintf('  保存位置: %s\n\n', test_filename);

%% 3. 可视化结果
fprintf('生成可视化图表...\n');

figure('Name', '3D打印仿真测试结果', 'Position', [100, 100, 1200, 800]);

% 子图1: 轨迹对比
subplot(2, 3, 1);
plot(trajectory.x_ref, trajectory.y_ref, 'b--', 'LineWidth', 1.5); hold on;
plot(trajectory_error.x_act, trajectory_error.y_act, 'r-', 'LineWidth', 1);
scatter(trajectory.x_ref(gcode_data.is_corner), trajectory.y_ref(gcode_data.is_corner), ...
    100, 'filled', 'MarkerEdgeColor', 'k');
xlabel('X (mm)');
ylabel('Y (mm)');
title('轨迹对比（参考 vs 实际）');
legend('参考轨迹', '实际轨迹', '转角', 'Location', 'best');
grid on;
axis equal;

% 子图2: 位置误差
subplot(2, 3, 2);
plot(trajectory.time, trajectory_error.epsilon_r, 'b-', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('位置误差幅值 (mm)');
title('位置误差随时间变化');
grid on;

% 子图3: 温度变化
subplot(2, 3, 3);
plot(trajectory.time, thermal_field.T_nozzle_path, 'r-', 'LineWidth', 1.5);
yline(params.melting_point, 'k--', 'LineWidth', 1);
xlabel('时间 (s)');
ylabel('温度 (°C)');
title('喷嘴路径温度');
legend('温度', '熔点', 'Location', 'best');
grid on;

% 子图4: 冷却速率
subplot(2, 3, 4);
plot(trajectory.time, thermal_field.cooling_rate, 'g-', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('冷却速率 (°C/s)');
title('冷却速率');
grid on;

% 子图5: 粘结强度
subplot(2, 3, 5);
plot(trajectory.time, adhesion_strength.strength, 'm-', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('粘结强度 (MPa)');
title('层间粘结强度');
grid on;

% 子图6: 惯性力
subplot(2, 3, 6);
plot(trajectory.time, trajectory_error.F_inertia_x, 'b-', 'LineWidth', 1); hold on;
plot(trajectory.time, trajectory_error.F_inertia_y, 'r-', 'LineWidth', 1);
xlabel('时间 (s)');
ylabel('惯性力 (N)');
title('惯性力');
legend('X方向', 'Y方向', 'Location', 'best');
grid on;

% 保存图像
saveas(gcf, './output/quick_test_results.png');
fprintf('  图像已保存: ./output/quick_test_results.png\n\n');

%% 4. 总结
fprintf('========================================\n');
fprintf('测试完成总结\n');
fprintf('========================================\n');
fprintf('各模块运行时间:\n');
fprintf('  1. G-code生成:      %.2f 秒\n', t1);
fprintf('  2. 轨迹误差仿真:    %.2f 秒\n', t2);
fprintf('  3. 温度场仿真:      %.2f 秒\n', t3);
fprintf('  4. 粘结力计算:      %.2f 秒\n', t4);
fprintf('  5. 数据保存:        %.2f 秒\n', t5);
fprintf('  总计:              %.2f 秒\n\n', t1 + t2 + t3 + t4 + t5);

fprintf('关键结果:\n');
fprintf('  最大轨迹误差: %.4f mm\n', trajectory_error.max_error);
fprintf('  平均粘结强度: %.2f MPa\n', adhesion_strength.mean);
fprintf('  弱粘结比例:   %.1f%%\n', adhesion_strength.weak_bond_ratio * 100);
fprintf('  质量评分:     %.2f / 1.00\n\n', adhesion_strength.quality_score);

fprintf('输出文件:\n');
fprintf('  数据: %s\n', test_filename);
fprintf('  图像: ./output/quick_test_results.png\n\n');

fprintf('========================================\n');
fprintf('✓ 测试成功！系统运行正常。\n');
fprintf('========================================\n');

% 估算完整仿真时间
single_sample_time = t1 + t2 + t3 + t4;
estimated_full_time = single_sample_time * 500;  % 500个样本

fprintf('\n时间估算:\n');
fprintf('  单个样本: %.1f 秒\n', single_sample_time);
fprintf('  100个样本: ~%.1f 分钟\n', single_sample_time * 100 / 60);
fprintf('  500个样本: ~%.1f 分钟\n', estimated_full_time / 60);
fprintf('  1000个样本: ~%.1f 小时\n\n', single_sample_time * 1000 / 3600);

fprintf('建议:\n');
fprintf('  - 首次运行建议使用 50-100 个样本\n');
fprintf('  - 正式训练建议使用 500+ 个样本\n');
fprintf('  - 可调整参数以适应你的打印机型号\n\n');

fprintf('运行完整仿真:\n');
fprintf('  run_full_simulation\n\n');
