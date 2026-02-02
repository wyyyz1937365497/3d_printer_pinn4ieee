% TEST_FIRMWARE_EFFECTS - 测试固件效应是否能达到0.1mm误差目标
%
% 验证新增的误差源（Junction Deviation、微步谐振、定时器抖动）
% 是否能将总误差提升到80-150μm范围

fprintf('=== 固件效应仿真测试 ===\n\n');

% 添加路径
addpath(genpath(fullfile(pwd, 'matlab_simulation')));

% 加载物理参数
params = physics_parameters();

% 查找现有的仿真数据（使用layer01）
data_dirs = dir('data_simulation_*');
if isempty(data_dirs)
    error('未找到仿真数据目录。请先运行数据收集脚本。');
end

% 尝试加载第一层
found = false;
for i = 1:length(data_dirs)
    mat_file = fullfile(data_dirs(i).name, 'layer01_ender3v2.mat');
    if exist(mat_file, 'file')
        fprintf('加载测试数据: %s\n', mat_file);
        load(mat_file, 'simulation_data');
        found = true;
        break;
    end
end

if ~found
    % 如果没有现有数据，生成一个简单的测试轨迹
    fprintf('未找到现有数据，生成测试轨迹...\n');
    t = linspace(0, 10, 1000)';
    simulation_data.x = 50 * sin(2*pi*t/5);
    simulation_data.y = 50 * cos(2*pi*t/5);
    simulation_data.vx = gradient(simulation_data.x, t);
    simulation_data.vy = gradient(simulation_data.y, t);
    simulation_data.ax = gradient(simulation_data.vx, t);
    simulation_data.ay = gradient(simulation_data.vy, t);
    simulation_data.time = t;
end

fprintf('\n运行固件增强仿真...\n');
fprintf('------------------------\n');

% 运行新的固件增强仿真
results = simulate_trajectory_error_with_firmware_effects(simulation_data, params);

% 分析结果
fprintf('\n=== 误差分析 ===\n');

error_mag = results.error_magnitude * 1000;  % 转换为μm

fprintf('  最大误差: %.1f μm\n', max(error_mag));
fprintf('  平均误差: %.1f μm\n', mean(error_mag));
fprintf('  RMS误差:  %.1f μm\n', rms(error_mag));
fprintf('  标准差:   %.1f μm\n', std(error_mag));

% 误差分布
fprintf('\n  误差分布:\n');
fprintf('    < 50μm:  %.1f%%\n', sum(error_mag < 50) / length(error_mag) * 100);
fprintf('    50-100μm: %.1f%%\n', sum(error_mag >= 50 & error_mag < 100) / length(error_mag) * 100);
fprintf('    100-150μm: %.1f%%\n', sum(error_mag >= 100 & error_mag < 150) / length(error_mag) * 100);
fprintf('    > 150μm: %.1f%%\n', sum(error_mag >= 150) / length(error_mag) * 100);

% 可视化
figure('Name', '固件效应测试', 'Position', [100, 100, 1200, 800]);

% 1. 轨迹和误差向量
subplot(2, 3, 1);
hold on;
plot(simulation_data.x, simulation_data.y, 'b-', 'LineWidth', 1);
quiver(simulation_data.x(1:50:end), simulation_data.y(1:50:end), ...
       results.error_x(1:50:end), results.error_y(1:50:end), ...
       'r', 'LineWidth', 1.5, 'AutoScale', 'on');
hold off;
xlabel('X (mm)');
ylabel('Y (mm)');
title('轨迹和误差向量');
grid on;
axis equal;

% 2. 误差幅值时间序列
subplot(2, 3, 2);
plot(simulation_data.time, error_mag, 'b-', 'LineWidth', 1);
xlabel('时间 (s)');
ylabel('误差 (μm)');
title('误差幅值时间序列');
grid on;
ylim([0, max(error_mag) * 1.1]);

% 3. 误差直方图
subplot(2, 3, 3);
histogram(error_mag, 50, 'FaceColor', [0.2, 0.6, 0.8]);
xlabel('误差 (μm)');
ylabel('频次');
title('误差分布直方图');
grid on;

% 4. 各误差源贡献
subplot(2, 3, 4);
error_sources = {
    rms(results.junction_deviation_x.^2 + results.junction_deviation_y.^2) * 1000;
    rms(results.resonance_x.^2 + results.resonance_y.^2) * 1000;
    rms(results.jitter_x.^2 + results.jitter_y.^2) * 1000
};
bar(error_sources);
set(gca, 'XTickLabel', {'Junction\\newlineDev', '微步\\newline谐振', '中断\\newline抖动'});
ylabel('RMS误差 (μm)');
title('各误差源贡献');
grid on;

% 5. X方向误差分量
subplot(2, 3, 5);
plot(simulation_data.time, results.junction_deviation_x * 1000, 'r-', 'LineWidth', 1);
hold on;
plot(simulation_data.time, results.resonance_x * 1000, 'g-', 'LineWidth', 1);
plot(simulation_data.time, results.jitter_x * 1000, 'b-', 'LineWidth', 1);
plot(simulation_data.time, results.error_x * 1000, 'k-', 'LineWidth', 1.5);
hold off;
xlabel('时间 (s)');
ylabel('X误差 (μm)');
title('X方向误差分量');
legend('Junction Dev', '微步谐振', '中断抖动', '总误差');
grid on;

% 6. 误差方向玫瑰图
subplot(2, 3, 6, 'projection', 'polar');
theta = results.error_direction;
r = error_mag;
polarhistogram(theta, 16, 'Weights', r);
title('误差方向分布');
rlim([0, max(error_mag)]);

% 结论
fprintf('\n=== 结论 ===\n');
target_min = 80;
target_max = 150;

if max(error_mag) >= target_min && max(error_mag) <= target_max
    fprintf('✅ 误差范围良好 (%.0f-%.0f μm)，满足0.1mm数量级要求\n', target_min, target_max);
    fprintf('   可以用于训练误差修复神经网络\n');
elseif max(error_mag) < target_min
    fprintf('⚠️ 误差偏小 (%.1f μm < %d μm)\n', max(error_mag), target_min);
    fprintf('   建议：增大junction_deviation参数或降低系统刚度\n');
else
    fprintf('⚠️ 误差偏大 (%.1f μm > %d μm)\n', max(error_mag), target_max);
    fprintf('   建议：减小junction_deviation参数或增大系统阻尼\n');
end

fprintf('\n测试完成！\n');
