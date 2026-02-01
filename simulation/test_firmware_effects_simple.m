% TEST_FIRMWARE_EFFECTS_SIMPLE - 简单测试固件效应（自包含轨迹）
%
% 生成测试轨迹并验证固件增强仿真是否达到0.1mm误差目标

fprintf('=== 固件效应仿真测试（自包含）===\n\n');

% 添加路径
cd('F:\TJ\3d_print\3d_printer_pinn4ieee');
addpath(genpath(fullfile(pwd, 'matlab_simulation')));

% 生成复杂测试轨迹（模拟实际打印）
fprintf('生成测试轨迹...\n');
t = linspace(0, 20, 2000)';  % 20秒，2000个点

% 混合直线、圆弧、转角
x = zeros(size(t));
y = zeros(size(t));

for i = 1:length(t)
    if t(i) < 5
        % 直线
        x(i) = 10 * t(i) / 5;
        y(i) = 0;
    elseif t(i) < 10
        % 90度转角 + 直线
        x(i) = 10;
        y(i) = 10 * (t(i) - 5) / 5;
    elseif t(i) < 15
        % 圆弧
        theta = pi/2 * (t(i) - 10) / 5;
        x(i) = 10 + 10 * cos(theta);
        y(i) = 10 + 10 * sin(theta);
    else
        % 45度转角
        x(i) = 10 + 10 * (t(i) - 15) / 5 / sqrt(2);
        y(i) = 20 + 10 * (t(i) - 15) / 5 / sqrt(2);
    end
end

% 计算速度和加速度
dt = mean(diff(t));
vx = gradient(x, dt);
vy = gradient(y, dt);
ax = gradient(vx, dt);
ay = gradient(vy, dt);

% 构建轨迹数据结构
trajectory_data = struct();
trajectory_data.time = t;
trajectory_data.x = x;
trajectory_data.y = y;
trajectory_data.z = zeros(size(t));
trajectory_data.vx = vx;
trajectory_data.vy = vy;
trajectory_data.vz = zeros(size(t));
trajectory_data.ax = ax;
trajectory_data.ay = ay;
trajectory_data.az = zeros(size(t));

fprintf('  轨迹点数: %d\n', length(t));
fprintf('  轨迹长度: %.1f mm\n', sum(sqrt(diff(x).^2 + diff(y).^2)));
fprintf('  转角数: 3 (90°, 90°, 45°)\n\n');

% 加载物理参数
fprintf('加载物理参数...\n');
params = physics_parameters();
fprintf('  X轴刚度: %.0f N/m\n', params.dynamics.x.stiffness);
fprintf('  Y轴刚度: %.0f N/m\n', params.dynamics.y.stiffness);
fprintf('  固有频率: %.1f Hz\n\n', params.dynamics.x.natural_freq / (2*pi));

% 运行固件增强仿真
fprintf('运行固件增强仿真...\n');
fprintf('========================\n');

results = simulate_trajectory_error_with_firmware_effects(trajectory_data, params);

% 分析结果
fprintf('\n=== 误差分析 ===\n');

error_mag = results.error_magnitude * 1000;  % 转换为μm

fprintf('  最大误差: %.1f μm\n', max(error_mag));
fprintf('  平均误差: %.1f μm\n', mean(error_mag));
fprintf('  RMS误差:  %.1f μm\n', rms(error_mag));
fprintf('  中位数:   %.1f μm\n', median(error_mag));
fprintf('  标准差:   %.1f μm\n', std(error_mag));

% 误差分布
fprintf('\n  误差分布:\n');
bins = [0, 50, 100, 150, Inf];
counts = histcounts(error_mag, bins);
total = length(error_mag);
fprintf('    < 50μm:   %.1f%% (%d点)\n', counts(1)/total*100, counts(1));
fprintf('    50-100μm: %.1f%% (%d点)\n', counts(2)/total*100, counts(2));
fprintf('    100-150μm: %.1f%% (%d点)\n', counts(3)/total*100, counts(3));
fprintf('    > 150μm:  %.1f%% (%d点)\n', counts(4)/total*100, counts(4));

% 各误差源RMS
fprintf('\n  各误差源RMS:\n');
if isfield(results, 'junction_deviation_x')
    jd_rms = rms(results.junction_deviation_x.^2 + results.junction_deviation_y.^2) * 1000;
    fprintf('    Junction Deviation: %.1f μm\n', jd_rms);
end
if isfield(results, 'resonance_x')
    res_rms = rms(results.resonance_x.^2 + results.resonance_y.^2) * 1000;
    fprintf('    微步谐振:          %.1f μm\n', res_rms);
end
if isfield(results, 'jitter_x')
    jit_rms = rms(results.jitter_x.^2 + results.jitter_y.^2) * 1000;
    fprintf('    定时器抖动:        %.1f μm\n', jit_rms);
end

% 可视化
fprintf('\n生成可视化...\n');
figure('Name', '固件效应测试 - 轨迹与误差', 'Position', [100, 100, 1400, 900]);

% 1. 轨迹和误差向量
subplot(2, 3, 1);
hold on;
plot(trajectory_data.x, trajectory_data.y, 'b-', 'LineWidth', 1.5);
% 每50个点画一个误差向量
skip = 50;
quiver(trajectory_data.x(1:skip:end), trajectory_data.y(1:skip:end), ...
       results.error_x(1:skip:end)*500, results.error_y(1:skip:end)*500, ...
       'r', 'LineWidth', 1.2, 'AutoScale', 'off');
hold off;
xlabel('X (mm)');
ylabel('Y (mm)');
title(sprintf('轨迹和误差向量 (放大500x)\n最大误差: %.1f μm', max(error_mag)));
grid on;
axis equal;
legend('名义轨迹', '误差向量', 'Location', 'best');

% 2. 误差幅值时间序列
subplot(2, 3, 2);
plot(trajectory_data.time, error_mag, 'b-', 'LineWidth', 1);
hold on;
yline(100, 'r--', 'LineWidth', 1.5, 'Label', '目标 (100μm)');
yline(50, 'g--', 'LineWidth', 1, 'Label', '下限 (50μm)');
hold off;
xlabel('时间 (s)');
ylabel('误差 (μm)');
title('误差幅值时间序列');
grid on;
ylim([0, max(error_mag) * 1.1]);

% 3. 误差直方图
subplot(2, 3, 3);
histogram(error_mag, 50, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'none');
hold on;
xline(100, 'r--', 'LineWidth', 1.5, 'Label', '目标');
xline(50, 'g--', 'LineWidth', 1, 'Label', '下限');
hold off;
xlabel('误差 (μm)');
ylabel('频次');
title(sprintf('误差分布 (RMS: %.1f μm)', rms(error_mag)));
grid on;

% 4. 各误差源对比
subplot(2, 3, 4);
if isfield(results, 'junction_deviation_x')
    error_sources = [
        rms(results.junction_deviation_x.^2 + results.junction_deviation_y.^2) * 1000;
        rms(results.resonance_x.^2 + results.resonance_y.^2) * 1000;
        rms(results.jitter_x.^2 + results.jitter_y.^2) * 1000
    ];
    bar(error_sources);
    set(gca, 'XTickLabel', {'Junction\\newlineDev', '微步\\newline谐振', '中断\\newline抖动'});
    ylabel('RMS误差 (μm)');
    title('各误差源贡献');
    grid on;
else
    text(0.5, 0.5, '误差源字段未找到', 'HorizontalAlignment', 'center');
end

% 5. X方向误差分量
subplot(2, 3, 5);
if isfield(results, 'junction_deviation_x')
    plot(trajectory_data.time, results.junction_deviation_x * 1000, 'r-', 'LineWidth', 0.8);
    hold on;
    plot(trajectory_data.time, results.resonance_x * 1000, 'g-', 'LineWidth', 0.8);
    plot(trajectory_data.time, results.jitter_x * 1000, 'b-', 'LineWidth', 0.8);
    plot(trajectory_data.time, results.error_x * 1000, 'k-', 'LineWidth', 1.5);
    hold off;
    xlabel('时间 (s)');
    ylabel('X误差 (μm)');
    title('X方向误差分量');
    legend('Junction Dev', '微步谐振', '中断抖动', '总误差', 'Location', 'best');
    grid on;
else
    plot(trajectory_data.time, results.error_x * 1000, 'k-', 'LineWidth', 1.5);
    xlabel('时间 (s)');
    ylabel('X误差 (μm)');
    title('X方向总误差');
    grid on;
end

% 6. 误差散点图（X vs Y）
subplot(2, 3, 6);
scatter(results.error_x * 1000, results.error_y * 1000, 10, error_mag, 'filled');
xlabel('X误差 (μm)');
ylabel('Y误差 (μm)');
title('误差散点图 (颜色=幅值)');
grid on;
colorbar;
axis equal;

% 结论
fprintf('\n=== 结论 ===\n');
target_min = 80;
target_max = 150;

if max(error_mag) >= target_min && max(error_mag) <= target_max
    fprintf('✅ 误差范围良好！\n');
    fprintf('   最大误差: %.1f μm 在目标范围 [%d, %d] μm内\n', max(error_mag), target_min, target_max);
    fprintf('   满足0.1mm数量级要求\n');
    fprintf('   ✓ 可以用于训练误差修复神经网络\n');
elseif max(error_mag) < target_min
    fprintf('⚠️ 误差偏小\n');
    fprintf('   最大误差: %.1f μm < %d μm (下限)\n', max(error_mag), target_min);
    fprintf('   建议：增大junction_deviation参数或降低系统刚度\n');
else
    fprintf('⚠️ 误差偏大\n');
    fprintf('   最大误差: %.1f μm > %d μm (上限)\n', max(error_mag), target_max);
    fprintf('   建议：减小junction_deviation参数或增大系统阻尼\n');
end

fprintf('\n测试完成！\n');
fprintf('========================\n');
