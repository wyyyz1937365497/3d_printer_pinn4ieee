% TEST_TRAJECTORY_ERROR - 测试轨迹误差范围
%
% 快速验证物理参数是否生成±0.1mm的轨迹误差
%
% Author: 3D Printer PINN Project
% Date: 2026-01-31

cd('F:\TJ\3d_print\3d_printer_pinn4ieee');
addpath(genpath(fullfile(pwd, 'matlab_simulation')));

fprintf('\n');
fprintf('============================================================\n');
fprintf('轨迹误差范围测试\n');
fprintf('============================================================\n');
fprintf('\n');

%% 加载物理参数
params = physics_parameters();

fprintf('当前物理参数:\n');
fprintf('  X轴 - 质量: %.2f kg, 刚度: %.0f N/m, 阻尼: %.1f N·s/m\n', ...
    params.dynamics.x.mass, params.dynamics.x.stiffness, params.dynamics.x.damping);
fprintf('  Y轴 - 质量: %.2f kg, 刚度: %.0f N/m, 阻尼: %.1f N·s/m\n', ...
    params.dynamics.y.mass, params.dynamics.y.stiffness, params.dynamics.y.damping);
fprintf('  X轴固有频率: %.1f Hz\n', params.dynamics.x.natural_freq / (2*pi));
fprintf('  Y轴固有频率: %.1f Hz\n', params.dynamics.y.natural_freq / (2*pi));
fprintf('\n');

%% 简单轨迹测试
fprintf('创建测试轨迹（高加速度场景）...\n');

% 创建高加速度轨迹
t = linspace(0, 1, 1000)';
trajectory = struct();
trajectory.time = t;

% 急转弯轨迹（最大加速度）
trajectory.x = sin(2*pi*5*t) * 50;  % 50mm振幅
trajectory.y = cos(2*pi*5*t) * 50;
trajectory.z = ones(size(t)) * 0.2;
trajectory.vx = gradient(trajectory.x, t);
trajectory.vy = gradient(trajectory.y, t);
trajectory.vz = zeros(size(t));
trajectory.ax = gradient(trajectory.vx, t);
trajectory.ay = gradient(trajectory.vy, t);
trajectory.az = zeros(size(t));
trajectory.jx = gradient(trajectory.ax, t);
trajectory.jy = gradient(trajectory.ay, t);
trajectory.jz = zeros(size(t));
trajectory.is_extruding = true(size(t));
trajectory.print_type = repmat({'Outer wall'}, size(t));
trajectory.layer_num = ones(size(t));

fprintf('  轨迹点数: %d\n', length(t));
fprintf('  X加速度范围: [%.1f, %.1f] mm/s²\n', min(trajectory.ax), max(trajectory.ax));
fprintf('  Y加速度范围: [%.1f, %.1f] mm/s²\n', min(trajectory.ay), max(trajectory.ay));
fprintf('\n');

%% 运行轨迹误差仿真
fprintf('运行轨迹误差仿真...\n');

gpu_info = setup_gpu(1);
if gpu_info.use_gpu
    fprintf('  使用GPU: %s\n', gpu_info.device_name);
else
    fprintf('  使用CPU\n');
end

tic;
results = simulate_trajectory_error(trajectory, params);
elapsed = toc;

fprintf('  仿真用时: %.2f 秒\n', elapsed);
fprintf('\n');

%% 分析误差
fprintf('误差统计:\n');
fprintf('  X误差范围: [%.3f, %.3f] mm\n', min(results.error_x), max(results.error_x));
fprintf('  Y误差范围: [%.3f, %.3f] mm\n', min(results.error_y), max(results.error_y));
fprintf('  X最大误差: %.3f mm\n', max(abs(results.error_x)));
fprintf('  Y最大误差: %.3f mm\n', max(abs(results.error_y)));
fprintf('  误差幅值最大: %.3f mm\n', max(results.error_magnitude));
fprintf('  误差幅值RMS: %.3f mm\n', rms(results.error_magnitude));
fprintf('\n');

%% 评估结果
fprintf('============================================================\n');
fprintf('评估结果:\n');
fprintf('============================================================\n');

max_error = max(results.error_magnitude) * 1000;  % 转换为μm
rms_error = rms(results.error_magnitude) * 1000;

fprintf('  最大误差: %.0f μm\n', max_error);
fprintf('  RMS误差: %.0f μm\n', rms_error);
fprintf('\n');

if max_error >= 80 && max_error <= 150
    fprintf('✅ 误差范围正常 (80-150μm)\n');
    fprintf('   符合±0.1mm的目标范围\n');
elseif max_error < 80
    fprintf('⚠️  误差偏小 (<80μm)\n');
    fprintf('   建议进一步降低刚度或阻尼\n');
else
    fprintf('⚠️  误差偏大 (>150μm)\n');
    fprintf('   建议增加刚度或阻尼\n');
end

fprintf('\n');

%% 保存测试数据
test_output = 'test_trajectory_error.mat';
save(test_output, 'trajectory', 'results', 'params', '-v7.3');
fprintf('测试数据已保存: %s\n', test_output);
fprintf('\n');

fprintf('============================================================\n');
fprintf('测试完成\n');
fprintf('============================================================\n');
fprintf('\n');
