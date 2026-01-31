% TEST_NEW_PARAMETERS - 验证更新后的物理参数
%
% 该脚本测试更新后的物理参数是否能生成合理的误差范围
% 目标: 误差应在 ±50-100 μm 范围内 (符合Ender-3实际精度)
%
% 引用文献:
% - Wozniak et al., Applied Sciences 2025
% - Wang et al., Robotics 2018
% - Grgić et al., Processes 2023

clear all; close all; clc;

fprintf('============================================================\n');
fprintf('测试更新后的物理参数\n');
fprintf('============================================================\n');
fprintf('\n');

%% 1. 加载更新后的物理参数
fprintf('步骤 1: 加载物理参数...\n');
params = physics_parameters();

fprintf('\n关键参数对比:\n');
fprintf('--------------------------------------------------------------\n');
fprintf('参数              | 旧值       | 新值       | 文献来源\n');
fprintf('--------------------------------------------------------------\n');
fprintf('X轴质量 (kg)      | 0.485      | %.3f     | Wozniak 2025\n', params.dynamics.x.mass);
fprintf('X轴刚度 (N/m)     | 150000     | %d        | Wang 2018\n', params.dynamics.x.stiffness);
fprintf('X轴阻尼 (N·s/m)   | 25.0       | %.1f      | Wozniak 2025\n', params.dynamics.x.damping);
fprintf('Y轴质量 (kg)      | 0.650      | %.3f     | 估算\n', params.dynamics.y.mass);
fprintf('Y轴刚度 (N/m)     | 150000     | %d        | Wang 2018\n', params.dynamics.y.stiffness);
fprintf('Y轴阻尼 (N·s/m)   | 25.0       | %.1f      | Wozniak 2025\n', params.dynamics.y.damping);
fprintf('--------------------------------------------------------------\n');

fprintf('\n系统特性:\n');
fprintf('  X轴固有频率: %.2f Hz (旧: 88 Hz)\n', params.dynamics.x.natural_freq / (2*pi));
fprintf('  Y轴固有频率: %.2f Hz (旧: 74 Hz)\n', params.dynamics.y.natural_freq / (2*pi));
fprintf('  X轴阻尼比: %.4f\n', params.dynamics.x.damping_ratio);
fprintf('  Y轴阻尼比: %.4f\n', params.dynamics.y.damping_ratio);
fprintf('\n');

%% 2. 准备测试配置
gcode_file = '../test_gcode_files/3DBenchy_PLA_1h28m.gcode';
output_dir = '../test_output_new_params';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% 3. 运行单个层仿真以验证参数
fprintf('步骤 2: 测试单个层的仿真...\n');
fprintf('  G-code: %s\n', gcode_file);

% 配置选项: 只仿真第一层
opts = struct();
opts.use_improved = true;
opts.layers = 1;  % 只仿真第1层
opts.include_skirt = false;

% 运行完整仿真
try
    simulation_data = run_full_simulation(gcode_file, [], opts);

    fprintf('\n仿真完成!\n');

    %% 4. 分析误差范围
    fprintf('\n步骤 3: 分析误差范围...\n');

    % 提取误差数据
    error_x = simulation_data.trajectory_results.error_x;
    error_y = simulation_data.trajectory_results.error_y;

    % 统计分析
    error_x_range = [min(error_x), max(error_x)];
    error_y_range = [min(error_y), max(error_y)];
    error_x_std = std(error_x);
    error_y_std = std(error_y);
    error_x_rms = rms(error_x);
    error_y_rms = rms(error_y);

    % 转换为微米
    error_x_range_um = error_x_range * 1000;
    error_y_range_um = error_y_range * 1000;
    error_x_std_um = error_x_std * 1000;
    error_y_std_um = error_y_std * 1000;
    error_x_rms_um = error_x_rms * 1000;
    error_y_rms_um = error_y_rms * 1000;

    fprintf('\n误差统计 (单位: μm):\n');
    fprintf('--------------------------------------------------\n');
    fprintf('X轴误差:\n');
    fprintf('  范围: [%.2f, %.2f] μm\n', error_x_range_um(1), error_x_range_um(2));
    fprintf('  标准差: %.2f μm\n', error_x_std_um);
    fprintf('  RMS: %.2f μm\n', error_x_rms_um);
    fprintf('\n');
    fprintf('Y轴误差:\n');
    fprintf('  范围: [%.2f, %.2f] μm\n', error_y_range_um(1), error_y_range_um(2));
    fprintf('  标准差: %.2f μm\n', error_y_std_um);
    fprintf('  RMS: %.2f μm\n', error_y_rms_um);
    fprintf('--------------------------------------------------\n');

    %% 5. 验证是否在目标范围内
    target_min = 50;   % μm
    target_max = 100;  % μm

    % 检查最大误差
    max_error_x = max(abs(error_x_range_um));
    max_error_y = max(abs(error_y_range_um));

    fprintf('\n目标验证 (目标: ±50-100 μm):\n');
    fprintf('--------------------------------------------------\n');

    if max_error_x >= target_min && max_error_x <= target_max
        fprintf('✓ X轴误差: %.2f μm - 在目标范围内!\n', max_error_x);
    elseif max_error_x < target_min
        fprintf('✗ X轴误差: %.2f μm - 太小 (目标: >=50 μm)\n', max_error_x);
        fprintf('  建议: 进一步降低刚度或质量\n');
    else
        fprintf('✗ X轴误差: %.2f μm - 太大 (目标: <=100 μm)\n', max_error_x);
        fprintf('  建议: 增加刚度或质量\n');
    end

    if max_error_y >= target_min && max_error_y <= target_max
        fprintf('✓ Y轴误差: %.2f μm - 在目标范围内!\n', max_error_y);
    elseif max_error_y < target_min
        fprintf('✗ Y轴误差: %.2f μm - 太小 (目标: >=50 μm)\n', max_error_y);
        fprintf('  建议: 进一步降低刚度或质量\n');
    else
        fprintf('✗ Y轴误差: %.2f μm - 太大 (目标: <=100 μm)\n', max_error_y);
        fprintf('  建议: 增加刚度或质量\n');
    end
    fprintf('--------------------------------------------------\n');

    %% 6. 可视化误差分布
    fprintf('\n步骤 4: 生成可视化...\n');

    figure('Position', [100, 100, 1200, 400]);

    subplot(1,3,1);
    histogram(error_x * 1000, 50, 'FaceColor', [0.2, 0.6, 0.8]);
    hold on;
    histogram(error_y * 1000, 50, 'FaceColor', [0.8, 0.4, 0.2]);
    xlabel('误差 (μm)');
    ylabel('频次');
    title('误差分布直方图');
    legend('X轴', 'Y轴');
    grid on;

    subplot(1,3,2);
    plot(error_x * 1000, 'b-', 'LineWidth', 0.5);
    hold on;
    plot(error_y * 1000, 'r-', 'LineWidth', 0.5);
    xlabel('时间步');
    ylabel('误差 (μm)');
    title('误差时间序列');
    legend('X轴', 'Y轴');
    grid on;

    subplot(1,3,3);
    scatter(error_x * 1000, error_y * 1000, 1, 'MarkerFaceAlpha', 0.3);
    xlabel('X轴误差 (μm)');
    ylabel('Y轴误差 (μm)');
    title('X-Y误差相关性');
    grid on;
    axis equal;

    % 保存图像
    saveas(gcf, fullfile(output_dir, 'error_analysis.png'));
    fprintf('  图像已保存: %s\n', fullfile(output_dir, 'error_analysis.png'));

    %% 7. 保存测试结果
    fprintf('\n步骤 5: 保存测试结果...\n');

    test_results = struct();
    test_results.params = params;
    test_results.error_x = error_x;
    test_results.error_y = error_y;
    test_results.statistics = struct();
    test_results.statistics.x_range_um = error_x_range_um;
    test_results.statistics.y_range_um = error_y_range_um;
    test_results.statistics.x_std_um = error_x_std_um;
    test_results.statistics.y_std_um = error_y_std_um;
    test_results.statistics.x_rms_um = error_x_rms_um;
    test_results.statistics.y_rms_um = error_y_rms_um;
    test_results.validation = struct();
    test_results.validation.target_min_um = target_min;
    test_results.validation.target_max_um = target_max;
    test_results.validation.x_in_range = max_error_x >= target_min && max_error_x <= target_max;
    test_results.validation.y_in_range = max_error_y >= target_min && max_error_y <= target_max;

    save(fullfile(output_dir, 'test_results.mat'), 'test_results');
    fprintf('  测试结果已保存: %s\n', fullfile(output_dir, 'test_results.mat'));

    %% 8. 总结和建议
    fprintf('\n============================================================\n');
    fprintf('测试总结\n');
    fprintf('============================================================\n');

    if test_results.validation.x_in_range && test_results.validation.y_in_range
        fprintf('✓ 参数验证成功!\n');
        fprintf('  误差范围符合Ender-3实际精度 (±50-100 μm)\n');
        fprintf('  可以使用这些参数重新生成所有数据集\n');
    else
        fprintf('✗ 参数需要调整\n');
        fprintf('  请根据上述建议修改 physics_parameters.m\n');
    end

    fprintf('\n参考文献:\n');
    fprintf('  [1] Wozniak et al., Applied Sciences, 2025, 15(24), 13140\n');
    fprintf('  [2] Wang et al., Robotics, 2018, 7(4), 75\n');
    fprintf('  [3] Grgić et al., Processes, 2023, 11(8), 2376\n');
    fprintf('============================================================\n');

catch ME
    fprintf('\n错误: 仿真失败!\n');
    fprintf('  %s\n', ME.message);
    fprintf('  位置: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
end
