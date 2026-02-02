function results = simulate_trajectory_error_with_firmware_effects(trajectory_data, params, gpu_info)
% SIMULATE_TRAJECTORY_ERROR_WITH_FIRMWARE_EFFECTS - 模拟轨迹误差（包含固件效应）
%
% 该函数结合了：
% 1. 基本动力学误差（惯性和皮带弹性）
% 2. 固件效应（junction deviation, microstep resonance, timer jitter）
%
% 输入：
%   trajectory_data - 来自parse_gcode的轨迹数据
%   params          - 物理参数
%   gpu_info        - GPU信息
%
% 输出：
%   results         - 包含误差信息的结构体

    fprintf('模拟轨迹误差（包含固件效应）...\n');

    %% 首先运行基本的动力学仿真
    fprintf('  1. 基本动力学仿真...\n');
    basic_results = simulate_trajectory_error(trajectory_data, params);

    %% 添加固件效应
    fprintf('  2. 添加固件效应...\n');

    % 提取基本误差
    error_x = basic_results.error_x;
    error_y = basic_results.error_y;

    % 计算固件效应引起的额外误差
    % 注意：parse_gcode_improved 返回的字段名是 x, y, z（不是 x_ref）
    fprintf('    - Junction Deviation...\n');
    [jd_x, jd_y] = junction_deviation_effect(trajectory_data.x, trajectory_data.y, params);

    fprintf('    - Microstep Resonance...\n');
    n_points = length(trajectory_data.time);
    [mr_x, mr_y] = microstep_resonance_effect(n_points, params);

    fprintf('    - Timer Jitter...\n');
    [tj_x, tj_y] = timer_jitter_effect(trajectory_data.vx, trajectory_data.vy, params);

    % 总误差 = 基本误差 + 固件效应误差
    total_error_x = error_x + jd_x + mr_x + tj_x;
    total_error_y = error_y + jd_y + mr_y + tj_y;

    % 计算总误差幅值
    total_error_mag = sqrt(total_error_x.^2 + total_error_y.^2);

    fprintf('    ✓ 固件效应已添加\n');

    %% 更新结果结构体
    results = basic_results;

    % 保存原始（未加固件效应）误差
    results.error_x_basic = error_x;
    results.error_y_basic = error_y;
    results.error_magnitude_basic = basic_results.error_magnitude;

    % 更新为包含固件效应的总误差
    results.error_x = total_error_x;
    results.error_y = total_error_y;
    results.error_magnitude = total_error_mag;

    % 更新实际轨迹
    results.x_act = trajectory_data.x_ref + total_error_x;
    results.y_act = trajectory_data.y_ref + total_error_y;
    results.z_act = trajectory_data.z_ref;  % Z轴误差忽略

    % 保存固件效应分解（用于分析）
    results.firmware_effects = struct();
    results.firmware_effects.junction_deviation_x = jd_x;
    results.firmware_effects.junction_deviation_y = jd_y;
    results.firmware_effects.microstep_resonance_x = mr_x;
    results.firmware_effects.microstep_resonance_y = mr_y;
    results.firmware_effects.timer_jitter_x = tj_x;
    results.firmware_effects.timer_jitter_y = tj_y;

    % 打印统计
    fprintf('\n  误差统计:\n');
    fprintf('    基本误差: %.3f mm (RMS)\n', rms(error_x.^2 + error_y.^2));
    fprintf('    Junction Deviation: %.3f mm (RMS)\n', rms(jd_x.^2 + jd_y.^2));
    fprintf('    Microstep Resonance: %.3f mm (RMS)\n', rms(mr_x.^2 + mr_y.^2));
    fprintf('    Timer Jitter: %.3f mm (RMS)\n', rms(tj_x.^2 + tj_y.^2));
    fprintf('    总误差: %.3f mm (RMS)\n', rms(total_error_mag));

    fprintf('  ✓ 轨迹误差模拟完成（包含固件效应）\n\n');
end
