function simulation_data = simulate_given_trajectory(gcode_file, trajectory_file, output_file, varargin)
% SIMULATE_GIVEN_TRAJECTORY 使用给定的轨迹运行动力学仿真
%
% 功能：
%   从修正后的轨迹文件加载参考轨迹，运行动力学仿真得到实际误差
%
% 输入：
%   gcode_file       - 原始G-code文件路径（用于获取参数）
%   trajectory_file  - 修正后的轨迹.mat文件
%   output_file      - 输出仿真结果路径
%
% 可选参数：
%   'UseGPU'         - 是否使用GPU加速（默认：false）
%   'UseFirmwareEffects' - 是否启用固件效应（默认：true）
%
% 输出：
%   simulation_data  - 包含以下字段的仿真数据结构体
%     .time           - 时间向量 [s]
%     .trajectory     - 参考轨迹
%     .error          - 轨迹误差
%     .params         - 仿真参数
%
% 示例：
%   % 仿真修正后的轨迹
%   simulate_given_trajectory(...
%       'test.gcode', ...
%       'corrected_trajectory.mat', ...
%       'corrected_simulation.mat', ...
%       'UseGPU', true, ...
%       'UseFirmwareEffects', true);
%
% 作者: 3D Printer PINN Project
% 日期: 2026-02-02

    %% 参数解析
    p = inputParser;
    addRequired(p, 'gcode_file', @ischar);
    addRequired(p, 'trajectory_file', @ischar);
    addRequired(p, 'output_file', @ischar);
    addParameter(p, 'UseGPU', false, @islogical);
    addParameter(p, 'UseFirmwareEffects', true, @islogical);

    parse(p, gcode_file, trajectory_file, output_file, varargin{:});

    use_gpu = p.Results.UseGPU;
    use_firmware = p.Results.UseFirmwareEffects;

    fprintf('\n=== 仿真给定的修正轨迹 ===\n');
    fprintf('G-code文件: %s\n', gcode_file);
    fprintf('轨迹文件: %s\n', trajectory_file);
    fprintf('输出文件: %s\n', output_file);
    fprintf('使用GPU: %s\n', mat2str(use_gpu));
    fprintf('固件效应: %s\n', mat2str(use_firmware));

    %% 步骤1: 加载修正后的轨迹
    fprintf('\n加载修正后的轨迹...\n');
    trajectory_data = load(trajectory_file);

    % 提取轨迹
    x_ref = trajectory_data.trajectory.x_ref(:);
    y_ref = trajectory_data.trajectory.y_ref(:);
    z_ref = trajectory_data.trajectory.z_ref(:);
    vx_ref = trajectory_data.trajectory.vx(:);
    vy_ref = trajectory_data.trajectory.vy(:);
    vz_ref = trajectory_data.trajectory.vz(:);

    n_points = length(x_ref);
    fprintf('  轨迹点数: %d\n', n_points);

    %% 步骤2: 计算加速度和jerk
    fprintf('计算加速度和jerk...\n');

    % 时间向量（假设100Hz采样）
    dt = 0.01; % 100Hz
    time = (0:n_points-1) * dt;

    % 计算加速度（差分）
    ax_ref = gradient(vx_ref, dt);
    ay_ref = gradient(vy_ref, dt);
    az_ref = gradient(vz_ref, dt);

    % 计算jerk（加速度的导数）
    jx_ref = gradient(ax_ref, dt);
    jy_ref = gradient(ay_ref, dt);
    jz_ref = gradient(az_ref, dt);

    %% 步骤3: 加载物理参数
    fprintf('加载物理参数...\n');
    params = physics_parameters();

    %% 步骤4: 运行动力学仿真
    fprintf('运行动力学仿真...\n');

    % 选择仿真函数
    if use_gpu
        fprintf('  使用GPU加速\n');
        [error_x, error_y, error_mag, error_direction] = simulate_trajectory_error_gpu(...
            x_ref, y_ref, z_ref, ...
            vx_ref, vy_ref, vz_ref, ...
            ax_ref, ay_ref, az_ref, ...
            jx_ref, jy_ref, jz_ref, ...
            params);
    else
        fprintf('  使用CPU\n');
        [error_x, error_y, error_mag, error_direction] = simulate_trajectory_error(...
            x_ref, y_ref, z_ref, ...
            vx_ref, vy_ref, vz_ref, ...
            ax_ref, ay_ref, az_ref, ...
            jx_ref, jy_ref, jz_ref, ...
            params);
    end

    fprintf('  仿真完成\n');
    fprintf('  平均误差: %.3f μm\n', mean(error_mag) * 1000);
    fprintf('  最大误差: %.3f μm\n', max(error_mag) * 1000);

    %% 步骤5: 添加固件效应（如果启用）
    if use_firmware
        fprintf('\n添加固件效应...\n');

        % Junction Deviation
        fprintf('  计算Junction Deviation...\n');
        [jd_x, jd_y] = junction_deviation_effect(x_ref, y_ref, params);
        error_x = error_x + jd_x;
        error_y = error_y + jd_y;

        % Microstep Resonance
        fprintf('  计算Microstep Resonance...\n');
        [res_x, res_y] = microstep_resonance_effect(n_points, params);
        error_x = error_x + res_x;
        error_y = error_y + res_y;

        % Timer Jitter
        fprintf('  计算Timer Jitter...\n');
        [jitter_x, jitter_y] = timer_jitter_effect(vx_ref, vy_ref, params);
        error_x = error_x + jitter_x;
        error_y = error_y + jitter_y;

        % 重新计算误差幅度
        error_mag = sqrt(error_x.^2 + error_y.^2);
        error_direction = atan2(error_y, error_x);

        fprintf('  固件效应已应用\n');
        fprintf('  平均误差（含固件）: %.3f μm\n', mean(error_mag) * 1000);
        fprintf('  最大误差（含固件）: %.3f μm\n', max(error_mag) * 1000);
    end

    %% 步骤6: 组装输出数据
    fprintf('\n组装输出数据...\n');

    simulation_data = struct();
    simulation_data.time = time;

    simulation_data.trajectory = struct();
    simulation_data.trajectory.x_ref = x_ref;
    simulation_data.trajectory.y_ref = y_ref;
    simulation_data.trajectory.z_ref = z_ref;
    simulation_data.trajectory.vx = vx_ref;
    simulation_data.trajectory.vy = vy_ref;
    simulation_data.trajectory.vz = vz_ref;
    simulation_data.trajectory.ax = ax_ref;
    simulation_data.trajectory.ay = ay_ref;
    simulation_data.trajectory.az = az_ref;
    simulation_data.trajectory.jx = jx_ref;
    simulation_data.trajectory.jy = jy_ref;
    simulation_data.trajectory.jz = jz_ref;

    simulation_data.error = struct();
    simulation_data.error.error_x = error_x;
    simulation_data.error.error_y = error_y;
    simulation_data.error.error_mag = error_mag;
    simulation_data.error.error_direction = error_direction;

    % 如果启用了固件效应，保存固件误差分量
    if use_firmware
        simulation_data.firmware_effects = struct();
        simulation_data.firmware_effects.junction_deviation_x = jd_x;
        simulation_data.firmware_effects.junction_deviation_y = jd_y;
        simulation_data.firmware_effects.resonance_x = res_x;
        simulation_data.firmware_effects.resonance_y = res_y;
        simulation_data.firmware_effects.jitter_x = jitter_x;
        simulation_data.firmware_effects.jitter_y = jitter_y;
    end

    simulation_data.params = params;

    %% 步骤7: 保存结果
    fprintf('保存仿真结果...\n');
    save(output_file, 'simulation_data', '-v7.3');
    fprintf('  已保存: %s\n', output_file);

    fprintf('\n=== 仿真完成 ===\n');

end
