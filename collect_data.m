% COLLECT_DATA_OPTIMIZED_V2 - Optimized data collection with trajectory reconstruction
%
% Key improvements:
% 1. Reconstructs full motion trajectory (not just waypoints)
% 2. Physics-based thermal accumulation model
% 3. Dense time interpolation (0.01s time step)
%
% Author: 3D Printer PINN Project
% Date: 2026-01-27

%% Setup
cd('F:\TJ\3d_print\3d_printer_pinn4ieee')
addpath(genpath(fullfile(pwd, 'matlab_simulation')))

fprintf('\n');
fprintf('============================================================\n');
fprintf('优化的数据收集策略 V2（含轨迹重建）\n');
fprintf('============================================================\n');
fprintf('\n');
fprintf('关键改进：\n');
fprintf('  ✅ 完整的S曲线运动轨迹重建\n');
fprintf('  ✅ 密集时间插值（0.01秒采样）\n');
fprintf('  ✅ 物理驱动的热累积模型\n');
fprintf('  ✅ 单层参数扫描 + 三层验证\n');
fprintf('\n');

%% GPU Configuration
gpu_id = 1;  % Use cuda1
fprintf('GPU配置：cuda1 (GPU %d)\n', gpu_id);
fprintf('\n');

%% Strategy Selection
strategy = 2;  % Standard dataset

%% Parameters
target_layer = 25;  % Middle layer (representative)

switch strategy
    case 1  % Quick validation
        fprintf('策略：快速验证（适合测试）\n');
        n_param_combos = 10;
        validation_layers = [1, 50];
        fprintf('目标：~10K 样本\n');
        fprintf('预计时间：30 分钟\n');

    case 2  % Standard (RECOMMENDED)
        fprintf('策略：标准数据集（推荐）\n');
        n_param_combos = 100;
        validation_layers = [1, 25, 50];
        fprintf('目标：~100K 样本（含增强）\n');
        fprintf('预计时间：1.5 小时\n');

    case 3  % Complete
        fprintf('策略：完整数据集（最大规模）\n');
        n_param_combos = 180;
        validation_layers = [1, 10, 25, 40, 50];
        fprintf('目标：~180K 样本（含增强）\n');
        fprintf('预计时间：3 小时\n');
end

fprintf('\n');

%% Parameter Grid Definition
accel_grid = 200:100:500;
velocity_grid = 100:100:400;
fan_grid = [0, 128, 255];
ambient_temp_grid = [20, 25, 30];

[accel_vals, vel_vals, fan_vals, temp_vals] = ...
    ndgrid(accel_grid, velocity_grid, fan_grid, ambient_temp_grid);

total_combos_available = length(accel_vals);
fprintf('可用参数组合：%d\n', total_combos_available);

if n_param_combos < total_combos_available
    fprintf('采样：%d 种组合\n', n_param_combos);
    indices = randperm(total_combos_available, n_param_combos);
else
    indices = 1:total_combos_available;
    n_param_combos = total_combos_available;
end

fprintf('实际仿真：%d 种配置\n', n_param_combos);
fprintf('\n');

%% Phase 1: Single Layer Parameter Scan
fprintf('============================================================\n');
fprintf('阶段1：单层参数扫描（主力数据）\n');
fprintf('============================================================\n');
fprintf('层号：%d\n', target_layer);
fprintf('参数组合：%d\n', n_param_combos);
fprintf('\n');

tic;
sim_count = 0;

output_dir = sprintf('data_simulation_layer%d', target_layer);
mkdir(output_dir);

for idx = 1:n_param_combos
    combo_idx = indices(idx);

    accel = accel_vals(combo_idx);
    velocity = vel_vals(combo_idx);
    fan = fan_vals(combo_idx);
    ambient_temp = temp_vals(combo_idx);

    fprintf('[%d/%d] a=%d, v=%d, fan=%d, T=%d°C...', ...
            idx, n_param_combos, accel, velocity, fan, ambient_temp);

    try
        params = physics_parameters();
        params.debug.verbose = false;  % 关闭图表
        params.motion.max_accel = accel;
        params.motion.max_velocity = velocity;

        if fan == 0
            params.heat_transfer.h_convection_with_fan = 10;
        elseif fan == 128
            params.heat_transfer.h_convection_with_fan = 25;
        else
            params.heat_transfer.h_convection_with_fan = 44;
        end

        params.environment.ambient_temp = ambient_temp;

        output_file = fullfile(output_dir, ...
            sprintf('combo_%04d_a%03d_v%03d_f%03d_t%02d.mat', ...
                    idx, accel, velocity, fan, ambient_temp));

        % Use new trajectory reconstruction
        options = struct();
        options.layers = target_layer;
        options.time_step = 0.01;  % 10ms time step
        options.include_type = {'Outer wall', 'Inner wall'};
        options.include_skirt = false;

        % Run simulation with trajectory reconstruction
        simulation_data = run_full_simulation_with_reconstruction(...
            'Tremendous Hillar_PLA_17m1s.gcode', ...
            output_file, ...
            options, ...
            params, ...
            gpu_id);

        sim_count = sim_count + 1;
        fprintf(' 完成\n');

        if mod(idx, 10) == 0
            fprintf('进度：%.1f%%\n', 100*idx/n_param_combos);
        end

    catch ME
        fprintf(' 错误：%s\n', ME.message);
        fprintf('错误位置：%s\n', ME.stack(1).name);
        fprintf('错误行号：%d\n', ME.stack(1).line);
        fprintf('\n');
        fprintf('============================================================\n');
        fprintf('发生错误，终止执行！\n');
        fprintf('============================================================\n');
        rethrow(ME);
    end
end

phase1_time = toc;
fprintf('\n阶段1完成！\n');
fprintf('仿真次数：%d\n', sim_count);
fprintf('用时：%.2f 分钟\n', phase1_time/60);
fprintf('\n');

%% Phase 2: Multi-layer validation
fprintf('============================================================\n');
fprintf('阶段2：多层验证（层间效应）\n');
fprintf('============================================================\n');
fprintf('层数：%s\n', mat2str(validation_layers));
fprintf('\n');

val_accel = [300, 400, 500];
val_velocity = [200, 300, 400];
val_fan = [0, 255];

[val_acc, val_vel, val_fan] = ndgrid(val_accel, val_velocity, val_fan);
n_val_combos = length(val_acc);

fprintf('每层参数：%d 种\n', n_val_combos);
fprintf('总验证仿真：%d 层 × %d 配置 = %d 次\n', ...
        length(validation_layers), n_val_combos, ...
        length(validation_layers) * n_val_combos);
fprintf('\n');

tic;
val_sim_count = 0;

for layer_idx = 1:length(validation_layers)
    layer_num = validation_layers(layer_idx);

    fprintf('验证层 %d (%d/%d)...\n', layer_num, layer_idx, length(validation_layers));

    for v_idx = 1:n_val_combos
        accel = val_acc(v_idx);
        velocity = val_vel(v_idx);
        fan = val_fan(v_idx);

        fprintf('  [%d/%d] a=%d, v=%d, fan=%d...', ...
                v_idx, n_val_combos, accel, velocity, fan);

        try
            params = physics_parameters();
            params.debug.verbose = false;
            params.motion.max_accel = accel;
            params.motion.max_velocity = velocity;

            if fan == 0
                params.heat_transfer.h_convection_with_fan = 10;
            else
                params.heat_transfer.h_convection_with_fan = 44;
            end

            options = struct();
            options.layers = layer_num;
            options.time_step = 0.01;
            options.include_type = {'Outer wall', 'Inner wall'};
            options.include_skirt = false;

            output_file = sprintf('validation_layer%d_%03d.mat', layer_num, v_idx);

            simulation_data = run_full_simulation_with_reconstruction(...
                'Tremendous Hillar_PLA_17m1s.gcode', ...
                output_file, ...
                options, ...
                params, ...
                gpu_id);

            val_sim_count = val_sim_count + 1;
            fprintf(' 完成\n');

        catch ME
            fprintf(' 错误：%s\n', ME.message);
            fprintf('错误位置：%s\n', ME.stack(1).name);
            fprintf('错误行号：%d\n', ME.stack(1).line);
            fprintf('\n');
            fprintf('============================================================\n');
            fprintf('发生错误，终止执行！\n');
            fprintf('============================================================\n');
            rethrow(ME);
        end
    end
end

phase2_time = toc;
fprintf('\n阶段2完成！\n');
fprintf('仿真次数：%d\n', val_sim_count);
fprintf('用时：%.2f 分钟\n', phase2_time/60);
fprintf('\n');

%% Summary
fprintf('============================================================\n');
fprintf('仿真总结\n');
fprintf('============================================================\n');
fprintf('\n');

total_time = (phase1_time + phase2_time) / 60;

fprintf('阶段1（单层扫描）：\n');
fprintf('  仿真次数：%d\n', sim_count);
fprintf('\n');

fprintf('阶段2（多层验证）：\n');
fprintf('  仿真次数：%d\n', val_sim_count);
fprintf('\n');

fprintf('时间统计：\n');
fprintf('  阶段1：%.2f 分钟\n', phase1_time/60);
fprintf('  阶段2：%.2f 分钟\n', phase2_time/60);
fprintf('  总计：%.2f 分钟 (%.2f 小时)\n', total_time, total_time/60);
fprintf('\n');

fprintf('下一步：\n');
fprintf('  1. 检查生成的数据文件\n');
fprintf('  2. 验证样本数量和密度\n');
fprintf('  3. 转换为Python格式进行训练\n');
fprintf('\n');

fprintf('============================================================\n');
fprintf('数据收集完成！\n');
fprintf('============================================================\n');

%% Helper function: Run simulation with trajectory reconstruction
function simulation_data = run_full_simulation_with_reconstruction(...
    gcode_file, output_file, options, params, gpu_id)

    % Step 1: Reconstruct trajectory
    trajectory_data = reconstruct_trajectory(gcode_file, params, options);

    % Step 2: Simulate trajectory error
    n_points = length(trajectory_data.time);

    gpu_info = setup_gpu(gpu_id);

    if gpu_info.use_gpu && n_points > 1000
        fprintf('  使用GPU加速仿真\n');
        trajectory_results = simulate_trajectory_error_gpu(trajectory_data, params, gpu_info);
    else
        fprintf('  使用CPU仿真\n');
        trajectory_results = simulate_trajectory_error(trajectory_data, params);
    end

    % Step 3: Simulate thermal field
    thermal_results = simulate_thermal_field(trajectory_data, params);

    % Step 4: Calculate adhesion strength
    adhesion_results = calculate_adhesion_strength(thermal_results, params);

    % Step 5: Combine results
    simulation_data = combine_results(trajectory_data, trajectory_results, ...
                                     thermal_results, adhesion_results, params);

    % Step 6: Save
    if nargin >= 2 && ~isempty(output_file)
        save(output_file, 'simulation_data', '-v7.3');
        fprintf('  已保存：%s\n', output_file);
    end
end

%% Helper function: Combine results
function simulation_data = combine_results(trajectory, trajectory_results, ...
                                          thermal, adhesion, params)

    t = trajectory.time;
    n_points = length(t);

    simulation_data = [];

    % Time
    simulation_data.time = t(:);

    % Reference trajectory
    simulation_data.x_ref = trajectory.x(:);
    simulation_data.y_ref = trajectory.y(:);
    simulation_data.z_ref = trajectory.z(:);

    % Actual trajectory (with dynamics)
    simulation_data.x_act = trajectory_results.x_act(:);
    simulation_data.y_act = trajectory_results.y_act(:);
    simulation_data.z_act = trajectory_results.z_act(:);

    % Kinematics (from reconstructed trajectory)
    simulation_data.vx_ref = trajectory.vx(:);
    simulation_data.vy_ref = trajectory.vy(:);
    simulation_data.vz_ref = trajectory.vz(:);
    simulation_data.v_mag_ref = sqrt(trajectory.vx(:).^2 + trajectory.vy(:).^2 + trajectory.vz(:).^2);

    simulation_data.ax_ref = trajectory.ax(:);
    simulation_data.ay_ref = trajectory.ay(:);
    simulation_data.az_ref = trajectory.az(:);
    simulation_data.a_mag_ref = sqrt(trajectory.ax(:).^2 + trajectory.ay(:).^2 + trajectory.az(:).^2);

    simulation_data.jx_ref = trajectory.jx(:);
    simulation_data.jy_ref = trajectory.jy(:);
    simulation_data.jz_ref = trajectory.jz(:);
    simulation_data.jerk_mag = sqrt(trajectory.jx(:).^2 + trajectory.jy(:).^2 + trajectory.jz(:).^2);

    % Dynamics
    simulation_data.F_inertia_x = trajectory_results.F_inertia_x(:);
    simulation_data.F_inertia_y = trajectory_results.F_inertia_y(:);
    simulation_data.F_elastic_x = trajectory_results.F_elastic_x(:);
    simulation_data.F_elastic_y = trajectory_results.F_elastic_y(:);
    simulation_data.belt_stretch_x = trajectory_results.belt_stretch_x(:);
    simulation_data.belt_stretch_y = trajectory_results.belt_stretch_y(:);

    % Trajectory error
    simulation_data.error_x = trajectory_results.error_x(:);
    simulation_data.error_y = trajectory_results.error_y(:);
    simulation_data.error_magnitude = trajectory_results.error_magnitude(:);
    simulation_data.error_direction = trajectory_results.error_direction(:);

    % Thermal
    simulation_data.T_nozzle = thermal.T_nozzle(:);
    simulation_data.T_interface = thermal.T_interface(:);
    simulation_data.T_surface = thermal.T_surface(:);
    simulation_data.cooling_rate = thermal.cooling_rate(:);
    simulation_data.temp_gradient_z = thermal.temp_gradient_z(:);
    simulation_data.interlayer_time = thermal.interlayer_time(:);

    % Adhesion
    simulation_data.adhesion_ratio = adhesion.adhesion_ratio(:);

    % G-code features
    simulation_data.is_extruding = trajectory.is_extruding(:);
    simulation_data.print_type = trajectory.print_type;
    simulation_data.layer_num = trajectory.layer_num(:);

    % Add params for reference
    simulation_data.params = params;
end
