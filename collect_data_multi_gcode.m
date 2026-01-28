% COLLECT_DATA_MULTI_GCODE - Batch data collection for multiple G-code files
%
% This script collects training data from multiple G-code files:
% - 3DBenchy (complex real model)
% - Cylinder (circles)
% - Spiral (continuous curvature change)
%
% Author: 3D Printer PINN Project
% Date: 2026-01-28

%% Setup
cd('F:\TJ\3d_print\3d_printer_pinn4ieee')
addpath(genpath(fullfile(pwd, 'matlab_simulation')))

fprintf('\n');
fprintf('============================================================\n');
fprintf('多G-code批量数据收集\n');
fprintf('============================================================\n');
fprintf('\n');

%% GPU Configuration
gpu_id = 1;  % Use cuda1
fprintf('GPU配置：cuda1 (GPU %d)\n', gpu_id);
fprintf('\n');

%% G-code Files Configuration (with individual strategies)
% Each entry: {filename, layers, n_param_combos}
% - filename: G-code file path
% - layers: array of layer numbers to collect (e.g., [5, 10, 15] or 10)
% - n_param_combos: number of parameter combinations per layer
gcode_config = {
    % 3DBenchy - 多层细致收集（复杂几何，层间差异大）
    {'test_gcode_files/3DBenchy_PLA_1h28m.gcode', [5, 10, 15, 20, 25], 80},
    % 圆柱 - 简化收集（简单几何，层间相似）
    {'test_gcode_files/圆柱_PLA_18m38s.gcode', 10, 30},
    % 螺旋 - 简化收集（简单几何，层间相似）
    {'test_gcode_files/螺旋_PLA_25m24s.gcode', 10, 30}
};

% Extract G-code file names for checking
gcode_files = cell(length(gcode_config), 1);
for i = 1:length(gcode_config)
    gcode_files{i} = gcode_config{i}{1};
end

% Check if files exist
fprintf('检查G-code文件...\n');
for i = 1:length(gcode_files)
    if exist(gcode_files{i}, 'file')
        fprintf('  ✓ %s\n', gcode_files{i});
    else
        fprintf('  ✗ %s (不存在！)\n', gcode_files{i});
        error('G-code文件不存在，请先准备好文件');
    end
end
fprintf('\n');

%% Collection Strategy Summary
fprintf('数据收集策略：\n');
fprintf('\n');
for i = 1:length(gcode_config)
    config = gcode_config{i};
    [~, name, ~] = fileparts(config{1});
    layers = config{2};
    n_params = config{3};

    if isscalar(layers)
        layer_str = sprintf('层 %d', layers);
        n_layers = 1;
    else
        layer_str = sprintf('%d层: %s', length(layers), mat2str(layers));
        n_layers = length(layers);
    end

    fprintf('%d. %s\n', i, name);
    fprintf('   层数: %s\n', layer_str);
    fprintf('   每层参数: %d 种组合\n', n_params);
    fprintf('   总仿真: %d 次\n', n_layers * n_params);
    fprintf('\n');
end

%% Calculate total simulations
total_sims = 0;
for i = 1:length(gcode_config)
    config = gcode_config{i};
    layers = config{2};
    n_params = config{3};
    n_layers = isscalar(layers) ? 1 : length(layers);
    total_sims = total_sims + n_layers * n_params;
end

fprintf('总仿真次数: %d 次\n', total_sims);
fprintf('\n');

%% Parameter Grid Definition (used for all G-codes)
accel_grid = 200:100:500;
velocity_grid = 100:100:400;
fan_grid = [0, 128, 255];
ambient_temp_grid = [20, 25, 30];

[accel_vals, vel_vals, fan_vals, temp_vals] = ...
    ndgrid(accel_grid, velocity_grid, fan_grid, ambient_temp_grid);

total_combos_available = length(accel_vals);
fprintf('可用参数组合：%d\n', total_combos_available);
fprintf('\n');

%% Main Loop: Process each G-code file with its configuration
total_start_time = tic;
total_sims_completed = 0;

for gcode_idx = 1:length(gcode_config)
    config = gcode_config{gcode_idx};
    gcode_file = config{1};
    layer_nums = config{2};
    n_param_combos = config{3};

    [~, gcode_name, ~] = fileparts(gcode_file);

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('处理G-code [%d/%d]: %s\n', gcode_idx, length(gcode_config), gcode_name);
    fprintf('============================================================\n');
    fprintf('\n');

    % Determine if single or multiple layers
    if isscalar(layer_nums)
        layers_to_collect = layer_nums;
        layer_str = sprintf('layer%d', layer_nums);
    else
        layers_to_collect = layer_nums;
        layer_str = sprintf('layers_%s', strrep(mat2str(layer_nums), ' ', '_'));
    end

    fprintf('目标层：%s\n', mat2str(layers_to_collect));
    fprintf('每层参数组合：%d\n', n_param_combos);
    fprintf('总仿真次数：%d\n', length(layers_to_collect) * n_param_combos);
    fprintf('\n');

    % Generate parameter indices for this G-code
    if n_param_combos < total_combos_available
        indices = randperm(total_combos_available, n_param_combos);
    else
        indices = 1:total_combos_available;
        n_param_combos = total_combos_available;
    end

    %% Process each layer
    for layer_idx = 1:length(layers_to_collect)
        target_layer = layers_to_collect(layer_idx);

        fprintf('\n--- 层 %d (%d/%d) ---\n', target_layer, layer_idx, length(layers_to_collect));

        %% Create output directory for this layer
        output_dir = sprintf('data_simulation_%s_%s', gcode_name, layer_str);
        mkdir(output_dir);

        fprintf('输出目录：%s\n', output_dir);

        %% Parameter Scan Loop for this layer
        sim_start_time = tic;
        sim_count = 0;

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
                    sprintf('combo_L%d_%04d_a%03d_v%03d_f%03d_t%02d.mat', ...
                            target_layer, idx, accel, velocity, fan, ambient_temp));

                % Simulation options
                options = struct();
                options.layers = target_layer;
                options.time_step = 0.01;  % 10ms time step
                options.include_type = {'Outer wall', 'Inner wall'};
                options.include_skirt = false;

                % Run simulation with trajectory reconstruction
                simulation_data = run_full_simulation_with_reconstruction(...
                    gcode_file, ...
                    output_file, ...
                    options, ...
                    params, ...
                    gpu_id);

                sim_count = sim_count + 1;
                total_sims_completed = total_sims_completed + 1;
                fprintf(' ✓\n');

                if mod(idx, 10) == 0
                    elapsed = toc(sim_start_time);
                    remaining = elapsed / idx * (n_param_combos - idx);
                    fprintf('  进度：%.1f%% | 已用：%.1f min | 预计剩余：%.1f min\n', ...
                            100*idx/n_param_combos, elapsed/60, remaining/60);
                end

            catch ME
                fprintf(' ✗\n');
                fprintf('  错误：%s\n', ME.message);
                if ~isempty(ME.stack)
                    fprintf('  位置：%s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
                end
                fprintf('  跳过此配置，继续下一个...\n');
            end
        end

        sim_elapsed = toc(sim_start_time);
        fprintf('\n层 %d 完成！\n', target_layer);
        fprintf('  成功仿真：%d/%d\n', sim_count, n_param_combos);
        fprintf('  用时：%.2f 分钟\n', sim_elapsed/60);
    end

    sim_elapsed_total = toc(sim_start_time);
    fprintf('\n');
    fprintf('完成 %s！\n', gcode_name);
    fprintf('  总用时：%.2f 分钟\n', sim_elapsed_total/60);
    fprintf('  数据保存到：%s\n', output_dir);
end

%% Overall Summary
total_elapsed = toc(total_start_time);

fprintf('\n');
fprintf('============================================================\n');
fprintf('批量数据收集完成！\n');
fprintf('============================================================\n');
fprintf('\n');

fprintf('处理G-code文件：%d\n', length(gcode_config));
for i = 1:length(gcode_config)
    config = gcode_config{i};
    [~, name, ~] = fileparts(config{1});
    fprintf('  - %s\n', name);
end
fprintf('\n');

fprintf('成功完成仿真：%d 次\n', total_sims_completed);
fprintf('总用时：%.2f 分钟 (%.2f 小时)\n', total_elapsed/60, total_elapsed/3600);
fprintf('\n');

fprintf('输出目录：\n');
for i = 1:length(gcode_config)
    config = gcode_config{i};
    [~, name, ~] = fileparts(config{1});
    layer_nums = config{2};

    if isscalar(layer_nums)
        layer_str = sprintf('layer%d', layer_nums);
    else
        layer_str = sprintf('layers_%s', strrep(mat2str(layer_nums), ' ', '_'));
    end

    fprintf('  %s\n', sprintf('data_simulation_%s_%s', name, layer_str));
end
fprintf('\n');

fprintf('下一步：\n');
fprintf('  1. 检查生成的数据文件\n');
fprintf('  2. 统计总样本数量\n');
fprintf('  3. 转换为Python格式：\n');
fprintf('     python matlab_simulation/convert_matlab_to_python.py \\\n');
fprintf('         data_simulation_* training -o training_data\n');
fprintf('  4. 开始PINN训练\n');
fprintf('\n');

%% Helper function: Run simulation with trajectory reconstruction
function simulation_data = run_full_simulation_with_reconstruction(...
    gcode_file, output_file, options, params, gpu_id)

    % Step 1: Reconstruct trajectory
    trajectory_data = reconstruct_trajectory(gcode_file, params, options);

    % Step 2: Simulate trajectory error
    n_points = length(trajectory_data.time);

    gpu_info = setup_gpu(gpu_id);

    if gpu_info.use_gpu && n_points > 500
        trajectory_results = simulate_trajectory_error_gpu(trajectory_data, params, gpu_info);
    else
        trajectory_results = simulate_trajectory_error(trajectory_data, params);
    end

    % Step 3: Simulate thermal field (already includes adhesion calculation)
    thermal_results = simulate_thermal_field(trajectory_data, params);

    % Step 4: Combine results
    simulation_data = combine_results(trajectory_data, trajectory_results, ...
                                     thermal_results, params);

    % Step 5: Save
    if nargin >= 2 && ~isempty(output_file)
        save(output_file, 'simulation_data', '-v7.3');
    end
end

%% Helper function: Combine results
function simulation_data = combine_results(trajectory, trajectory_results, ...
                                          thermal, params)

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

    % Kinematics
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
    simulation_data.T_nozzle = thermal.T_nozzle_history(:);
    simulation_data.T_interface = thermal.T_interface(:);
    simulation_data.T_surface = thermal.T_surface(:);
    simulation_data.cooling_rate = thermal.cooling_rate(:);
    simulation_data.temp_gradient_z = thermal.temp_gradient_z(:);
    simulation_data.interlayer_time = thermal.interlayer_time(:);

    % Adhesion
    simulation_data.adhesion_ratio = thermal.adhesion_ratio(:);

    % G-code features
    simulation_data.is_extruding = trajectory.is_extruding(:);
    simulation_data.print_type = trajectory.print_type;
    simulation_data.layer_num = trajectory.layer_num(:);

    % Add params for reference
    simulation_data.params = params;
end
