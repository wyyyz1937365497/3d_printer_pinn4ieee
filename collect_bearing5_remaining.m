% COLLECT_BEARING5_REMAINING - 完成bearing5剩余层数
%
% 完成bearing5_PLA_2h27m.gcode的剩余30层（层46-75）
%
% Author: 3D Printer PINN Project
% Date: 2026-01-29

%% Setup
cd('F:\TJ\3d_print\3d_printer_pinn4ieee');
addpath(genpath(fullfile(pwd, 'matlab_simulation')));

fprintf('\n');
fprintf('============================================================\n');
fprintf('完成bearing5剩余层数（46-75）\n');
fprintf('============================================================\n');
fprintf('\n');

%% GPU Configuration
gpu_id = 1;  % Use cuda1
fprintf('GPU配置：cuda1 (GPU %d)\n\n', gpu_id);

%% Configuration
gcode_file = 'test_gcode_files/bearing5_PLA_2h27m.gcode';
start_layer = 46;
end_layer = 75;

%% 检查已完成的层
output_dir = 'data_simulation_bearing5_PLA_2h27m_sampled_75layers';
existing_files = dir(fullfile(output_dir, 'layer*.mat'));
completed_layers = [];
for i = 1:length(existing_files)
    fname = existing_files(i).name;
    % Extract layer number from filename like "layer46_ender3v2.mat"
    tokens = regexp(fname, 'layer(\d+)_ender3v2\.mat', 'tokens');
    if ~isempty(tokens)
        completed_layers = [completed_layers; str2double(tokens{1}{1})];
    end
end

% 确定需要处理的层
layers_to_process = start_layer:end_layer;
layers_to_process = setdiff(layers_to_process, completed_layers);

if isempty(layers_to_process)
    fprintf('所有层（%d-%d）已完成！\n', start_layer, end_layer);
    return;
end

fprintf('已完成层: %s\n', mat2str(sort(completed_layers)));
fprintf('待处理层: %s\n', mat2str(layers_to_process));
fprintf('待处理数量: %d层\n\n', length(layers_to_process));

%% 预提取所有层的轨迹数据（一次性解析gcode，避免重复）
fprintf('============================================================\n');
fprintf('预提取所有层的轨迹数据...\n');
fprintf('============================================================\n');
fprintf('这将一次性解析gcode文件，提取所有%d层的轨迹\n', length(layers_to_process));
fprintf('预计需要1-3分钟，但会大幅加快后续仿真速度\n\n');

preextract_start = tic;
trajectory_options = struct();
trajectory_options.layers = layers_to_process;  % 一次性提取所有层
trajectory_options.time_step = 0.01;
trajectory_options.include_type = {'Outer wall', 'Inner wall'};
trajectory_options.include_skirt = false;

base_params = physics_parameters();
base_params.debug.verbose = false;

% 一次性提取所有层的轨迹
all_trajectory_data = reconstruct_trajectory(gcode_file, base_params, trajectory_options);
preextract_time = toc(preextract_start);

fprintf('✓ 轨迹预提取完成！用时: %.1f 分钟\n\n', preextract_time/60);

% 组织数据：按层号索引
fprintf('组织轨迹数据...\n');
trajectory_cache = struct();
for i = 1:length(layers_to_process)
    layer_num = layers_to_process(i);
    mask = all_trajectory_data.layer_num == layer_num;
    trajectory_cache(layer_num).time = all_trajectory_data.time(mask);
    trajectory_cache(layer_num).x = all_trajectory_data.x(mask);
    trajectory_cache(layer_num).y = all_trajectory_data.y(mask);
    trajectory_cache(layer_num).z = all_trajectory_data.z(mask);
    trajectory_cache(layer_num).vx = all_trajectory_data.vx(mask);
    trajectory_cache(layer_num).vy = all_trajectory_data.vy(mask);
    trajectory_cache(layer_num).vz = all_trajectory_data.vz(mask);
    trajectory_cache(layer_num).ax = all_trajectory_data.ax(mask);
    trajectory_cache(layer_num).ay = all_trajectory_data.ay(mask);
    trajectory_cache(layer_num).az = all_trajectory_data.az(mask);
    trajectory_cache(layer_num).jx = all_trajectory_data.jx(mask);
    trajectory_cache(layer_num).jy = all_trajectory_data.jy(mask);
    trajectory_cache(layer_num).jz = all_trajectory_data.jz(mask);
    trajectory_cache(layer_num).is_extruding = all_trajectory_data.is_extruding(mask);
    trajectory_cache(layer_num).print_type = all_trajectory_data.print_type;
    trajectory_cache(layer_num).layer_num = all_trajectory_data.layer_num(mask);
end
fprintf('✓ 数据组织完成！共%d层，总计%d个数据点\n\n', length(layers_to_process), length(all_trajectory_data.time));

%% Main Loop
total_start_time = tic;

for layer_idx = 1:length(layers_to_process)
    target_layer = layers_to_process(layer_idx);

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('[%s] 进度: %.1f%% (%d/%d) - 层 %d\n', ...
        datetime('now'), ...
        layer_idx/length(layers_to_process)*100, ...
        layer_idx, length(layers_to_process), target_layer);
    fprintf('============================================================\n');

    %% 使用预提取的轨迹数据
    fprintf('使用预提取轨迹数据（层 %d）...\n', target_layer);
    trajectory_data = trajectory_cache(target_layer);
    n_points = length(trajectory_data.time);
    fprintf('  ✓ 轨迹数据：%d 个数据点\n', n_points);

    %% Run simulation
    fprintf('运行仿真（Ender 3 V2参数）...\n');

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    output_file = fullfile(output_dir, ...
        sprintf('layer%02d_ender3v2.mat', target_layer));

    simulation_data = run_simulation_with_cached_trajectory(...
        trajectory_data, ...
        output_file, ...
        base_params, ...
        gpu_id);

    fprintf('✓ 层 %d 仿真完成！\n', target_layer);
    fprintf('  已保存：%s\n', output_file);

    % Show estimated remaining time
    elapsed = toc(total_start_time);
    sims_done = layer_idx;
    avg_time = elapsed / sims_done;
    remaining = (length(layers_to_process) - sims_done) * avg_time;
    fprintf('  已用时: %.1f 分钟 | 预计剩余: %.1f 分钟\n', ...
        elapsed/60, remaining/60);
end

%% Summary
total_elapsed = toc(total_start_time);

fprintf('\n');
fprintf('============================================================\n');
fprintf('bearing5剩余层完成！\n');
fprintf('============================================================\n');
fprintf('\n');
fprintf('成功仿真：%d 次\n', length(layers_to_process));
fprintf('总用时：%.2f 分钟 (%.2f 小时)\n', total_elapsed/60, total_elapsed/3600);
fprintf('平均每次仿真：%.2f 秒\n', total_elapsed / length(layers_to_process));
fprintf('\n');

%% Helper function: Run simulation with cached trajectory
function simulation_data = run_simulation_with_cached_trajectory(...
    trajectory_data, output_file, params, gpu_id)

    % Step 1: Simulate trajectory error (using cached trajectory)
    n_points = length(trajectory_data.time);

    gpu_info = setup_gpu(gpu_id);

    if gpu_info.use_gpu && n_points > 500
        trajectory_results = simulate_trajectory_error_gpu(trajectory_data, params, gpu_info);
    else
        trajectory_results = simulate_trajectory_error(trajectory_data, params);
    end

    % Step 2: Simulate thermal field (already includes adhesion calculation)
    thermal_results = simulate_thermal_field(trajectory_data, params);

    % Step 2.5: Calculate quality metrics
    quality_results = calculate_quality_metrics(trajectory_data, thermal_results, params);

    % Step 3: Combine results
    simulation_data = combine_results(trajectory_data, trajectory_results, ...
                                     thermal_results, quality_results, params);

    % Step 4: Save
    if nargin >= 2 && ~isempty(output_file)
        save(output_file, 'simulation_data', '-v7.3');
    end
end

%% Helper function: Combine results
function simulation_data = combine_results(trajectory, trajectory_results, ...
                                          thermal, quality, params)

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

    % Quality metrics
    simulation_data.adhesion_ratio = quality.adhesion_ratio(:);
    simulation_data.internal_stress = quality.internal_stress(:);
    simulation_data.porosity = quality.porosity(:);
    simulation_data.dimensional_accuracy = quality.dimensional_accuracy(:);
    simulation_data.quality_score = quality.quality_score(:);

    % G-code features
    simulation_data.is_extruding = trajectory.is_extruding(:);
    simulation_data.print_type = trajectory.print_type;
    simulation_data.layer_num = trajectory.layer_num(:);

    % Add params for reference
    simulation_data.params = params;
end
