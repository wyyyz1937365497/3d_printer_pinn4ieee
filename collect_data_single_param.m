% COLLECT_DATA_SINGLE_PARAM - 单参数配置数据收集
%
% 使用Ender 3 V2的真实参数进行仿真
% 不需要多组参数组合，因为gcode已经是针对Ender 3 V2切片的
%
% Author: 3D Printer PINN Project
% Date: 2026-01-28

%% Setup
cd('F:\TJ\3d_print\3d_printer_pinn4ieee')
addpath(genpath(fullfile(pwd, 'matlab_simulation')))

fprintf('\n');
fprintf('============================================================\n');
fprintf('Ender 3 V2 单参数数据收集\n');
fprintf('============================================================\n');
fprintf('\n');

%% GPU Configuration
gpu_id = 1;  % Use cuda1
fprintf('GPU配置：cuda1 (GPU %d)\n\n', gpu_id);

%% G-code Files Configuration
% 目标：52,000样本（超过50K）
gcode_config = {
    % 3DBenchy - 收集25层（每2层取1层）
    {'test_gcode_files/3DBenchy_PLA_1h28m.gcode', 2:2:50},  % 25层
    % 圆柱 - 收集10层（每5层取1层）
    {'test_gcode_files/圆柱_PLA_18m38s.gcode', 1:5:50},  % 10层
    % 螺旋 - 收集25层（每2层取1层，点数少所以多收集）
    {'test_gcode_files/螺旋_PLA_25m24s.gcode', 1:2:50}   % 25层
};

% 预期结果：
% 3DBenchy: 25层 × 1200点 = 30,000点
% 圆柱:     10层 × 1200点 = 12,000点
% 螺旋:     25层 × 400点  = 10,000点
% 总计:                      52,000点 ✅

% Check files
fprintf('检查G-code文件...\n');
for i = 1:length(gcode_config)
    config = gcode_config{i};
    if exist(config{1}, 'file')
        [~, name, ~] = fileparts(config{1});
        fprintf('  ✓ %s\n', name);
    else
        error('文件不存在: %s', config{1});
    end
end
fprintf('\n');

%% Summary
fprintf('数据收集策略：\n\n');
total_sims = 0;
for i = 1:length(gcode_config)
    config = gcode_config{i};
    [~, name, ~] = fileparts(config{1});
    layers = config{2};

    if isscalar(layers)
        n_layers = 1;
        layer_str = sprintf('层%d', layers);
    else
        n_layers = length(layers);
        layer_str = sprintf('%d层: %s', n_layers, mat2str(layers));
    end

    n_sims = n_layers;  % 每层只仿真1次
    total_sims = total_sims + n_sims;

    fprintf('%d. %s\n', i, name);
    fprintf('   层数: %s\n', layer_str);
    fprintf('   仿真次数: %d\n\n', n_sims);
end

fprintf('总仿真次数: %d\n', total_sims);
fprintf('预计时间: %.1f 分钟\n\n', total_sims * 0.5);
fprintf('预期样本数: ~52,000（超过50K目标）\n\n');

%% Main Loop
total_start_time = tic;

for gcode_idx = 1:length(gcode_config)
    config = gcode_config{gcode_idx};
    gcode_file = config{1};
    layer_nums = config{2};

    [~, gcode_name, ~] = fileparts(gcode_file);

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('处理 [%d/%d]: %s\n', gcode_idx, length(gcode_config), gcode_name);
    fprintf('============================================================\n');
    fprintf('\n');

    % Determine layers
    if isscalar(layer_nums)
        layers_to_collect = layer_nums;
        layer_str = sprintf('layer%d', layer_nums);
    else
        layers_to_collect = layer_nums;
        layer_str = sprintf('layers_%s', strrep(mat2str(layer_nums), ' ', '_'));
    end

    fprintf('目标层：%s\n', mat2str(layers_to_collect));
    fprintf('\n');

    % Process each layer
    for layer_idx = 1:length(layers_to_collect)
        target_layer = layers_to_collect(layer_idx);

        fprintf('\n--- 层 %d (%d/%d) ---\n', target_layer, layer_idx, length(layers_to_collect));

        %% Extract trajectory ONCE for this layer
        fprintf('提取轨迹数据...\n');
        trajectory_options = struct();
        trajectory_options.layers = target_layer;
        trajectory_options.time_step = 0.01;
        trajectory_options.include_type = {'Outer wall', 'Inner wall'};
        trajectory_options.include_skirt = false;

        base_params = physics_parameters();
        base_params.debug.verbose = false;

        trajectory_data = reconstruct_trajectory(gcode_file, base_params, trajectory_options);
        n_points = length(trajectory_data.time);
        fprintf('  轨迹提取完成：%d 个数据点\n\n', n_points);

        %% Setup output directory
        output_dir = sprintf('data_simulation_%s_%s', gcode_name, layer_str);
        mkdir(output_dir);

        %% Run simulation with Ender 3 V2 parameters
        fprintf('运行仿真（Ender 3 V2参数）...\n');

        output_file = fullfile(output_dir, ...
            sprintf('layer%02d_ender3v2.mat', target_layer));

        simulation_data = run_simulation_with_cached_trajectory(...
            trajectory_data, ...
            output_file, ...
            base_params, ...
            gpu_id);

        fprintf('✓ 层 %d 完成！\n', target_layer);
        fprintf('  已保存：%s\n', output_file);
    end
end

%% Summary
total_elapsed = toc(total_start_time);

fprintf('\n');
fprintf('============================================================\n');
fprintf('数据收集完成！\n');
fprintf('============================================================\n');
fprintf('\n');

% Final progress bar (100%)
bar_length = 40;
progress_bar = repmat('=', 1, bar_length);
fprintf('总进度: [%s] %d/%d (100.0%%)\n\n', progress_bar, total_sims, total_sims);

fprintf('处理G-code文件：%d\n', length(gcode_config));
for i = 1:length(gcode_config)
    config = gcode_config{i};
    [~, name, ~] = fileparts(config{1});
    fprintf('  - %s\n', name);
end
fprintf('\n');

fprintf('成功仿真：%d 次\n', total_sims);
fprintf('总用时：%.2f 分钟 (%.2f 小时)\n', total_elapsed/60, total_elapsed/3600);
fprintf('平均每次仿真：%.2f 秒\n', total_elapsed / total_sims);
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

    fprintf('  data_simulation_%s_%s/\n', name, layer_str);
end
fprintf('\n');

fprintf('下一步：\n');
fprintf('  1. 转换为Python格式：\n');
fprintf('     python matlab_simulation/convert_to_trajectory_features.py \\\n');
fprintf('         data_simulation_* -o trajectory_data.h5\n');
fprintf('  2. 验证数据：\n');
fprintf('     python matlab_simulation/test_conversion.py\n');
fprintf('  3. 开始训练\n');
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
