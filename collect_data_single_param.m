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
% 优化策略：使用所有层（自动检测每个文件的实际层数）
% 总计4个gcode文件

% === 配置选项 ===
% 采样间隔：每N层收集1层（1=收集所有层，2=每隔1层收集，等等）
LAYER_SAMPLING_INTERVAL = 5;  % 建议值：5（收集20%的层，平衡数据量和时间）

% 最小收集层数：即使采样，也至少收集这么多层
MIN_LAYERS_PER_FILE = 20;

% 最大收集层数：即使文件很大，也最多收集这么多层
MAX_LAYERS_PER_FILE = 100;

gcode_files = {
    'test_gcode_files/3DBenchy_PLA_1h28m.gcode',
    'test_gcode_files/bearing5_PLA_2h27m.gcode',
    'test_gcode_files/Nautilus_Gears_Plate_PLA_3h36m.gcode',
    'test_gcode_files/simple_boat5_PLA_4h4m.gcode'
};

% 自动检测每个文件的总层数并生成采样配置
fprintf('检测G-code文件的层数...\n');
fprintf('采样间隔: 每%d层收集1层\n', LAYER_SAMPLING_INTERVAL);
fprintf('最小收集: %d层/文件\n', MIN_LAYERS_PER_FILE);
fprintf('最大收集: %d层/文件\n\n', MAX_LAYERS_PER_FILE);

gcode_config = {};
total_expected_points = 0;

for i = 1:length(gcode_files)
    gcode_file = gcode_files{i};
    [~, name, ~] = fileparts(gcode_file);

    % 快速检测层数
    max_layer = detect_max_layer(gcode_file);

    if max_layer == 0
        warning('无法检测 %s 的层数，跳过', name);
        continue;
    end

    % 计算要收集的层数（采样）
    if max_layer <= MAX_LAYERS_PER_FILE
        % 文件不大，收集所有层
        layers_to_collect = 1:max_layer;
        sampling_str = sprintf('全部%d层', max_layer);
    else
        % 文件很大，采样收集
        % 方法：在1:max_layer范围内，每隔LAYER_SAMPLING_INTERVAL取1层
        layers_to_collect = 1:LAYER_SAMPLING_INTERVAL:max_layer;

        % 确保至少收集MIN_LAYERS_PER_FILE层
        if length(layers_to_collect) < MIN_LAYERS_PER_FILE
            % 重新计算采样间隔
            new_interval = floor(max_layer / MIN_LAYERS_PER_FILE);
            layers_to_collect = 1:new_interval:max_layer;
        end

        % 限制到MAX_LAYERS_PER_FILE
        if length(layers_to_collect) > MAX_LAYERS_PER_FILE
            layers_to_collect = layers_to_collect(1:MAX_LAYERS_PER_FILE);
        end

        sampling_str = sprintf('采样%d/%d层 (%.0f%%)', ...
            length(layers_to_collect), max_layer, ...
            length(layers_to_collect)/max_layer*100);
    end

    fprintf('  ✓ %s: %d层 → 收集%s\n', name, max_layer, sampling_str);

    % 估算每层的点数
    if contains(name, 'Nautilus') || contains(name, 'Gears')
        points_per_layer = 2000;
    elseif contains(name, 'bearing')
        points_per_layer = 1000;
    elseif contains(name, 'Benchy') || contains(name, 'boat')
        points_per_layer = 1200;
    else
        points_per_layer = 1000;
    end

    expected_points = length(layers_to_collect) * points_per_layer;
    total_expected_points = total_expected_points + expected_points;

    % 添加到配置
    gcode_config{end+1} = {gcode_file, layers_to_collect};
end

fprintf('\n');

% 计算总仿真次数
total_sims = 0;
for i = 1:length(gcode_config)
    layers = gcode_config{i}{2};
    total_sims = total_sims + length(layers);
end

fprintf('总计 %d 个文件\n', length(gcode_config));
fprintf('预期仿真次数: %d层\n', total_sims);
fprintf('预计时间: %.1f 分钟 (GPU) 或 %.1f 分钟 (CPU)\n\n', total_sims * 0.5, total_sims * 1.5);
fprintf('预期原始数据点: ~%d\n', total_expected_points);
fprintf('预期Python训练样本（stride=5）: ~%d\n', round(total_expected_points / 5));
sample_ratio = 896030 / round(total_expected_points / 5);
if sample_ratio < 20
    rating = '优秀！';
else
    rating = '良好';
end
fprintf('参数/样本比: %.1f:1（%s）\n\n', sample_ratio, rating);

%% Summary
fprintf('数据收集策略：\n\n');
for i = 1:length(gcode_config)
    config = gcode_config{i};
    [~, name, ~] = fileparts(config{1});
    layers = config{2};

    if isscalar(layers)
        n_layers = 1;
        layer_str = sprintf('层%d', layers);
    else
        n_layers = length(layers);
        layer_str = sprintf('%d层: %s', n_layers, mat2str(layers(1:min(5,end))));
        if n_layers > 5
            layer_str = [layer_str sprintf('... (共%d层)', n_layers)];
        end
    end

    fprintf('%d. %s\n', i, name);
    fprintf('   层数: %s\n', layer_str);
    fprintf('   仿真次数: %d\n\n', n_layers);
end

fprintf('============================================================\n\n');

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
        n_layers = length(layers_to_collect);
        layer_str = sprintf('sampled_%dlayers', n_layers);
    end

    fprintf('目标层：%s\n', mat2str(layers_to_collect));
    fprintf('\n');

    % Process each layer
    for layer_idx = 1:length(layers_to_collect)
        target_layer = layers_to_collect(layer_idx);

        % Show progress
        global_progress = ((gcode_idx-1) * length(layers_to_collect) + layer_idx) / total_sims * 100;
        fprintf('\n');
        fprintf('============================================================\n');
        fprintf('[%s] 进度: %.1f%% (%d/%d) - 文件 %d/%d, 层 %d\n', ...
            datetime('now'), global_progress, ...
            ((gcode_idx-1) * length(layers_to_collect) + layer_idx), ...
            total_sims, gcode_idx, length(gcode_config), target_layer);
        fprintf('============================================================\n');

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
        fprintf('  ✓ 轨迹提取完成：%d 个数据点\n', n_points);

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

        fprintf('✓ 层 %d 仿真完成！\n', target_layer);
        fprintf('  已保存：%s\n', output_file);

        % Show estimated remaining time
        elapsed = toc(total_start_time);
        sims_done = (gcode_idx-1) * length(layers_to_collect) + layer_idx;
        avg_time = elapsed / sims_done;
        remaining = (total_sims - sims_done) * avg_time;
        fprintf('  已用时: %.1f 分钟 | 预计剩余: %.1f 分钟\n', ...
            elapsed/60, remaining/60);
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
        n_layers = length(layer_nums);
        layer_str = sprintf('sampled_%dlayers', n_layers);
    end

    fprintf('  data_simulation_%s_%s/\n', name, layer_str);
end
fprintf('\n');

fprintf('下一步：\n');
fprintf('  1. 验证数据生成：\n');
fprintf('     python -c "from data.simulation import PrinterSimulationDataset; import glob; files = glob.glob(''data_simulation_*/*.mat''); print(f''找到 {len(files)} 个.mat文件''); ds = PrinterSimulationDataset(files, seq_len=200, pred_len=50, stride=5, mode=''train'', fit_scaler=True); print(f''训练样本: {len(ds)}'')"\n');
fprintf('\n');
fprintf('  2. 重新训练模型（使用修复后的物理损失 + 新数据）：\n');
fprintf('     python experiments/train_implicit_state_tcn_optimized.py \\\n');
fprintf('         --data_dir "data_simulation_*" \\\n');
fprintf('         --epochs 100 \\\n');
fprintf('         --batch_size 256 \\\n');
fprintf('         --lr 1e-3 \\\n');
fprintf('         --lambda_physics 0.05 \\\n');
fprintf('         --num_workers 8\n');
fprintf('\n');
fprintf('  3. 评估模型性能：\n');
fprintf('     python experiments/evaluate_implicit_state_tcn_comprehensive.py \\\n');
fprintf('         --checkpoint checkpoints/implicit_state_tcn_optimized/best_model.pth \\\n');
fprintf('         --data_dir "data_simulation_*" \\\n');
fprintf('         --batch_size 256\n');
fprintf('\n');

%% Helper function: Detect maximum layer number in G-code file
function max_layer = detect_max_layer(gcode_file)
    % 快速扫描gcode文件，查找最大层号
    % 方法1: 查找 "; total layer number: X" 注释（最准确）
    % 方法2: 查找LAYER_CHANGE注释计数
    % 方法3: 使用parse_gcode_improved解析

    max_layer = 0;

    try
        % 方法1: 查找 "; total layer number" 注释
        fid = fopen(gcode_file, 'r');
        if fid ~= -1
            while ~feof(fid)
                line = fgetl(fid);
                if ischar(line)
                    % 查找 "; total layer number: X"
                    tokens = regexp(line, 'total layer number:\s*(\d+)', 'tokens');
                    if ~isempty(tokens)
                        max_layer = str2double(tokens{1}{1});
                        fclose(fid);
                        return;
                    end
                end
            end
            fclose(fid);
        end
    catch ME
        % 继续尝试其他方法
    end

    % 方法2: 如果方法1失败，统计LAYER_CHANGE注释
    try
        fid = fopen(gcode_file, 'r');
        if fid ~= -1
            layer_count = 0;
            while ~feof(fid)
                line = fgetl(fid);
                if ischar(line) && contains(line, 'LAYER_CHANGE')
                    layer_count = layer_count + 1;
                end
            end
            fclose(fid);

            if layer_count > 0
                max_layer = layer_count;
                return;
            end
        end
    catch ME
        % 继续尝试方法3
    end

    % 方法3: 使用parse_gcode_improved（慢但准确）
    try
        params = physics_parameters();
        options = struct();
        options.layers = 'all';
        options.include_skirt = false;

        temp_data = parse_gcode_improved(gcode_file, params, options);
        if isfield(temp_data, 'layer_num') && ~isempty(temp_data.layer_num)
            max_layer = max(temp_data.layer_num);
            return;
        end
    catch ME
        % 如果所有方法都失败
    end

    % 如果所有方法都失败，使用默认值
    max_layer = 50;
    warning('无法检测 %s 的层数，使用默认值50', gcode_file);
end

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
