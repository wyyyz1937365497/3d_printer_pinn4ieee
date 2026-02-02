function collect_data_parallel(gcode_files, layer_configs, varargin)
% COLLECT_DATA_PARALLEL - 并行数据收集（使用多核CPU）
%
% 使用 parfor 并行处理多层，大幅提升速度
%
% 用法:
%   collect_data_parallel(gcode_file, layer_config)
%   collect_data_parallel(gcode_file, layer_config, 'OptionName', OptionValue)
%
% 可选参数:
%   'NumWorkers'       - 并行worker数量（默认：自动检测）
%   'OutputDir'        - 输出目录名
%   'Resume'           - 是否跳过已完成的层 (默认: true)
%   'UseFirmwareEffects' - 是否使用固件增强仿真 (默认: true)
%
% 示例:
%   % 收集3DBenchy的采样层数据（并行）
%   collect_data_parallel('test_gcode_files/3DBenchy_PLA_1h28m.gcode', 'sampled:5');
%
%   % 收集指定范围（并行）
%   collect_data_parallel('test.gcode', 1:50, 'NumWorkers', 8);

%% Setup
cd('F:\TJ\3d_print\3d_printer_pinn4ieee');
addpath(genpath(fullfile(pwd, 'simulation')));

%% 解析输入参数
p = inputParser;
addRequired(p, 'gcode_files', @(x) ischar(x) || iscell(x));
addRequired(p, 'layer_configs', @(x) ischar(x) || isnumeric(x) || iscell(x));
addParameter(p, 'NumWorkers', [], @isnumeric);
addParameter(p, 'OutputDir', '', @ischar);
addParameter(p, 'Resume', true, @islogical);
addParameter(p, 'UseFirmwareEffects', true, @islogical);

parse(p, gcode_files, layer_configs, varargin{:});

num_workers = p.Results.NumWorkers;
resume_flag = p.Results.Resume;
use_firmware_effects = p.Results.UseFirmwareEffects;

% 标准化为cell数组
if ischar(gcode_files)
    gcode_files = {gcode_files};
end

if ischar(layer_configs) || isnumeric(layer_configs)
    layer_configs = {layer_configs};
end

% 验证
if numel(gcode_files) ~= numel(layer_configs)
    error('gcode_files和layer_configs数量必须相同');
end

%% 启动并行池
fprintf('\n');
fprintf('============================================================\n');
fprintf('3D打印数据收集 - 并行版本\n');
fprintf('============================================================\n');
fprintf('\n');

% 检查 Parallel Computing Toolbox
if ~license('test', 'Distrib_Computing_Toolbox')
    error('需要 Parallel Computing Toolbox 才能使用并行功能');
end

% 设置worker数量
if isempty(num_workers)
    % 自动检测：使用核心数-1（保留一个核心给系统）
    num_cores = feature('numcores');
    num_workers = max(1, num_cores - 1);
end

fprintf('并行配置:\n');
fprintf('  核心数: %d\n', feature('numcores'));
fprintf('  Worker数: %d\n', num_workers);
fprintf('  仿真模式: CPU (固件效应=%s)\n', mat2str(use_firmware_effects));
fprintf('\n');

% 启动并行池
try
    pool = gcp('nocreate');
    if isempty(pool)
        parpool('local', num_workers);
        fprintf('✓ 并行池已启动 (%d workers)\n\n', num_workers);
    else
        fprintf('✓ 使用现有并行池 (%d workers)\n\n', pool.NumWorkers);
    end
catch ME
    warning('启动并行池失败: %s\n回退到单线程模式...', ME.message);
    num_workers = 0;
end

%% 处理每个文件
for file_idx = 1:length(gcode_files)
    gcode_file = gcode_files{file_idx};
    layer_config = layer_configs{file_idx};

    [~, gcode_name, ~] = fileparts(gcode_file);

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('[%d/%d] 处理文件: %s\n', file_idx, length(gcode_files), gcode_name);
    fprintf('============================================================\n');
    fprintf('\n');

    % 解析层配置
    layers_to_process = parse_layer_config(gcode_file, layer_config);

    % 生成输出目录
    output_dir = generate_output_dir(gcode_name, layers_to_process);

    % 移除已完成的层
    if resume_flag
        layers_to_process = remove_completed_layers(output_dir, layers_to_process);
    end

    if isempty(layers_to_process)
        fprintf('所有层已完成！\n');
        continue;
    end

    fprintf('待处理层数: %d\n', length(layers_to_process));
    fprintf('输出目录: %s\n\n', output_dir);

    % 创建输出目录
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % 预提取轨迹数据（所有worker共享）
    fprintf('预提取轨迹数据（所有层）...\n');
    trajectory_cache = preextract_trajectory(gcode_file, layers_to_process);
    fprintf('✓ 轨迹预提取完成\n\n');

    % 并行处理每一层
    if num_workers > 0 && length(layers_to_process) > 1
        fprintf('开始并行处理...\n\n');
        process_layers_parallel(layers_to_process, output_dir, ...
                              use_firmware_effects, num_workers, trajectory_cache);
    else
        fprintf('单线程模式处理...\n\n');
        process_layers_serial(layers_to_process, output_dir, ...
                            use_firmware_effects, trajectory_cache);
    end
end

fprintf('\n');
fprintf('============================================================\n');
fprintf('所有文件处理完成！\n');
fprintf('============================================================\n');
fprintf('\n');

end

%% ========== 辅助函数 ==========

function trajectory_cache = preextract_trajectory(gcode_file, layers)
% 预提取所有层的轨迹数据（所有worker共享）

fprintf('  解析gcode文件，提取%d层的轨迹...\n', length(layers));

tic;
trajectory_options = struct();
trajectory_options.layers = layers;
trajectory_options.time_step = 0.01;
trajectory_options.include_type = {'Outer wall', 'Inner wall'};
trajectory_options.include_skirt = false;

base_params = physics_parameters();
base_params.debug.verbose = false;

% 一次性提取所有层
all_trajectory_data = reconstruct_trajectory(gcode_file, base_params, trajectory_options);

elapsed = toc;
fprintf('  用时: %.1f 分钟\n', elapsed/60);

% 组织数据（按层号索引）
fprintf('  组织轨迹数据到缓存...\n');
trajectory_cache = containers.Map('KeyType', 'double', 'ValueType', 'any');

for i = 1:length(layers)
    layer_num = layers(i);
    mask = all_trajectory_data.layer_num == layer_num;

    % 提取该层的数据（只提取实际存在的字段）
    layer_data = struct();
    layer_data.time = all_trajectory_data.time(mask);
    layer_data.x = all_trajectory_data.x(mask);
    layer_data.y = all_trajectory_data.y(mask);
    layer_data.z = all_trajectory_data.z(mask);
    layer_data.vx = all_trajectory_data.vx(mask);
    layer_data.vy = all_trajectory_data.vy(mask);
    layer_data.vz = all_trajectory_data.vz(mask);
    layer_data.ax = all_trajectory_data.ax(mask);
    layer_data.ay = all_trajectory_data.ay(mask);
    layer_data.az = all_trajectory_data.az(mask);
    layer_data.jx = all_trajectory_data.jx(mask);
    layer_data.jy = all_trajectory_data.jy(mask);
    layer_data.jz = all_trajectory_data.jz(mask);
    layer_data.is_extruding = all_trajectory_data.is_extruding(mask);
    layer_data.print_type = all_trajectory_data.print_type(mask);
    layer_data.layer_num = all_trajectory_data.layer_num(mask);

    % 存储到缓存（使用层号作为键）
    trajectory_cache(layer_num) = layer_data;
end

fprintf('  缓存完成: %d层, 共%d数据点\n', length(layers), length(all_trajectory_data.time));
fprintf('\n');

end

function process_layers_parallel(layers, output_dir, use_firmware_effects, num_workers, trajectory_cache)
% 并行处理多层（使用共享轨迹缓存）

tic;

% 准备任务参数
n_tasks = length(layers);

% 显示进度信息
fprintf('任务数: %d\n', n_tasks);
fprintf('Worker数: %d\n', num_workers);
fprintf('\n');

% 结果存储
completed = false(1, n_tasks);
errors = cell(1, n_tasks);

% 并行循环（共享轨迹缓存）
parfor task_idx = 1:n_tasks
    target_layer = layers(task_idx);

    try
        % 从缓存获取轨迹数据（不需要重新解析gcode）
        process_single_layer_with_cache(target_layer, output_dir, use_firmware_effects, trajectory_cache);
        completed(task_idx) = true;
    catch ME
        errors{task_idx} = ME.message;
        completed(task_idx) = false;
    end
end

% 统计结果
elapsed = toc;
n_success = sum(completed);
n_fail = n_tasks - n_success;

fprintf('\n');
fprintf('============================================================\n');
fprintf('并行处理完成！\n');
fprintf('============================================================\n');
fprintf('总耗时: %.1f 分钟 (%.2f 秒/层)\n', elapsed/60, elapsed/n_tasks);
fprintf('成功: %d/%d\n', n_success, n_tasks);
fprintf('失败: %d/%d\n', n_fail, n_tasks);

if n_fail > 0
    fprintf('\n失败的任务:\n');
    for i = 1:n_tasks
        if ~completed(i)
            fprintf('  层 %d: %s\n', layers(i), errors{i});
        end
    end
end
fprintf('\n');

end

function process_layers_serial(layers, output_dir, use_firmware_effects, trajectory_cache)
% 单线程处理（fallback）

total_start = tic;

for layer_idx = 1:length(layers)
    target_layer = layers(layer_idx);

    fprintf('[%d/%d] 处理层 %d...\n', layer_idx, length(layers), target_layer);

    try
        process_single_layer_with_cache(target_layer, output_dir, use_firmware_effects, trajectory_cache);
        fprintf('  ✓ 完成\n');
    catch ME
        fprintf('  ✗ 失败: %s\n', ME.message);
    end

    % 预计剩余时间
    elapsed = toc(total_start);
    avg_time = elapsed / layer_idx;
    remaining = (length(layers) - layer_idx) * avg_time;
    fprintf('  剩余时间: %.1f 分钟\n', remaining/60);
end

end

function process_single_layer_with_cache(target_layer, output_dir, use_firmware_effects, trajectory_cache)
% 处理单个层（从缓存获取轨迹）

% 生成输出文件名
output_file = fullfile(output_dir, sprintf('layer%02d_ender3v2.mat', target_layer));

% 检查是否已存在
if exist(output_file, 'file')
    return;  % 跳过
end

% 从缓存获取轨迹数据（不需要重新解析gcode）
trajectory_data = trajectory_cache(target_layer);

% 运行仿真（CPU模式，不使用GPU）
base_params = physics_parameters();
base_params.debug.verbose = false;  % 关闭详细输出（并行环境）

if use_firmware_effects
    % 使用固件增强仿真（静默模式）
    evalc('trajectory_results = simulate_trajectory_error_with_firmware_effects(trajectory_data, base_params);');
else
    % 使用基础仿真（CPU）
    trajectory_results = simulate_trajectory_error(trajectory_data, base_params);
end

% 合并结果
simulation_data = combine_results(trajectory_data, trajectory_results, base_params);

% 保存
save(output_file, 'simulation_data', '-v7.3');

end

function simulation_data = combine_results(trajectory, trajectory_err, params)

    data = [];

    % Time
    data.time = trajectory.time(:);

    % Reference trajectory
    data.x_ref = trajectory.x(:);
    data.y_ref = trajectory.y(:);
    data.z_ref = trajectory.z(:);

    % Actual trajectory (with errors)
    data.x_act = trajectory.x(:) + trajectory_err.error_x(:);
    data.y_act = trajectory.y(:) + trajectory_err.error_y(:);
    data.z_act = trajectory.z(:);

    % Kinematics - Reference
    data.vx_ref = trajectory.vx(:);
    data.vy_ref = trajectory.vy(:);
    data.vz_ref = trajectory.vz(:);
    data.v_mag_ref = sqrt(trajectory.vx(:).^2 + trajectory.vy(:).^2 + trajectory.vz(:).^2);

    data.ax_ref = trajectory.ax(:);
    data.ay_ref = trajectory.ay(:);
    data.az_ref = trajectory.az(:);
    data.a_mag_ref = sqrt(trajectory.ax(:).^2 + trajectory.ay(:).^2 + trajectory.az(:).^2);

    data.jx_ref = trajectory.jx(:);
    data.jy_ref = trajectory.jy(:);
    data.jz_ref = trajectory.jz(:);
    data.jerk_mag = sqrt(trajectory.jx(:).^2 + trajectory.jy(:).^2 + trajectory.jz(:).^2);

    % Errors
    data.error_x = trajectory_err.error_x(:);
    data.error_y = trajectory_err.error_y(:);
    data.error_magnitude = trajectory_err.error_magnitude(:);

    if isfield(trajectory_err, 'error_direction')
        data.error_direction = trajectory_err.error_direction(:);
    end

    % G-code features（只提取存在的字段）
    data.is_extruding = trajectory.is_extruding(:);
    data.is_travel = ~trajectory.is_extruding;
    data.layer_num = trajectory.layer_num(:);

    % System info
    data.params = params;

    simulation_data = data;

end

function layers = parse_layer_config(gcode_file, config)
% 解析层配置

% 检测总层数
max_layer = detect_max_layer(gcode_file);

if ischar(config)
    if strcmpi(config, 'all')
        % 收集所有层
        layers = 1:max_layer;

    elseif startsWith(config, 'sampled:')
        % 采样配置
        interval = str2double(config(strfind(config, ':')+1:end));
        layers = 1:interval:max_layer;

    else
        error('未知配置: %s', config);
    end

elseif isnumeric(config)
    if isscalar(config)
        % 单层
        layers = config;
    else
        % 范围或数组
        layers = config;
    end
end

% 限制范围
layers = layers(layers >= 1 & layers <= max_layer);

end

function max_layer = detect_max_layer(gcode_file)
% 检测gcode文件的最大层数

max_layer = 0;

try
    fid = fopen(gcode_file, 'r');
    if fid ~= -1
        while ~feof(fid)
            line = fgetl(fid);
            if ischar(line)
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
    warning('检测层数失败: %s', ME.message);
end

end

function output_dir = generate_output_dir(gcode_name, layers)
% 生成输出目录名

if length(layers) == 1
    layer_str = sprintf('layer%d', layers);
elseif length(layers) > 20
    layer_str = sprintf('sampled_%dlayers', length(layers));
else
    layer_str = sprintf('layers_%d-%d', min(layers), max(layers));
end

output_dir = sprintf('data_simulation_%s_%s', gcode_name, layer_str);

end

function layers = remove_completed_layers(output_dir, layers)
% 移除已完成的层

if ~exist(output_dir, 'dir')
    return;
end

existing_files = dir(fullfile(output_dir, 'layer*.mat'));
completed_layers = [];

for i = 1:length(existing_files)
    fname = existing_files(i).name;
    tokens = regexp(fname, 'layer(\d+)_ender3v2\.mat', 'tokens');
    if ~isempty(tokens)
        completed_layers = [completed_layers; str2double(tokens{1}{1})];
    end
end

if ~isempty(completed_layers)
    fprintf('已完成层: %s\n', mat2str(sort(completed_layers)));
    layers = setdiff(layers, completed_layers);
end

end
