function collect_data(gcode_files, layer_configs, varargin)
% COLLECT_DATA - 统一的数据收集核心函数
%
% 用法:
%   collect_data(gcode_file, layer_config)
%   collect_data(gcode_file, layer_config, 'OptionName', OptionValue)
%
% 输入:
%   gcode_files  - 字符串或cell数组，gcode文件路径
%   layer_configs - 层数配置，可以是:
%                   - 标量: 收集该单层
%                   - 向量: [start:end] 收集指定范围
%                   - 'all': 收集所有层
%                   - 'sampled:N': 采样间隔N层
%
% 可选参数 (Name-Value pairs):
%   'GPU'              - GPU ID (默认: 1)
%   'OutputDir'        - 输出目录名 (默认: 自动生成)
%   'Resume'           - 是否跳过已完成的层 (默认: true)
%   'PreExtract'       - 是否预提取轨迹 (默认: true)
%   'UseFirmwareEffects' - 是否使用固件增强仿真(0.1mm误差) (默认: true)
%
% 示例:
%   % 收集单个文件的所有层
%   collect_data('test.gcode', 'all');
%
%   % 收集指定范围
%   collect_data('test.gcode', 1:50);
%
%   % 采样收集（每5层）
%   collect_data('test.gcode', 'sampled:5');
%
%   % 批量处理多个文件
%   files = {'file1.gcode', 'file2.gcode'};
%   configs = {'all', 'sampled:5'};
%   collect_data(files, configs);
%
% Author: 3D Printer PINN Project
% Date: 2026-01-31

%% Setup
cd('F:\TJ\3d_print\3d_printer_pinn4ieee');
addpath(genpath(fullfile(pwd, 'simulation')));

%% 解析输入参数
p = inputParser;
addRequired(p, 'gcode_files', @is_gcode_file);
addRequired(p, 'layer_configs', @is_valid_config);
addParameter(p, 'GPU', 1, @isnumeric);
addParameter(p, 'OutputDir', '', @ischar);
addParameter(p, 'Resume', true, @islogical);
addParameter(p, 'PreExtract', true, @islogical);
addParameter(p, 'UseFirmwareEffects', true, @islogical);

parse(p, gcode_files, layer_configs, varargin{:});

gpu_id = p.Results.GPU;
resume_flag = p.Results.Resume;
preextract_flag = p.Results.PreExtract;
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

%% GPU配置
fprintf('\n');
fprintf('============================================================\n');
fprintf('3D打印数据收集 - 统一接口\n');
fprintf('============================================================\n');
fprintf('\n');
fprintf('GPU配置：cuda%d (GPU %d)\n\n', gpu_id, gpu_id);

%% 处理每个文件
for file_idx = 1:length(gcode_files)
    gcode_file = gcode_files{file_idx};
    layer_config = layer_configs{file_idx};

    % 获取文件信息
    [~, gcode_name, ~] = fileparts(gcode_file);

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('[%d/%d] 处理文件: %s\n', file_idx, length(gcode_files), gcode_name);
    fprintf('============================================================\n');
    fprintf('\n');

    % 解析层配置
    layers_to_process = parse_layer_config(gcode_file, layer_config);

    if isempty(layers_to_process)
        fprintf('跳过: 无层需要处理\n');
        continue;
    end

    % 确定输出目录
    if isempty(p.Results.OutputDir)
        output_dir = generate_output_dir(gcode_name, layers_to_process);
    else
        output_dir = p.Results.OutputDir;
    end

    % 检查已完成的层
    if resume_flag
        layers_to_process = filter_completed_layers(output_dir, layers_to_process);

        if isempty(layers_to_process)
            fprintf('所有层已完成，跳过\n');
            continue;
        end
    end

    fprintf('待处理层数: %d\n', length(layers_to_process));
    fprintf('输出目录: %s\n\n', output_dir);

    % 创建输出目录
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % 预提取轨迹数据
    if preextract_flag
        fprintf('预提取轨迹数据...\n');
        trajectory_cache = preextract_trajectory(gcode_file, layers_to_process);
        fprintf('✓ 轨迹预提取完成\n\n');
    else
        trajectory_cache = [];
    end

    % 处理每一层
    process_layers(gcode_file, layers_to_process, output_dir, ...
                   gpu_id, trajectory_cache, use_firmware_effects);
end

fprintf('\n');
fprintf('============================================================\n');
fprintf('所有文件处理完成！\n');
fprintf('============================================================\n');
fprintf('\n');

end

%% ========== 辅助函数 ==========

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

if isscalar(layers)
    layer_str = sprintf('layer%d', layers);
elseif length(layers) == (max(layers) - min(layers) + 1)
    % 连续范围
    layer_str = sprintf('layers_%d-%d', min(layers), max(layers));
else
    % 采样
    n_layers = length(layers);
    layer_str = sprintf('sampled_%dlayers', n_layers);
end

output_dir = sprintf('data_simulation_%s_%s', gcode_name, layer_str);

end

function layers = filter_completed_layers(output_dir, layers)
% 过滤已完成的层

if ~exist(output_dir, 'dir')
    layers = layers;
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

function trajectory_cache = preextract_trajectory(gcode_file, layers)
% 预提取所有层的轨迹数据

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

% 组织数据
fprintf('  组织轨迹数据...\n');
trajectory_cache = struct();

for i = 1:length(layers)
    layer_num = layers(i);
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

fprintf('  数据点: %d\n', length(all_trajectory_data.time));

end

function process_layers(gcode_file, layers, output_dir, gpu_id, trajectory_cache, use_firmware_effects)
% 处理每一层

use_cache = ~isempty(trajectory_cache);
total_start = tic;

for layer_idx = 1:length(layers)
    target_layer = layers(layer_idx);

    fprintf('\n');
    fprintf('[%s] 进度: %.1f%% (%d/%d) - 层 %d\n', ...
        datetime('now'), ...
        layer_idx/length(layers)*100, ...
        layer_idx, length(layers), target_layer);

    % 获取轨迹数据
    if use_cache
        trajectory_data = trajectory_cache(target_layer);
    else
        % 实时提取
        trajectory_options = struct();
        trajectory_options.layers = target_layer;
        trajectory_options.time_step = 0.01;
        trajectory_options.include_type = {'Outer wall', 'Inner wall'};
        trajectory_options.include_skirt = false;

        base_params = physics_parameters();
        base_params.debug.verbose = false;

        trajectory_data = reconstruct_trajectory(gcode_file, base_params, trajectory_options);
    end

    n_points = length(trajectory_data.time);
    fprintf('  轨迹点: %d\n', n_points);

    % 运行仿真
    output_file = fullfile(output_dir, ...
        sprintf('layer%02d_ender3v2.mat', target_layer));

    run_simulation(trajectory_data, output_file, gpu_id, use_firmware_effects);

    fprintf('  ✓ 完成\n');

    % 预计剩余时间
    elapsed = toc(total_start);
    avg_time = elapsed / layer_idx;
    remaining = (length(layers) - layer_idx) * avg_time;
    fprintf('  剩余时间: %.1f 分钟\n', remaining/60);
end

end

function run_simulation(trajectory_data, output_file, gpu_id, use_firmware_effects)
% 运行单个层的仿真（仅轨迹误差）

base_params = physics_parameters();

% 轨迹误差仿真
fprintf('  仿真模式: ');
if use_firmware_effects
    fprintf('固件增强（目标误差~0.1mm）\n');
    gpu_info = setup_gpu(gpu_id);

    % 使用固件增强仿真（集成Junction Deviation、微步谐振、定时器抖动）
    trajectory_results = simulate_trajectory_error_with_firmware_effects(trajectory_data, base_params);
else
    fprintf('基础动力学（目标误差~0.05mm）\n');
    gpu_info = setup_gpu(gpu_id);
    n_points = length(trajectory_data.time);

    if gpu_info.use_gpu && n_points > 500
        trajectory_results = simulate_trajectory_error_gpu(trajectory_data, base_params, gpu_info);
    else
        trajectory_results = simulate_trajectory_error(trajectory_data, base_params);
    end
end

% 合并结果
simulation_data = combine_results(trajectory_data, trajectory_results, base_params);

% 保存
save(output_file, 'simulation_data', '-v7.3');

end

function simulation_data = combine_results(trajectory, trajectory_results, params)
% 合并仿真结果（仅轨迹误差相关）

simulation_data = [];

% 时间和轨迹
simulation_data.time = trajectory.time(:);
simulation_data.x_ref = trajectory.x(:);
simulation_data.y_ref = trajectory.y(:);
simulation_data.z_ref = trajectory.z(:);
simulation_data.x_act = trajectory_results.x_act(:);
simulation_data.y_act = trajectory_results.y_act(:);
simulation_data.z_act = trajectory_results.z_act(:);

% 运动学
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

% 动力学
simulation_data.F_inertia_x = trajectory_results.F_inertia_x(:);
simulation_data.F_inertia_y = trajectory_results.F_inertia_y(:);
simulation_data.F_elastic_x = trajectory_results.F_elastic_x(:);
simulation_data.F_elastic_y = trajectory_results.F_elastic_y(:);
simulation_data.belt_stretch_x = trajectory_results.belt_stretch_x(:);
simulation_data.belt_stretch_y = trajectory_results.belt_stretch_y(:);

% 轨迹误差
simulation_data.error_x = trajectory_results.error_x(:);
simulation_data.error_y = trajectory_results.error_y(:);
simulation_data.error_magnitude = trajectory_results.error_magnitude(:);
simulation_data.error_direction = trajectory_results.error_direction(:);

% 元数据
simulation_data.is_extruding = trajectory.is_extruding(:);
simulation_data.print_type = trajectory.print_type;
simulation_data.layer_num = trajectory.layer_num(:);
simulation_data.params = params;

end

%% ========== 验证函数 ==========

function result = is_gcode_file(val)
result = ischar(val) || iscell(val);
if iscell(val)
    result = all(cellfun(@ischar, val));
end
end

function result = is_valid_config(val)
result = ischar(val) || isnumeric(val) || iscell(val);
if iscell(val)
    result = all(cellfun(@(x) ischar(x) || isnumeric(x), val));
end
end
