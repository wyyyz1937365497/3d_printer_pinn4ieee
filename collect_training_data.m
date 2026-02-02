%% 数据收集脚本 - 批量生成训练数据（自适应层数）
%
% 使用方法：
%   1. 配置下面的参数（gcode文件列表、采样间隔等）
%   2. 运行此脚本
%   3. 脚本会自动检测每个gcode文件的总层数，并自适应采样
%
% 输出：
%   - data/simulation/ 目录下生成多个 .mat 文件
%   - 文件命名格式: <gcode_abbrev>_layer<NN>.mat
%
% 作者: 3D Printer PINN Project
% 日期: 2026-02-02

%% =========================== 辅助函数 ===========================

function total_layers = detect_total_layers(gcode_file, params)
    % 检测gcode文件的总层数
    % 优先使用文件头注释，备用解析器方法

    % 方法1: 读取文件头的 "total layer number" 注释（最快速）
    total_layers = read_total_layer_from_header(gcode_file);

    if total_layers > 0
        return;  % 成功找到
    end

    % 方法2: 如果方法1失败，尝试使用解析器
    try
        fprintf(' [使用解析器]');
        parser_options = struct();
        parser_options.use_improved = true;
        parser_options.layers = 'all';
        parser_options.include_skirt = false;
        parser_options.include_type = {'Outer wall', 'Inner wall'};

        trajectory_data = parse_gcode_improved(gcode_file, params, parser_options);
        total_layers = max(trajectory_data.layer_num);

    catch ME
        % 方法3: 解析器也失败，扫描文件统计层注释
        try
            fprintf(' [扫描文件]');
            total_layers = count_layers_by_scanning(gcode_file);
        catch ME2
            warning('检测层数失败: %s (扫描: %s)', ME.message, ME2.message);
            total_layers = 0;
        end
    end
end

function n_layers = read_total_layer_from_header(gcode_file)
    % 从文件头读取 "total layer number" 注释
    % 这是Cura/PrusaSlicer等切片软件的标准格式

    n_layers = 0;

    % 尝试UTF-8编码
    fid = fopen(gcode_file, 'r', 'n', 'UTF-8');
    if fid == -1
        % 失败则用系统默认编码
        fid = fopen(gcode_file, 'r');
    end

    if fid == -1
        return;
    end

    % 只读取前100行，查找总层数注释
    max_lines = 100;
    line_count = 0;

    while ~feof(fid) && line_count < max_lines
        line = fgetl(fid);
        line_count = line_count + 1;

        if ischar(line)
            % 匹配 "; total layer number: XX" 或类似的格式
            % 大小写不敏感，允许空格变化
            match = regexp(line, 'total\s+layer\s+number\s*:\s*(\d+)', 'tokens', 'ignorecase');

            if ~isempty(match)
                n_layers = str2double(match{1}{1});
                break;  % 找到后立即退出
            end
        end
    end

    fclose(fid);
end

function n_layers = count_layers_by_scanning(gcode_file)
    % 通过扫描gcode文件统计层数
    % 查找包含 "LAYER:" 或 "LAYER " 的注释行

    % 读取文件（使用自动编码检测）
    fid = fopen(gcode_file, 'r', 'n', 'UTF-8');
    if fid == -1
        % 如果UTF-8失败，尝试系统默认编码
        fid = fopen(gcode_file, 'r');
    end

    if fid == -1
        error('无法打开文件');
    end

    n_layers = 0;
    layer_numbers = [];

    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line)
            % 查找层注释：LAYER:X, LAYER X, 或者 ;LAYER:X
            tokens = regexp(line, '(?:LAYER[:\s]|;LAYER[:\s])\s*(\d+)', 'tokens');

            if ~isempty(tokens)
                for i = 1:length(tokens)
                    layer_num = str2double(tokens{i}{1});
                    if ~isempty(layer_num) && layer_num > 0
                        layer_numbers = [layer_numbers; layer_num];
                    end
                end
            end
        end
    end

    fclose(fid);

    if ~isempty(layer_numbers)
        n_layers = max(layer_numbers);
    else
        % 如果找不到层标记，尝试另一种方法：统计 Z 高度变化
        n_layers = count_layers_by_z_height(gcode_file);
    end
end

function n_layers = count_layers_by_z_height(gcode_file)
    % 通过Z高度变化统计层数

    fid = fopen(gcode_file, 'r', 'n', 'UTF-8');
    if fid == -1
        fid = fopen(gcode_file, 'r');
    end

    if fid == -1
        error('无法打开文件');
    end

    z_heights = [];
    layer_height = 0.2;  % 默认层高

    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line)
            % 查找 G1 Z... 命令
            z_match = regexp(line, 'G[01].*Z([\d.\-]+)', 'tokens');
            if ~isempty(z_match)
                z = str2double(z_match{1}{1});
                if ~isempty(z) && z > 0
                    z_heights = [z_heights; z];
                end
            end

            % 尝试从注释中获取层高
            layer_h_match = regexp(line, 'layer.height[:\s]+([\d\.]+)', 'tokens', 'ignorecase');
            if ~isempty(layer_h_match)
                layer_height = str2double(layer_h_match{1}{1});
            end
        end
    end

    fclose(fid);

    if ~isempty(z_heights)
        % 估计层数
        min_z = min(z_heights);
        max_z = max(z_heights);

        if layer_height > 0
            n_layers = round((max_z - min_z) / layer_height) + 1;
        else
            n_layers = length(unique(round(z_heights * 10) / 10));  % 粗略估计
        end
    else
        n_layers = 0;
    end
end

%% =========================== 脚本开始 ===========================

clear; clc;

%% =========================== 配置参数 ===========================

% 1. G-code文件配置
% 指定要处理的gcode文件列表
GCODE_FILES = {
    'test_gcode_files/3DBenchy_PLA_1h28m.gcode',      % 3DBenchy PLA
    'test_gcode_files/bearing5_PLA_2h27m.gcode',      % Boat (如果存在)
    'test_gcode_files/Nautilus_Gears_Plate_PLA_3h36m.gcode', % Bearing (如果存在)
    'test_gcode_files/simple_boat5_PLA_4h4m.gcode',  % Nautilus (如果存在)
    % 添加更多gcode文件...
};

% 2. 层采样配置（自适应）
LAYER_START = 1;       % 起始层（建议从第1层开始）
LAYER_STEP = 2;        % 采样间隔（每隔几层采样一次）
                      % 1 = 每层都采样（数据最多，时间最长）
                      % 2 = 隔层采样（推荐，平衡数据量和时间）
                      % 3 = 每3层采样1次（快速，数据较少）
MAX_LAYERS = 50;       % 每个文件最多采集多少层（防止过大模型）

% 3. 仿真配置
USE_GPU = true;        % 是否使用GPU加速（如果可用）
FIRMWARE_EFFECTS = true;  % 包含固件效应（junction deviation, resonance, timer jitter）
TIME_STEP = 0.01;      % 仿真时间步长（秒）
VERBOSE = false;       % 是否显示详细进度（false可加快速度）

% 4. 输出配置
OUTPUT_DIR = 'data/simulation';  % 输出目录

% 添加路径
addpath('simulation');

% 创建输出目录
if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
    fprintf('创建输出目录: %s\n', OUTPUT_DIR);
end

%% 打印配置信息
fprintf('\n');
fprintf('==============================================================\n');
fprintf('数据收集配置（自适应层数）\n');
fprintf('==============================================================\n');
fprintf('\n');

fprintf('G-code文件 (%d个):\n', length(GCODE_FILES));
for i = 1:length(GCODE_FILES)
    exists_str = '';
    if ~exist(GCODE_FILES{i}, 'file')
        exists_str = ' [不存在]';
    end
    fprintf('  %d. %s%s\n', i, GCODE_FILES{i}, exists_str);
end

fprintf('\n层采样配置:\n');
fprintf('  起始层: %d\n', LAYER_START);
fprintf('  采样间隔: %d (每隔%d层采样一次)\n', LAYER_STEP, LAYER_STEP);
fprintf('  每文件最多采集: %d 层\n', MAX_LAYERS);

fprintf('\n仿真配置:\n');
fprintf('  GPU加速: %s\n', mat2str(USE_GPU));
fprintf('  固件效应: %s\n', mat2str(FIRMWARE_EFFECTS));
fprintf('  时间步长: %.3f ms (%.0f Hz)\n', TIME_STEP*1000, 1/TIME_STEP);
fprintf('  输出目录: %s\n', OUTPUT_DIR);

fprintf('\n==============================================================\n');
fprintf('第1步: 检测文件层数...\n');
fprintf('==============================================================\n');
fprintf('\n');

%% 加载物理参数（用于解析）
params = physics_parameters();

%% 检测每个文件的总层数
file_info = {};
total_tasks = 0;

for gcode_idx = 1:length(GCODE_FILES)
    gcode_file = GCODE_FILES{gcode_idx};

    fprintf('[%d/%d] %s\n', gcode_idx, length(GCODE_FILES), gcode_file);

    % 检查文件是否存在
    if ~exist(gcode_file, 'file')
        fprintf('  ⚠ 文件不存在，跳过\n');
        continue;
    end

    % 检测总层数
    fprintf('  检测层数...');
    total_layers = detect_total_layers(gcode_file, params);

    if total_layers > 0
        fprintf(' ✓ 共 %d 层\n', total_layers);

        % 计算要采集的层
        layer_end = min(total_layers, LAYER_START + (MAX_LAYERS-1) * LAYER_STEP);
        layers_to_collect = LAYER_START:LAYER_STEP:layer_end;
        n_layers = length(layers_to_collect);

        fprintf('  将采集: 第 %s 层 (共%d层)\n', ...
            mat2str(layers_to_collect(1:min(5, end))), n_layers);
        if n_layers > 5
            fprintf('         ... 第 %s 层\n', mat2str(layers_to_collect(end-4:end)));
        end

        % 保存信息
        file_info{end+1} = struct(...
            'gcode_file', gcode_file, ...
            'total_layers', total_layers, ...
            'layers_to_collect', layers_to_collect, ...
            'n_layers', n_layers);

        total_tasks = total_tasks + n_layers;
    else
        fprintf(' ✗ 检测失败\n');
    end

    fprintf('\n');
end

if isempty(file_info)
    fprintf('❌ 没有有效的gcode文件！\n');
    return;
end

fprintf('\n==============================================================\n');
fprintf('第2步: 开始数据采集...\n');
fprintf('==============================================================\n');
fprintf('\n');

fprintf('汇总:\n');
fprintf('  有效文件数: %d\n', length(file_info));
fprintf('  总任务数: %d 个层\n', total_tasks);
fprintf('\n');

%% 统计变量
completed_files = 0;
failed_files = 0;
start_time = tic;

%% 循环处理每个gcode文件
for file_idx = 1:length(file_info)
    info = file_info{file_idx};
    gcode_file = info.gcode_file;
    layers_to_collect = info.layers_to_collect;
    n_layers = info.n_layers;

    % 提取gcode文件名缩写
    [~, gcode_name, ~] = fileparts(gcode_file);
    gcode_abbrev = gcode_name;

    % 如果文件名太长，截断
    if length(gcode_abbrev) > 20
        gcode_abbrev = gcode_abbrev(1:20);
    end

    fprintf('\n[%d/%d] 处理文件: %s (共%d层)\n', ...
        file_idx, length(file_info), gcode_name, info.total_layers);
    fprintf('--------------------------------------------------------------\n');

    % 循环处理每一层
    for layer_idx = 1:n_layers
        layer_num = layers_to_collect(layer_idx);

        % 生成输出文件名
        output_file = fullfile(OUTPUT_DIR, ...
            sprintf('%s_layer%02d.mat', gcode_abbrev, layer_num));

        % 检查文件是否已存在
        if exist(output_file, 'file')
            fprintf('  [%2d/%2d] 第%2d层: 已存在，跳过\n', ...
                layer_idx, n_layers, layer_num);
            completed_files = completed_files + 1;
            continue;
        end

        fprintf('  [%2d/%2d] 第%2d层: 仿真中...', ...
            layer_idx, n_layers, layer_num);

        try
            % 运行仿真
            simulation_data = run_simulation(...
                gcode_file, ...
                'Layers', layer_num, ...
                'OutputFile', output_file, ...
                'UseGPU', USE_GPU, ...
                'FirmwareEffects', FIRMWARE_EFFECTS, ...
                'TimeStep', TIME_STEP, ...
                'Verbose', VERBOSE);

            fprintf(' ✓ 完成\n');
            completed_files = completed_files + 1;

        catch ME
            fprintf(' ✗ 失败\n');
            fprintf('      错误: %s\n', strrep(ME.message, newline, ' '));
            failed_files = failed_files + 1;
        end
    end
end

%% 完成统计
elapsed_time = toc(start_time);

fprintf('\n');
fprintf('==============================================================\n');
fprintf('数据收集完成！\n');
fprintf('==============================================================\n');
fprintf('\n');

fprintf('统计信息:\n');
fprintf('  总任务数: %d\n', total_tasks);
fprintf('  成功: %d\n', completed_files);
fprintf('  失败: %d\n', failed_files);
if total_tasks > 0
    fprintf('  成功率: %.1f%%\n', completed_files/total_tasks*100);
end
fprintf('\n');

fprintf('时间统计:\n');
fprintf('  总耗时: %.1f 分钟\n', elapsed_time/60);
if completed_files > 0
    fprintf('  平均每个文件: %.1f 秒\n', elapsed_time/completed_files);
end
fprintf('\n');

fprintf('输出目录: %s\n', OUTPUT_DIR);

% 列出生成的文件
output_files = dir(fullfile(OUTPUT_DIR, '*.mat'));
fprintf('生成文件数: %d\n', length(output_files));
fprintf('\n');

% 计算总大小
total_size = 0;
for i = 1:length(output_files)
    total_size = total_size + output_files(i).bytes;
end

fprintf('总数据大小: %.2f MB\n', total_size/1024/1024);
fprintf('\n');

%% 提示后续步骤
fprintf('==============================================================\n');
fprintf('后续步骤:\n');
fprintf('==============================================================\n');
fprintf('\n');
fprintf('1. 检查生成的数据:\n');
fprintf('   >> python check_training_data.py --data_dir "%s/*.mat"\n', OUTPUT_DIR);
fprintf('\n');
fprintf('2. 开始训练模型:\n');
fprintf('   >> python experiments/train_realtime.py --data_dir "%s/*.mat"\n', OUTPUT_DIR);
fprintf('\n');
fprintf('3. 可视化热图对比（训练完成后）:\n');
fprintf('   >> python experiments/visualize_realtime_correction.py \\\n');
fprintf('        --checkpoint checkpoints/realtime_corrector/best_model.pth \\\n');
fprintf('        --gcode test_gcode_files/3DBenchy_PLA_1h28m.gcode --layer 25\n');
fprintf('\n');
