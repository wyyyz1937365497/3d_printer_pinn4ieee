% REGENERATE_ALL_DATASETS - 使用更新后的参数重新生成所有数据集
%
% 该脚本批量重新生成所有仿真数据,使用更新后的物理参数
% 新参数基于真实文献,产生 ±50-100 μm 的轨迹误差
%
% 执行前请确认:
% 1. 已运行 test_new_parameters.m 并验证参数正确
% 2. 备份旧数据(可选)
% 3. 确保有足够的磁盘空间(约500MB)
%
% 预计时间: 取决于硬件,CPU版本约1-2小时,GPU版本约30分钟

clear all; close all; clc;

fprintf('============================================================\n');
fprintf('批量重新生成所有仿真数据集\n');
fprintf('============================================================\n');
fprintf('\n');

%% 配置
% G-code文件列表
gcode_files = {
    '../test_gcode_files/3DBenchy_PLA_1h28m.gcode'
    '../test_gcode_files/bearing5_PLA_2h27m.gcode'
    '../test_gcode_files/Nautilus_Gears_Plate_PLA_3h36m.gcode'
    '../test_gcode_files/simple_boat5_PLA_4h4m.gcode'
};

% 输出目录前缀
output_prefix = 'data_simulation_';

% 仿真选项
opts = struct();
opts.use_improved = true;
opts.layers = 'all';  % 仿真所有层
opts.include_skirt = false;

%% 显示配置信息
fprintf('配置信息:\n');
fprintf('--------------------------------------------------\n');
fprintf('G-code文件数量: %d\n', length(gcode_files));
fprintf('仿真模式: 所有层\n');
fprintf('使用改进解析器: 是\n');
fprintf('--------------------------------------------------\n');
fprintf('\n');

% 显示文件列表
fprintf('将要处理的文件:\n');
for i = 1:length(gcode_files)
    [~, name, ~] = fileparts(gcode_files{i});
    fprintf('  %d. %s\n', i, name);
end
fprintf('\n');

%% 确认继续
fprintf('警告: 这将重新生成所有数据,覆盖现有文件!\n');
response = input('继续? (y/n): ', 's');

if ~strcmpi(response, 'y')
    fprintf('已取消。\n');
    return;
end

fprintf('\n');

%% 开始处理
start_time = datetime('now');
fprintf('开始时间: %s\n', datestr(start_time));
fprintf('\n');

% 初始化统计
total_files = length(gcode_files);
success_count = 0;
fail_count = 0;
results = cell(total_files, 1);

%% 处理每个G-code文件
for i = 1:total_files
    fprintf('============================================================\n');
    fprintf('处理文件 %d/%d\n', i, total_files);
    fprintf('============================================================\n');

    gcode_file = gcode_files{i};
    [~, name, ~] = fileparts(gcode_file);

    fprintf('文件: %s\n', gcode_file);
    fprintf('名称: %s\n', name);
    fprintf('\n');

    % 创建输出目录
    output_dir = [output_prefix, name];
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
        fprintf('创建输出目录: %s\n', output_dir);
    end

    % 运行仿真
    try
        fprintf('开始仿真...\n');
        tic;

        % 运行完整仿真
        simulation_data = run_full_simulation(gcode_file, [], opts);

        elapsed = toc;
        fprintf('仿真完成! 用时: %.1f 秒\n', elapsed);

        % 保存每层数据到单独的.mat文件
        fprintf('保存数据...\n');

        % 获取层数
        num_layers = length(simulation_data.trajectory_results.layer_data);

        for layer_idx = 1:num_layers
            % 提取该层数据
            layer_data = simulation_data.trajectory_results.layer_data{layer_idx};

            % 构造输出文件名
            layer_num = layer_data.layer_number;
            output_file = sprintf('%s/layer%03d_ender3v2.mat', output_dir, layer_num);

            % 保存
            save(output_file, 'layer_data', '-v7.3');
        end

        fprintf('已保存 %d 层数据到 %s\n', num_layers, output_dir);

        % 记录成功
        success_count = success_count + 1;
        results{i} = struct(...
            'name', name, ...
            'status', 'success', ...
            'num_layers', num_layers, ...
            'time', elapsed, ...
            'output_dir', output_dir);

        fprintf('✓ 文件 %d 处理成功\n', i);

    catch ME
        % 记录失败
        fail_count = fail_count + 1;
        results{i} = struct(...
            'name', name, ...
            'status', 'failed', ...
            'error', ME.message, ...
            'error_file', ME.stack(1).name);

        fprintf('✗ 文件 %d 处理失败!\n', i);
        fprintf('  错误: %s\n', ME.message);
        fprintf('  位置: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end

    fprintf('\n');
end

%% 完成总结
end_time = datetime('now');
total_time = end_time - start_time;

fprintf('============================================================\n');
fprintf('批量处理完成!\n');
fprintf('============================================================\n');
fprintf('\n');
fprintf('统计信息:\n');
fprintf('--------------------------------------------------\n');
fprintf('总文件数: %d\n', total_files);
fprintf('成功: %d\n', success_count);
fprintf('失败: %d\n', fail_count);
fprintf('成功率: %.1f%%\n', 100 * success_count / total_files);
fprintf('总用时: %s\n', char(total_time));
fprintf('平均用时: %.1f 分钟/文件\n', minutes(total_time) / total_files);
fprintf('--------------------------------------------------\n');
fprintf('\n');

% 显示详细结果
fprintf('详细结果:\n');
fprintf('--------------------------------------------------\n');
for i = 1:total_files
    if strcmp(results{i}.status, 'success')
        fprintf('✓ %s: %d 层, %.1f 秒\n', ...
            results{i}.name, results{i}.num_layers, results{i}.time);
    else
        fprintf('✗ %s: 失败 - %s\n', ...
            results{i}.name, results{i}.error);
    end
end
fprintf('--------------------------------------------------\n');
fprintf('\n');

% 保存处理日志
log_file = 'regeneration_log.mat';
save(log_file, 'results', 'start_time', 'end_time', 'total_time', ...
     'success_count', 'fail_count');
fprintf('处理日志已保存: %s\n', log_file);

fprintf('\n完成时间: %s\n', datestr(end_time));
fprintf('============================================================\n');

%% 如果有失败的文件,提供选项重新处理
if fail_count > 0
    fprintf('\n有 %d 个文件处理失败。\n', fail_count);
    response = input('是否仅重新处理失败的文件? (y/n): ', 's');

    if strcmpi(response, 'y')
        fprintf('\n重新处理失败的文件...\n');
        % TODO: 实现重新处理逻辑
    end
end
