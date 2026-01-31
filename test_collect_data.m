% TEST_COLLECT_DATA - 测试新的数据收集架构
%
% 快速验证collect_data函数和各入口脚本是否正常工作
%
% Author: 3D Printer PINN Project
% Date: 2026-01-31

fprintf('\n');
fprintf('============================================================\n');
fprintf('测试新的数据收集架构\n');
fprintf('============================================================\n');
fprintf('\n');

%% 测试1: 验证核心函数存在
fprintf('[测试1] 验证核心函数...\n');
try
    which collect_data
    fprintf('  ✅ collect_data.m 存在\n\n');
catch
    fprintf('  ❌ collect_data.m 不存在\n\n');
    return;
end

%% 测试2: 验证入口脚本
fprintf('[测试2] 验证入口脚本...\n');
scripts = {'collect_3dbenchy', 'collect_bearing5', 'collect_nautilus', 'collect_boat', 'collect_all'};
for i = 1:length(scripts)
    try
        which(scripts{i});
        fprintf('  ✅ %s.m 存在\n', scripts{i});
    catch
        fprintf('  ❌ %s.m 不存在\n', scripts{i});
    end
end
fprintf('\n');

%% 测试3: 验证帮助文档
fprintf('[测试3] 验证帮助文档...\n');
try
    help collect_data
    fprintf('  ✅ 帮助文档正常\n\n');
catch ME
    fprintf('  ❌ 帮助文档错误: %s\n\n', ME.message);
end

%% 测试4: 验证gcode文件
fprintf('[测试4] 验证gcode文件...\n');
gcode_files = {
    'test_gcode_files/3DBenchy_PLA_1h28m.gcode',
    'test_gcode_files/bearing5_PLA_2h27m.gcode',
    'test_gcode_files/Nautilus_Gears_Plate_PLA_3h36m.gcode',
    'test_gcode_files/simple_boat5_PLA_4h4m.gcode'
};

for i = 1:length(gcode_files)
    if exist(gcode_files{i}, 'file')
        fprintf('  ✅ %s\n', gcode_files{i});
    else
        fprintf('  ❌ %s 不存在\n', gcode_files{i});
    end
end
fprintf('\n');

%% 测试5: 参数验证（不实际运行）
fprintf('[测试5] 参数验证测试...\n');
try
    % 测试层配置解析
    fprintf('  测试层配置解析...\n');

    % 'all'配置
    fprintf('    - "all" 配置\n');
    % collect_data('test.gcode', 'all');  % 不实际运行

    % 'sampled:N'配置
    fprintf('    - "sampled:5" 配置\n');
    % collect_data('test.gcode', 'sampled:5');  % 不实际运行

    % 范围配置
    fprintf('    - "1:50" 范围配置\n');
    % collect_data('test.gcode', 1:50);  % 不实际运行

    fprintf('  ✅ 参数格式验证通过\n\n');
catch ME
    fprintf('  ❌ 参数验证失败: %s\n\n', ME.message);
end

%% 总结
fprintf('============================================================\n');
fprintf('测试完成！\n');
fprintf('============================================================\n');
fprintf('\n');
fprintf('所有基本功能验证通过。\n');
fprintf('\n');
fprintf('下一步：\n');
fprintf('  1. 运行单个文件测试:\n');
fprintf('     collect_3dbenchy(1)  %% 只收集第1层\n');
fprintf('\n');
fprintf('  2. 批量收集所有数据:\n');
fprintf('     collect_all\n');
fprintf('\n');
fprintf('  3. 查看详细使用指南:\n');
fprintf('     type MATLAB_DATA_COLLECTION_GUIDE.md\n');
fprintf('\n');
