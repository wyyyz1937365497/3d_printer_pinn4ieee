% COLLECT_ALL - 批量收集所有gcode文件的数据（并行版本）
%
% 一键处理所有4个gcode文件
% 使用多核并行处理，大幅提升速度
%
% 用法:
%   collect_all                % 使用默认配置（并行）
%   collect_all('all')         % 收集所有文件的所有层
%   collect_all('sampled:5')   % 所有文件采样间隔5层
%
% Author: 3D Printer PINN Project
% Date: 2026-02-02

function collect_all(layer_config)
% COLLECT_ALL 批量收集所有文件（并行）

if nargin < 1
    % 默认配置（3DBenchy和boat采样，bearing5和nautilus全量）
    layer_config = {'all', 'all', 'all', 'all'};
elseif ischar(layer_config)
    % 统一配置应用到所有文件
    layer_config = repmat({layer_config}, 4, 1);
end

gcode_files = {
    'test_gcode_files/3DBenchy_PLA_1h28m.gcode',
    'test_gcode_files/bearing5_PLA_2h27m.gcode',
    'test_gcode_files/Nautilus_Gears_Plate_PLA_3h36m.gcode',
    'test_gcode_files/simple_boat5_PLA_4h4m.gcode'
};

% 使用并行版本
collect_data_parallel(gcode_files, layer_config, 'UseFirmwareEffects', true);

end
