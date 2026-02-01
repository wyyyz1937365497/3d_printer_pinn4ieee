% COLLECT_BOAT - 收集simple_boat5数据
%
% 收集simple_boat5_PLA_4h4m.gcode的仿真数据
% 默认配置：采样间隔5层（共74层）
%
% 用法:
%   collect_boat               % 使用默认配置（采样）
%   collect_boat('all')        % 收集所有369层
%   collect_boat('sampled:3')  % 采样间隔3层
%   collect_boat(1:100)        % 收集指定范围
%
% Author: 3D Printer PINN Project
% Date: 2026-01-31

function collect_boat(layer_config)
% COLLECT_BOAT 收集simple_boat5数据

if nargin < 1
    % 默认：采样间隔5层
    layer_config = 'sampled:5';
end

gcode_file = 'test_gcode_files/simple_boat5_PLA_4h4m.gcode';

collect_data(gcode_file, layer_config, 'GPU', 1);

end
