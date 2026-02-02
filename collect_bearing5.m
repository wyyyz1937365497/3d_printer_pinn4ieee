% COLLECT_BEARING5 - 收集bearing5数据
%
% 收集bearing5_PLA_2h27m.gcode的仿真数据
% 默认配置：收集所有75层
%
% 用法:
%   collect_bearing5           % 使用默认配置（所有层）
%   collect_bearing5('all')    % 收集所有层
%   collect_bearing5('sampled:5')  % 采样间隔5层
%   collect_bearing5(46:75)    % 收集指定范围
%
% Author: 3D Printer PINN Project
% Date: 2026-01-31

function collect_bearing5(layer_config)
% COLLECT_BEARING5 收集bearing5数据

if nargin < 1
    % 默认：收集所有层
    layer_config = 'all';
end

gcode_file = 'test_gcode_files/bearing5_PLA_2h27m.gcode';

collect_data(gcode_file, layer_config, 'GPU', 1);

end
