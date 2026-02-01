% COLLECT_NAUTILUS - 收集Nautilus数据
%
% 收集Nautilus_Gears_Plate_PLA_3h36m.gcode的仿真数据
% 默认配置：收集所有56层
%
% 用法:
%   collect_nautilus           % 使用默认配置（所有层）
%   collect_nautilus('all')    % 收集所有层
%   collect_nautilus('sampled:5')  % 采样间隔5层
%   collect_nautilus(1:28)     % 收集指定范围
%
% Author: 3D Printer PINN Project
% Date: 2026-01-31

function collect_nautilus(layer_config)
% COLLECT_NAUTILUS 收集Nautilus数据

if nargin < 1
    % 默认：收集所有层
    layer_config = 'all';
end

gcode_file = 'test_gcode_files/Nautilus_Gears_Plate_PLA_3h36m.gcode';

collect_data(gcode_file, layer_config, 'GPU', 1);

end
