% COLLECT_3DBENCHY - 收集3DBenchy数据
%
% 收集3DBenchy_PLA_1h28m.gcode的仿真数据
% 默认配置：采样间隔5层（共48层）
%
% 用法:
%   collect_3dbenchy           % 使用默认配置
%   collect_3dbenchy('all')    % 收集所有240层
%   collect_3dbenchy(1:50)     % 收集指定范围
%
% Author: 3D Printer PINN Project
% Date: 2026-01-31

function collect_3dbenchy(layer_config)
% COLLECT_3DBENCHY 收集3DBenchy数据

if nargin < 1
    % 默认：采样间隔5层
    layer_config = 'sampled:5';
end

gcode_file = 'test_gcode_files/3DBenchy_PLA_1h28m.gcode';

collect_data(gcode_file, layer_config, 'GPU', 1);

end
