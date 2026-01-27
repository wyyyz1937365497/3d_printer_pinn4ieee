cd('f:\TJ\3d_print\3d_printer_pinn4ieee')
addpath(genpath(fullfile(pwd, 'matlab_simulation')))
% 设置仿真选项
options = struct();
options.layers = 10;                                    % 指定第10层
options.include_type = {'Outer wall'};                  % 仅包含外墙
options.include_skirt = false;                          % 不包含裙边
options.use_improved = true;                            % 使用改进的解析器

% 运行完整仿真
gcode_file = 'Tremendous Hillar_PLA_17m1s.gcode';
output_file = 'layer10_outer_wall_simulation.mat';

simulation_data = run_full_simulation(gcode_file, output_file, options);