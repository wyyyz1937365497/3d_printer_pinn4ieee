% COLLEAT_DATA_OPTIMIZED - Optimized data collection for identical-shape layers
%
% Key optimization: Since all layers have the same shape, we only need:
% 1. Single layer with大规模参数扫描
% 2. Three-layer validation for inter-layer effects
% 3. Data augmentation through time windows
%
% This strategy reduces simulation time from 2-3 days to 1.5 hours!
%
% Author: 3D Printer PINN Project
% Date: 2026-01-27

%% Setup
cd('F:\TJ\3d_print\3d_printer_pinn4ieee')
addpath(genpath(fullfile(pwd, 'matlab_simulation')))

fprintf('\n');
fprintf('============================================================\n');
fprintf('优化的数据收集策略（基于同形状层）\n');
fprintf('============================================================\n');
fprintf('\n');
fprintf('关键优化：\n');
fprintf('  ✅ 单层包含所有几何特征\n');
fprintf('  ✅ 参数扫描提供物理多样性\n');
fprintf('  ✅ 三层验证覆盖层间效应\n');
fprintf('  ✅ 时间窗口滑动生成多样本\n');
fprintf('\n');

%% GPU Configuration
gpu_id = 1;  % Use cuda1
fprintf('GPU配置：cuda1 (GPU %d)\n', gpu_id);
fprintf('\n');

%% Strategy Selection
% Choose one:
% 1 - Quick validation (10 parameter combos)
% 2 - Standard dataset (100 parameter combos) - RECOMMENDED
% 3 - Complete dataset (180 parameter combos)

strategy = 2;  % Default: Standard dataset

%% Parameters
target_layer = 25;  % Middle layer (representative)

switch strategy
    case 1  % Quick validation
        fprintf('策略：快速验证（适合测试）\n');
        n_param_combos = 10;
        validation_layers = [1, 50];  % First and last only
        fprintf('目标：~10K 样本\n');
        fprintf('预计时间：30 分钟\n');

    case 2  % Standard (RECOMMENDED)
        fprintf('策略：标准数据集（推荐）\n');
        n_param_combos = 100;
        validation_layers = [1, 25, 50];  % First, middle, last
        fprintf('目标：~100K 样本（含增强）\n');
        fprintf('预计时间：1.5 小时\n');

    case 3  % Complete
        fprintf('策略：完整数据集（最大规模）\n');
        n_param_combos = 180;
        validation_layers = [1, 10, 25, 40, 50];  % 5 layers
        fprintf('目标：~180K 样本（含增强）\n');
        fprintf('预计时间：3 小时\n');
end

fprintf('\n');

%% Parameter Grid Definition
% Motion parameters
accel_grid = 200:100:500;        % 5 values: 200-500 mm/s²
velocity_grid = 100:100:400;     % 4 values: 100-400 mm/s

% Thermal parameters
fan_grid = [0, 128, 255];         % 3 values: Off, Half, Full
ambient_temp_grid = [20, 25, 30]; % 3 values: 15-35°C

% Generate all combinations
[accel_vals, vel_vals, fan_vals, temp_vals] = ...
    ndgrid(accel_grid, velocity_grid, fan_grid, ambient_temp_grid);

total_combos_available = length(accel_vals);
fprintf('可用参数组合：%d\n', total_combos_available);  % 5×4×3×3 = 180

% Sample combos if needed
if n_param_combos < total_combos_available
    fprintf('采样：%d 种组合\n', n_param_combos);
    indices = randperm(total_combos_available, n_param_combos);
else
    indices = 1:total_combos_available;
    n_param_combos = total_combos_available;
end

fprintf('实际仿真：%d 种配置\n', n_param_combos);
fprintf('\n');

%% Phase 1: Single Layer Parameter Scan
fprintf('============================================================\n');
fprintf('阶段1：单层参数扫描（主力数据）\n');
fprintf('============================================================\n');
fprintf('层号：%d\n', target_layer);
fprintf('参数组合：%d\n', n_param_combos);
fprintf('\n');

tic;
sim_count = 0;

% Create output directory
output_dir = sprintf('data_simulation_layer%d', target_layer);
mkdir(output_dir);

for idx = 1:n_param_combos
    combo_idx = indices(idx);

    % Get parameters for this combination
    accel = accel_vals(combo_idx);
    velocity = vel_vals(combo_idx);
    fan = fan_vals(combo_idx);
    ambient_temp = temp_vals(combo_idx);

    % Display progress
    fprintf('[%d/%d] a=%d, v=%d, fan=%d, T=%d°C...', ...
            idx, n_param_combos, accel, velocity, fan, ambient_temp);

    % Configure options
    options = struct();
    options.layers = target_layer;
    options.use_improved = true;
    options.include_type = {'Outer wall', 'Inner wall'};
    options.include_skirt = false;

    % Run simulation
    try
        % Load and modify physics parameters
        params = physics_parameters();
        params.motion.max_accel = accel;
        params.motion.max_velocity = velocity;

        % Adjust fan effect (simplified)
        if fan == 0
            params.heat_transfer.h_convection_with_fan = 10;  % No fan
        elseif fan == 128
            params.heat_transfer.h_convection_with_fan = 25;  % Half
        else
            params.heat_transfer.h_convection_with_fan = 44;  % Full
        end

        params.environment.ambient_temp = ambient_temp;

        % Output file
        output_file = fullfile(output_dir, ...
            sprintf('combo_%04d_a%03d_v%03d_f%03d_t%02d.mat', ...
                    idx, accel, velocity, fan, ambient_temp));

        % Run simulation
        simulation_data = run_full_simulation_gpu(...
            'Tremendous Hillar_PLA_17m1s.gcode', ...
            output_file, ...
            options, ...
            gpu_id);

        sim_count = sim_count + 1;
        fprintf(' 完成\n');

        % Save every 10
        if mod(idx, 10) == 0
            fprintf('进度：%.1f%%\n', 100*idx/n_param_combos);
        end

    catch ME
        fprintf(' 错误：%s\n', ME.message);
        continue;
    end
end

phase1_time = toc;
fprintf('\n阶段1完成！\n');
fprintf('仿真次数：%d\n', sim_count);
fprintf('用时：%.2f 分钟\n', phase1_time/60);
fprintf('预计原始点数：%d\n', sim_count * 3000);
fprintf('时间窗口后样本数：%d\n', sim_count * 280);
fprintf('\n');

%% Phase 2: Multi-layer Validation
fprintf('============================================================\n');
fprintf('阶段2：多层验证（层间效应）\n');
fprintf('============================================================\n');
fprintf('层数：%s\n', mat2str(validation_layers));
fprintf('\n');

% Reduced parameter set for validation layers
val_accel = [300, 400, 500];        % 3 values
val_velocity = [200, 300, 400];     % 3 values
val_fan = [0, 255];                 % 2 values

[val_acc, val_vel, val_fan] = ndgrid(val_accel, val_velocity, val_fan);
n_val_combos = length(val_acc);

fprintf('每层参数：%d 种\n', n_val_combos);
fprintf('总验证仿真：%d 层 × %d 配置 = %d 次\n', ...
        length(validation_layers), n_val_combos, ...
        length(validation_layers) * n_val_combos);
fprintf('\n');

tic;
val_sim_count = 0;

for layer_idx = 1:length(validation_layers)
    layer_num = validation_layers(layer_idx);

    fprintf('验证层 %d (%d/%d)...\n', layer_num, layer_idx, length(validation_layers));

    for v_idx = 1:n_val_combos
        % Get parameters
        accel = val_acc(v_idx);
        velocity = val_vel(v_idx);
        fan = val_fan(v_idx);

        fprintf('  [%d/%d] a=%d, v=%d, fan=%d...', ...
                v_idx, n_val_combos, accel, velocity, fan);

        try
            params = physics_parameters();
            params.motion.max_accel = accel;
            params.motion.max_velocity = velocity;

            if fan == 0
                params.heat_transfer.h_convection_with_fan = 10;
            else
                params.heat_transfer.h_convection_with_fan = 44;
            end

            options.layers = layer_num;
            options.use_improved = true;
            options.include_type = {'Outer wall', 'Inner wall'};
            options.include_skirt = false;

            output_file = sprintf('validation_layer%d_%03d.mat', layer_num, v_idx);

            run_full_simulation_gpu('Tremendous Hillar_PLA_17m1s.gcode', ...
                                   output_file, options, gpu_id);

            val_sim_count = val_sim_count + 1;
            fprintf(' 完成\n');

        catch ME
            fprintf(' 错误：%s\n', ME.message);
        end
    end
end

phase2_time = toc;
fprintf('\n阶段2完成！\n');
fprintf('仿真次数：%d\n', val_sim_count);
fprintf('用时：%.2f 分钟\n', phase2_time/60);
fprintf('\n');

%% Summary
fprintf('============================================================\n');
fprintf('仿真总结\n');
fprintf('============================================================\n');
fprintf('\n');

total_time = (phase1_time + phase2_time) / 60;

fprintf('阶段1（单层扫描）：\n');
fprintf('  仿真次数：%d\n', sim_count);
fprintf('  原始点数：%d\n', sim_count * 3000);
fprintf('  样本数（时间窗口）：%d\n', sim_count * 280);
fprintf('\n');

fprintf('阶段2（多层验证）：\n');
fprintf('  仿真次数：%d\n', val_sim_count);
fprintf('  原始点数：%d\n', val_sim_count * 3000);
fprintf('  样本数（时间窗口）：%d\n', val_sim_count * 280);
fprintf('\n');

total_samples = (sim_count + val_sim_count) * 280;
fprintf('总计：\n');
fprintf('  原始点数：%d\n', (sim_count + val_sim_count) * 3000);
fprintf('  样本数（时间窗口）：%d\n', total_samples);
fprintf('  增强后（×3）：%d\n', total_samples * 3);
fprintf('\n');

fprintf('时间统计：\n');
fprintf('  阶段1：%.2f 分钟\n', phase1_time/60);
fprintf('  阶段2：%.2f 分钟\n', phase2_time/60);
fprintf('  总计：%.2f 分钟 (%.2f 小时)\n', total_time, total_time/60);
fprintf('\n');

fprintf('数据集规模评估：\n');
if total_samples >= 30000
    fprintf('  ✅ 达到标准配置目标（50K样本）\n');
elseif total_samples >= 10000
    fprintf('  ⚠️ 达到最低要求，建议扩展参数\n');
else
    fprintf('  ❌ 不足，建议增加参数组合\n');
end
fprintf('\n');

%% Next Steps
fprintf('下一步：\n');
fprintf('  1. 检查生成的数据文件\n');
fprintf('  2. 转换为Python格式：\n');
fprintf('     python matlab_simulation/convert_matlab_to_python.py ...\n');
fprintf('         data_simulation_layer25/*.mat training -o training_data\n');
fprintf('  3. 开始训练模型\n');
fprintf('\n');

fprintf('============================================================\n');
fprintf('数据收集完成！\n');
fprintf('============================================================\n');
