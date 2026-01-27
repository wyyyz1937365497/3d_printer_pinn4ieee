%% 3D打印机完整仿真主脚本
% 功能：整合轨迹误差和温度场仿真，生成PINN训练数据
% 作者：自动生成
% 日期：2025-01-27

clear; clc; close all;

%% 1. 仿真参数设置
fprintf('========================================\n');
fprintf('3D打印机PINN数据生成仿真系统\n');
fprintf('========================================\n\n');

% 仿真控制参数
params.simulation_name = '3d_print_simulation_v1';
params.num_samples = 500;  % 生成样本数量

% G-code轨迹参数
params.trajectory_type = 'random_rectangles';  % 'random_rectangles', 'sine_wave', 'spiral'
params.num_corners = 20;  % 转角数量
params.bed_size = [200, 200];  % 打印床尺寸 (mm)
params.layer_height = 0.2;  % 层高 (mm)
params.num_layers = 50;  % 层数

% 运动参数
params.print_speed = 50;  % 打印速度 (mm/s)
params.travel_speed = 150;  % 移动速度 (mm/s)
params.acceleration = 1500;  % 加速度 (mm/s^2)
params.jerk = 10;  % 加加速度 (mm/s^3)

% 传动系统参数（二阶系统）
% 参考文献: Wang et al. (2018) - Nominal Stiffness of GT-2 Timing Belts
% GT2皮带刚度: ~2,000,000 N/m (实验测量值)
params.mass_x = 0.5;  % X轴运动质量 (kg)
params.mass_y = 0.5;  % Y轴运动质量 (kg)
params.stiffness_x = 2000000;  % X轴皮带刚度 (N/m) - 基于GT2皮带实验数据
params.stiffness_y = 2000000;  % Y轴皮带刚度 (N/m)
params.damping_x = 40;  % X轴阻尼 (N·s/m) - ζ≈0.02
params.damping_y = 40;  % Y轴阻尼 (N·s/m)

% 热学参数
params.T_nozzle = 220;  % 喷嘴温度 (°C)
params.T_bed = 60;  % 热床温度 (°C)
params.T_ambient = 25;  % 环境温度 (°C) - 重要！影响冷却速率
params.fan_speed = 255;  % 风扇转速 (0-255)

% 材料参数（PLA）
params.material_density = 1240;  % 密度 (kg/m^3)
params.material_specific_heat = 1800;  % 比热容 (J/kg·K)
params.material_thermal_conductivity = 0.13;  % 热导率 (W/m·K)
params.melting_point = 150;  % 熔点 (°C)

% 挤出参数
params.nozzle_diameter = 0.4;  % 喷嘴直径 (mm)
params.extrusion_width = 0.45;  % 挤出宽度 (mm)
params.extrusion_multiplier = 1.0;  % 挤出倍率

fprintf('仿真参数设置完成\n\n');

%% 2. 故障模式和噪声配置
fprintf('配置故障模式和噪声...\n');

% 故障模式配置
fault_config = struct();
fault_config.enable = true;  % 启用故障模式

% 故障模式概率（每个样本随机选择）
% 0-20%样本：正常（无故障）
% 30-50%样本：轻微故障
% 30-40%样本：中度故障
% 10-20%样本：严重故障

% 定义所有可能的故障模式
fault_modes = {
    'belt_wear', ...
    'motor_failure', ...
    'bearing_wear', ...
    'nozzle_temp_fault', ...
    'bed_temp_fault', ...
    'fan_failure', ...
    'nozzle_clog', ...
    'feeding_failure', ...
    'ambient_temp_fault', ...
    'vibration'
};

% 噪声配置
noise_config = struct();
noise_config.enable = true;  % 启用噪声
noise_config.position_noise.enable = true;
noise_config.position_noise.sigma = 0.02;  % mm (步进电机精度)
noise_config.velocity_noise.enable = true;
noise_config.velocity_noise.sigma_percent = 2.0;  % %
noise_config.temperature_noise.enable = true;
noise_config.temperature_noise.sigma = 2.0;  % °C (热电偶精度)
noise_config.extrusion_noise.enable = true;
noise_config.extrusion_noise.sigma_percent = 3.0;  % %
noise_config.force_noise.enable = false;  % 默认无力传感器
noise_config.layer_height_noise.enable = true;
noise_config.layer_height_noise.sigma = 0.01;  % mm
noise_config.ambient_fluctuation.enable = true;
noise_config.ambient_fluctuation.sigma = 1.5;  % °C
noise_config.fan_speed_variation.enable = true;
noise_config.fan_speed_variation.sigma = 8.0;  % RPM

fprintf('  故障模式: %s\n', mat2str(fault_modes'));
fprintf('  噪声已启用\n\n');

%% 3. 运行批量仿真
fprintf('开始批量仿真...\n');
fprintf('样本数量: %d\n', params.num_samples);
fprintf('----------------------------------------\n\n');

% 预分配存储空间
all_data = cell(params.num_samples, 1);

% 故障统计
fault_statistics = struct();
for i = 1:length(fault_modes)
    fault_statistics.(fault_modes{i}) = 0;
end
fault_statistics.normal = 0;

% 进度条
progress = waitbar(0, '仿真进度...', 'Name', '3D打印仿真');

for sample_idx = 1:params.num_samples
    % 随机化部分参数（增加数据多样性）
    current_params = params;

    % 随机化环境温度 (15-35°C)
    current_params.T_ambient = 15 + rand() * 20;

    % 随机化打印速度 (30-80 mm/s)
    current_params.print_speed = 30 + rand() * 50;

    % 随机化层高 (0.1-0.3 mm)
    current_params.layer_height = 0.1 + rand() * 0.2;

    % 随机化喷嘴温度 (200-230°C)
    current_params.T_nozzle = 200 + rand() * 30;

    % 随机化加速度 (1000-3000 mm/s^2)
    current_params.acceleration = 1000 + rand() * 2000;

    %% 3.1 随机选择故障模式
    sample_fault_config = struct();
    sample_fault_config.enable = false;

    % 决定是否添加故障（80%概率有故障）
    if rand() < 0.8
        sample_fault_config.enable = true;

        % 随机选择1-2种故障模式
        num_faults = randi([1, 2]);
        selected_faults = randperm(length(fault_modes), num_faults);

        for i = 1:num_faults
            fault_name = fault_modes{selected_faults(i)};

            % 配置故障严重程度（轻微、中度、严重）
            severity_type = randi([1, 3]);  % 1=轻微, 2=中度, 3=严重

            switch fault_name
                case 'belt_wear'
                    sample_fault_config.belt_wear.enable = true;
                    sample_fault_config.belt_wear.severity = [0.25, 0.5, 0.75](severity_type);
                    fault_statistics.belt_wear = fault_statistics.belt_wear + 1;

                case 'motor_failure'
                    sample_fault_config.motor_failure.enable = true;
                    sample_fault_config.motor_failure.severity = [0.2, 0.4, 0.6](severity_type);
                    fault_statistics.motor_failure = fault_statistics.motor_failure + 1;

                case 'bearing_wear'
                    sample_fault_config.bearing_wear.enable = true;
                    sample_fault_config.bearing_wear.severity = [1.5, 2.0, 3.0](severity_type);
                    fault_statistics.bearing_wear = fault_statistics.bearing_wear + 1;

                case 'nozzle_temp_fault'
                    sample_fault_config.nozzle_temp_fault.enable = true;
                    if randi([0, 1]) == 1
                        sample_fault_config.nozzle_temp_fault.type = 'high';
                    else
                        sample_fault_config.nozzle_temp_fault.type = 'low';
                    end
                    sample_fault_config.nozzle_temp_fault.delta_T = [10, 20, 40](severity_type);
                    fault_statistics.nozzle_temp_fault = fault_statistics.nozzle_temp_fault + 1;

                case 'bed_temp_fault'
                    sample_fault_config.bed_temp_fault.enable = true;
                    sample_fault_config.bed_temp_fault.delta_T = [10, 15, 20](severity_type);
                    fault_statistics.bed_temp_fault = fault_statistics.bed_temp_fault + 1;

                case 'fan_failure'
                    sample_fault_config.fan_failure.enable = true;
                    sample_fault_config.fan_failure.severity = [0.3, 0.6, 0.9](severity_type);
                    fault_statistics.fan_failure = fault_statistics.fan_failure + 1;

                case 'nozzle_clog'
                    sample_fault_config.nozzle_clog.enable = true;
                    sample_fault_config.nozzle_clog.severity = [0.7, 0.5, 0.3](severity_type);
                    fault_statistics.nozzle_clog = fault_statistics.nozzle_clog + 1;

                case 'feeding_failure'
                    sample_fault_config.feeding_failure.enable = true;
                    sample_fault_config.feeding_failure.rate = [0.1, 0.2, 0.3](severity_type);
                    fault_statistics.feeding_failure = fault_statistics.feeding_failure + 1;

                case 'ambient_temp_fault'
                    sample_fault_config.ambient_temp_fault.enable = true;
                    if randi([0, 1]) == 1
                        sample_fault_config.ambient_temp_fault.type = 'high';
                    else
                        sample_fault_config.ambient_temp_fault.type = 'low';
                    end
                    sample_fault_config.ambient_temp_fault.delta_T = [5, 10, 15](severity_type);
                    fault_statistics.ambient_temp_fault = fault_statistics.ambient_temp_fault + 1;

                case 'vibration'
                    sample_fault_config.vibration.enable = true;
                    sample_fault_config.vibration.amplitude = [0.05, 0.1, 0.2](severity_type);
                    sample_fault_config.vibration.frequency = 20 + rand() * 30;  % 20-50Hz
                    fault_statistics.vibration = fault_statistics.vibration + 1;
            end
        end
    else
        % 正常样本
        fault_statistics.normal = fault_statistics.normal + 1;
    end

    %% 3.2 应用故障模式
    fprintf('[样本 %d/%d] ', sample_idx, params.num_samples);
    current_params_fault = add_fault_modes(current_params, sample_fault_config);

    %% 3.3 生成/解析G-code轨迹
    fprintf('生成G-code轨迹...\n');
    [gcode_data, trajectory] = generate_or_parse_gcode(current_params_fault);

    %% 3.4 仿真轨迹误差（二阶系统）
    fprintf('  仿真轨迹误差...\n');
    trajectory_error = simulate_trajectory_error(trajectory, current_params_fault);

    %% 3.5 仿真温度场
    fprintf('  仿真温度场...\n');
    thermal_field = simulate_thermal_field(trajectory, current_params_fault);

    %% 3.6 计算粘结强度
    fprintf('  计算粘结强度...\n');
    adhesion_strength = calculate_adhesion_strength(thermal_field, current_params_fault);

    %% 3.7 整合数据
    sample_data = struct();
    sample_data.sample_id = sample_idx;
    sample_data.params = current_params_fault;
    sample_data.gcode_data = gcode_data;
    sample_data.trajectory = trajectory;
    sample_data.trajectory_error = trajectory_error;
    sample_data.thermal_field = thermal_field;
    sample_data.adhesion_strength = adhesion_strength;

    %% 3.8 添加传感器噪声
    sample_data = add_sensor_noise(sample_data, noise_config);

    % 保存故障标记
    sample_data.has_fault = current_params_fault.has_fault;
    sample_data.fault_names = current_params_fault.fault_names;
    sample_data.fault_count = current_params_fault.fault_count;

    all_data{sample_idx} = sample_data;

    % 更新进度条
    waitbar(sample_idx / params.num_samples, progress, ...
        sprintf('仿真进度: %d/%d', sample_idx, params.num_samples));
end

close(progress);
fprintf('----------------------------------------\n');
fprintf('批量仿真完成！\n\n');

%% 4. 故障统计报告
fprintf('========================================\n');
fprintf('故障模式统计\n');
fprintf('========================================\n');

total_faults = 0;
for i = 1:length(fault_modes)
    fault_count = fault_statistics.(fault_modes{i});
    total_faults = total_faults + fault_count;
    fprintf('  %-20s: %4d (%.1f%%)\n', ...
        fault_modes{i}, fault_count, fault_count/params.num_samples*100);
end
fprintf('  %-20s: %4d (%.1f%%)\n', ...
    '正常（无故障）', fault_statistics.normal, fault_statistics.normal/params.num_samples*100);
fprintf('----------------------------------------\n');
fprintf('  总故障数: %d\n', total_faults);
fprintf('  故障样本率: %.1f%%\n', (params.num_samples - fault_statistics.normal) / params.num_samples * 100);
fprintf('  平均每样本故障数: %.2f\n', total_faults / params.num_samples);
fprintf('\n');

%% 5. 保存数据为MATLAB格式
fprintf('========================================\n');
fprintf('保存数据\n');
fprintf('========================================\n');
output_filename = fullfile('./output', [params.simulation_name '_data.mat']);
if ~exist('./output', 'dir')
    mkdir('./output');
end
save(output_filename, 'all_data', 'params', 'fault_statistics', '-v7.3');
fprintf('数据已保存: %s\n', output_filename);

%% 4. 转换为Python格式（可选）
fprintf('\n是否转换为Python格式？\n');
user_response = input('输入 y 转换，其他键跳过: ', 's');

if strcmpi(user_response, 'y')
    fprintf('转换数据为Python格式...\n');
    export_to_python(output_filename, params);
end

%% 5. 生成数据统计报告
fprintf('\n========================================\n');
fprintf('数据统计摘要\n');
fprintf('========================================\n');

% 轨迹误差统计
all_max_errors = zeros(params.num_samples, 1);
all_rms_errors = zeros(params.num_samples, 1);

for i = 1:params.num_samples
    all_max_errors(i) = max(all_data{i}.trajectory_error.position_error_magnitude);
    all_rms_errors(i) = sqrt(mean(all_data{i}.trajectory_error.position_error_magnitude.^2));
end

fprintf('轨迹误差统计:\n');
fprintf('  最大误差: %.4f mm (平均), %.4f mm (最大)\n', ...
    mean(all_max_errors), max(all_max_errors));
fprintf('  RMS误差: %.4f mm (平均), %.4f mm (最大)\n', ...
    mean(all_rms_errors), max(all_rms_errors));

% 温度统计
fprintf('\n温度场统计:\n');
fprintf('  层间温度范围: %.1f - %.1f °C\n', ...
    params.T_bed, params.T_nozzle);

% 粘结强度统计
all_avg_adhesion = zeros(params.num_samples, 1);
for i = 1:params.num_samples
    all_avg_adhesion(i) = mean(all_data{i}.adhesion_strength.strength);
end

fprintf('\n粘结强度统计:\n');
fprintf('  平均粘结强度: %.2f MPa (平均), %.2f MPa (最大)\n', ...
    mean(all_avg_adhesion), max(all_avg_adhesion));

fprintf('\n========================================\n');
fprintf('仿真完成！\n');
fprintf('========================================\n');
