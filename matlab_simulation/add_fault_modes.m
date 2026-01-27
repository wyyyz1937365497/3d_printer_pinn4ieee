function params_fault = add_fault_modes(params, fault_config)
%% 为仿真添加故障模式
% 功能：模拟3D打印过程中的各种故障
% 输入：
%   params - 原始参数
%   fault_config - 故障配置结构体
% 输出：
%   params_fault - 包含故障的参数

fprintf('  添加故障模式...\n');

% 默认：无故障
if nargin < 2
    fault_config = struct();
    fault_config.enable = false;
end

% 如果未启用故障，直接返回原参数
if isfield(fault_config, 'enable') && ~fault_config.enable
    params_fault = params;
    fprintf('    故障模式未启用\n');
    return;
end

% 复制参数
params_fault = params;

% 故障统计
fault_count = 0;
fault_names = {};

%% F1. 皮带磨损 (Belt Wear)
if isfield(fault_config, 'belt_wear') && fault_config.belt_wear.enable
    fault_count = fault_count + 1;
    fault_names{end+1} = 'Belt Wear';

    % 磨损程度：轻微(25%)、中度(50%)、严重(75%)
    severity = fault_config.belt_wear.severity;  % 0.25, 0.5, 0.75

    % 刚度下降
    params_fault.stiffness_x = params_fault.stiffness_x * (1 - severity);
    params_fault.stiffness_y = params_fault.stiffness_y * (1 - severity);

    % 阻尼可能略微增加（摩擦增大）
    params_fault.damping_x = params_fault.damping_x * (1 + severity*0.5);
    params_fault.damping_y = params_fault.damping_y * (1 + severity*0.5);

    fprintf('    [F1] 皮带磨损: 刚度下降%.0f%%\n', severity*100);
end

%% F2. 电机故障 (Motor Failure)
if isfield(fault_config, 'motor_failure') && fault_config.motor_failure.enable
    fault_count = fault_count + 1;
    fault_names{end+1} = 'Motor Failure';

    % 扭矩不足
    severity = fault_config.motor_failure.severity;  % 0.2-0.6

    % 降低最大加速度（在仿真中会用到）
    params_fault.acceleration_max = params_fault.acceleration * (1 - severity);

    fprintf('    [F2] 电机故障: 最大加速度下降%.0f%%\n', severity*100);
end

%% F3. 轴承磨损 (Bearing Wear)
if isfield(fault_config, 'bearing_wear') && fault_config.bearing_wear.enable
    fault_count = fault_count + 1;
    fault_names{end+1} = 'Bearing Wear';

    % 摩擦增大
    severity = fault_config.bearing_wear.severity;  % 1-3 (倍数)

    % 阻尼增加
    params_fault.damping_x = params_fault.damping_x * severity;
    params_fault.damping_y = params_fault.damping_y * severity;

    fprintf('    [F3] 轴承磨损: 阻尼增加%.0f倍\n', severity);
end

%% F4. 喷嘴温度异常 (Nozzle Temperature Fault)
if isfield(fault_config, 'nozzle_temp_fault') && fault_config.nozzle_temp_fault.enable
    fault_count = fault_count + 1;
    fault_names{end+1} = 'Nozzle Temp Fault';

    fault_type = fault_config.nozzle_temp_fault.type;  % 'high' or 'low'
    delta_T = fault_config.nozzle_temp_fault.delta_T;  % ±10-40°C

    if strcmp(fault_type, 'high')
        % 温度过高
        params_fault.T_nozzle = params_fault.T_nozzle + delta_T;
        fprintf('    [F4a] 喷嘴温度过高: +%d°C\n', delta_T);
    else
        % 温度过低
        params_fault.T_nozzle = params_fault.T_nozzle - delta_T;
        fprintf('    [F4b] 喷嘴温度过低: -%d°C\n', delta_T);
    end
end

%% F5. 热床温度故障 (Bed Temperature Fault)
if isfield(fault_config, 'bed_temp_fault') && fault_config.bed_temp_fault.enable
    fault_count = fault_count + 1;
    fault_names{end+1} = 'Bed Temp Fault';

    % 热床温度不均匀
    delta_T = fault_config.bed_temp_fault.delta_T;  % ±10-20°C

    params_fault.T_bed_variation = delta_T;
    fprintf('    [F5] 热床温度变化: ±%d°C\n', delta_T);
end

%% F6. 冷却风扇故障 (Cooling Fan Failure)
if isfield(fault_config, 'fan_failure') && fault_config.fan_failure.enable
    fault_count = fault_count + 1;
    fault_names{end+1} = 'Fan Failure';

    % 风扇转速不足
    severity = fault_config.fan_failure.severity;  % 0-100%

    params_fault.fan_speed = params_fault.fan_speed * (1 - severity);
    fprintf('    [F6] 风扇故障: 转速下降%.0f%%\n', severity*100);
end

%% F7. 喷嘴堵塞 (Nozzle Clogging)
if isfield(fault_config, 'nozzle_clog') && fault_config.nozzle_clog.enable
    fault_count = fault_count + 1;
    fault_names{end+1} = 'Nozzle Clogging';

    % 挤出不足
    severity = fault_config.nozzle_clog.severity;  % 0.3-0.7

    params_fault.extrusion_multiplier = params_fault.extrusion_multiplier * severity;
    fprintf('    [F7] 喷嘴堵塞: 挤出量降至%.0f%%\n', severity*100);
end

%% F8. 进料故障 (Feeding Failure)
if isfield(fault_config, 'feeding_failure') && fault_config.feeding_failure.enable
    fault_count = fault_count + 1;
    fault_names{end+1} = 'Feeding Failure';

    % 间歇性挤出
    params_fault.feeding_intermittent = true;
    params_fault.feeding_failure_rate = fault_config.feeding_failure.rate;  % 10-30%

    fprintf('    [F8] 进料故障: 间歇性失效率%.0f%%\n', ...
        fault_config.feeding_failure.rate*100);
end

%% F9. 环境温度异常 (Ambient Temperature Fault)
if isfield(fault_config, 'ambient_temp_fault') && fault_config.ambient_temp_fault.enable
    fault_count = fault_count + 1;
    fault_names{end+1} = 'Ambient Temp Fault';

    fault_type = fault_config.ambient_temp_fault.type;  % 'high' or 'low'
    delta_T = fault_config.ambient_temp_fault.delta_T;  % ±5-15°C

    if strcmp(fault_type, 'high')
        % 温度过高（夏天）
        params_fault.T_ambient = params_fault.T_ambient + delta_T;
        fprintf('    [F9a] 环境温度过高: +%d°C\n', delta_T);
    else
        % 温度过低（冬天）
        params_fault.T_ambient = params_fault.T_ambient - delta_T;
        fprintf('    [F9b] 环境温度过低: -%d°C\n', delta_T);
    end
end

%% F10. 振动干扰 (Vibration Interference)
if isfield(fault_config, 'vibration') && fault_config.vibration.enable
    fault_count = fault_count + 1;
    fault_names{end+1} = 'Vibration Interference';

    params_fault.external_vibration = true;
    params_fault.vibration_amplitude = fault_config.vibration.amplitude;  % mm
    params_fault.vibration_frequency = fault_config.vibration.frequency;  % Hz

    fprintf('    [F10] 振动干扰: 幅值%.3fmm, 频率%dHz\n', ...
        params_fault.vibration_amplitude, params_fault.vibration_frequency);
end

%% 故障总结
if fault_count == 0
    fprintf('    无故障模式\n');
else
    fprintf('    总计: %d种故障模式\n', fault_count);
    fprintf('    故障: %s\n', strjoin(fault_names, ', '));
end

% 添加故障标记
params_fault.fault_count = fault_count;
params_fault.fault_names = fault_names;
params_fault.has_fault = fault_count > 0;

end
