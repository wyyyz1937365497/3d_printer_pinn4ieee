function data_noisy = add_sensor_noise(data, noise_config)
%% 为仿真数据添加传感器噪声和过程噪声
% 功能：模拟真实环境中的随机性
% 输入：
%   data - 原始数据（包含trajectory, thermal_field等）
%   noise_config - 噪声配置结构体
% 输出：
%   data_noisy - 添加噪声后的数据

fprintf('  添加噪声...\n');

% 默认噪声配置
if nargin < 2
    noise_config = struct();
    noise_config.enable = false;
end

% 如果未启用噪声，直接返回原数据
if isfield(noise_config, 'enable') && ~noise_config.enable
    data_noisy = data;
    fprintf('    噪声未启用\n');
    return;
end

% 复制数据
data_noisy = data;

% 获取时间点数量
n_points = length(data.trajectory.time);

%% 1. 位置传感器噪声
if isfield(noise_config, 'position_noise') && noise_config.position_noise.enable
    sigma_pos = noise_config.position_noise.sigma;  % mm

    % 添加高斯噪声到位置
    noise_x = sigma_pos * randn(n_points, 1);
    noise_y = sigma_pos * randn(n_points, 1);

    data_noisy.trajectory.x_ref = data_noisy.trajectory.x_ref + noise_x;
    data_noisy.trajectory.y_ref = data_noisy.trajectory.y_ref + noise_y;

    fprintf('    位置噪声: σ=%.3f mm\n', sigma_pos);
else
    % 默认位置噪声（步进电机精度）
    sigma_pos = 0.02;  % mm
    noise_x = sigma_pos * randn(n_points, 1);
    noise_y = sigma_pos * randn(n_points, 1);

    data_noisy.trajectory.x_ref = data_noisy.trajectory.x_ref + noise_x;
    data_noisy.trajectory.y_ref = data_noisy.trajectory.y_ref + noise_y;
    fprintf('    位置噪声: σ=%.3f mm (默认)\n', sigma_pos);
end

%% 2. 速度传感器噪声
if isfield(noise_config, 'velocity_noise') && noise_config.velocity_noise.enable
    sigma_vel_percent = noise_config.velocity_noise.sigma_percent;  % %

    % 添加高斯噪声到速度
    noise_vx = sigma_vel_percent/100 * data.trajectory.vx_ref .* randn(n_points, 1);
    noise_vy = sigma_vel_percent/100 * data.trajectory.vy_ref .* randn(n_points, 1);

    data_noisy.trajectory.vx_ref = data_noisy.trajectory.vx_ref + noise_vx;
    data_noisy.trajectory.vy_ref = data_noisy.trajectory.vy_ref + noise_vy;

    fprintf('    速度噪声: σ=%.1f%%\n', sigma_vel_percent);
end

%% 3. 温度传感器噪声
if isfield(noise_config, 'temperature_noise') && noise_config.temperature_noise.enable
    sigma_T = noise_config.temperature_noise.sigma;  % °C

    % 添加噪声到温度数据
    noise_T_nozzle = sigma_T * randn(n_points, 1);
    noise_T_bed = sigma_T * randn(n_points, 1);
    noise_T_ambient = sigma_T * randn(n_points, 1);

    % 温度场噪声（空间相关）
    if isfield(data, 'thermal_field')
        [ny, nx] = size(data.thermal_field.T_final);
        noise_T_field = sigma_T * randn(ny, nx);

        data_noisy.thermal_field.T_final = data_noisy.thermal_field.T_final + noise_T_field;
        data_noisy.thermal_field.T_nozzle_path = ...
            data_noisy.thermal_field.T_nozzle_path + noise_T_nozzle;
    end

    fprintf('    温度噪声: σ=%.1f°C\n', sigma_T);
end

%% 4. 挤出流量噪声
if isfield(noise_config, 'extrusion_noise') && noise_config.extrusion_noise.enable
    sigma_ext_percent = noise_config.extrusion_noise.sigma_percent;  % %

    % 添加噪声到挤出参数
    noise_ext = sigma_ext_percent/100 * randn(n_points, 1);

    % 挤出倍率波动
    if isfield(data_noisy.params, 'extrusion_multiplier')
        base_ext_mult = data_noisy.params.extrusion_multiplier;
        data_noisy.params.extrusion_multiplier = base_ext_mult * (1 + noise_ext);
    end

    fprintf('    挤出噪声: σ=%.1f%%\n', sigma_ext_percent);
end

%% 5. 力传感器噪声（如果有）
if isfield(data, 'trajectory_error')
    if isfield(noise_config, 'force_noise') && noise_config.force_noise.enable
        sigma_F = noise_config.force_noise.sigma;  % N

        % 添加噪声到惯性力
        noise_Fx = sigma_F * randn(n_points, 1);
        noise_Fy = sigma_F * randn(n_points, 1);

        data_noisy.trajectory_error.F_inertia_x = ...
            data_noisy.trajectory_error.F_inertia_x + noise_Fx;
        data_noisy.trajectory_error.F_inertia_y = ...
            data_noisy.trajectory_error.F_inertia_y + noise_Fy;

        fprintf('    力传感器噪声: σ=%.2f N\n', sigma_F);
    end
end

%% 6. 层高变化噪声
if isfield(noise_config, 'layer_height_noise') && noise_config.layer_height_noise.enable
    sigma_layer = noise_config.layer_height_noise.sigma;  % mm

    % 添加噪声到层高
    noise_layer = sigma_layer * randn(n_points, 1);

    data_noisy.params.layer_height = data_noisy.params.layer_height + ...
        mean(noise_layer);

    fprintf('    层高噪声: σ=%.3f mm\n', sigma_layer);
end

%% 7. 环境温度波动
if isfield(noise_config, 'ambient_fluctuation') && noise_config.ambient_fluctuation.enable
    sigma_Tamb = noise_config.ambient_fluctuation.sigma;  % °C

    % 环境温度随时间波动（低频）
    % 使用平滑随机游走
    noise_Tamb = sigma_Tamb * cumsum(randn(n_points, 1)) / sqrt(n_points);

    if isfield(data_noisy.params, 'T_ambient')
        data_noisy.params.T_ambient = data_noisy.params.T_ambient + mean(noise_Tamb);
    end

    fprintf('    环境温度波动: σ=%.1f°C\n', sigma_Tamb);
end

%% 8. 风扇转速波动
if isfield(noise_config, 'fan_speed_variation') && noise_config.fan_speed_variation.enable
    sigma_fan = noise_config.fan_speed_variation.sigma;  % RPM

    % 风扇转速波动
    noise_fan = sigma_fan * randn(n_points, 1);

    if isfield(data_noisy.params, 'fan_speed')
        data_noisy.params.fan_speed = max(0, ...
            data_noisy.params.fan_speed + mean(noise_fan));
    end

    fprintf('    风扇转速波动: σ=%.1f RPM\n', sigma_fan);
end

%% 9. 添加噪声标记
data_noisy.noise_added = true;
data_noisy.noise_config = noise_config;

fprintf('    噪声添加完成\n');

end
