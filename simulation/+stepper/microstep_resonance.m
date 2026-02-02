function error = microstep_resonance(step_rate, params, axis)
% MICROSTEP_RESONANCE - 计算微步引起的谐振误差
%
% **关键**：16x微步在特定频率下会引起谐振，显著增大位置误差
% 这是实际打印机误差的重要来源
%
% 机制:
%   - 步进电机16x微步 → 电感效应 → 谐振
%   - 微步频率 = step_rate / 16
%   - 如果接近固有频率(21Hz) → 谐振放大
%
% 输入:
%   step_rate - 步速率 (steps/s)
%   params - 物理参数结构体
%   axis - 'x' 或 'y'（轴标识）
%
% 输出:
%   error - 位置误差向量 (mm)，与step_rate同长度
%
% 参考：步进电机谐振理论

    % 确保step_rate是列向量
    step_rate = step_rate(:);

    % 微步设置（Ender 3 V2固件）
    microsteps = 16;  % 16x微步

    % 微步频率 (Hz)
    microstep_freq = step_rate / microsteps;

    % 固有频率（从物理参数获取）
    if strcmp(axis, 'x')
        natural_freq = params.dynamics.x.natural_freq;  % rad/s
        damping_ratio = params.dynamics.x.damping_ratio;
    else  % y轴
        natural_freq = params.dynamics.y.natural_freq;
        damping_ratio = params.dynamics.y.damping_ratio;
    end

    % 转换为Hz
    natural_freq_hz = natural_freq / (2*pi);

    % 频率比
    frequency_ratio = microstep_freq / natural_freq_hz;

    % 初始化误差向量
    error = zeros(size(step_rate));

    % 基础步进误差（单步）
    step_angle_deg = 1.8 / microsteps;  % 16x微步
    step_angle_rad = deg2rad(step_angle_deg);

    % 皮带参数（GT2）
    pulley_teeth = 20;
    pitch = 2;  % mm
    effective_radius = (pulley_teeth * pitch) / (2*pi);  % mm

    % 转换为线性位移误差
    base_error = step_angle_rad * effective_radius;

    % 对每个点计算误差
    for i = 1:length(step_rate)
        fr = frequency_ratio(i);

        % 谐振范围（0.5-2.0倍固有频率）
        if fr > 0.5 && fr < 2.0
            % 计算品质因数Q
            Q_factor = 1 / (2 * damping_ratio);

            % 谐振放大倍数（二阶系统频率响应）
            denominator = sqrt((1 - fr^2)^2 + (2 * damping_ratio * fr)^2);

            if denominator > 1e-6
                magnification = Q_factor / denominator;
            else
                magnification = 1.0;
            end

            % 谐振放大后的误差（mm）
            error_i = base_error * magnification;

            % 添加随机相位（不同谐振模态）
            phase = 2 * pi * rand();
            resonance_component = sin(2 * pi * fr * i + phase);

            % 误差随频率变化（包络线）
            envelope = exp(-0.5 * abs(log(fr)));  % 谐振峰附近最强

            error(i) = error_i * envelope * abs(resonance_component);

            % 高频限制（实际机械系统有高频衰减）
            max_freq_cutoff = 100;  % Hz
            if microstep_freq(i) > max_freq_cutoff
                attenuation = exp(-(microstep_freq(i) - max_freq_cutoff) / 20);
                error(i) = error(i) * attenuation;
            end
        else
            % 非谐振区，只有基础误差
            error(i) = base_error * 0.01;  % 1%的量化误差

            % 添加随机噪声（模拟其他扰动）
            noise_level = 0.002;  % mm
            error(i) = error(i) + noise_level * randn();
        end
    end

    % 确保误差在合理范围内（0.15mm以内）
    max_error_limit = 0.15;  % mm
    error = max(min(error, max_error_limit), -max_error_limit);
end
