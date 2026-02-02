function [error_x, error_y] = microstep_resonance_effect(n_points, params)
% MICROSTEP_RESONANCE_EFFECT - 计算微步谐振引起的X/Y位置误差
%
% 输入:
%   n_points - 点数
%   params - 物理参数结构体
%
% 输出:
%   error_x, error_y - X/Y方向的位置误差 (mm)
%
% 参考: 步进电机16x微步谐振理论

    % 假设中等步进频率
    step_rate_x = 5000 * ones(n_points, 1);  % steps/s (X轴)
    step_rate_y = 5000 * ones(n_points, 1);  % steps/s (Y轴)

    % 添加一些变化（模拟实际速度变化）
    step_rate_x = step_rate_x + 1000 * sin(linspace(0, 10*pi, n_points)');
    step_rate_y = step_rate_y + 1000 * sin(linspace(0, 8*pi, n_points)');

    % 调用microstep resonance函数（使用包前缀）
    error_x = stepper.microstep_resonance(step_rate_x, params, 'x');
    error_y = stepper.microstep_resonance(step_rate_y, params, 'y');
end
