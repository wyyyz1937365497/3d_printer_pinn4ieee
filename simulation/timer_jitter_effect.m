function [error_x, error_y] = timer_jitter_effect(vx_ref, vy_ref, params)
% TIMER_JITTER_EFFECT - 计算定时器抖动引起的X/Y位置误差
%
% 输入:
%   vx_ref, vy_ref - 参考速度 (mm/s)
%   params - 物理参数结构体
%
% 输出:
%   error_x, error_y - X/Y方向的位置误差 (mm)
%
% 参考: STM32定时器中断延迟

    n_points = length(vx_ref);

    % 假设100Hz采样频率
    dt = 0.01;  % 秒
    nominal_times = (0:n_points-1)' * dt;

    % 调用timer jitter函数（使用包前缀）
    [~, position_errors] = stepper.timer_jitter(nominal_times, params);

    % 根据速度方向分配误差
    speed = sqrt(vx_ref.^2 + vy_ref.^2);
    speed(speed < 1e-6) = 1e-6;  % 避免除零

    % 按速度分量分配
    error_x = position_errors .* (vx_ref ./ speed);
    error_y = position_errors .* (vy_ref ./ speed);

    % 修正NaN值
    error_x(isnan(error_x)) = 0;
    error_y(isnan(error_y)) = 0;
end
