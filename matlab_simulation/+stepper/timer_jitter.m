function [actual_times, position_errors] = timer_jitter(nominal_times, params)
% TIMER_JITTER - 模拟定时器中断引起的实际位置误差
%
% **关键**：实时系统中断延迟会导致脉冲时间不规则
% 这会引入额外的位置误差，特别是在高步进频率时
%
% 机制：
%   - STM32定时器中断有固定开销（50-100周期）
%   - 中断延迟导致脉冲间隔不均匀
%   - 累积的位置偏差
%
% 输入:
%   nominal_times - 计划的脉冲时间向量 (秒)
%   params - 物理参数结构体
%
% 输出:
%   actual_times - 实际脉冲时间（考虑抖动）
%   position_errors - 位置误差向量 (mm)

    % STM32F103参数（Ender 3 V2）
    cpu_freq = 72e6;  % Hz
    timer_resolution = 1 / cpu_freq;

    % 中断开销（周期数）
    interrupt_overhead_min = 50;   % 最优情况
    interrupt_overhead_max = 150;  % 最差情况（缓存未命中等）

    % 计算名义脉冲间隔
    nominal_times = nominal_times(:);  % 确保是列向量
    if length(nominal_times) > 1
        nominal_intervals = diff(nominal_times);
    else
        nominal_intervals = nominal_times(1);
    end

    % 生成随机中断延迟
    num_pulses = length(nominal_intervals);
    jitter_delays = zeros(num_pulses, 1);

    for i = 1:num_pulses
        % 随机延迟（模拟中断冲突、缓存未命中等）
        overhead = interrupt_overhead_min + ...
                  rand() * (interrupt_overhead_max - interrupt_overhead_min);

        % 转换为时间
        jitter_delays(i) = overhead * timer_resolution;

        % 高频时延迟更明显（累积效应）
        if nominal_intervals(i) < 0.001  % 1kHz以上
            jitter_delays(i) = jitter_delays(i) * 2;  % 放大
        end
    end

    % 实际脉冲间隔
    actual_intervals = nominal_intervals + jitter_delays;

    % 确保间隔为正
    actual_intervals = max(actual_intervals, 0.5 * nominal_intervals);

    % 累积时间
    actual_times = cumsum([nominal_times(1); actual_intervals]);

    % 计算位置误差
    % 原理：时间偏差 → 速度偏差 → 位置偏差
    % Δx ≈ v * Δt

    % 估算平均速度（基于运动学）
    avg_velocity = params.motion.max_velocity / 2;  % mm/s

    % 时间误差
    time_errors = actual_times - nominal_times;

    % 转换为位置误差
    position_errors = avg_velocity * time_errors;

    % 低通滤波（机械系统不能响应极高频）
    % 截止频率约100Hz
    cutoff_freq = 100;  % Hz
    dt_mean = mean(nominal_intervals);
    alpha = dt_mean / (dt_mean + 1/(2*pi*cutoff_freq));

    position_errors = filter([alpha, 1-alpha], 1, position_errors);

    % 限制最大误差（不超过0.05mm）
    max_jitter_error = 0.05;  % mm
    position_errors = max(min(position_errors, max_jitter_error), -max_jitter_error);
end
