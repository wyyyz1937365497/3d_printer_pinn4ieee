function results = simulate_trajectory_error_with_firmware_effects(trajectory_data, params)
% SIMULATE_TRAJECTORY_ERROR_WITH_FIRMWARE_EFFECTS
%
% 基于固件特性的轨迹误差仿真，专门增大误差到0.1mm级别
%
% **目标**：为误差修复神经网络生成真实的训练数据
%
% 集成的误差源（按效果排序）：
%   1. Junction Deviation - 转角圆化 (+20-50μm)
%   2. 微步谐振 - 高频振动 (+10-30μm)
%   3. 定时器中断抖动 - 脉冲不规则 (+5-15μm)
%   4. 基础动力学误差 - 惯性/弹性 (+50-80μm)
%
% 输入:
%   trajectory_data - 轨迹数据结构
%   params - 物理参数
%
% 输出:
%   results - 仿真结果（包含误差向量）

    fprintf('应用固件效应以增大轨迹误差...\n');

    %% 1. 基础动力学仿真（CPU模式，用于并行处理）
    fprintf('  [1/4] 基础动力学误差...\n');
    % 并行环境不使用GPU（避免多个worker竞争GPU资源）
    trajectory_results = simulate_trajectory_error(trajectory_data, params);

    % 初始误差
    error_x = trajectory_results.error_x;
    error_y = trajectory_results.error_y;
    fprintf('    基础误差RMS: %.3f mm\n', rms(sqrt(error_x.^2 + error_y.^2)));

    %% 2. 应用Junction Deviation（转角处额外误差）
    fprintf('  [2/4] Junction Deviation效应...\n');

    % 检测转角
    vx = trajectory_data.vx;
    vy = trajectory_data.vy;
    speed = sqrt(vx.^2 + vy.^2);

    % 计算方向变化
    dx = diff([trajectory_data.x; trajectory_data.x(end)]);
    dy = diff([trajectory_data.y; trajectory_data.y(end)]);
    angles = atan2(dy, dx);  % 移动方向
    angle_changes = abs(diff([angles; angles(end)]));  % 转角

    % 计算转角误差（Junction Deviation）
    junction_deviation_mm = 0.013;  % 固件参数
    max_accel = 500;  % mm/s²

    junction_errors_x = zeros(size(error_x));
    junction_errors_y = zeros(size(error_y));

    for i = 2:length(angle_changes)-1
        if angle_changes(i) > deg2rad(5)  % 只处理显著转角（>5度）
            % 前后速度向量
            v1 = [vx(i-1), vy(i-1)];
            v2 = [vx(i), vy(i)];

            % 计算允许的转角速度
            vmax = planner.junction_deviation(v1, v2, rad2deg(angle_changes(i)), ...
                                           max_accel, junction_deviation_mm);

            % 如果实际速度超过转角限制，产生额外误差
            actual_speed = speed(i);
            if actual_speed > vmax && vmax > 1  % vmax必须合理
                % 速度超限比例（限制最大为2，避免极端情况）
                overspeed_ratio = min(actual_speed / vmax, 2.0);

                % Junction Deviation: 固件会偏离转角一定距离
                % 偏差量 ≈ JD参数，但受速度超限影响
                base_deviation = junction_deviation_mm;  % 0.013mm
                corner_deviation = base_deviation * (overspeed_ratio - 1);

                % 限制最大偏差（不超过0.05mm）
                corner_deviation = min(corner_deviation, 0.05);

                % 分配到X/Y（沿角平分线方向）
                % 简化：根据速度方向分配
                if abs(v1(1)) > abs(v1(2))  % 主要是X方向运动
                    junction_errors_x(i) = corner_deviation * sign(v1(1));
                else
                    junction_errors_y(i) = corner_deviation * sign(v1(2));
                end
            end
        end
    end

    fprintf('    Junction Deviation RMS: %.3f mm\n', rms(sqrt(junction_errors_x.^2 + junction_errors_y.^2)));

    %% 3. 应用微步谐振（高频振动）
    fprintf('  [3/4] 微步谐振效应...\n');

    % 估算步进频率
    % 步数 = 距离 × steps_per_mm
    steps_per_mm = 80;  % Ender 3 V2配置

    distance_per_point = sqrt(diff(trajectory_data.x).^2 + diff(trajectory_data.y).^2);
    distance_per_point = [distance_per_point; distance_per_point(end)];  % 补齐长度
    steps_per_point = round(distance_per_point * steps_per_mm);

    % 估算步进频率
    dt = mean(diff(trajectory_data.time));
    step_rate = steps_per_point / dt;  % steps/s

    % 计算谐振误差
    resonance_error_x = stepper.microstep_resonance(step_rate, params, 'x');
    resonance_error_y = stepper.microstep_resonance(step_rate, params, 'y');

    fprintf('    微步谐振RMS: %.3f mm\n', rms(sqrt(resonance_error_x.^2 + resonance_error_y.^2)));

    %% 4. 应用了定时器中断抖动
    fprintf('  [4/4] 定时器中断抖动...\n');

    % 生成名义脉冲时间（简化：使用实际数据时间间隔）
    nominal_pulse_times = trajectory_data.time;

    % 计算抖动
    [~, jitter_position_errors] = stepper.timer_jitter(nominal_pulse_times, params);

    % 将时间误差转换为X/Y误差（按速度分量分配）
    % 确保尺寸匹配
    jitter_position_errors = jitter_position_errors(:);  % 列向量
    if length(jitter_position_errors) < length(trajectory_data.vx)
        % 补齐长度（重复最后一个值）
        jitter_position_errors = [jitter_position_errors; jitter_position_errors(end)];
    end

    velocity_x = trajectory_data.vx;
    velocity_y = trajectory_data.vy;
    v_mag = sqrt(velocity_x.^2 + velocity_y.^2);

    % 避免除零
    v_mag(v_mag < 1) = 1;

    jitter_errors_x = jitter_position_errors .* (velocity_x ./ v_mag);
    jitter_errors_y = jitter_position_errors .* (velocity_y ./ v_mag);

    fprintf('    中断抖动RMS: %.3f mm\n', rms(sqrt(jitter_errors_x.^2 + jitter_errors_y.^2)));

    %% 5. 叠加所有误差源
    fprintf('\n');
    fprintf('误差源汇总:\n');

    final_error_x = error_x + junction_errors_x + resonance_error_x + jitter_errors_x;
    final_error_y = error_y + junction_errors_y + resonance_error_y + jitter_errors_y;

    fprintf('  基础动力学:    %.3f mm\n', rms(sqrt(error_x.^2 + error_y.^2)));
    fprintf('  Junction Dev:  %.3f mm\n', rms(sqrt(junction_errors_x.^2 + junction_errors_y.^2)));
    fprintf('  微步谐振:      %.3f mm\n', rms(sqrt(resonance_error_x.^2 + resonance_error_y.^2)));
    fprintf('  中断抖动:      %.3f mm\n', rms(sqrt(jitter_errors_x.^2 + jitter_errors_y.^2)));
    fprintf('  --------------------------\n');
    fprintf('  总误差RMS:     %.3f mm\n', rms(sqrt(final_error_x.^2 + final_error_y.^2)));

    % 统计
    max_error = max(sqrt(final_error_x.^2 + final_error_y.^2)) * 1000;  % μm
    fprintf('  最大误差:       %.0f μm\n', max_error);

    %% 6. 构建输出结果（保持兼容格式）
    results = trajectory_results;  % 包含基础动力学结果

    % 更新误差向量（加入固件效应）
    results.error_x = final_error_x(:);
    results.error_y = final_error_y(:);
    results.error_magnitude = sqrt(final_error_x.^2 + final_error_y.^2);
    results.error_direction = atan2(final_error_y, final_error_x);

    % 添加新字段（记录各误差源）
    results.junction_deviation_x = junction_errors_x;
    results.junction_deviation_y = junction_errors_y;
    results.resonance_x = resonance_error_x;
    results.resonance_y = resonance_error_y;
    results.jitter_x = jitter_errors_x;
    results.jitter_y = jitter_errors_y;

    fprintf('\n');
    if max_error >= 80 && max_error <= 150
        fprintf('✅ 误差范围良好 (80-150μm)\n');
    elseif max_error < 80
        fprintf('⚠️ 误差偏小 (<80μm)，可能需要调整参数\n');
    else
        fprintf('⚠️ 误差偏大 (>150μm)\n');
    end
    fprintf('\n');
end
