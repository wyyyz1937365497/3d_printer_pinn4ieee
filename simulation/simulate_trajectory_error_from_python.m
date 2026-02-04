function results = simulate_trajectory_error_from_python(input_traj, ideal_traj, params)
% SIMULATE_TRAJECTORY_ERROR_FROM_PYTHON - 从Python解析的轨迹仿真误差
%
% **关键改进**：
%   1. 接受Python解析的完整轨迹数据（包含速度变化）
%   2. 计算相对于理想轨迹的误差（而不是输入轨迹）
%   3. 支持非恒定速度的G-code
%
% 输入：
%   input_traj  - Python解析的输入轨迹（将发送给打印机的）
%   ideal_traj  - 原始理想轨迹（我们想要的最终形状）
%   params      - 物理参数
%
% 输出：
%   results     - 包含actual轨迹和相对于ideal_traj的误差
%
% 调用方式（从Python）：
%   results = simulate_trajectory_error_from_python(traj_data, ideal_data, params)

    fprintf('正在模拟轨迹误差（二阶系统动力学）...\n');
    fprintf('  X轴: ωn = %.2f rad/s, ζ = %.4f\n', ...
        sqrt(params.dynamics.x.stiffness / params.dynamics.x.mass), ...
        params.dynamics.x.damping / (2 * sqrt(params.dynamics.x.stiffness * params.dynamics.x.mass)));
    fprintf('  Y轴: ωn = %.2f rad/s, ζ = %.4f\n', ...
        sqrt(params.dynamics.y.stiffness / params.dynamics.y.mass), ...
        params.dynamics.y.damping / (2 * sqrt(params.dynamics.y.stiffness * params.dynamics.y.mass)));

    %% 提取输入轨迹参数（来自Python解析）
    t = input_traj.time;
    x_in = input_traj.x;
    y_in = input_traj.y;
    z_in = input_traj.z;
    vx_in = input_traj.vx;
    vy_in = input_traj.vy;
    ax_in = input_traj.ax;
    ay_in = input_traj.ay;

    % 理想轨迹（目标）
    t_ideal = ideal_traj.time;
    x_ideal = ideal_traj.x;
    y_ideal = ideal_traj.y;
    z_ideal = ideal_traj.z;

    % 统一插值到相同时间点（使用输入轨迹的时间）
    x_ideal_interp = interp1(t_ideal, x_ideal, t, 'linear', 'extrap');
    y_ideal_interp = interp1(t_ideal, y_ideal, t, 'linear', 'extrap');
    z_ideal_interp = interp1(t_ideal, z_ideal, t, 'linear', 'extrap');

    %% 系统参数
    % X轴
    mx = params.dynamics.x.mass;
    kx = params.dynamics.x.stiffness;
    cx = params.dynamics.x.damping;

    % Y轴
    my = params.dynamics.y.mass;
    ky = params.dynamics.y.stiffness;
    cy = params.dynamics.y.damping;

    % 固有频率和阻尼比
    omega_nx = sqrt(kx / mx);
    zeta_x = cx / (2 * sqrt(kx * mx));

    omega_ny = sqrt(ky / my);
    zeta_y = cy / (2 * sqrt(ky * my));

    fprintf('  模拟X轴动力学...\n');

    % 状态空间形式：ẋ = Ax + Bu
    % 状态向量：[position_error; velocity_error]
    Ax = [0, 1; -omega_nx^2, -2*zeta_x*omega_nx];
    Bx = [0; -1];  % 输入是加速度

    % 初始化误差状态（从零误差开始）
    n = length(t);
    error_state_x = zeros(2, n);
    error_state_x(:, 1) = [0; 0];

    % 时间积分（RK4）
    fprintf('    正在积分X轴动力学...\n');
    for i = 2:n
        dt = t(i) - t(i-1);

        if dt > 0
            % 参考加速度（输入轨迹的加速度）
            ax_ref = ax_in(i);

            % RK4积分步骤
            k1 = Ax * error_state_x(:, i-1) + Bx * ax_ref;
            k2 = Ax * (error_state_x(:, i-1) + 0.5*dt*k1) + Bx * ax_ref;
            k3 = Ax * (error_state_x(:, i-1) + 0.5*dt*k2) + Bx * ax_ref;
            k4 = Ax * (error_state_x(:, i-1) + dt*k3) + Bx * ax_ref;

            error_state_x(:, i) = error_state_x(:, i-1) + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);

            % 数值保护
            if any(~isfinite(error_state_x(:, i))) || any(abs(error_state_x(:, i)) > 100)
                error_state_x(:, i) = error_state_x(:, i-1);
            end
        end
    end

    fprintf('  模拟Y轴动力学...\n');

    Ay = [0, 1; -omega_ny^2, -2*zeta_y*omega_ny];
    By = [0; -1];

    error_state_y = zeros(2, n);
    error_state_y(:, 1) = [0; 0];

    fprintf('    正在积分Y轴动力学...\n');
    for i = 2:n
        dt = t(i) - t(i-1);

        if dt > 0
            ay_ref = ay_in(i);

            k1 = Ay * error_state_y(:, i-1) + By * ay_ref;
            k2 = Ay * (error_state_y(:, i-1) + 0.5*dt*k1) + By * ay_ref;
            k3 = Ay * (error_state_y(:, i-1) + 0.5*dt*k2) + By * ay_ref;
            k4 = Ay * (error_state_y(:, i-1) + dt*k3) + By * ay_ref;

            error_state_y(:, i) = error_state_y(:, i-1) + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);

            if any(~isfinite(error_state_y(:, i))) || any(abs(error_state_y(:, i)) > 100)
                error_state_y(:, i) = error_state_y(:, i-1);
            end
        end
    end

    %% 计算实际轨迹
    % 实际位置 = 输入位置 + 动力学误差
    x_act = x_in + error_state_x(1, :);
    y_act = y_in + error_state_y(1, :);
    z_act = z_in;  % Z轴假设没有误差

    %% 关键：计算相对于理想轨迹的误差
    error_x = x_act - x_ideal_interp;  % 实际轨迹 vs 理想轨迹
    error_y = y_act - y_ideal_interp;  % 实际轨迹 vs 理想轨迹

    error_magnitude = sqrt(error_x.^2 + error_y.^2);

    %% 误差统计分析
    rms_error = sqrt(mean(error_magnitude.^2));
    max_error = max(error_magnitude);
    mean_error = mean(error_magnitude);

    fprintf('  计算误差统计分析...\n');
    fprintf('    最大X误差: %.3f mm\n', max(abs(error_x)));
    fprintf('    最大Y误差: %.3f mm\n', max(abs(error_y)));
    fprintf('    最大误差幅值: %.3f mm\n', max_error);
    fprintf('    RMS误差幅值: %.3f mm\n', rms_error);
    fprintf('    平均误差幅值: %.3f mm\n', mean_error);

    %% 频率分析
    fprintf('  执行频率分析...\n');

    % 计算采样率
    dt_array = diff(t);
    dt_actual = median(dt_array);
    fs = 1 / dt_actual;

    % 功率谱密度
    if length(error_x) > 100
        try
            [psd_x, freq_x] = pwelch(error_x, [], [], [], fs);
            [psd_y, freq_y] = pwelch(error_y, [], [], [], fs);

            [max_psd_x, idx_x] = max(psd_x);
            [max_psd_y, idx_y] = max(psd_y);

            dominant_freq_x = freq_x(idx_x);
            dominant_freq_y = freq_y(idx_y);

            fprintf('    主导误差频率 (X): %.2f Hz\n', dominant_freq_x);
            fprintf('    主导误差频率 (Y): %.2f Hz\n', dominant_freq_y);
        catch
            fprintf('    频率分析跳过（数据不足）\n');
            dominant_freq_x = 0;
            dominant_freq_y = 0;
        end
    else
        dominant_freq_x = 0;
        dominant_freq_y = 0;
    end

    fprintf('    X轴固有频率: %.2f Hz\n', omega_nx / (2*pi));
    fprintf('    Y轴固有频率: %.2f Hz\n', omega_ny / (2*pi));

    fprintf('  轨迹误差模拟完成！\n\n');

    %% 返回结果
    results.x_ideal = x_ideal_interp;      % 理想轨迹（目标）
    results.y_ideal = y_ideal_interp;
    results.z_ideal = z_ideal_interp;

    results.x_input = x_in;                 % 输入轨迹（发送给打印机的）
    results.y_input = y_in;
    results.z_input = z_in;

    results.x_act = x_act;                  % 实际轨迹（打印出来的）
    results.y_act = y_act;
    results.z_act = z_act;

    results.vx_act = vx_in;                 % 速度（简化）
    results.vy_act = vy_in;
    results.ax_act = ax_in;
    results.ay_act = ay_in;

    % 关键：误差是相对于理想轨迹的
    results.error_x = error_x;              % mm - X方向误差（actual - ideal）
    results.error_y = error_y;              % mm - Y方向误差（actual - ideal）
    results.error_magnitude = error_magnitude; % mm - 误差幅值

    results.rms_error = rms_error;
    results.max_error = max_error;
    results.mean_error = mean_error;

    results.dominant_freq_x = dominant_freq_x;
    results.dominant_freq_y = dominant_freq_y;

    results.natural_freq_x = omega_nx / (2*pi);
    results.natural_freq_y = omega_ny / (2*pi);

end
