function results = simulate_trajectory_error(trajectory_data, params)
% SIMULATE_TRAJECTORY_ERROR - 模拟由于惯性和皮带弹性引起的轨迹误差
%
% 该函数将打印机动力学建模为二阶质量-弹簧-阻尼系统：
%   m·x'' + c·x' + k·x = F(t)
%
% 其中：
%   m - 移动系统的有效质量
%   c - 阻尼系数（皮带和摩擦）
%   k - 传动刚度（皮带弹性）
%   F(t) - 激励函数（加速度引起的惯性力）
%
% 系统对参考轨迹的响应包括：
% 1. 由于加速度引起的滞后（惯性）
% 2. 由于欠阻尼响应引起的振荡
% 3. 稳态误差（如有）
%
% 输入：
%   trajectory_data - 来自parse_gcode.m的结构
%   params          - 来自physics_parameters.m的物理参数
%
% 输出：
%   results         - 包含实际轨迹和误差向量的结构
%
% 参考：二阶系统的控制理论

    fprintf('正在模拟轨迹误差（二阶系统动力学）...\n');

    %% 提取时间序列数据
    t = trajectory_data.time;
    n_points = length(t);

    % 参考轨迹（来自G代码）
    x_ref = trajectory_data.x;
    y_ref = trajectory_data.y;
    z_ref = trajectory_data.z;

    % 参考加速度（激励函数）
    ax_ref = trajectory_data.ax;
    ay_ref = trajectory_data.ay;

    %% 提取系统参数
    % X轴动力学
    mx = params.dynamics.x.mass;
    kx = params.dynamics.x.stiffness;
    cx = params.dynamics.x.damping;
    wn_x = params.dynamics.x.natural_freq;
    zeta_x = params.dynamics.x.damping_ratio;

    % Y轴动力学
    my = params.dynamics.y.mass;
    ky = params.dynamics.y.stiffness;
    cy = params.dynamics.y.damping;
    wn_y = params.dynamics.y.natural_freq;
    zeta_y = params.dynamics.y.damping_ratio;

    fprintf('  X轴: ωn = %.2f rad/s, ζ = %.4f\n', wn_x, zeta_x);
    fprintf('  Y轴: ωn = %.2f rad/s, ζ = %.4f\n', wn_y, zeta_y);

    %% 仿真时间步长（使用均匀网格）
    dt = params.simulation.time_step;

    % 创建均匀时间网格
    t_uniform = linspace(t(1), t(end), ceil((t(end) - t(1)) / dt) + 1);
    dt_actual = t_uniform(2) - t_uniform(1);
    n_uniform = length(t_uniform);

    % 数值稳定性检查
    % 对于RK4，时间步长应该满足: dt < 0.1 * T_period = 0.1 * 2*pi/wn
    dt_max_x = 0.1 * 2 * pi / wn_x;
    dt_max_y = 0.1 * 2 * pi / wn_y;
    dt_max = min(dt_max_x, dt_max_y);

    if dt_actual > dt_max
        fprintf('  警告：时间步长过大，可能导致数值不稳定\n');
        fprintf('    dt_actual = %.4f ms, dt_max = %.4f ms\n', dt_actual*1000, dt_max*1000);
        fprintf('    重新采样以使用更小的时间步长...\n');
        % 重新计算采样点数
        n_new = ceil((t(end) - t(1)) / dt_max) + 1;
        t_uniform = linspace(t(1), t(end), n_new);
        dt_actual = t_uniform(2) - t_uniform(1);
        n_uniform = length(t_uniform);
        fprintf('    新时间步长: %.4f ms (%.1f Hz)\n', dt_actual*1000, 1/dt_actual);
    end

    % 插值参考到均匀网格
    x_ref_uniform = interp1(t, x_ref, t_uniform, 'linear', 'extrap');
    y_ref_uniform = interp1(t, y_ref, t_uniform, 'linear', 'extrap');
    ax_ref_uniform = interp1(t, ax_ref, t_uniform, 'linear', 'extrap');
    ay_ref_uniform = interp1(t, ay_ref, t_uniform, 'linear', 'extrap');

    % 检查输入数据的有效性
    if any(~isfinite(ax_ref_uniform)) || any(~isfinite(ay_ref_uniform))
        error('输入加速度包含NaN或Inf值');
    end

    %% 模拟X轴动力学（二阶系统）
    fprintf('  模拟X轴动力学...\n');

    % 状态空间: [x; v] 其中x是位置误差，v是速度误差
    % 对于受加速度驱动的二阶系统：
    % x' = v
    % v' = -(c/m)*v - (k/m)*x - a_ref
    %
    % 注意：a_ref上的负号是因为误差定义为：
    % error = actual - reference
    % 实际位置响应于惯性力 F = -m*a_ref

    % 初始化状态
    x_state = zeros(2, n_uniform);
    x_state(:, 1) = [0; 0];  % 从零误差开始

    % 系统矩阵
    Ax = [0, 1;
          -kx/mx, -cx/mx];
    Bx = [0; -1];  % 输入是加速度

    % 时间积分（RK4方法 - 与GPU版本一致）
    for i = 2:n_uniform
        % RK4积分步骤
        k1 = Ax * x_state(:, i-1) + Bx * ax_ref_uniform(i-1);
        k2 = Ax * (x_state(:, i-1) + 0.5*dt_actual*k1) + Bx * ax_ref_uniform(i-1);
        k3 = Ax * (x_state(:, i-1) + 0.5*dt_actual*k2) + Bx * ax_ref_uniform(i-1);
        k4 = Ax * (x_state(:, i-1) + dt_actual*k3) + Bx * ax_ref_uniform(i-1);

        x_state(:, i) = x_state(:, i-1) + (dt_actual/6) * (k1 + 2*k2 + 2*k3 + k4);

        % 数值保护：防止误差无限增长
        if any(~isfinite(x_state(:, i))) || any(abs(x_state(:, i)) > 100)
            x_state(:, i) = x_state(:, i-1);  % 保持前一个值
        end
    end

    % 提取实际位置
    x_act_uniform = x_ref_uniform + x_state(1, :);
    vx_act_uniform = gradient(x_act_uniform, dt_actual);

    %% 模拟Y轴动力学
    fprintf('  模拟Y轴动力学...\n');

    y_state = zeros(2, n_uniform);
    y_state(:, 1) = [0; 0];

    Ay = [0, 1;
          -ky/my, -cy/my];
    By = [0; -1];

    % RK4积分
    for i = 2:n_uniform
        k1 = Ay * y_state(:, i-1) + By * ay_ref_uniform(i-1);
        k2 = Ay * (y_state(:, i-1) + 0.5*dt_actual*k1) + By * ay_ref_uniform(i-1);
        k3 = Ay * (y_state(:, i-1) + 0.5*dt_actual*k2) + By * ay_ref_uniform(i-1);
        k4 = Ay * (y_state(:, i-1) + dt_actual*k3) + By * ay_ref_uniform(i-1);

        y_state(:, i) = y_state(:, i-1) + (dt_actual/6) * (k1 + 2*k2 + 2*k3 + k4);

        % 数值保护
        if any(~isfinite(y_state(:, i))) || any(abs(y_state(:, i)) > 100)
            y_state(:, i) = y_state(:, i-1);
        end
    end

    y_act_uniform = y_ref_uniform + y_state(1, :);
    vy_act_uniform = gradient(y_act_uniform, dt_actual);

    %% 计算误差向量（作为向量，而不仅仅是幅值）
    fprintf('  计算误差向量...\n');

    % 位置误差向量（分量）
    error_x_uniform = x_state(1, :);
    error_y_uniform = y_state(1, :);

    % 位置误差幅值
    error_magnitude_uniform = sqrt(error_x_uniform.^2 + error_y_uniform.^2);

    % 误差方向角度
    error_direction = atan2(error_y_uniform, error_x_uniform);

    %% 动态力
    fprintf('  计算动态力...\n');

    % 惯性力: F = m * a_ref
    F_inertia_x = mx * ax_ref_uniform;
    F_inertia_y = my * ay_ref_uniform;

    % 弹性力（皮带拉伸）: F = k * error
    F_elastic_x = kx * error_x_uniform;
    F_elastic_y = ky * error_y_uniform;

    % 阻尼力: F = c * v_error
    v_error_x = x_state(2, :);
    v_error_y = y_state(2, :);
    F_damping_x = cx * v_error_x;
    F_damping_y = cy * v_error_y;

    %% 插值回原始时间网格
    x_act = interp1(t_uniform, x_act_uniform, t, 'linear', 'extrap');
    y_act = interp1(t_uniform, y_act_uniform, t, 'linear', 'extrap');
    vx_act = interp1(t_uniform, vx_act_uniform, t, 'linear', 'extrap');
    vy_act = interp1(t_uniform, vy_act_uniform, t, 'linear', 'extrap');

    error_x = interp1(t_uniform, error_x_uniform, t, 'linear', 'extrap');
    error_y = interp1(t_uniform, error_y_uniform, t, 'linear', 'extrap');
    error_magnitude = interp1(t_uniform, error_magnitude_uniform, t, 'linear', 'extrap');
    error_dir = interp1(t_uniform, error_direction, t, 'linear', 'extrap');

    F_inertia_x = interp1(t_uniform, F_inertia_x, t, 'linear', 'extrap');
    F_inertia_y = interp1(t_uniform, F_inertia_y, t, 'linear', 'extrap');
    F_elastic_x = interp1(t_uniform, F_elastic_x, t, 'linear', 'extrap');
    F_elastic_y = interp1(t_uniform, F_elastic_y, t, 'linear', 'extrap');

    %% 统计分析
    fprintf('  误差统计分析：\n');
    fprintf('    最大X误差: %.3f mm\n', max(abs(error_x)));
    fprintf('    最大Y误差: %.3f mm\n', max(abs(error_y)));
    fprintf('    最大误差幅值: %.3f mm\n', max(error_magnitude));
    fprintf('    RMS误差幅值: %.3f mm\n', rms(error_magnitude));
    fprintf('    平均误差幅值: %.3f mm\n', mean(error_magnitude));

    %% 频率分析（检测共振激发）
    fprintf('  执行频率分析...\n');

    % 误差的功率谱密度
    [psd_x, freq_x] = pwelch(error_x, [], [], [], 1/dt_actual);
    [psd_y, freq_y] = pwelch(error_y, [], [], [], 1/dt_actual);

    % 查找主导频率
    [max_psd_x, idx_x] = max(psd_x);
    [max_psd_y, idx_y] = max(psd_y);

    dominant_freq_x = freq_x(idx_x);
    dominant_freq_y = freq_y(idx_y);

    fprintf('    主导误差频率 (X): %.2f Hz\n', dominant_freq_x);
    fprintf('    主导误差频率 (Y): %.2f Hz\n', dominant_freq_y);
    fprintf('    X轴固有频率: %.2f Hz\n', wn_x / (2*pi));
    fprintf('    Y轴固有频率: %.2f Hz\n', wn_y / (2*pi));

    %% 创建输出结构
    results.time = t;

    % 参考轨迹
    results.x_ref = x_ref;
    results.y_ref = y_ref;
    results.z_ref = z_ref;

    % 实际轨迹（含动力学）
    results.x_act = x_act;
    results.y_act = y_act;
    results.z_act = z_ref;  % Z轴未在动力学中建模

    % 实际速度
    results.vx_act = vx_act;
    results.vy_act = vy_act;
    results.vz_act = zeros(size(vy_act));

    % 实际加速度（数值计算）
    results.ax_act = gradient(vx_act, dt);
    results.ay_act = gradient(vy_act, dt);
    results.az_act = zeros(size(vy_act));

    % 误差向量（不仅仅是幅值！）
    results.error_x = error_x;           % mm - 误差向量的X分量
    results.error_y = error_y;           % mm - 误差向量的Y分量
    results.error_magnitude = error_magnitude;  % mm - |误差向量|
    results.error_direction = error_dir; % rad - 误差向量的方向

    % 动态力
    results.F_inertia_x = F_inertia_x;   % N
    results.F_inertia_y = F_inertia_y;   % N
    results.F_elastic_x = F_elastic_x;   % N
    results.F_elastic_y = F_elastic_y;   % N
    results.F_damping_x = F_damping_x;   % N (在均匀网格上)
    results.F_damping_y = F_damping_y;   % N (在均匀网格上)

    % 皮带拉伸（位移）
    results.belt_stretch_x = error_x;    % mm
    results.belt_stretch_y = error_y;    % mm

    % 系统响应指标
    results.settling_time_x = params.dynamics.x.settling_time;
    results.settling_time_y = params.dynamics.y.settling_time;
    results.overshoot_x = exp(-pi * zeta_x / sqrt(1 - zeta_x^2)) * 100;  % %
    results.overshoot_y = exp(-pi * zeta_y / sqrt(1 - zeta_y^2)) * 100;  % %

    % 频率分析
    results.frequency_x = freq_x;
    results.frequency_y = freq_y;
    results.psd_x = psd_x;
    results.psd_y = psd_y;
    results.dominant_freq_x = dominant_freq_x;  % Hz
    results.dominant_freq_y = dominant_freq_y;  % Hz

    fprintf('  轨迹误差模拟完成！\n\n');

    %% 可选：绘图（如果处于调试模式）
    if params.debug.plot_trajectory
        figure('Name', '轨迹误差分析', 'Position', [100, 100, 1200, 800]);

        % 参考与实际轨迹（顶视图）
        subplot(2, 3, 1);
        plot(x_ref, y_ref, 'b--', 'LineWidth', 1.5); hold on;
        plot(x_act, y_act, 'r-', 'LineWidth', 1);
        axis equal;
        grid on;
        xlabel('X (mm)');
        ylabel('Y (mm)');
        title('参考 vs 实际轨迹');
        legend('参考', '实际', 'Location', 'best');

        % 误差幅值随时间变化
        subplot(2, 3, 2);
        plot(t, error_magnitude, 'r-', 'LineWidth', 1);
        grid on;
        xlabel('时间 (s)');
        ylabel('误差幅值 (mm)');
        title('位置误差幅值');

        % 误差分量（向量可视化）
        subplot(2, 3, 3);
        % 每10个点显示一个箭头，避免太密集
        skip = max(1, floor(length(t) / 100));
        quiver(x_ref(1:skip:end), y_ref(1:skip:end), ...
                error_x(1:skip:end), error_y(1:skip:end), ...
                'AutoScale', 'on', 'MaxHeadSize', 0.5);
        hold on;
        plot(x_ref, y_ref, 'k--', 'LineWidth', 0.5);
        axis equal;
        grid on;
        xlabel('X (mm)');
        ylabel('Y (mm)');
        title('误差向量（采样显示）');

        % X轴误差
        subplot(2, 3, 4);
        plot(t, error_x, 'b-', 'LineWidth', 1);
        grid on;
        xlabel('时间 (s)');
        ylabel('X 误差 (mm)');
        title('X轴位置误差');

        % Y轴误差
        subplot(2, 3, 5);
        plot(t, error_y, 'r-', 'LineWidth', 1);
        grid on;
        xlabel('时间 (s)');
        ylabel('Y 误差 (mm)');
        title('Y轴位置误差');

        % 功率谱密度
        subplot(2, 3, 6);
        semilogy(freq_x, psd_x, 'b-', 'LineWidth', 1.5); hold on;
        semilogy(freq_y, psd_y, 'r-', 'LineWidth', 1.5);
        xline(wn_x / (2*pi), 'b--', 'X \omega_n', 'LineWidth', 1.5);
        xline(wn_y / (2*pi), 'r--', 'Y \omega_n', 'LineWidth', 1.5);
        grid on;
        xlabel('频率 (Hz)');
        ylabel('PSD (mm²/Hz)');
        title('误差功率谱密度');
        legend('X误差', 'Y误差', 'Location', 'best');
        xlim([0, 100]);

        drawnow;
    end

end