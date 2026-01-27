function trajectory_error = simulate_trajectory_error(trajectory, params)
%% 仿真轨迹误差（二阶震荡系统）
% 物理模型：质量-弹簧-阻尼系统
% 方程：m*x'' + c*x' + k*x = F(t)
%
% 输入：
%   trajectory - 参考轨迹数据
%   params - 仿真参数
% 输出：
%   trajectory_error - 轨迹误差数据

fprintf('    求解二阶系统响应...\n');

%% 1. 提取参考轨迹
t = trajectory.time;
x_ref = trajectory.x_ref;
y_ref = trajectory.y_ref;
ax_ref = trajectory.ax_ref;
ay_ref = trajectory.ay_ref;

n_points = length(t);
dt = t(2) - t(1);

%% 2. 二阶系统参数（X轴和Y轴）
% X轴参数
m_x = params.mass_x;
k_x = params.stiffness_x;
c_x = params.damping_x;

% Y轴参数
m_y = params.mass_y;
k_y = params.stiffness_y;
c_y = params.damping_y;

% 计算系统特性
omega_n_x = sqrt(k_x / m_x);  % 固有频率
zeta_x = c_x / (2 * sqrt(m_x * k_x));  % 阻尼比

omega_n_y = sqrt(k_y / m_y);
zeta_y = c_y / (2 * sqrt(m_y * k_y));

%% 3. 状态空间表示
% 状态向量：[x; vx] 和 [y; vy]
% X轴：dx/dt = [vx; (-k/m)*x + (-c/m)*vx + (1/m)*F]
% Y轴：dy/dt = [vy; (-k/m)*y + (-c/m)*vy + (1/m)*F]

% X轴状态空间矩阵
A_x = [0, 1;
       -k_x/m_x, -c_x/m_x];
B_x = [0;
       1/m_x];

% Y轴状态空间矩阵
A_y = [0, 1;
       -k_y/m_y, -c_y/m_y];
B_y = [0;
       1/m_y];

%% 4. 计算惯性力输入
% 惯性力 F_inertia = m * a_ref
F_inertia_x = m_x * ax_ref;
F_inertia_y = m_y * ay_ref;

%% 5. 时间响应求解（欧拉法）
% 初始状态
state_x = [x_ref(1); 0];  % [位置; 速度]
state_y = [y_ref(1); 0];

% 预分配数组
x_act = zeros(n_points, 1);
y_act = zeros(n_points, 1);
vx_act = zeros(n_points, 1);
vy_act = zeros(n_points, 1);
ax_act = zeros(n_points, 1);
ay_act = zeros(n_points, 1);

% 动力学量存储
F_inertia_x_out = zeros(n_points, 1);
F_inertia_y_out = zeros(n_points, 1);
delta_L_x = zeros(n_points, 1);
delta_L_y = zeros(n_points, 1);
F_elastic_x = zeros(n_points, 1);
F_elastic_y = zeros(n_points, 1);
F_damping_x = zeros(n_points, 1);
F_damping_y = zeros(n_points, 1);

% 时间积分
for i = 1:n_points
    % 保存当前状态
    x_act(i) = state_x(1);
    y_act(i) = state_y(1);
    vx_act(i) = state_x(2);
    vy_act(i) = state_y(2);

    % 计算实际加速度（从状态方程）
    ax_act(i) = A_x(2,:) * state_x + B_x * F_inertia_x(i);
    ay_act(i) = A_y(2,:) * state_y + B_y * F_inertia_y(i);

    % 计算动力学量
    F_inertia_x_out(i) = F_inertia_x(i);
    F_inertia_y_out(i) = F_inertia_y(i);

    % 皮带弹性伸长 delta_L = F / k
    delta_L_x(i) = F_inertia_x(i) / k_x;
    delta_L_y(i) = F_inertia_y(i) / k_y;

    % 弹性力 F_elastic = k * delta_L
    F_elastic_x(i) = k_x * (x_ref(i) - x_act(i));
    F_elastic_y(i) = k_y * (y_ref(i) - y_act(i));

    % 阻尼力 F_damping = c * v
    F_damping_x(i) = c_x * vx_act(i);
    F_damping_y(i) = c_y * vy_act(i);

    % 状态更新（欧拉法）
    if i < n_points
        % X轴
        dxdt_x = A_x * state_x + B_x * F_inertia_x(i);
        state_x = state_x + dxdt_x * dt;

        % Y轴
        dxdt_y = A_y * state_y + B_y * F_inertia_y(i);
        state_y = state_y + dxdt_y * dt;
    end
end

%% 6. 计算误差
epsilon_x = x_act - x_ref;
epsilon_y = y_act - y_ref;
epsilon_r = sqrt(epsilon_x.^2 + epsilon_y.^2);  % 位置误差幅值

% 速度误差
epsilon_vx = vx_act - trajectory.vx_ref;
epsilon_vy = vy_act - trajectory.vy_ref;

%% 7. 计算系统性能指标
% 上升时间（近似）
[~, idx_10] = max(epsilon_r > 0.1 * max(epsilon_r));
[~, idx_90] = max(epsilon_r > 0.9 * max(epsilon_r));
if idx_90 > idx_10
    rise_time = t(idx_90) - t(idx_10);
else
    rise_time = 0;
end

% 超调量
steady_state_error = mean(epsilon_r(round(end/2):end));
max_error = max(epsilon_r);
if steady_state_error > 0
    overshoot = (max_error - steady_state_error) / steady_state_error * 100;
else
    overshoot = 0;
end

%% 8. 构建输出数据结构
trajectory_error = struct();

% 系统参数
trajectory_error.omega_n_x = omega_n_x;
trajectory_error.omega_n_y = omega_n_y;
trajectory_error.zeta_x = zeta_x;
trajectory_error.zeta_y = zeta_y;
trajectory_error.rise_time = rise_time;
trajectory_error.overshoot = overshoot;

% 参考轨迹
trajectory_error.x_ref = x_ref;
trajectory_error.y_ref = y_ref;
trajectory_error.vx_ref = trajectory.vx_ref;
trajectory_error.vy_ref = trajectory.vy_ref;
trajectory_error.ax_ref = ax_ref;
trajectory_error.ay_ref = ay_ref;

% 实际轨迹
trajectory_error.x_act = x_act;
trajectory_error.y_act = y_act;
trajectory_error.vx_act = vx_act;
trajectory_error.vy_act = vy_act;
trajectory_error.ax_act = ax_act;
trajectory_error.ay_act = ay_act;

% 误差
trajectory_error.epsilon_x = epsilon_x;
trajectory_error.epsilon_y = epsilon_y;
trajectory_error.position_error_magnitude = epsilon_r;
trajectory_error.epsilon_vx = epsilon_vx;
trajectory_error.epsilon_vy = epsilon_vy;

% 动力学量
trajectory_error.F_inertia_x = F_inertia_x_out;
trajectory_error.F_inertia_y = F_inertia_y_out;
trajectory_error.delta_L_x = delta_L_x;
trajectory_error.delta_L_y = delta_L_y;
trajectory_error.F_elastic_x = F_elastic_x;
trajectory_error.F_elastic_y = F_elastic_y;
trajectory_error.F_damping_x = F_damping_x;
trajectory_error.F_damping_y = F_damping_y;

% 统计
trajectory_error.max_error = max(epsilon_r);
trajectory_error.rms_error = sqrt(mean(epsilon_r.^2));
trajectory_error.steady_state_error = steady_state_error;

fprintf('    轨迹误差仿真完成\n');
fprintf('    最大误差: %.4f mm, RMS误差: %.4f mm\n', ...
    trajectory_error.max_error, trajectory_error.rms_error);
fprintf('    系统特性: ω_n_x=%.2f rad/s, ζ_x=%.3f\n', omega_n_x, zeta_x);

end
