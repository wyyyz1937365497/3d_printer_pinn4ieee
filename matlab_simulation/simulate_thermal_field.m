function thermal_field = simulate_thermal_field(trajectory, params)
%% 仿真温度场（移动热源热传导）
% 物理模型：三维非稳态热传导方程
% ∂T/∂t = α·∇²T + Q_source - Q_cooling
%
% 简化：2D有限差分（XY平面），单层
%
% 输入：
%   trajectory - 轨迹数据
%   params - 仿真参数
% 输出：
%   thermal_field - 温度场数据

fprintf('    求解热传导方程...\n');

%% 1. 网格设置
% 空间网格
dx = 2;  % 网格间距 (mm)
dy = 2;

nx = ceil(params.bed_size(1) / dx);
ny = ceil(params.bed_size(2) / dy);

x_grid = 0:dx:params.bed_size(1);
y_grid = 0:dy:params.bed_size(2);
[X, Y] = meshgrid(x_grid, y_grid);

% 时间网格
dt = 0.01;  % 时间步长 (s)
t_end = trajectory.time(end);
nt = ceil(t_end / dt);
t_grid = 0:dt:t_end;

fprintf('    网格: %d x %d, 时间步数: %d\n', nx, ny, nt);

%% 2. 热物性参数
% 热扩散率 α = k / (ρ * c)
alpha = params.material_thermal_conductivity / ...
        (params.material_density * params.material_specific_heat);

% 稳定性检查（CFL条件）
alpha_dt_dx2 = alpha * dt / (dx^2);
if alpha_dt_dx2 > 0.25
    warning('时间步长太大，可能不稳定！建议 dt < %.6f s', dx^2 / (4*alpha));
end

%% 3. 初始和边界条件
% 初始温度场（等于热床温度）
T = ones(ny, nx) * params.T_bed;

% 边界条件（固定温度）
T_left = params.T_ambient;
T_right = params.T_ambient;
T_bottom = params.T_bed;
T_top = params.T_ambient;

%% 4. 冷却系数（对流 + 辐射）
% 对流换热系数（取决于风扇转速）
% 简化模型：h = h_base + h_fan * (fan_speed / 255)
h_base = 10;  % 自然对流 (W/m^2·K)
h_fan = 50;  % 强制对流最大值 (W/m^2·K)
h = h_base + h_fan * (params.fan_speed / 255);

% 表面密度（每单位体积的表面积，考虑层高）
surface_area_per_volume = 2 / params.layer_height;  % 1/m

% 冷却系数 (1/s)
cooling_coefficient = h * surface_area_per_volume / ...
                     (params.material_density * params.material_specific_heat);

% 辐射冷却（简化）
epsilon = 0.9;  % 发射率
sigma = 5.67e-8;  % 斯特藩-玻尔兹曼常数
% 线性化辐射冷却（在T_bed附近）
T_avg = (params.T_nozzle + params.T_bed) / 2 + 273.15;  % 转换为K
radiation_coefficient = 4 * epsilon * sigma * T_avg^3 * surface_area_per_volume / ...
                        (params.material_density * params.material_specific_heat);

% 总冷却系数
cooling_total = cooling_coefficient + radiation_coefficient;

%% 5. 热源模型（移动喷嘴）
% 喷嘴位置插值到仿真时间网格
t_traj = trajectory.time;
x_traj = trajectory.x_ref;
y_traj = trajectory.y_ref;

% 挤出流量 (mm^3/s)
nozzle_area = pi * (params.nozzle_diameter / 2)^2;
Q_flow = trajectory.v_ref .* nozzle_area * params.extrusion_multiplier;

% 热输入功率 (W)
% Q_in = v_extrude * rho * c * (T_nozzle - T_bed)
Q_heat = Q_flow * (params.material_density / 1e9) * ...
         params.material_specific_heat * ...
         (params.T_nozzle - params.T_bed);

% 插值到仿真时间网格
x_traj_sim = interp1(t_traj, x_traj, t_grid, 'linear', 'extrap');
y_traj_sim = interp1(t_traj, y_traj, t_grid, 'linear', 'extrap');
Q_heat_sim = interp1(t_traj, Q_heat, t_grid, 'linear', 'extrap');

%% 6. 热源分布（高斯热源）
% 热源半径（近似为喷嘴直径）
r_source = params.nozzle_diameter / 2;
sigma_source = r_source / 3;  % 高斯分布标准差

% 预计算高斯权重
[X_grid, Y_grid] = meshgrid(x_grid, y_grid);

%% 7. 时间积分（显式有限差分）
fprintf('    时间积分中...\n');

% 存储选定时间点的温度场（节省内存）
save_interval = max(1, floor(nt / 100));  % 保存约100个时间点
T_history = zeros(ceil(nt/save_interval) + 1, ny, nx);
T_history(1, :, :) = T;

% 存储喷嘴路径上的温度
T_nozzle_path = zeros(length(t_traj), 1);
T_interface = zeros(length(t_traj), 1);  % 层间温度（简化为当前层温度）
cooling_rate = zeros(length(t_traj), 1);
time_above_melting = zeros(length(t_traj), 1);

% 记录熔融时间（累积）
t_above_Tm = zeros(ny, nx);

current_save_idx = 1;

for n = 1:nt
    % 当前时间
    t_current = t_grid(n);

    %% 7.1 计算热源项
    % 喷嘴位置
    x_nozzle = x_traj_sim(n);
    y_nozzle = y_traj_sim(n);

    % 高斯热源分布
    dist_sq = (X_grid - x_nozzle).^2 + (Y_grid - y_nozzle).^2;
    heat_source = (Q_heat_sim(n) / (2 * pi * sigma_source^2 * params.layer_height * dx * dy)) * ...
                  exp(-dist_sq / (2 * sigma_source^2));

    % 转换为温度变化率 (°C/s)
    heat_source_rate = heat_source / (params.material_density * params.material_specific_heat);

    %% 7.2 计算拉普拉斯算子（热扩散）
    % 内部点
    T_center = T(2:end-1, 2:end-1);
    T_left_inner = T(2:end-1, 1:end-2);
    T_right_inner = T(2:end-1, 3:end);
    T_bottom_inner = T(1:end-2, 2:end-1);
    T_top_inner = T(3:end, 2:end-1);

    laplacian = (T_left_inner + T_right_inner + T_bottom_inner + T_top_inner - 4*T_center) / (dx^2);

    %% 7.3 计算冷却项
    % 对流 + 辐射冷却：h*(T - T_ambient)
    cooling_rate_field = cooling_total * (T - params.T_ambient);

    %% 7.4 更新温度场
    % ∂T/∂t = α·∇²T + Q_source - Q_cooling
    dT_dt = alpha * laplacian + heat_source_rate(2:end-1, 2:end-1) - cooling_rate_field(2:end-1, 2:end-1);

    % 更新内部点
    T(2:end-1, 2:end-1) = T(2:end-1, 2:end-1) + dT_dt * dt;

    % 更新边界点（固定温度）
    T(1, :) = T_bottom;  % 底部
    T(end, :) = T_top;   % 顶部
    T(:, 1) = T_left;    % 左侧
    T(:, end) = T_right; % 右侧

    %% 7.5 记录熔融时间
    t_above_Tm(T > params.melting_point) = t_above_Tm(T > params.melting_point) + dt;

    %% 7.6 保存数据
    if mod(n, save_interval) == 0
        current_save_idx = current_save_idx + 1;
        T_history(current_save_idx, :, :) = T;
    end

    % 进度显示
    if mod(n, floor(nt/10)) == 0
        fprintf('      进度: %.0f%%, T_max: %.1f°C\n', n/nt*100, max(T(:)));
    end
end

%% 8. 提取轨迹点上的温度数据
% 找到每个轨迹时刻对应的温度
for i = 1:length(t_traj)
    % 找到最近的网格点
    [~, idx_x] = min(abs(x_grid - trajectory.x_ref(i)));
    [~, idx_y] = min(abs(y_grid - trajectory.y_ref(i)));

    % 当前层温度
    T_current = T(idx_y, idx_x);
    T_nozzle_path(i) = T_current;

    % 层间温度（简化：假设下层温度略低）
    T_interface(i) = T_current * 0.95;

    % 冷却速率（简化：当前温度与环境温度的差值 * 冷却系数）
    cooling_rate(i) = cooling_total * (T_current - params.T_ambient);

    % 时间高于熔点
    time_above_melting(i) = t_above_Tm(idx_y, idx_x);
end

%% 9. 计算温度梯度
% 层间温度梯度（Z方向，简化）
gradient_z = abs(T_nozzle_path - params.T_bed) / params.layer_height;

% 面内温度梯度（XY平面）
gradient_xy_mag = zeros(length(t_traj), 1);
for i = 1:length(t_traj)
    [~, idx_x] = min(abs(x_grid - trajectory.x_ref(i)));
    [~, idx_y] = min(abs(y_grid - trajectory.y_ref(i)));

    if idx_x > 1 && idx_x < nx && idx_y > 1 && idx_y < ny
        dTdx = (T(idx_y, idx_x+1) - T(idx_y, idx_x-1)) / (2*dx);
        dTdy = (T(idx_y+1, idx_x) - T(idx_y-1, idx_x)) / (2*dy);
        gradient_xy_mag(i) = sqrt(dTdx^2 + dTdy^2);
    end
end

%% 10. 热累积指标
% 热累积时间：在喷嘴附近的停留时间
thermal_accumulation_time = zeros(length(t_traj), 1);
window_size = 10;  % 时间窗口
for i = 1:length(t_traj)
    % 计算过去window_size秒内，喷嘴在当前位置附近的次数
    time_window = t_traj(i) - window_size;
    idx_in_window = find(t_traj >= time_window & t_traj <= t_traj(i));

    if ~isempty(idx_in_window)
        distances = sqrt((trajectory.x_ref(idx_in_window) - trajectory.x_ref(i)).^2 + ...
                        (trajectory.y_ref(idx_in_window) - trajectory.y_ref(i)).^2);
        thermal_accumulation_time(i) = sum(distances < params.nozzle_diameter * 2) * dt;
    end
end

%% 11. 构建输出数据结构
thermal_field = struct();

% 网格信息
thermal_field.x_grid = x_grid;
thermal_field.y_grid = y_grid;
thermal_field.X = X;
thermal_field.Y = Y;
thermal_field.dx = dx;
thermal_field.dy = dy;

% 时间信息
thermal_field.t_grid = t_grid;
thermal_field.dt = dt;

% 完整温度场历史
thermal_field.T_history = T_history;
thermal_field.T_final = T;

% 轨迹点温度
thermal_field.T_nozzle_path = T_nozzle_path;
thermal_field.T_interface = T_interface;
thermal_field.cooling_rate = cooling_rate;
thermal_field.time_above_melting = time_above_melting;

% 温度梯度
thermal_field.gradient_z = gradient_z;
thermal_field.gradient_xy = gradient_xy_mag;

% 热累积
thermal_field.thermal_accumulation_time = thermal_accumulation_time;

% 喷嘴位置和热输入
thermal_field.x_nozzle = trajectory.x_ref;
thermal_field.y_nozzle = trajectory.y_ref;
thermal_field.Q_heat = Q_heat;

% 统计
thermal_field.T_max = max(T_nozzle_path);
thermal_field.T_min = min(T_nozzle_path);
thermal_field.T_mean = mean(T_nozzle_path);
thermal_field.T_std = std(T_nozzle_path);

fprintf('    温度场仿真完成\n');
fprintf('    温度范围: %.1f - %.1f °C\n', thermal_field.T_min, thermal_field.T_max);

end
