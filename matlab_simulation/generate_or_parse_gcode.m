function [gcode_data, trajectory] = generate_or_parse_gcode(params)
%% 生成或解析G-code轨迹
% 输入：
%   params - 仿真参数结构体
% 输出：
%   gcode_data - G-code特征数据
%   trajectory - 轨迹数据（时间序列）

fprintf('  生成虚拟G-code轨迹...\n');

%% 1. 根据参数类型生成轨迹
switch params.trajectory_type
    case 'random_rectangles'
        [x_ref, y_ref] = generate_random_rectangles(params);
    case 'sine_wave'
        [x_ref, y_ref] = generate_sine_wave(params);
    case 'spiral'
        [x_ref, y_ref] = generate_spiral(params);
    otherwise
        [x_ref, y_ref] = generate_random_rectangles(params);
end

%% 2. 生成时间序列（根据速度和加速度）
% 计算线段长度
n_points = length(x_ref);
segment_lengths = zeros(n_points-1, 1);
for i = 1:n_points-1
    segment_lengths(i) = sqrt((x_ref(i+1)-x_ref(i))^2 + (y_ref(i+1)-y_ref(i))^2);
end

% 梯形速度规划：加速-匀速-减速
v_current = 0;
t = zeros(1, 1);
x_traj = x_ref(1);
y_traj = y_ref(1);
v_traj = 0;
a_traj = 0;

time_points = [0];
x_points = [x_ref(1)];
y_points = [y_ref(1)];
v_points = [0];
a_points = [0];

for i = 1:n_points-1
    % 当前线段长度和方向
    L = segment_lengths(i);
    dx = x_ref(i+1) - x_ref(i);
    dy = y_ref(i+1) - y_ref(i);
    direction = atan2(dy, dx);

    % 梯形速度曲线
    % 加速距离
    d_accel = v_current^2 / (2 * params.acceleration);

    if L < 2 * d_accel
        % 线段太短，无法达到目标速度
        v_max = sqrt(v_current^2 / 2 + L * params.acceleration / 2);
        t_accel = (v_max - v_current) / params.acceleration;
        t_decel = v_max / params.acceleration;
    else
        % 可以达到目标速度
        v_max = min(params.print_speed, sqrt((v_current^2 + params.print_speed^2)/2 + params.acceleration * L / 2));

        % 加速时间
        t_accel = (v_max - v_current) / params.acceleration;

        % 匀速距离
        d_cruise = L - (v_max^2 - v_current^2) / (2 * params.acceleration) - v_max^2 / (2 * params.acceleration);

        % 匀速时间
        t_cruise = d_cruise / v_max;

        % 减速时间
        t_decel = v_max / params.acceleration;
    end

    % 生成时间序列点
    dt = 0.001;  % 1ms时间步长
    n_steps = ceil((t_accel + t_cruise + t_decel) / dt);

    for step = 1:n_steps
        t_current = step * dt;

        if t_current <= t_accel
            % 加速段
            v = v_current + params.acceleration * t_current;
            a = params.acceleration;
            x_local = v_current * t_current + 0.5 * params.acceleration * t_current^2;
        elseif t_current <= t_accel + t_cruise
            % 匀速段
            t_c = t_current - t_accel;
            v = v_max;
            a = 0;
            x_local = v_current * t_accel + 0.5 * params.acceleration * t_accel^2 + v_max * t_c;
        else
            % 减速段
            t_d = t_current - t_accel - t_cruise;
            v = v_max - params.acceleration * t_d;
            a = -params.acceleration;
            x_local = v_current * t_accel + 0.5 * params.acceleration * t_accel^2 + ...
                      v_max * t_cruise + v_max * t_d - 0.5 * params.acceleration * t_d^2;
        end

        % 转换为全局坐标
        time_points(end+1) = time_points(end) + dt;
        x_points(end+1) = x_points(end) + (v * cos(direction) * dt);
        y_points(end+1) = y_points(end) + (v * sin(direction) * dt);
        v_points(end+1) = v;
        a_points(end+1) = a;
    end

    v_current = 0;  % 每段结束时减速到0（简化）
end

%% 3. 计算导数和特征
% 速度分量
vx_points = gradient(x_points, 0.001);
vy_points = gradient(y_points, 0.001);

% 加速度分量
ax_points = gradient(vx_points, 0.001);
ay_points = gradient(vy_points, 0.001);

% 加加速度
jx_points = gradient(ax_points, 0.001);
jy_points = gradient(ay_points, 0.001);

%% 4. 检测转角和特征
% 计算方向变化
direction_angles = atan2(vy_points, vx_points);
direction_changes = [0; mod(diff(direction_angles + pi), 2*pi) - pi];  % 归一化到[-pi, pi]

% 转角检测：方向变化大于阈值
corner_threshold = deg2rad(15);  % 15度以上认为是转角
is_corner = abs(direction_changes) > corner_threshold;

% 转角角度
corner_angles = zeros(size(direction_changes));
corner_angles(is_corner) = abs(direction_changes(is_corner));

% 曲率（近似）
curvature = abs(direction_changes) ./ sqrt(vx_points.^2 + vy_points.^2 + eps);
curvature(1) = 0;

%% 5. 构建G-code特征数据结构
gcode_data = struct();
gcode_data.is_corner = is_corner;
gcode_data.corner_angles = corner_angles;
gcode_data.curvature = curvature;
gcode_data.direction_changes = direction_changes;
gcode_data.direction_angles = direction_angles;

% 距离上次/下次转角的距离
d_last_corner = zeros(length(x_points), 1);
d_next_corner = zeros(length(x_points), 1);
corner_indices = find(is_corner);

if ~isempty(corner_indices)
    for i = 1:length(x_points)
        % 距离上次转角
        last_corners = corner_indices(corner_indices < i);
        if ~isempty(last_corners)
            last_corner_idx = last_corners(end);
            d_last_corner(i) = sqrt((x_points(i) - x_points(last_corner_idx))^2 + ...
                                   (y_points(i) - y_points(last_corner_idx))^2);
        else
            d_last_corner(i) = inf;
        end

        % 距离下次转角
        next_corners = corner_indices(corner_indices > i);
        if ~isempty(next_corners)
            next_corner_idx = next_corners(1);
            d_next_corner(i) = sqrt((x_points(i) - x_points(next_corner_idx))^2 + ...
                                   (y_points(i) - y_points(next_corner_idx))^2);
        else
            d_next_corner(i) = inf;
        end
    end
end

gcode_data.d_last_corner = d_last_corner;
gcode_data.d_next_corner = d_next_corner;

%% 6. 构建轨迹数据结构
trajectory = struct();
trajectory.time = time_points;

% 参考轨迹（规划轨迹）
trajectory.x_ref = x_points;
trajectory.y_ref = y_points;
trajectory.z_ref = ones(size(x_points)) * params.layer_height;  % 单层仿真

trajectory.vx_ref = vx_points;
trajectory.vy_ref = vy_points;
trajectory.v_ref = v_points;

trajectory.ax_ref = ax_points;
trajectory.ay_ref = ay_points;

trajectory.jx_ref = jx_points;
trajectory.jy_ref = jy_points;

fprintf('  轨迹生成完成: %d 个时间点, %d 个转角\n', ...
    length(x_points), sum(is_corner));

%% 子函数
function [x, y] = generate_random_rectangles(params)
    % 生成随机矩形轨迹
    n_corners = params.num_corners;
    x = zeros(n_corners + 1, 1);
    y = zeros(n_corners + 1, 1);

    % 在打印床范围内随机生成点
    margin = 20;
    for i = 1:n_corners
        x(i) = margin + rand() * (params.bed_size(1) - 2*margin);
        y(i) = margin + rand() * (params.bed_size(2) - 2*margin);
    end
    x(n_corners + 1) = x(1);
    y(n_corners + 1) = y(1);
end

function [x, y] = generate_sine_wave(params)
    % 生成正弦波轨迹
    x = linspace(20, params.bed_size(1)-20, 100)';
    amplitude = 30;
    frequency = 0.1;
    y = params.bed_size(2)/2 + amplitude * sin(frequency * x);
end

function [x, y] = generate_spiral(params)
    % 生成螺旋轨迹
    center_x = params.bed_size(1) / 2;
    center_y = params.bed_size(2) / 2;
    max_radius = min(params.bed_size) / 2 - 20;

    theta = linspace(0, 6*pi, 200)';
    r = linspace(5, max_radius, 200)';

    x = center_x + r .* cos(theta);
    y = center_y + r .* sin(theta);
end

end
