function thermal_results = simulate_thermal_field(trajectory_data, params)
% SIMULATE_THERMAL_FIELD - 仿真FDM打印过程中的温度场
%
% 该函数使用以下方法建模打印件的热演化：
% 1. 移动热源模型（喷嘴）
% 2. 3D瞬态热传导方程
% 3. 对流和辐射冷却
% 4. 层间热传递
%
% 热方程：
%   ∂T/∂t = α·∇²T + Q_source - Q_cooling
%
% 其中：
%   α - 热扩散率
%   Q_source - 挤出的热输入
%   Q_cooling - 对流+辐射冷却
%
% 输入：
%   trajectory_data - 来自parse_gcode.m的结构
%   params          - 来自physics_parameters.m的物理参数
%
% 输出：
%   thermal_results - 包含温度场数据的结构
%
% 参考：增材制造传热文献

    fprintf('正在模拟温度场（移动热源模型）...\n');

    %% 提取轨迹数据
    t = trajectory_data.time;
    x_nozzle = trajectory_data.x;  % 使用重建的理想轨迹
    y_nozzle = trajectory_data.y;
    z_nozzle = trajectory_data.z;

    n_points = length(t);

    %% 仿真域
    fprintf('  设置仿真域...\n');

    % 从轨迹确定边界
    x_min = min(x_nozzle) - 10;  % 10mm边距
    x_max = max(x_nozzle) + 10;
    y_min = min(y_nozzle) - 10;
    y_max = max(y_nozzle) + 10;
    z_min = 0;
    z_max = max(z_nozzle) + 2;

    % 空间分辨率
    dx = params.simulation.dx;    % mm
    dy = params.simulation.dy;    % mm
    dz = params.simulation.dz;    % mm (层分辨率)

    % 创建网格
    x_grid = x_min:dx:x_max;
    y_grid = y_min:dy:y_max;
    z_grid = z_min:dz:z_max;

    nx = length(x_grid);
    ny = length(y_grid);
    nz = length(z_grid);

    fprintf('    网格大小：%d × %d × %d = %.1f 百万个点\n', ...
            nx, ny, nz, (nx*ny*nz)/1e6);
    fprintf('    X范围：%.1f - %.1f mm\n', x_min, x_max);
    fprintf('    Y范围：%.1f - %.1f mm\n', y_min, y_max);
    fprintf('    Z范围：%.1f - %.1f mm\n', z_min, z_max);

    %% 时间步进
    dt_thermal = params.simulation.dt_thermal;  % s
    fprintf('    时间步长：%.4f s\n', dt_thermal);

    %% 初始化温度场
    fprintf('  初始化温度场...\n');

    % 获取层信息
    if isfield(trajectory_data, 'layer_num')
        current_layer = trajectory_data.layer_num(1);
    else
        current_layer = 1;
    end

    % 计算热历史以获得初始温度
    print_times = [];  % 可以传入或估算
    layer_intervals = [];  % 可以传入或估算

    % 对于多层，总是尝试使用热累积模型
    if current_layer > 1
        try
            T_initial = calculate_thermal_history(current_layer, ...
                                                  print_times, ...
                                                  layer_intervals, ...
                                                  params);
            fprintf('    使用热累积模型：初始温度 = %.1f°C\n', T_initial);
        catch ME
            fprintf('    警告：热历史计算失败：%s\n', ME.message);
            fprintf('    回退到环境温度\n');
            T_initial = params.environment.ambient_temp;
        end
    else
        T_initial = params.environment.ambient_temp;
        fprintf('    第1层：使用环境温度：初始温度 = %.1f°C\n', T_initial);
    end

    T = ones(ny, nx, nz) * T_initial;  % °C - 使用计算出的初始温度

    % 在底部设置热床温度
    T(:,:,1) = params.printing.bed_temp;

    % 温度场存储（在选定点和时间）
    % 我们将在喷嘴位置和层界面跟踪温度

    T_interface = zeros(n_points, 1);  % 层界面温度
    T_surface = zeros(n_points, 1);    % 表面温度
    cooling_rate = zeros(n_points, 1); % dT/dt

    %% 材料属性
    alpha = params.material.thermal_diffusivity;      % m²/s
    k_thermal = params.material.thermal_conductivity; % W/(m·K)
    rho = params.material.density;                    % kg/m³
    cp = params.material.specific_heat;               % J/(kg·K)

    %% 传热系数
    h_conv = params.heat_transfer.h_convection_with_fan;  % W/(m²·K)
    h_rad = params.heat_transfer.h_radiation;            % W/(m²·K) (线性化)
    h_total = h_conv + h_rad;

    %% 移动热源参数
    T_nozzle = params.printing.nozzle_temp;  % °C
    nozzle_dia = params.nozzle.diameter;     % mm

    % 热源半径（高斯分布）
    r_source = nozzle_dia;  % mm

    %% 简化的热模型（点跟踪）
    % 全3D有限差分对于Python集成来说太慢
    % 相反，我们使用解析解在关键位置跟踪温度

    fprintf('  使用解析移动热源模型...\n');

    % 初始化层温度跟踪
    current_layer = 0;
    layer_deposition_time = [];
    layer_center_temp = [];

    T_nozzle_history = zeros(n_points, 1);  % 喷嘴位置的温度

    % 用于冷却速率计算的前一个温度
    T_prev = params.environment.ambient_temp;

    %% 主时间循环
    fprintf('  运行热仿真...\n');

    for i = 1:n_points
        % 当前喷嘴位置
        xi = x_nozzle(i);
        yi = y_nozzle(i);
        zi = z_nozzle(i);

        % 检查是否正在挤出
        is_extruding = trajectory_data.is_extruding(i);

        % 当前层
        layer_i = trajectory_data.layer_num(i);

        % 层变化检测
        if layer_i > current_layer
            current_layer = layer_i;
            layer_deposition_time(current_layer) = t(i);
            fprintf('    第%d层在t=%.2f s\n', current_layer, t(i));
        end

        %% 喷嘴位置的温度（简化）
        % 这是一个简化的模型，考虑了：
        % 1. 来自喷嘴的热输入
        % 2. 来自环境的冷却
        % 3. 材料的热惯性

        if is_extruding
            % 挤出：材料是热的
            T_local = T_nozzle;
        else
            % 行走：材料冷却
            % 使用牛顿冷却定律：dT/dt = -h*A/(m*cp) * (T - T_amb)
            time_since_extrusion = 0;
            for j = i-1:-1:1
                if trajectory_data.is_extruding(j)
                    time_since_extrusion = t(i) - t(j);
                    break;
                end
            end

            % 冷却模型
            cooling_constant = h_total / (rho * cp * (dz*1e-3));  % 1/s
            T_local = (T_nozzle - params.environment.ambient_temp) * ...
                      exp(-cooling_constant * time_since_extrusion) + ...
                      params.environment.ambient_temp * (1 - exp(-cooling_constant * time_since_extrusion));
        end

        T_nozzle_history(i) = T_local;

        %% 层界面温度
        % 当前层与前一层之间界面的温度
        if layer_i > 1
            % 从前一层在此位置沉积以来的时间
            % 查找喷嘴在前一层接近此(x,y)的位置
            prev_layer_mask = trajectory_data.layer_num == layer_i - 1;

            if sum(prev_layer_mask) > 0
                % 查找前一层最接近的点
                dist_prev = sqrt((trajectory_data.x(prev_layer_mask) - xi).^2 + ...
                                 (trajectory_data.y(prev_layer_mask) - yi).^2);

                [min_dist, idx_min] = min(dist_prev);

                if min_dist < 5  % 在5mm以内
                    prev_layer_indices = find(prev_layer_mask);
                    prev_idx = prev_layer_indices(idx_min);
                    time_diff = t(i) - t(prev_idx);

                    % 此位置前一层的温度
                    T_prev_layer = T_nozzle_history(prev_idx) * exp(-cooling_constant * time_diff) + ...
                                   params.environment.ambient_temp * (1 - exp(-cooling_constant * time_diff));

                    T_interface(i) = (T_local + T_prev_layer) / 2;  % 平均
                else
                    % 没有前一层轨迹数据，使用热累积模型的温度作为估算
                    % 前一层应该已经冷却了一些，但仍保持一定温度
                    T_prev_layer_est = T_initial * 0.7 + params.environment.ambient_temp * 0.3;
                    T_interface(i) = (T_local + T_prev_layer_est) / 2;
                end
            else
                % 没有前一层轨迹数据（例如只仿真单层），使用热累积模型的温度作为估算
                % 前一层应该已经冷却了一些，但仍保持一定温度
                T_prev_layer_est = T_initial * 0.7 + params.environment.ambient_temp * 0.3;
                T_interface(i) = (T_local + T_prev_layer_est) / 2;
            end
        else
            T_interface(i) = T_local;  % 第一层 - 下面没有界面
        end

        %% 表面温度
        T_surface(i) = T_local;

        %% 冷却速率
        if i > 1
            cooling_rate(i) = (T_local - T_prev) / (t(i) - t(i-1));
        end

        T_prev = T_local;
    end

    %% 计算热指标
    fprintf('  计算热指标...\n');

    % 熔点以上的温度（用于分子扩散）
    time_above_melting = sum(T_nozzle_history > params.material.melting_point) * mean(diff(t));

    % 玻璃化转变温度以上的温度
    time_above_tg = sum(T_nozzle_history > params.material.glass_transition) * mean(diff(t));

    % 最大冷却速率
    max_cooling_rate = max(cooling_rate(T_nozzle_history < params.printing.nozzle_temp - 50));

    % 打印期间的平均界面温度
    mean_interface_temp = mean(T_interface(trajectory_data.is_extruding));

    fprintf('    Tm以上时间：%.2f s\n', time_above_melting);
    fprintf('    Tg以上时间：%.2f s\n', time_above_tg);
    fprintf('    最大冷却速率：%.2f °C/s\n', abs(max_cooling_rate));
    fprintf('    平均界面温度：%.2f °C\n', mean_interface_temp);

    %% 计算粘合强度（简化模型）
    fprintf('  估算层间粘合强度...\n');

    % Wool-O'Connor愈合模型（简化）
    % 结合强度取决于：
    % 1. 界面温度
    % 2. 关键温度以上的时间
    % 3. 冷却速率

    adhesion_ratio = zeros(n_points, 1);

    for i = 1:n_points
        if trajectory_data.layer_num(i) > 1 && trajectory_data.is_extruding(i)
            T_int = T_interface(i);

            % 基于温度的愈合比
            if T_int < params.material.glass_transition
                healing_ratio = 0;
            elseif T_int < params.material.melting_point
                % 从Tg到Tm的线性斜坡
                healing_ratio = (T_int - params.material.glass_transition) / ...
                               (params.material.melting_point - params.material.glass_transition);
            else
                healing_ratio = 1.0;
            end

            % 根据冷却速率调整（快速冷却=愈合不良）
            cooling_factor = min(1.0, 10 / (abs(cooling_rate(i)) + 1));

            adhesion_ratio(i) = healing_ratio * cooling_factor;
        end
    end

    mean_adhesion = mean(adhesion_ratio(adhesion_ratio > 0));
    min_adhesion = min(adhesion_ratio(adhesion_ratio > 0));

    fprintf('    平均粘合比：%.2f\n', mean_adhesion);
    fprintf('    最小粘合比：%.2f\n', min_adhesion);

    %% 层间时间间隔
    fprintf('  计算层间时间间隔...\n');

    interlayer_time = zeros(n_points, 1);

    for i = 2:n_points
        if trajectory_data.layer_num(i) > trajectory_data.layer_num(i-1)
            % 这是新层的开始
            % 计算从此位置上次层以来的时间
            current_layer = trajectory_data.layer_num(i);

            % 查找我们在前一层相似(x,y)的位置
            prev_layer_mask = trajectory_data.layer_num == current_layer - 1;

            if sum(prev_layer_mask) > 0
                xi = x_nozzle(i);
                yi = y_nozzle(i);

                dist_prev = sqrt((trajectory_data.x(prev_layer_mask) - xi).^2 + ...
                                 (trajectory_data.y(prev_layer_mask) - yi).^2);

                [min_dist, idx_min] = min(dist_prev);

                if min_dist < 10  % 在10mm以内
                    prev_layer_indices = find(prev_layer_mask);
                    prev_idx = prev_layer_indices(idx_min);
                    interlayer_time(i) = t(i) - t(prev_idx);
                end
            end
        end
    end

    mean_interlayer_time = mean(interlayer_time(interlayer_time > 0));

    fprintf('    平均层间时间：%.2f s\n', mean_interlayer_time);

    %% 温度梯度（简化）
    fprintf('  估算温度梯度...\n');

    % 垂直梯度（层间）
    temp_gradient_z = zeros(n_points, 1);

    for i = 2:n_points
        if trajectory_data.is_extruding(i) && trajectory_data.layer_num(i) > 1
            % 估算与前一层的温度差
            T_current = T_nozzle_history(i);

            % 查找前一层的对应点
            prev_layer_mask = trajectory_data.layer_num == trajectory_data.layer_num(i) - 1;

            if sum(prev_layer_mask) > 0
                xi = x_nozzle(i);
                yi = y_nozzle(i);

                dist_prev = sqrt((trajectory_data.x(prev_layer_mask) - xi).^2 + ...
                                 (trajectory_data.y(prev_layer_mask) - yi).^2);

                [min_dist, idx_min] = min(dist_prev);

                if min_dist < 10
                    prev_layer_indices = find(prev_layer_mask);
                    prev_idx = prev_layer_indices(idx_min);
                    T_prev_layer_local = T_nozzle_history(prev_idx);

                    % 垂直梯度（°C/mm）
                    temp_gradient_z(i) = abs(T_current - T_prev_layer_local) / params.extrusion.height;
                end
            end
        end
    end

    %% 创建输出结构
    thermal_results.time = t;

    % 喷嘴位置（实际轨迹）
    thermal_results.x_nozzle = x_nozzle;
    thermal_results.y_nozzle = y_nozzle;
    thermal_results.z_nozzle = z_nozzle;

    % 温度场
    thermal_results.T_nozzle_history = T_nozzle_history;  % °C
    thermal_results.T_interface = T_interface;            % °C
    thermal_results.T_surface = T_surface;                % °C

    % 温度梯度
    thermal_results.temp_gradient_z = temp_gradient_z;    % °C/mm

    % 冷却
    thermal_results.cooling_rate = cooling_rate;          % °C/s
    thermal_results.time_above_melting = time_above_melting;  % s
    thermal_results.time_above_tg = time_above_tg;        % s

    % 层间
    thermal_results.interlayer_time = interlayer_time;    % s
    thermal_results.mean_interlayer_time = mean_interlayer_time;  % s

    % 粘合强度
    thermal_results.adhesion_ratio = adhesion_ratio;      % -
    thermal_results.mean_adhesion = mean_adhesion;        % -
    thermal_results.min_adhesion = min_adhesion;          % -

    % 环境
    thermal_results.T_ambient = params.environment.ambient_temp;
    thermal_results.T_bed = params.printing.bed_temp;
    thermal_results.T_nozzle_setpoint = params.printing.nozzle_temp;

    % 网格信息
    thermal_results.x_grid = x_grid;
    thermal_results.y_grid = y_grid;
    thermal_results.z_grid = z_grid;
    thermal_results.dx = dx;
    thermal_results.dy = dy;
    thermal_results.dz = dz;

    fprintf('  温度场仿真完成！\n\n');

    %% 可选：绘图
    if params.debug.plot_temperature
        figure('Name', '热场分析', 'Position', [100, 100, 1200, 800]);

        % 喷嘴位置的温度
        subplot(2, 3, 1);
        plot(t, T_nozzle_history, 'r-', 'LineWidth', 1.5);
        yline(params.material.melting_point, 'b--', '熔点', 'LineWidth', 1.5);
        yline(params.material.glass_transition, 'g--', 'T_g', 'LineWidth', 1.5);
        yline(params.environment.ambient_temp, 'k--', '环境', 'LineWidth', 1);
        grid on;
        xlabel('时间 (s)');
        ylabel('温度 (°C)');
        title('喷嘴位置的温度');
        ylim([params.environment.ambient_temp - 10, T_nozzle + 10]);

        % 界面温度
        subplot(2, 3, 2);
        plot(t, T_interface, 'b-', 'LineWidth', 1.5);
        yline(params.material.glass_transition, 'g--', 'T_g', 'LineWidth', 1.5);
        grid on;
        xlabel('时间 (s)');
        ylabel('界面温度 (°C)');
        title('层界面温度');

        % 冷却速率
        subplot(2, 3, 3);
        plot(t, cooling_rate, 'r-', 'LineWidth', 1);
        grid on;
        xlabel('时间 (s)');
        ylabel('冷却速率 (°C/s)');
        title('冷却速率');
        ylim([min(cooling_rate)*1.1, max(abs(cooling_rate))*1.1]);

        % 粘合比
        subplot(2, 3, 4);
        valid_adhesion = adhesion_ratio > 0;
        plot(t(valid_adhesion), adhesion_ratio(valid_adhesion), 'b.-', 'LineWidth', 1);
        grid on;
        xlabel('时间 (s)');
        ylabel('粘合比');
        title('层间粘合强度');
        ylim([0, 1.1]);

        % 层间时间
        subplot(2, 3, 5);
        valid_interlayer = interlayer_time > 0;
        plot(t(valid_interlayer), interlayer_time(valid_interlayer), 'g.-', 'LineWidth', 1);
        grid on;
        xlabel('时间 (s)');
        ylabel('时间 (s)');
        title('层间时间间隔');

        % 温度梯度
        subplot(2, 3, 6);
        valid_grad = temp_gradient_z > 0;
        plot(t(valid_grad), temp_gradient_z(valid_grad), 'm-', 'LineWidth', 1);
        grid on;
        xlabel('时间 (s)');
        ylabel('梯度 (°C/mm)');
        title('垂直温度梯度');

        drawnow;
    end

end