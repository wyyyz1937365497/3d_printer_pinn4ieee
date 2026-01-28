function params = physics_parameters()
% PHYSICS_PARAMETERS - Ender-3 V2 FDM 3D打印机仿真物理参数
%
% 该文件包含以下方面准确仿真所需的所有物理参数：
% 1. 由于惯性和皮带弹性引起的轨迹误差（二阶系统）
% 2. 热场和层间粘合（移动热源模型）
%
% 来源：
% - Ender-3 V2技术规格
% - PLA材料属性（热学和力学）
% - GT2皮带特性
% - NEMA 17 42-34步进电机规格
% - 文献中的FDM传热系数
%
% 输出：包含所有参数的params结构

%% ========================================================================
% 1. 打印机机械参数 (Ender-3 V2)
% ===========================================================================

% --- 移动质量 ---
% 打印期间移动的有效质量
params.mass.extruder_assy = 0.350;      % kg - 挤出机组件（热端+风道+支架）
params.mass.x_carriage = 0.120;          % kg - X滑车
params.mass.y_belt = 0.015;             % kg - 移动皮带质量（估计）
params.mass.total_x = 0.485;            % kg - X方向总移动质量
params.mass.total_y = 0.650;            % kg - Y方向总移动质量（包括X）

% --- GT2皮带规格 ---
% GT2同步带特性（影响弹性）
params.belt.pitch = 2.0;                % mm - 齿距
params.belt.width = 6.0;                % mm - 皮带宽度
params.belt.length_x = 0.420;           % m - X轴皮带长度
params.belt.length_y = 0.520;           % m - Y轴皮带长度
params.belt.stiffness = 150000;         % N/m - 有效皮带刚度（实验得出）
params.belt.damping = 25.0;             % N·s/m - 皮带阻尼系数

% 皮带刚度计算参考：
% 对于橡胶同步带：k ≈ EA/L 其中 E≈2GPa, A≈width*pitch
% k ≈ (2e9 × 6e-3 × 2e-3) / 0.45 ≈ 53,000 N/m (单股)
% 预紧和张紧后，有效刚度更高
% 来源：GT2皮带规格，机械工程手册

% --- 步进电机规格 (NEMA 17 42-34) ---
% 用于X轴和Y轴
params.motor.step_angle = 1.8;          % 度 - 步距角
params.motor.steps_per_rev = 200;       % 步/转
params.motor.holding_torque = 0.40;     % N·m - 保持扭矩（额定）
params.motor.current = 1.5;             % A - 额定电流
params.motor.voltage = 12.0;            % V - 供电电压
params.motor.inductance = 3.0;          % mH - 相电感
params.motor.resistance = 1.5;          % 欧姆 - 相电阻

% 电机转子惯量
params.motor.rotor_inertia = 54e-6;     % kg·m² - 转子惯量（NEMA 17典型值）
params.motor.detent_torque = 0.02;      % N·m - 失步转矩

% --- 传动系统 ---
% 减速齿轮和皮带轮系统
params.pulley.teeth = 20;               % 齿 - GT2皮带轮
params.pulley.diameter = 12.13;         % mm - 有效直径（20齿 × 2mm / π）
params.pulley.radius = 6.065e-3;        % m - 有效半径
params.microstepping = 16;              % - - 细分设置（1/16）

% --- 急停和加速度限制 ---
% 来自Marlin固件配置（M205, M201）
params.motion.max_accel = 500;          % mm/s² - 最大加速度
params.motion.max_accel_x = 500;        % mm/s² - X轴最大加速度
params.motion.max_accel_y = 500;        % mm/s² - Y轴最大加速度
params.motion.max_jerk = 10.0;          % mm/s - 瞬时速度变化（急停）
params.motion.max_jerk_x = 8.0;         % mm/s - X轴急停限制
params.motion.max_jerk_y = 8.0;         % mm/s - Y轴急停限制
params.motion.max_velocity = 500;       % mm/s - 最大速度（来自M203）

% --- 系统动力学参数 ---
% 用于轨迹误差建模的二阶系统参数
% m·x'' + c·x' + k·x = F(t)

% X轴动力学
params.dynamics.x.mass = 0.485;         % kg - 有效质量
params.dynamics.x.stiffness = 150000;   % N/m - 皮带刚度
params.dynamics.x.damping = 25.0;       % N·s/m - 阻尼系数
params.dynamics.x.natural_freq = sqrt(params.dynamics.x.stiffness / params.dynamics.x.mass);  % rad/s
params.dynamics.x.damping_ratio = params.dynamics.x.damping / (2 * sqrt(params.dynamics.x.mass * params.dynamics.x.stiffness));
params.dynamics.x.settling_time = 4 / (params.dynamics.x.damping_ratio * params.dynamics.x.natural_freq);  % s

% Y轴动力学
params.dynamics.y.mass = 0.650;         % kg - 有效质量
params.dynamics.y.stiffness = 150000;   % N/m - 皮带刚度
params.dynamics.y.damping = 25.0;       % N·s/m - 阻尼系数
params.dynamics.y.natural_freq = sqrt(params.dynamics.y.stiffness / params.dynamics.y.mass);  % rad/s
params.dynamics.y.damping_ratio = params.dynamics.y.damping / (2 * sqrt(params.dynamics.y.mass * params.dynamics.y.stiffness));
params.dynamics.y.settling_time = 4 / (params.dynamics.y.damping_ratio * params.dynamics.y.natural_freq);  % s

%% ========================================================================
% 2. PLA材料属性
% ===========================================================================

% --- 机械属性 ---
params.material.name = 'PLA';
params.material.density = 1240;          % kg/m³ - 密度（来自G代码）
params.material.elastic_modulus = 3.5e9; % Pa - 弹性模量（3.5 GPa）
params.material.poisson_ratio = 0.36;   % - - 泊松比
params.material.yield_strength = 60e6;   % Pa - 屈服强度（60 MPa）
params.material.tensile_strength = 70e6; % Pa - 抗拉强度（70 MPa）

% --- 热学属性 ---
% 来源：PLA技术数据表和研究论文
params.material.thermal_conductivity = 0.13;  % W/(m·K) - 热导率
params.material.specific_heat = 1200;         % J/(kg·K) - 比热容
params.material.thermal_diffusivity = params.material.thermal_conductivity / ...
                                      (params.material.density * params.material.specific_heat);  % m²/s

% PLA热扩散率 ≈ 0.13 / (1240 × 1200) ≈ 8.7e-8 m²/s

% --- 相变温度 ---
params.material.glass_transition = 60;   % °C - 玻璃化转变温度（Tg）
params.material.melting_point = 171;     % °C - 熔点（Tm）
params.material.cold_crystallization = 107;  % °C - 冷结晶

% --- 打印温度 ---
params.printing.nozzle_temp = 220;       % °C - 喷嘴温度（来自G代码）
params.printing.bed_temp = 60;           % °C - 热床温度（来自G代码）
params.printing.min_fan_temp = 220;      % °C - 风扇启动温度
params.printing.chamber_temp = 25;       % °C - 环境腔室温度（典型值）

% --- 耗材规格 ---
params.filament.diameter = 1.75;         % mm - 标称耗材直径
params.filament.density = 1.24;          % g/cm³ - 耗材密度（来自G代码）

%% ========================================================================
% 3. 挤出和流动参数
% ===========================================================================

% --- 喷嘴规格 ---
params.nozzle.diameter = 0.4;           % mm - 喷嘴直径
params.nozzle.material = 'Brass';       % - - 喷嘴材料

% --- 挤出参数 ---
params.extrusion.width = 0.45;          % mm - 挤出宽度（来自G代码）
params.extrusion.height = 0.2;          % mm - 层高（来自G代码）
params.extrusion.length_ratio = 1.0;    % - - 挤出倍数

% --- 挤出几何 ---
params.extrusion.cross_section_area = params.extrusion.width * params.extrusion.height;  % mm²
params.extrusion.volume_flow_max = 15;  % mm³/s - 最大体积流量

% --- 热输入模型 ---
% 挤出热输入：Q = ṁ × c × ΔT
% 其中 ṁ = ρ × A × v（质量流率）
params.extrusion.heat_capacity_flow = params.material.density * ...
                                      params.extrusion.cross_section_area * 1e-9 * ...
                                      params.material.specific_heat;  % J/(m·K)

%% ========================================================================
% 4. 热模型参数（移动热源）
% ===========================================================================

% --- 传热系数 ---
% 来源：FDM传热文献，实验测量
params.heat_transfer.h_convection_no_fan = 10;    % W/(m²·K) - 自然对流（无风扇）
params.heat_transfer.h_convection_with_fan = 44;  % W/(m²·K) - 强制对流（风扇开启）
params.heat_transfer.h_conduction_bed = 150;      % W/(m²·K) - 与热床接触
params.heat_transfer.h_radiation = 10;            % W/(m²·K) - 有效辐射（线性化）

% 组合传热系数（典型打印条件）
params.heat_transfer.h_combined = params.heat_transfer.h_convection_with_fan + ...
                                   params.heat_transfer.h_radiation;  % W/(m²·K)

% 斯特藩-玻尔兹曼常数用于辐射
params.heat_transfer.sigma = 5.67e-8;     % W/(m²·K⁴) - 斯特藩-玻尔兹曼常数

% --- 发射率 ---
params.heat_transfer.emissivity_pla = 0.92;  % - - PLA发射率
params.heat_transfer.emissivity_bed = 0.95;  % - - 打印表面发射率

% --- 冷却风扇规格 ---
params.fan.max_speed = 255;               % - - PWM最大值（0-255）
params.fan.typical_speed = 255;           % - - 典型工作速度
params.fan.diameter = 40;                 % mm - 风扇直径
params.fan.flow_rate = 5.5;               % CFM - 气流（典型40mm风扇）

% --- 环境条件 ---
params.environment.ambient_temp = 25;     % °C - 环境温度
params.environment.humidity = 50;         % % - 相对湿度（影响冷却）
params.environment.chamber_temp = 25;     % °C - 腔室温度（Ender-3 V2是开放式结构）

%% ========================================================================
% 5. 数值仿真参数
% ===========================================================================

% --- 时间步进 ---
params.simulation.time_step = 0.001;     % s - 仿真时间步长（1 ms）
params.simulation.max_time = 1000;       % s - 最大仿真时间

% --- 空间离散化（用于热模型） ---
params.simulation.dx = 1.0;              % mm - X方向空间分辨率
params.simulation.dy = 1.0;              % mm - Y方向空间分辨率
params.simulation.dz = 0.1;              % mm - Z方向空间分辨率（层分辨率）

% --- 稳定性准则 ---
% 显式有限差分：Δt ≤ Δx² / (4α)
params.simulation.dt_stability_limit = (params.simulation.dx * 1e-3)^2 / ...
                                      (4 * params.material.thermal_diffusivity);  % s

% 用于热仿真的自适应时间步长
params.simulation.dt_thermal = min(params.simulation.time_step, ...
                                   params.simulation.dt_stability_limit * 0.9);  % s

% --- 输出配置 ---
params.output.save_interval = 100;       % - - 每N步保存一次
params.output.interpolate = true;        % - - 插值到均匀时间网格

%% ========================================================================
% 6. 层间粘合模型参数
% ===========================================================================

% Wool-O'Connor聚合物愈合模型
% 来源：关于FDM层间结合的研究
%
% 愈合模型：H = H∞ × exp(-Ea/RT) × t^n
% 其中：
%   H - 愈合比（结合强度发展）
%   H∞ - 最大愈合
%   Ea - 活化能
%   R - 气体常数
%   T - 温度（K）
%   t - 时间
%   n - 时间指数

params.adhesion.activation_energy = 50e3;    % J/mol - PLA扩散活化能
params.adhesion.gas_constant = 8.314;        % J/(mol·K) - 通用气体常数
params.adhesion.time_exponent = 0.5;         % - - 菲克扩散通常为0.5
params.adhesion.max_healing = 1.0;           % - - 最大愈合比
params.adhesion.reference_temp = 220;        % °C - 参考温度

% 简化的粘合强度模型（温度相关）
% σ_adhesion = σ_bulk × [1 - exp(-t/τ(T))]
% 其中 τ(T) = τ₀ × exp(Ea/RT)

params.adhesion.bulk_strength = 70e6;        % Pa - 块状材料强度
params.adhesion.pre_exponential = 1e-3;      % s - 指前因子

% 分子扩散的临界温度
params.adhesion.min_diffusion_temp = params.material.glass_transition + 10;  % °C
params.adhesion.optimal_temp = params.material.melting_point - 20;  % °C

% --- 愈合时间阈值 ---
params.adhesion.min_healing_time = 0.5;      % s - 任何结合的最小时间
params.adhesion.optimal_healing_time = 2.0;  % s - 最大强度的最佳时间

%% ========================================================================
% 7. G代码处理参数
% ===========================================================================

% G代码解析配置
params.gcode.coordinate_system = 'absolute';  % - - G90（绝对）或G91（相对）
params.gcode.extrusion_mode = 'relative';     % - - E值（通常为相对）

% 角点检测参数
params.gcode.corner_angle_threshold = 15;     % 度 - 检测角点的最小角度
params.gcode.min_segment_length = 0.1;        % mm - 要处理的最小段长

% 行走vs挤出分类
params.gcode.extrusion_threshold = 0.01;     % mm - 成为挤出的最小E变化

%% ========================================================================
% 8. 诊断和调试参数
% ===========================================================================

params.debug.plot_trajectory = false;        % - - 绘制参考vs实际轨迹
params.debug.plot_temperature = false;       % - - 绘制温度场演变
params.debug.plot_forces = false;            % - - 绘制惯性力和弹性力
params.debug.verbose = false;                % - - 打印进度消息（也控制绘图生成）

params.debug.save_intermediate = false;      % - - 保存中间结果
params.debug.check_stability = true;         % - - 检查数值稳定性

%% ========================================================================
% 9. 验证和参考数据
% ===========================================================================

% 实验验证数据（来自文献）
% 来源：关于Ender-3 V2性能的研究论文

params.validation.typical_corner_error = 0.3;  % mm - 典型角点圆化误差
params.validation.resonance_freq_x = 45;       % Hz - X轴共振频率
params.validation.resonance_freq_y = 35;       % Hz - Y轴共振频率

% 打印质量指标
params.quality.max_allowable_error = 0.5;     % mm - 最大可接受尺寸误差
params.quality.min_adhesion_ratio = 0.7;       % - - 最小层间强度比

%% ========================================================================
% 参数定义结束
% ===========================================================================

% 如果详细模式显示摘要
if params.debug.verbose
    fprintf('========================================\n');
    fprintf('物理参数加载成功\n');
    fprintf('========================================\n');
    fprintf('打印机：Ender-3 V2\n');
    fprintf('材料：%s\n', params.material.name);
    fprintf('\n');
    fprintf('X轴动力学：\n');
    fprintf('  质量：%.3f kg\n', params.dynamics.x.mass);
    fprintf('  固有频率：%.2f rad/s (%.2f Hz)\n', ...
            params.dynamics.x.natural_freq, ...
            params.dynamics.x.natural_freq / (2*pi));
    fprintf('  阻尼比：%.4f\n', params.dynamics.x.damping_ratio);
    fprintf('\n');
    fprintf('Y轴动力学：\n');
    fprintf('  质量：%.3f kg\n', params.dynamics.y.mass);
    fprintf('  固有频率：%.2f rad/s (%.2f Hz)\n', ...
            params.dynamics.y.natural_freq, ...
            params.dynamics.y.natural_freq / (2*pi));
    fprintf('  阻尼比：%.4f\n', params.dynamics.y.damping_ratio);
    fprintf('\n');
    fprintf('热学属性：\n');
    fprintf('  热导率：%.2f W/(m·K)\n', params.material.thermal_conductivity);
    fprintf('  比热容：%.0f J/(kg·K)\n', params.material.specific_heat);
    fprintf('  扩散率：%.2e m²/s\n', params.material.thermal_diffusivity);
    fprintf('\n');
    fprintf('传热：\n');
    fprintf('  对流（风扇）：%d W/(m²·K)\n', params.heat_transfer.h_convection_with_fan);
    fprintf('  对流（无风扇）：%d W/(m²·K)\n', params.heat_transfer.h_convection_no_fan);
    fprintf('========================================\n');
end

end