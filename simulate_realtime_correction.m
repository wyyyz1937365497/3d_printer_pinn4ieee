function results = simulate_realtime_correction(gcode_file, layer_num, model_path, output_dir)
% SIMULATE_REALTIME_CORRECTION 仿真实时轨迹修正过程
%
% 功能：
%   模拟真实3D打印过程中的实时轨迹修正：
%   1. 对于每个轨迹点，使用LSTM模型预测误差
%   2. 在发送给"电机"之前修正轨迹（补偿预测的误差）
%   3. 运行物理仿真得到实际位置
%   4. 记录修正前后的误差对比
%
% 输入：
%   gcode_file  - G-code文件路径
%   layer_num   - 要仿真的层编号
%   model_path  - PyTorch模型检查点路径
%   output_dir  - 输出目录
%
% 输出：
%   results     - 包含所有仿真结果的结构体
%
% 工作流程：
%   对于每个时间步 t:
%     1. 获取参考轨迹 r_ref(t)
%     2. 调用Python模型预测误差: e_pred = model(history)
%     3. 修正轨迹: r_corrected = r_ref - e_pred
%     4. 运行物理仿真得到实际位置: r_actual
%     5. 计算实际误差: e_actual = r_actual - r_ref
%
% 作者: 3D Printer PINN Project
% 日期: 2026-02-02

    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('实时轨迹修正仿真（模拟真实3D打印过程）\n');
    fprintf('%s\n\n', repmat('=', 1, 70));

    % 参数配置
    seq_len = 20;  % LSTM序列长度
    dt = 0.01;     % 时间步长 (100Hz)

    fprintf('配置:\n');
    fprintf('  G-code: %s\n', gcode_file);
    fprintf('  层编号: %d\n', layer_num);
    fprintf('  模型: %s\n', model_path);
    fprintf('  序列长度: %d\n', seq_len);
    fprintf('  采样频率: %.0f Hz\n', 1/dt);

    % 确保输出目录存在
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    %% 步骤1: 加载物理参数
    fprintf('\n[1/6] 加载物理参数...\n');
    params = physics_parameters();
    fprintf('  ✓ 参数加载完成\n');

    %% 步骤2: 解析G-code获取参考轨迹
    fprintf('\n[2/6] 解析G-code获取参考轨迹...\n');
    fprintf('  文件: %s\n', gcode_file);

    % 调用轨迹重建函数（参数顺序：gcode_file, params, options）
    options = struct();
    options.layers = layer_num;
    trajectory_data = reconstruct_trajectory(gcode_file, params, options);

    % 提取参考轨迹
    time_ref = trajectory_data.time;
    x_ref = trajectory_data.x;
    y_ref = trajectory_data.y;
    z_ref = trajectory_data.z;
    vx_ref = trajectory_data.vx;
    vy_ref = trajectory_data.vy;
    vz_ref = trajectory_data.vz;
    ax_ref = trajectory_data.ax;
    ay_ref = trajectory_data.ay;
    az_ref = trajectory_data.az;

    n_points = length(time_ref);
    fprintf('  轨迹点数: %d\n', n_points);
    fprintf('  时间跨度: %.2f 秒\n', time_ref(end));
    fprintf('  ✓ 轨迹重建完成\n');

    %% 步骤3: 初始化Python环境
    fprintf('\n[3/6] 初始化Python环境...\n');

    % 设置Python环境为3dprint conda环境
    try
        % 检测conda环境路径（Windows）
        conda_env_path = fullfile('G:', 'Miniconda3', 'envs', '3dprint', 'python.exe');
        if exist(conda_env_path, 'file')
            pyenv('Version', conda_env_path);
            fprintf('  设置Python环境: %s\n', conda_env_path);
        end
    catch ME
        fprintf('  警告: 无法设置conda环境，使用默认Python\n');
    end

    % 添加Python路径
    pe = pyenv;
    % 检查Python环境
    try
        py_version = pe.Version;
        fprintf('  Python版本: %s\n', char(py_version));
    catch
        error('无法找到Python环境。请确保已安装并配置Python。');
    end

    % 导入必要的Python库
    try
        py.importlib.import_module('torch');
        py.importlib.import_module('numpy');
        py.importlib.import_module('sys');

        % 添加项目路径到Python
        project_root = fileparts(mfilename('fullpath'));
        py.sys.path().insert(0, project_root);
        py.sys.path().insert(0, fullfile(project_root, 'experiments'));
        py.sys.path().insert(0, fullfile(project_root, 'models'));
        py.sys.path().insert(0, fullfile(project_root, 'data'));

        fprintf('  ✓ Python环境初始化成功\n');
    catch ME
        error('Python库导入失败: %s', ME.message);
    end

    % 加载PyTorch模型
    fprintf('\n  加载LSTM模型: %s\n', model_path);

    try
        % 创建模型实例
        py.sys.path().insert(0, project_root);
        model_module = py.importlib.import_module('models.realtime_corrector');

        % 实例化模型（使用feval调用Python类构造函数）
        model = feval(model_module.RealTimeCorrector, ...
            int32(4), ...       % input_size
            int32(56), ...      % hidden_size
            int32(2), ...       % num_layers
            0.1);               % dropout

        % 加载检查点（使用CPU加载，避免CUDA错误）
        map_location = py.torch.device('cpu');
        checkpoint = py.torch.load(model_path, map_location);
        model.load_state_dict(checkpoint{'model_state_dict'});
        model.eval();

        fprintf('  ✓ 模型加载成功\n');

        % 获取归一化参数
        if isfield(checkpoint, 'scaler_mean') && isfield(checkpoint, 'scaler_scale')
            scaler_mean = double(checkpoint.scaler_mean);
            scaler_scale = double(checkpoint.scaler_scale);
        else
            % 使用默认参数
            scaler_mean = [110.0, 110.0, 85.3, 85.3];
            scaler_scale = [30.5, 30.5, 45.2, 45.2];
        end

        fprintf('  ✓ 归一化参数加载成功\n');

    catch ME
        error('模型加载失败: %s', ME.message);
    end

    %% 步骤4: 实时仿真循环
    fprintf('\n[4/6] 实时修正仿真...\n');

    %% === 第一次仿真：完全未修正（直接用G-code） ===
    fprintf('  步骤4a: 仿真未修正轨迹（完整独立运行）...\n');

    % 初始化未修正系统的状态
    x_state_uncorrected = [0; 0];  % [位置误差; 速度误差]
    y_state_uncorrected = [0; 0];

    % 系统矩阵（使用与数据收集时相同的动力学参数）
    Ax = [0, 1; -params.dynamics.x.stiffness/params.dynamics.x.mass, -params.dynamics.x.damping/params.dynamics.x.mass];
    Ay = [0, 1; -params.dynamics.y.stiffness/params.dynamics.y.mass, -params.dynamics.y.damping/params.dynamics.y.mass];
    Bx = [0; -1];
    By = [0; -1];

    % 存储未修正结果
    x_actual_uncorrected = zeros(n_points, 1);
    y_actual_uncorrected = zeros(n_points, 1);
    error_x_uncorrected = zeros(n_points, 1);
    error_y_uncorrected = zeros(n_points, 1);

    % 第一次完整仿真循环
    for i = 1:n_points
        x_r = x_ref(i);
        y_r = y_ref(i);
        ax_r = ax_ref(i);
        ay_r = ay_ref(i);

        % RK4单步积分
        if i > 1
            k1_x = Ax * x_state_uncorrected + Bx * ax_r;
            k2_x = Ax * (x_state_uncorrected + 0.5*dt*k1_x) + Bx * ax_r;
            k3_x = Ax * (x_state_uncorrected + 0.5*dt*k2_x) + Bx * ax_r;
            k4_x = Ax * (x_state_uncorrected + dt*k3_x) + Bx * ax_r;
            x_state_uncorrected = x_state_uncorrected + (dt/6) * (k1_x + 2*k2_x + 2*k3_x + k4_x);

            k1_y = Ay * y_state_uncorrected + By * ay_r;
            k2_y = Ay * (y_state_uncorrected + 0.5*dt*k1_y) + By * ay_r;
            k3_y = Ay * (y_state_uncorrected + 0.5*dt*k2_y) + By * ay_r;
            k4_y = Ay * (y_state_uncorrected + dt*k3_y) + By * ay_r;
            y_state_uncorrected = y_state_uncorrected + (dt/6) * (k1_y + 2*k2_y + 2*k3_y + k4_y);
        end

        % 计算未修正的实际位置
        x_actual_uncorrected(i) = x_r + x_state_uncorrected(1);
        y_actual_uncorrected(i) = y_r + y_state_uncorrected(1);
        error_x_uncorrected(i) = x_state_uncorrected(1);
        error_y_uncorrected(i) = y_state_uncorrected(1);
    end

    fprintf('  ✓ 未修正仿真完成\n');
    fprintf('    平均误差: %.2f μm\n', mean(sqrt(error_x_uncorrected.^2 + error_y_uncorrected.^2)) * 1000);

    %% === 第二次仿真：使用LSTM模型进行实时修正 ===
    fprintf('\n  步骤4b: 实时修正仿真（基于未修正参考轨迹的历史）...\n');

    % 初始化修正后系统的状态
    x_state_corrected = [0; 0];
    y_state_corrected = [0; 0];

    % 存储修正后结果
    x_actual_corrected = zeros(n_points, 1);
    y_actual_corrected = zeros(n_points, 1);
    x_corrected = zeros(n_points, 1);
    y_corrected = zeros(n_points, 1);
    error_x_corrected = zeros(n_points, 1);
    error_y_corrected = zeros(n_points, 1);
    pred_error_x = zeros(n_points, 1);
    pred_error_y = zeros(n_points, 1);
    inference_times = zeros(n_points, 1);

    % 历史缓冲区（用参考轨迹的历史，因为模型是这样训练的）
    history_x = zeros(seq_len, 1);
    history_y = zeros(seq_len, 1);
    history_vx = zeros(seq_len, 1);
    history_vy = zeros(seq_len, 1);

    % 存储修正后的命令位置序列（用于计算加速度）
    x_command_buffer = zeros(n_points, 1);
    y_command_buffer = zeros(n_points, 1);

    fprintf('  进度:\n');

    % 第二次完整仿真循环（实时修正）
    for i = 1:n_points
        x_r = x_ref(i);
        y_r = y_ref(i);

        % === LSTM预测误差（基于参考轨迹的历史，与训练时一致） ===
        tic;

        if i < seq_len
            pred_ex = 0;
            pred_ey = 0;
        else
            % 使用参考轨迹的历史（与模型训练时一致）
            seq = [history_x, history_y, history_vx, history_vy];
            seq_norm = (seq - scaler_mean) ./ scaler_scale;
            seq_tensor = py.torch.from_numpy(seq_norm);
            seq_tensor = seq_tensor.unsqueeze(int32(0)).float();
            pred = model(seq_tensor);
            pred_numpy = double(py.numpy.array(pred.detach()));
            pred_ex = pred_numpy(1);
            pred_ey = pred_numpy(2);
        end

        inference_time = toc * 1000;
        inference_times(i) = inference_time;
        pred_error_x(i) = pred_ex;
        pred_error_y(i) = pred_ey;

        % === 修正轨迹命令（理想轨迹减去预测的误差） ===
        x_c = x_r - pred_ex;
        y_c = y_r - pred_ey;
        x_corrected(i) = x_c;
        y_corrected(i) = y_c;

        % 存储修正后的命令位置
        x_command_buffer(i) = x_c;
        y_command_buffer(i) = y_c;

        % === 计算修正后轨迹的加速度 ===
        if i >= 3
            % 使用中心差分计算加速度（更准确）
            % a[i] ≈ (x[i+1] - 2*x[i] + x[i-1]) / dt^2
            % 但由于是实时预测，我们只有当前和历史数据
            % 使用后向差分：
            ax_c = (x_c - 2*x_command_buffer(i-1) + x_command_buffer(i-2)) / (dt^2);
            ay_c = (y_c - 2*y_command_buffer(i-1) + y_command_buffer(i-2)) / (dt^2);
        elseif i == 2
            % 简单差分
            ax_c = (x_c - 2*x_command_buffer(i-1) + x_ref(i-2)) / (dt^2);
            ay_c = (y_c - 2*y_command_buffer(i-1) + y_ref(i-2)) / (dt^2);
        else
            % 第一个点，使用参考加速度
            ax_c = ax_ref(i);
            ay_c = ay_ref(i);
        end

        % === 物理仿真（使用修正后轨迹的加速度） ===
        if i > 1
            k1_xc = Ax * x_state_corrected + Bx * ax_c;
            k2_xc = Ax * (x_state_corrected + 0.5*dt*k1_xc) + Bx * ax_c;
            k3_xc = Ax * (x_state_corrected + 0.5*dt*k2_xc) + Bx * ax_c;
            k4_xc = Ax * (x_state_corrected + dt*k3_xc) + Bx * ax_c;
            x_state_corrected = x_state_corrected + (dt/6) * (k1_xc + 2*k2_xc + 2*k3_xc + k4_xc);

            k1_yc = Ay * y_state_corrected + By * ay_c;
            k2_yc = Ay * (y_state_corrected + 0.5*dt*k1_yc) + By * ay_c;
            k3_yc = Ay * (y_state_corrected + 0.5*dt*k2_yc) + By * ay_c;
            k4_yc = Ay * (y_state_corrected + dt*k3_yc) + By * ay_c;
            y_state_corrected = y_state_corrected + (dt/6) * (k1_yc + 2*k2_yc + 2*k3_yc + k4_yc);
        end

        % 计算修正后的实际位置（修正后的指令 + 误差状态）
        x_actual_corrected(i) = x_c + x_state_corrected(1);
        y_actual_corrected(i) = y_c + y_state_corrected(1);

        % 修正后的误差（相对于原始参考）
        error_x_corrected(i) = x_actual_corrected(i) - x_r;
        error_y_corrected(i) = y_actual_corrected(i) - y_r;

        % === 更新历史（用参考轨迹，因为模型是这样训练的） ===
        if i < seq_len
            history_x(i) = x_r;
            history_y(i) = y_r;
            history_vx(i) = vx_ref(i);
            history_vy(i) = vy_ref(i);
        else
            history_x = circshift(history_x, -1);
            history_x(end) = x_r;
            history_y = circshift(history_y, -1);
            history_y(end) = y_r;
            history_vx = circshift(history_vx, -1);
            history_vx(end) = vx_ref(i);
            history_vy = circshift(history_vy, -1);
            history_vy(end) = vy_ref(i);
        end

        % 进度显示
        if mod(i, 500) == 0 || i == n_points
            pct = i / n_points * 100;
            fprintf('    [%d/%d] %.1f%% - 推理时间: %.3f ms\n', ...
                    i, n_points, pct, inference_time);
        end
    end

    fprintf('  ✓ 修正仿真完成\n');
    fprintf('    修正后平均误差: %.2f μm\n', mean(sqrt(error_x_corrected.^2 + error_y_corrected.^2)) * 1000);

    fprintf('\n  ✓ 实时修正仿真完成\n');

    %% 步骤5: 添加固件效应
    fprintf('\n[5/6] 添加固件效应...\n');

    % Junction Deviation
    fprintf('  计算Junction Deviation效应...\n');
    [jd_x, jd_y] = junction_deviation_effect(x_ref, y_ref, params);
    error_x_uncorrected = error_x_uncorrected + jd_x;
    error_y_uncorrected = error_y_uncorrected + jd_y;
    error_x_corrected = error_x_corrected + jd_x;
    error_y_corrected = error_y_corrected + jd_y;

    % Microstep Resonance
    fprintf('  计算Microstep Resonance效应...\n');
    [res_x, res_y] = microstep_resonance_effect(n_points, params);
    error_x_uncorrected = error_x_uncorrected + res_x;
    error_y_uncorrected = error_y_uncorrected + res_y;
    error_x_corrected = error_x_corrected + res_x;
    error_y_corrected = error_y_corrected + res_y;

    % Timer Jitter
    fprintf('  计算Timer Jitter效应...\n');
    [jitter_x, jitter_y] = timer_jitter_effect(vx_ref, vy_ref, params);
    error_x_uncorrected = error_x_uncorrected + jitter_x;
    error_y_uncorrected = error_y_uncorrected + jitter_y;
    error_x_corrected = error_x_corrected + jitter_x;
    error_y_corrected = error_y_corrected + jitter_y;

    fprintf('  ✓ 固件效应已应用\n');

    % 计算误差幅度
    error_mag_uncorrected = sqrt(error_x_uncorrected.^2 + error_y_uncorrected.^2);
    error_mag_corrected = sqrt(error_x_corrected.^2 + error_y_corrected.^2);

    %% 步骤6: 保存结果
    fprintf('\n[6/6] 保存结果...\n');

    % 组装结果结构体
    results = struct();
    results.time = time_ref;

    % 参考轨迹
    results.trajectory = struct();
    results.trajectory.x_ref = x_ref;
    results.trajectory.y_ref = y_ref;
    results.trajectory.z_ref = z_ref;
    results.trajectory.vx = vx_ref;
    results.trajectory.vy = vy_ref;
    results.trajectory.vz = vz_ref;

    % 修正后的轨迹（发送给电机的指令）
    results.trajectory_corrected = struct();
    results.trajectory_corrected.x = x_corrected;
    results.trajectory_corrected.y = y_corrected;

    % 实际轨迹（仿真得到的）- 修正后的实际位置
    results.trajectory_actual = struct();
    results.trajectory_actual.x = x_actual_corrected;
    results.trajectory_actual.y = y_actual_corrected;

    % 未修正误差（参考）
    results.error_uncorrected = struct();
    results.error_uncorrected.x = error_x_uncorrected;
    results.error_uncorrected.y = error_y_uncorrected;
    results.error_uncorrected.mag = error_mag_uncorrected;

    % 修正后误差
    results.error_corrected = struct();
    results.error_corrected.x = error_x_corrected;
    results.error_corrected.y = error_y_corrected;
    results.error_corrected.mag = error_mag_corrected;

    % 预测误差
    results.predicted_error = struct();
    results.predicted_error.x = pred_error_x;
    results.predicted_error.y = pred_error_y;

    % 性能统计
    results.performance = struct();
    results.performance.mean_inference_time_ms = mean(inference_times);
    results.performance.max_inference_time_ms = max(inference_times);
    results.performance.throughput_pred_per_sec = 1000 / mean(inference_times);

    % 参数
    results.params = params;

    % 保存为.mat文件
    output_file = fullfile(output_dir, sprintf('realtime_correction_layer_%d.mat', layer_num));
    save(output_file, 'results', '-v7.3');
    fprintf('  ✓ 结果已保存: %s\n', output_file);

    % 打印统计
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('仿真完成 - 统计摘要\n');
    fprintf('%s\n\n', repmat('=', 1, 70));

    fprintf('未修正误差:\n');
    fprintf('  平均: %.3f μm  最大: %.3f μm\n', ...
            mean(error_mag_uncorrected)*1000, max(error_mag_uncorrected)*1000);

    fprintf('\n修正后误差:\n');
    fprintf('  平均: %.3f μm  最大: %.3f μm\n', ...
            mean(error_mag_corrected)*1000, max(error_mag_corrected)*1000);

    fprintf('\n改善效果:\n');
    improvement = (1 - mean(error_mag_corrected) / mean(error_mag_uncorrected)) * 100;
    fprintf('  平均误差降低: %.1f%%\n', improvement);

    fprintf('\n实时性能:\n');
    fprintf('  平均推理时间: %.3f ms\n', mean(inference_times));
    fprintf('  最大推理时间: %.3f ms\n', max(inference_times));
    fprintf('  吞吐量: %.0f predictions/s\n', 1000/mean(inference_times));
    fprintf('  实时性: %s\n', iif(mean(inference_times) < 1, '✓ 满足 (<1ms)', '✗ 不满足'));

    fprintf('\n%s\n', repmat('=', 1, 70));

end


%% 辅助函数

function [x_new, v_new, a_new] = simulate_second_order_step(x_prev, v_prev, x_cmd, v_cmd, a_cmd, m, c, k, dt)
% SIMULATE_SECOND_ORDER_STEP 单步二阶系统仿真（连续动态系统）
%
% 物理模型：
%   打印头是一个二阶质量-弹簧-阻尼系统
%   m*a + c*v + k*(x - x_cmd) = 0
%   其中 x_cmd 是指令位置，x 是实际位置
%
% 输入:
%   x_prev, v_prev - 上一时刻的实际位置和速度
%   x_cmd, v_cmd, a_cmd - 当前时刻的指令位置、速度、加速度
%   m, c, k  - 质量、阻尼、刚度
%   dt       - 时间步长
%
% 输出:
%   x_new, v_new, a_new - 当前时刻的实际位置、速度、加速度

    % 计算误差
    error_pos = x_prev - x_cmd;

    % 二阶系统动力学：m*a + c*v + k*(x - x_cmd) = 0
    % a = -(c*v + k*(x - x_cmd)) / m
    a = -(c * v_prev + k * error_pos) / m;

    % 欧拉积分（简单但有效）
    v_new = v_prev + a * dt;
    x_new = x_prev + v_new * dt;
    a_new = a;

end


function result = iif(condition, true_val, false_val)
% IIF 立即if函数
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
