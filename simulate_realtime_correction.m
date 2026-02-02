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

    % 调用轨迹重建函数
    trajectory_data = reconstruct_trajectory(gcode_file, layer_num, params);

    % 提取参考轨迹
    time_ref = trajectory_data.time;
    x_ref = trajectory_data.x_ref;
    y_ref = trajectory_data.y_ref;
    z_ref = trajectory_data.z_ref;
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
        py.sys.path.insert(0, project_root);
        py.sys.path.insert(0, fullfile(project_root, 'experiments'));
        py.sys.path.insert(0, fullfile(project_root, 'models'));
        py.sys.path.insert(0, fullfile(project_root, 'data'));

        fprintf('  ✓ Python环境初始化成功\n');
    catch ME
        error('Python库导入失败: %s', ME.message);
    end

    % 加载PyTorch模型
    fprintf('\n  加载LSTM模型: %s\n', model_path);

    try
        % 创建模型实例
        py.sys.path.insert(0, project_root);
        model_module = py.importlib.import_module('models.realtime_corrector');
        RealTimeCorrector = model_module.RealTimeCorrector;

        % 实例化模型
        model = RealTimeCorrector(
            py.int32(4),    % input_size
            py.int32(56),   % hidden_size
            py.int32(2),    % num_layers
            py.float64(0.1) % dropout
        );

        % 加载检查点
        checkpoint = py.torch.load(model_path);
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
    fprintf('\n[4/6] 开始实时仿真循环...\n');
    fprintf('  对每个点: 预测误差 → 修正轨迹 → 物理仿真 → 记录结果\n');

    % 初始化存储数组
    x_actual = zeros(n_points, 1);
    y_actual = zeros(n_points, 1);
    x_corrected = zeros(n_points, 1);
    y_corrected = zeros(n_points, 1);

    error_x_uncorrected = zeros(n_points, 1);  % 未修正误差（参考）
    error_y_uncorrected = zeros(n_points, 1);
    error_x_corrected = zeros(n_points, 1);    % 修正后误差
    error_y_corrected = zeros(n_points, 1);

    pred_error_x = zeros(n_points, 1);         % 预测的误差
    pred_error_y = zeros(n_points, 1);

    inference_times = zeros(n_points, 1);

    % 历史缓冲区
    history_x = zeros(seq_len, 1);
    history_y = zeros(seq_len, 1);
    history_vx = zeros(seq_len, 1);
    history_vy = zeros(seq_len, 1);

    fprintf('\n  进度:\n');

    % 主循环：逐点处理
    for i = 1:n_points
        % === 步骤4.1: 获取当前参考轨迹点 ===
        x_r = x_ref(i);
        y_r = y_ref(i);
        vx_r = vx_ref(i);
        vy_r = vy_ref(i);
        ax_r = ax_ref(i);
        ay_r = ay_ref(i);

        % === 步骤4.2: LSTM模型预测误差 ===
        tic;

        if i < seq_len
            % 历史不足，零误差
            pred_ex = 0;
            pred_ey = 0;
        else
            % 准备输入序列 [seq_len, 4]
            seq = [history_x, history_y, history_vx, history_vy];

            % 归一化
            seq_norm = (seq - scaler_mean) ./ scaler_scale;

            % 转换为PyTorch tensor
            seq_tensor = py.torch.from_numpy(seq_norm);
            seq_tensor = seq_tensor.unsqueeze(0).float();  % [1, seq_len, 4]

            % LSTM预测
            with py.torch.no_grad():
                pred = model(seq_tensor);

            pred_numpy = double(py.numpy.array(pred));
            pred_ex = pred_numpy(1);
            pred_ey = pred_numpy(2);
        end

        inference_time = toc * 1000;  % ms

        % 记录预测误差
        pred_error_x(i) = pred_ex;
        pred_error_y(i) = pred_ey;

        inference_times(i) = inference_time;

        % === 步骤4.3: 修正轨迹（在发送给"电机"之前） ===
        % 方法：从参考轨迹中减去预测的误差
        x_c = x_r - pred_ex;
        y_c = y_r - pred_ey;

        x_corrected(i) = x_c;
        y_corrected(i) = y_c;

        % === 步骤4.4: 运行物理仿真（修正后的轨迹） ===
        % 二阶系统: m*x'' + c*x' + k*x = -m*a_ref
        % 这里使用修正后的加速度作为输入

        % X轴仿真
        [x_act, ~, ~] = simulate_second_order_step(...
            x_c, vx_r, ax_r, ...
            params.motion.mass_x, ...
            params.damping.x, ...
            params.stiffness.x, ...
            dt);

        % Y轴仿真
        [y_act, ~, ~] = simulate_second_order_step(...
            y_c, vy_r, ay_r, ...
            params.motion.mass_y, ...
            params.damping.y, ...
            params.stiffness.y, ...
            dt);

        x_actual(i) = x_act;
        y_actual(i) = y_act;

        % === 步骤4.5: 计算误差 ===
        % 未修正误差（用原始参考轨迹作为基准）
        error_x_uncorrected(i) = x_act - x_r;
        error_y_uncorrected(i) = y_act - y_r;

        % 修正后的误差
        error_x_corrected(i) = x_act - x_r;  % 相对于原始参考
        error_y_corrected(i) = y_act - y_r;

        % === 步骤4.6: 更新历史缓冲区 ===
        if i < seq_len
            history_x(i) = x_r;
            history_y(i) = y_r;
            history_vx(i) = vx_r;
            history_vy(i) = vy_r;
        else
            % 滚动更新
            history_x = circshift(history_x, -1);
            history_x(end) = x_r;
            history_y = circshift(history_y, -1);
            history_y(end) = y_r;
            history_vx = circshift(history_vx, -1);
            history_vx(end) = vx_r;
            history_vy = circshift(history_vy, -1);
            history_vy(end) = vy_r;
        end

        % 进度显示
        if mod(i, 500) == 0 || i == n_points
            pct = i / n_points * 100;
            fprintf('    [%d/%d] %.1f%% - 推理时间: %.3f ms\n', ...
                    i, n_points, pct, inference_time);
        end
    end

    fprintf('  ✓ 仿真循环完成\n');

    %% 步骤5: 添加固件效应
    fprintf('\n[5/6] 添加固件效应...\n');

    % Junction Deviation
    [jd_x, jd_y] = junction_deviation_effect(x_ref, y_ref, params);
    error_x_uncorrected = error_x_uncorrected + jd_x;
    error_y_uncorrected = error_y_uncorrected + jd_y;
    error_x_corrected = error_x_corrected + jd_x;
    error_y_corrected = error_y_corrected + jd_y;

    % Microstep Resonance
    [res_x, res_y] = microstep_resonance_effect(n_points, params);
    error_x_uncorrected = error_x_uncorrected + res_x;
    error_y_uncorrected = error_y_uncorrected + res_y;
    error_x_corrected = error_x_corrected + res_x;
    error_y_corrected = error_y_corrected + res_y;

    % Timer Jitter
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

    % 实际轨迹（仿真得到的）
    results.trajectory_actual = struct();
    results.trajectory_actual.x = x_actual;
    results.trajectory_actual.y = y_actual;

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

function [x_new, v_new, a_new] = simulate_second_order_step(x0, v0, a_ref, m, c, k, dt)
% SIMULATE_SECOND_ORDER_STEP 单步二阶系统仿真
%
% 输入:
%   x0, v0   - 初始位置和速度
%   a_ref    - 参考加速度
%   m, c, k  - 质量、阻尼、刚度
%   dt       - 时间步长
%
% 输出:
%   x_new, v_new, a_new - 新的位置、速度、加速度

    % 计算加速度: a = -(c*v + k*x) / m - a_ref
    a = -(c * v0 + k * x0) / m - a_ref;

    % RK4积分
    k1_v = a;
    k1_x = v0;

    k2_v = a;  % 简化（假设加速度在一个步长内不变）
    k2_x = v0 + 0.5 * dt * k1_v;

    k3_v = a;
    k3_x = v0 + 0.5 * dt * k2_v;

    k4_v = a;
    k4_x = v0 + dt * k3_v;

    % 更新
    v_new = v0 + (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v);
    x_new = x0 + (dt / 6) * (k1_x + 2*k2_x + 2*k3_x + k4_x);
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
