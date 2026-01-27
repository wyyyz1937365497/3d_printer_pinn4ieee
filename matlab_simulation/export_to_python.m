function export_to_python(mat_filename, params)
%% 将MATLAB数据转换为Python格式
% 输入：
%   mat_filename - .mat文件路径
%   params - 仿真参数
% 输出：
%   生成Python可读的数据文件

fprintf('  转换数据为Python格式...\n');

%% 1. 加载MATLAB数据
fprintf('    加载MATLAB数据...\n');
loaded_data = load(mat_filename);
all_data = loaded_data.all_data;

num_samples = length(all_data);

%% 2. 准备Python数据结构
% 使用Python兼容的格式（.mat v7.3或.h5）
% 这里我们使用.mat格式，但组织为Python友好的结构

fprintf('    准备数据结构...\n');

%% 3. 提取并组织数据
% 为每个样本提取关键特征

% 输入特征（X）
X_features = cell(num_samples, 1);

% 输出目标（y）
y_targets = cell(num_samples, 1);

% 元数据
metadata = struct();

fprintf('    提取特征...\n');

for i = 1:num_samples
    sample = all_data{i};

    %% 3.1 输入特征（50个核心状态量）
    features = zeros(1, 50);

    idx = 1;

    % 轨迹误差模块（20个）
    features(idx) = mean(sample.trajectory_error.epsilon_x); idx = idx + 1;      % 1. X位置误差均值
    features(idx) = mean(sample.trajectory_error.epsilon_y); idx = idx + 1;      % 2. Y位置误差均值
    features(idx) = max(sample.trajectory_error.epsilon_r); idx = idx + 1;       % 3. 最大位置误差
    features(idx) = sample.trajectory_error.rms_error; idx = idx + 1;            % 4. RMS位置误差
    features(idx) = mean(sample.trajectory_error.vx_act); idx = idx + 1;         % 5. X速度均值
    features(idx) = mean(sample.trajectory_error.vy_act); idx = idx + 1;         % 6. Y速度均值
    features(idx) = max(sample.trajectory_error.v_ref); idx = idx + 1;           % 7. 最大速度
    features(idx) = mean(abs(sample.trajectory_error.ax_ref)); idx = idx + 1;   % 8. 平均X加速度
    features(idx) = mean(abs(sample.trajectory_error.ay_ref)); idx = idx + 1;   % 9. 平均Y加速度
    features(idx) = max(abs(sample.trajectory_error.jx_ref)); idx = idx + 1;    % 10. 最大X加加速度
    features(idx) = max(abs(sample.trajectory_error.jy_ref)); idx = idx + 1;    % 11. 最大Y加加速度
    features(idx) = mean(abs(sample.trajectory_error.F_inertia_x)); idx = idx + 1; % 12. 平均X惯性力
    features(idx) = mean(abs(sample.trajectory_error.F_inertia_y)); idx = idx + 1; % 13. 平均Y惯性力
    features(idx) = max(abs(sample.trajectory_error.delta_L_x)); idx = idx + 1;  % 14. 最大X皮带伸长
    features(idx) = max(abs(sample.trajectory_error.delta_L_y)); idx = idx + 1;  % 15. 最大Y皮带伸长
    features(idx) = sample.trajectory_error.omega_n_x; idx = idx + 1;            % 16. X轴固有频率
    features(idx) = sample.trajectory_error.zeta_x; idx = idx + 1;               % 17. X轴阻尼比
    features(idx) = sample.params.print_speed; idx = idx + 1;                    % 18. 打印速度
    features(idx) = sample.params.acceleration; idx = idx + 1;                   % 19. 加速度设置
    features(idx) = sample.params.jerk; idx = idx + 1;                           % 20. 加加速度设置

    % 温度场模块（18个）
    features(idx) = mean(sample.thermal_field.T_nozzle_path); idx = idx + 1;     % 21. 平均喷嘴路径温度
    features(idx) = max(sample.thermal_field.T_nozzle_path); idx = idx + 1;      % 22. 最高温度
    features(idx) = min(sample.thermal_field.T_nozzle_path); idx = idx + 1;      % 23. 最低温度
    features(idx) = std(sample.thermal_field.T_nozzle_path); idx = idx + 1;      % 24. 温度标准差
    features(idx) = mean(sample.thermal_field.T_interface); idx = idx + 1;       % 25. 平均层间温度
    features(idx) = mean(sample.thermal_field.cooling_rate); idx = idx + 1;      % 26. 平均冷却速率
    features(idx) = max(sample.thermal_field.cooling_rate); idx = idx + 1;       % 27. 最大冷却速率
    features(idx) = mean(sample.thermal_field.time_above_melting); idx = idx + 1; % 28. 平均时间高于熔点
    features(idx) = mean(sample.thermal_field.gradient_z); idx = idx + 1;        % 29. 平均Z方向温度梯度
    features(idx) = mean(sample.thermal_field.gradient_xy); idx = idx + 1;       % 30. 平均XY平面温度梯度
    features(idx) = mean(sample.thermal_field.thermal_accumulation_time); idx = idx + 1; % 31. 平均热累积时间
    features(idx) = sample.params.T_nozzle; idx = idx + 1;                       % 32. 喷嘴温度设置
    features(idx) = sample.params.T_bed; idx = idx + 1;                          % 33. 热床温度设置
    features(idx) = sample.params.T_ambient; idx = idx + 1;                      % 34. 环境温度设置
    features(idx) = sample.params.fan_speed; idx = idx + 1;                      % 35. 风扇转速设置
    features(idx) = mean(sample.trajectory.vx_ref); idx = idx + 1;               % 36. 平均X速度（用于热计算）
    features(idx) = mean(sample.trajectory.vy_ref); idx = idx + 1;               % 37. 平均Y速度（用于热计算）
    features(idx) = sample.params.layer_height; idx = idx + 1;                   % 38. 层高设置

    % G-code特征模块（8个）
    features(idx) = sum(sample.gcode_data.is_corner) / length(sample.gcode_data.is_corner); idx = idx + 1; % 39. 转角密度
    features(idx) = mean(sample.gcode_data.corner_angles(sample.gcode_data.corner_angles > 0)); idx = idx + 1; % 40. 平均转角角度
    features(idx) = max(sample.gcode_data.curvature); idx = idx + 1;             % 41. 最大曲率
    features(idx) = mean(sample.gcode_data.curvature); idx = idx + 1;            % 42. 平均曲率
    features(idx) = mean(sample.gcode_data.d_last_corner(sample.gcode_data.d_last_corner < inf)); idx = idx + 1; % 43. 平均距上次转角距离
    features(idx) = sample.params.num_layers; idx = idx + 1;                     % 44. 层数
    features(idx) = sum(sample.gcode_data.is_corner); idx = idx + 1;             % 45. 转角总数
    features(idx) = sample.params.extrusion_width; idx = idx + 1;                % 46. 挤出宽度

    % 其他参数（4个）
    features(idx) = sample.params.nozzle_diameter; idx = idx + 1;                % 47. 喷嘴直径
    features(idx) = sample.params.extrusion_multiplier; idx = idx + 1;           % 48. 挤出倍率
    features(idx) = sample.params.mass_x; idx = idx + 1;                         % 49. X轴质量
    features(idx) = sample.params.stiffness_x; idx = idx + 1;                    % 50. X轴刚度

    % 修正：应该是50个特征，但索引会到51，说明多算了，重新检查
    % 实际上应该刚好50个

    X_features{i} = features;

    %% 3.2 输出目标（4个）
    targets = zeros(1, 4);

    targets(1) = sample.trajectory_error.max_error;  % 最大轨迹误差
    targets(2) = sample.adhesion_strength.mean;     % 平均粘结强度
    targets(3) = sample.adhesion_strength.weak_bond_ratio;  % 弱粘结比例
    targets(4) = sample.adhesion_strength.quality_score;   % 综合质量评分

    y_targets{i} = targets;
end

%% 4. 转换为矩阵格式
X_matrix = cell2mat(X_features);  % (num_samples, 50)
y_matrix = cell2mat(y_targets);   % (num_samples, 4)

fprintf('    数据矩阵: %d 样本, %d 特征, %d 目标\n', ...
    size(X_matrix, 1), size(X_matrix, 2), size(y_matrix, 2));

%% 5. 保存为Python格式
% 方法1：保存为.mat（Python可用scipy.io.loadmat读取）
python_mat_filename = strrep(mat_filename, '.mat', '_python.mat');
fprintf('    保存为MAT格式（Python兼容）...\n');

python_data = struct();
python_data.X = X_matrix;
python_data.y = y_matrix;
python_data.feature_names = get_feature_names();
python_data.target_names = {'max_trajectory_error_mm', 'mean_adhesion_strength_MPa', ...
                            'weak_bond_ratio', 'quality_score'};
python_data.num_samples = num_samples;
python_data.params = params;

save(python_mat_filename, '-struct', 'python_data', '-v7.3');
fprintf('    已保存: %s\n', python_mat_filename);

% 方法2：保存为CSV（更通用）
python_csv_X = strrep(mat_filename, '.mat', '_X.csv');
python_csv_y = strrep(mat_filename, '.mat', '_y.csv');

fprintf('    保存为CSV格式...\n');
csvwrite(python_csv_X, X_matrix);
csvwrite(python_csv_y, y_matrix);
fprintf('    已保存: %s\n', python_csv_X);
fprintf('    已保存: %s\n', python_csv_y);

% 方法3：生成Python加载脚本
python_loader_script = strrep(mat_filename, '.mat', '_loader.py');
generate_python_loader_script(python_loader_script, python_mat_filename, params);

fprintf('    已生成Python加载脚本: %s\n', python_loader_script);

%% 6. 生成数据说明文档
doc_filename = strrep(mat_filename, '.mat', '_README.txt');
generate_data_documentation(doc_filename, params);
fprintf('    已生成数据说明: %s\n', doc_filename);

fprintf('  数据转换完成！\n');

%% 子函数

function feature_names = get_feature_names()
    feature_names = {
        'mean_epsilon_x_mm', 'mean_epsilon_y_mm', 'max_epsilon_r_mm', 'rms_error_mm', ...
        'mean_vx_act_mm_s', 'mean_vy_act_mm_s', 'max_v_ref_mm_s', ...
        'mean_abs_ax_ref_mm_s2', 'mean_abs_ay_ref_mm_s2', ...
        'max_abs_jx_ref_mm_s3', 'max_abs_jy_ref_mm_s3', ...
        'mean_abs_F_inertia_x_N', 'mean_abs_F_inertia_y_N', ...
        'max_abs_delta_L_x_mm', 'max_abs_delta_L_y_mm', ...
        'omega_n_x_rad_s', 'zeta_x', ...
        'print_speed_mm_s', 'acceleration_mm_s2', 'jerk_mm_s3', ...
        'mean_T_nozzle_path_C', 'max_T_nozzle_path_C', 'min_T_nozzle_path_C', 'std_T_nozzle_path_C', ...
        'mean_T_interface_C', 'mean_cooling_rate_C_s', 'max_cooling_rate_C_s', ...
        'mean_time_above_melting_s', 'mean_gradient_z_C_mm', 'mean_gradient_xy_C_mm', ...
        'mean_thermal_accumulation_time_s', ...
        'T_nozzle_C', 'T_bed_C', 'T_ambient_C', 'fan_speed', ...
        'mean_vx_ref_mm_s', 'mean_vy_ref_mm_s', 'layer_height_mm', ...
        'corner_density', 'mean_corner_angle_deg', 'max_curvature_1_mm', ...
        'mean_curvature_1_mm', 'mean_d_last_corner_mm', ...
        'num_layers', 'num_corners', 'extrusion_width_mm', ...
        'nozzle_diameter_mm', 'extrusion_multiplier', 'mass_x_kg', 'stiffness_x_N_m'
    };
end

function generate_python_loader_script(filename, mat_filename, params)
    fid = fopen(filename, 'w');
    fprintf(fid, '"""\n');
    fprintf(fid, 'Python加载脚本 - 3D打印PINN训练数据\n');
    fprintf(fid, '自动生成于: %s\n', datestr(now));
    fprintf(fid, '"""\n\n');
    fprintf(fid, 'import numpy as np\n');
    fprintf(fid, 'from scipy.io import loadmat\n');
    fprintf(fid, 'import matplotlib.pyplot as plt\n\n');
    fprintf(fid, 'def load_data(mat_file):\n');
    fprintf(fid, '    """\n');
    fprintf(fid, '    加载MATLAB导出的数据\n');
    fprintf(fid, '\n');
    fprintf(fid, '    参数:\n');
    fprintf(fid, '        mat_file: .mat文件路径\n');
    fprintf(fid, '\n');
    fprintf(fid, '    返回:\n');
    fprintf(fid, '        X: 特征矩阵 (num_samples, 50)\n');
    fprintf(fid, '        y: 目标矩阵 (num_samples, 4)\n');
    fprintf(fid, '        feature_names: 特征名称列表\n');
    fprintf(fid, '        target_names: 目标名称列表\n');
    fprintf(fid, '    """\n');
    fprintf(fid, '    data = loadmat(mat_file)\n');
    fprintf(fid, '    X = data[''X'']\n');
    fprintf(fid, '    y = data[''y'']\n');
    fprintf(fid, '    feature_names = [name[0] for name in data[''feature_names'']]\n');
    fprintf(fid, '    target_names = [name[0] for name in data[''target_names'']]\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    print(f"数据形状: X={X.shape}, y={y.shape}")\n');
    fprintf(fid, '    print(f"特征数量: {len(feature_names)}")\n');
    fprintf(fid, '    print(f"目标数量: {len(target_names)}")\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    return X, y, feature_names, target_names\n\n');
    fprintf(fid, 'if __name__ == "__main__":\n');
    fprintf(fid, '    # 加载数据\n');
    fprintf(fid, '    mat_file = r"%s"\n', mat_filename);
    fprintf(fid, '    X, y, feature_names, target_names = load_data(mat_file)\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    # 数据统计\n');
    fprintf(fid, '    print("\\n特征统计:")\n');
    fprintf(fid, '    print(f"  均值: {np.mean(X, axis=0)}")\n');
    fprintf(fid, '    print(f"  标准差: {np.std(X, axis=0)}")\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    print("\\n目标统计:")\n');
    fprintf(fid, '    for i, name in enumerate(target_names):\n');
    fprintf(fid, '        print(f"  {name}: mean={np.mean(y[:, i]):.4f}, std={np.std(y[:, i]):.4f}")\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    # 可视化\n');
    fprintf(fid, '    fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n');
    fprintf(fid, '    for i, ax in enumerate(axes.flat):\n');
    fprintf(fid, '        ax.scatter(X[:, i] if i < X.shape[1] else np.arange(len(y)), y[:, i], alpha=0.5)\n');
    fprintf(fid, '        ax.set_xlabel("Feature" if i < X.shape[1] else "Sample Index")\n');
    fprintf(fid, '        ax.set_ylabel(target_names[i])\n');
    fprintf(fid, '        ax.set_title(f"{target_names[i]} vs Feature")\n');
    fprintf(fid, '    plt.tight_layout()\n');
    fprintf(fid, '    plt.savefig("data_exploration.png", dpi=150)\n');
    fprintf(fid, '    print("\\n图像已保存: data_exploration.png")\n');
    fclose(fid);
end

function generate_data_documentation(filename, params)
    fid = fopen(filename, 'w');
    fprintf(fid, '3D打印PINN训练数据集说明文档\n');
    fprintf(fid, '====================================\n\n');
    fprintf(fid, '生成时间: %s\n\n', datestr(now));
    fprintf(fid, '数据集概述:\n');
    fprintf(fid, '-----------\n');
    fprintf(fid, '本数据集通过MATLAB仿真生成，包含以下两个核心问题：\n');
    fprintf(fid, '1. 轨迹误差：由于速度突变导致的惯性+皮带弹性二阶震荡响应\n');
    fprintf(fid, '2. 层间粘结力：基于温度历史的分子扩散模型\n\n');
    fprintf(fid, '数据格式:\n');
    fprintf(fid, '--------\n');
    fprintf(fid, '- X: 特征矩阵 (num_samples, 50)\n');
    fprintf(fid, '- y: 目标矩阵 (num_samples, 4)\n\n');
    fprintf(fid, '特征列表 (50个):\n');
    fprintf(fid, '---------------\n');
    fprintf(fid, 'A. 轨迹误差模块 (20个):\n');
    fprintf(fid, '   1. mean_epsilon_x: X方向平均位置误差 (mm)\n');
    fprintf(fid, '   2. mean_epsilon_y: Y方向平均位置误差 (mm)\n');
    fprintf(fid, '   3. max_epsilon_r: 最大位置误差幅值 (mm)\n');
    fprintf(fid, '   4. rms_error: RMS位置误差 (mm)\n');
    fprintf(fid, '   5-6. 平均实际速度 (mm/s)\n');
    fprintf(fid, '   7. 最大参考速度 (mm/s)\n');
    fprintf(fid, '   8-9. 平均参考加速度绝对值 (mm/s²)\n');
    fprintf(fid, '   10-11. 最大参考加加速度绝对值 (mm/s³)\n');
    fprintf(fid, '   12-13. 平均惯性力绝对值 (N)\n');
    fprintf(fid, '   14-15. 最大皮带伸长量 (mm)\n');
    fprintf(fid, '   16. X轴固有频率 (rad/s)\n');
    fprintf(fid, '   17. X轴阻尼比\n');
    fprintf(fid, '   18. 打印速度设置 (mm/s)\n');
    fprintf(fid, '   19. 加速度设置 (mm/s²)\n');
    fprintf(fid, '   20. 加加速度设置 (mm/s³)\n\n');
    fprintf(fid, 'B. 温度场模块 (18个):\n');
    fprintf(fid, '   21-24. 喷嘴路径温度统计: 均值/最大/最小/标准差 (°C)\n');
    fprintf(fid, '   25. 平均层间温度 (°C)\n');
    fprintf(fid, '   26-27. 平均/最大冷却速率 (°C/s)\n');
    fprintf(fid, '   28. 平均时间高于熔点 (s)\n');
    fprintf(fid, '   29. 平均Z方向温度梯度 (°C/mm)\n');
    fprintf(fid, '   30. 平均XY平面温度梯度 (°C/mm)\n');
    fprintf(fid, '   31. 平均热累积时间 (s)\n');
    fprintf(fid, '   32-35. 温度设置: 喷嘴/热床/环境/风扇 (°C, °C, °C, RPM)\n');
    fprintf(fid, '   36-37. 平均参考速度 (mm/s)\n');
    fprintf(fid, '   38. 层高设置 (mm)\n\n');
    fprintf(fid, 'C. G-code特征模块 (8个):\n');
    fprintf(fid, '   39. 转角密度 (转角数/总点数)\n');
    fprintf(fid, '   40. 平均转角角度 (度)\n');
    fprintf(fid, '   41. 最大曲率 (1/mm)\n');
    fprintf(fid, '   42. 平均曲率 (1/mm)\n');
    fprintf(fid, '   43. 平均距离上次转角 (mm)\n');
    fprintf(fid, '   44. 层数\n');
    fprintf(fid, '   45. 转角总数\n');
    fprintf(fid, '   46. 挤出宽度 (mm)\n\n');
    fprintf(fid, 'D. 其他参数 (4个):\n');
    fprintf(fid, '   47. 喷嘴直径 (mm)\n');
    fprintf(fid, '   48. 挤出倍率\n');
    fprintf(fid, '   49. X轴运动质量 (kg)\n');
    fprintf(fid, '   50. X轴刚度 (N/m)\n\n');
    fprintf(fid, '目标列表 (4个):\n');
    fprintf(fid, '---------------\n');
    fprintf(fid, '1. max_trajectory_error: 最大轨迹误差 (mm)\n');
    fprintf(fid, '2. mean_adhesion_strength: 平均层间粘结强度 (MPa)\n');
    fprintf(fid, '3. weak_bond_ratio: 弱粘结区域比例 (0-1)\n');
    fprintf(fid, '4. quality_score: 综合质量评分 (0-1)\n\n');
    fprintf(fid, '使用方法:\n');
    fprintf(fid, '--------\n');
    fprintf(fid, '在Python中加载:\n');
    fprintf(fid, '```python\n');
    fprintf(fid, 'from scipy.io import loadmat\n');
    fprintf(fid, 'data = loadmat(''filename_python.mat'')\n');
    fprintf(fid, 'X = data[''X'']\n');
    fprintf(fid, 'y = data[''y'']\n');
    fprintf(fid, '```\n\n');
    fprintf(fid, '或者运行生成的加载脚本:\n');
    fprintf(fid, '```bash\n');
    fprintf(fid, 'python filename_loader.py\n');
    fprintf(fid, '```\n');
    fclose(fid);
end

end
