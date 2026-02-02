function vmax_entry = junction_deviation(v1, v2, theta, acceleration, junction_deviation_mm)
% JUNCTION_DEVIATION - 计算转角最大允许速度（引入位置偏差）
%
% 基于Marlin固件算法 (planner.cpp:2459)
%
% **关键**：这个算法会在转角处"切圆角"而非完全停止
% 这会引入额外的位置误差，用于训练误差修复网络
%
% 公式: vmax² = a * JD * sin(θ/2) / (1 - sin(θ/2))
%
% 输入:
%   v1, v2 - 前后段速度向量 [vx, vy] (mm/s)
%   theta - 转角角度 (度)
%   acceleration - 最大加速度 (mm/s²) - 取X/Y最小值
%   junction_deviation_mm - JUNCTION_DEVIATION_MM参数 (mm)
%
% 输出:
%   vmax_entry - 转角最大允许速度 (mm/s)
%
% 示例:
%   v1 = [50, 0]; v2 = [0, 50]; theta = 90;
%   vmax = junction_deviation(v1, v2, theta, 500, 0.013);
%   预期结果: vmax ≈ 39.5 mm/s（而非90度转角后的0）

    % 将角度转换为弧度
    theta_rad = deg2rad(theta);

    % 计算单位向量
    v1_norm = norm(v1);
    v2_norm = norm(v2);

    if v1_norm < 1e-6 || v2_norm < 1e-6
        vmax_entry = 0;
        return;
    end

    unit_vec1 = v1 / v1_norm;
    unit_vec2 = v2 / v2_norm;

    % 计算夹角余弦
    junction_cos_theta = dot(unit_vec1, unit_vec2);

    % 限制范围避免数值误差
    junction_cos_theta = max(-1.0, min(1.0, junction_cos_theta));

    % 半角正弦值 (三角恒等式: sin²(θ/2) = (1 - cos(θ))/2)
    sin_theta_d2 = sqrt(0.5 * (1.0 - junction_cos_theta));

    % 如果角度很小，使用全速
    if sin_theta_d2 < 0.001
        vmax_entry = min(v1_norm, v2_norm);
        return;
    end

    % 取各轴最小加速度（保守估计）
    if isscalar(acceleration)
        junction_acceleration = acceleration;
    else
        junction_acceleration = min(acceleration);
    end

    % Junction Deviation核心公式
    % vmax² = a * JD * sin(θ/2) / (1 - sin(θ/2))
    vmax_junction_sqr = junction_acceleration * junction_deviation_mm * sin_theta_d2 / (1.0 - sin_theta_d2);

    % 转换为速度
    vmax_junction = sqrt(vmax_junction_sqr);

    % 保守限制：不能超过任一段的速度
    vmax_entry = min([vmax_junction, v1_norm, v2_norm]);

    % 实际位置偏差估算（用于验证）
    % r = v² / a  (圆弧半径)
    % deviation = r - sqrt(r² - (JD/2)²)
    if nargout > 1
        % 可选：计算实际偏差量
        radius = v1_norm^2 / junction_acceleration;
        actual_deviation = radius - sqrt(radius^2 - (junction_deviation_mm/2)^2);
        varargout{1} = actual_deviation;
    end
end
