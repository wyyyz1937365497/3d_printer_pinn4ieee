function [error_x, error_y] = junction_deviation_effect(x_ref, y_ref, params)
% JUNCTION_DEVIATION_EFFECT - 计算Junction Deviation引起的X/Y位置误差
%
% 输入:
%   x_ref, y_ref - 参考轨迹位置 (mm)
%   params - 物理参数结构体
%
% 输出:
%   error_x, error_y - X/Y方向的位置误差 (mm)
%
% 参考: Marlin固件的junction deviation算法

    n_points = length(x_ref);
    error_x = zeros(n_points, 1);
    error_y = zeros(n_points, 1);

    % 计算速度向量（使用位置差分）
    if n_points > 1
        vx = [0; diff(x_ref) * 100];  % 假设100Hz采样，转换为mm/s
        vy = [0; diff(y_ref) * 100];
    else
        vx = zeros(n_points, 1);
        vy = zeros(n_points, 1);
    end

    % 计算转角角度
    for i = 2:n_points-1
        % 前后速度向量
        v1 = [vx(i), vy(i)];
        v2 = [vx(i+1), vy(i+1)];

        % 计算夹角
        v1_norm = norm(v1);
        v2_norm = norm(v2);

        if v1_norm > 1e-6 && v2_norm > 1e-6
            cos_theta = dot(v1, v2) / (v1_norm * v2_norm);
            cos_theta = max(-1, min(1, cos_theta));
            theta = rad2deg(acos(cos_theta));

            % 调用junction deviation函数计算偏差
            junction_dev_mm = 0.013;  % Marlin默认值
            acceleration = params.motion.max_accel;

            % 估算位置偏差（简化模型）
            % 在转角处，实际路径会比指令路径"切圆角"
            % 偏差量与转角角度和速度相关

            % 偏差方向（垂直于角平分线）
            if theta > 5  % 只在明显转角处有偏差
                % 估算偏差量
                sin_theta_d2 = sind(theta / 2);
                if sin_theta_d2 > 0.001
                    % 半径估算
                    radius = v1_norm^2 / acceleration;
                    % 实际偏差
                    deviation = radius - sqrt(max(0, radius^2 - (junction_dev_mm/2)^2));

                    % 分解到X/Y方向
                    angle_avg = atan2(vy(i) + vy(i+1), vx(i) + vx(i+1));
                    error_x(i) = deviation * sin(angle_avg) * sin_theta_d2;
                    error_y(i) = -deviation * cos(angle_avg) * sin_theta_d2;
                end
            end
        end
    end
end
