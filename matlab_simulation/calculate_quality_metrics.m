function quality_data = calculate_quality_metrics(trajectory_data, thermal_data, params)
% CALCULATE_QUALITY_METRICS - Calculate implicit quality parameters
%
% This function computes quality metrics that cannot be directly measured
% during printing but are critical for final part quality.
%
% These are IMPLICIT QUALITY PARAMETERS that must be inferred from
% observable sensor data (trajectory, temperature, forces).
%
% Inputs:
%   trajectory_data - Structure from reconstruct_trajectory.m
%   thermal_data    - Structure from simulate_thermal_field.m
%   params          - Physics parameters
%
% Output:
%   quality_data - Structure containing:
%     .adhesion_ratio       - Interlayer adhesion strength (ratio 0-1)
%     .internal_stress      - Residual stress (MPa)
%     .porosity             - Porosity percentage (0-100%)
%     .dimensional_accuracy - Dimensional error (mm)
%     .quality_score        - Overall quality score (0-1)
%
% Reference:
%   - Wool-O'Connor model for adhesion
%   - Thermal stress models for residual stress
%   - Empirical models for porosity and dimensional accuracy

    fprintf('Calculating implicit quality metrics...\n');

    n_points = length(trajectory_data.time);

    %% Initialize output structure
    quality_data.adhesion_ratio = zeros(n_points, 1);
    quality_data.internal_stress = zeros(n_points, 1);
    quality_data.porosity = zeros(n_points, 1);
    quality_data.dimensional_accuracy = zeros(n_points, 1);
    quality_data.quality_score = zeros(n_points, 1);

    %% 1. Interlayer Adhesion Strength
    % Already calculated in thermal_data, copy it
    if isfield(thermal_data, 'adhesion_ratio')
        quality_data.adhesion_ratio = thermal_data.adhesion_ratio;
    else
        % Fallback: calculate from interface temperature
        T_interface = thermal_data.T_interface;
        T_glass = params.material.glass_transition;  % PLA: 60°C

        % Higher interface temperature → better adhesion
        % Model: adhesion increases exponentially above Tg
        quality_data.adhesion_ratio = 1 - exp(-0.1 * (T_interface - T_glass));
        quality_data.adhesion_ratio(T_interface < T_glass) = 0.1;  % Minimum baseline
        quality_data.adhesion_ratio = min(quality_data.adhesion_ratio, 1.0);
    end

    %% 2. Internal Stress (Residual Stress)
    % Caused by thermal contraction during cooling
    % Model: Stress ~ E * alpha * delta_T
    %
    % where:
    %   E = Young's modulus (increases as temperature drops)
    %   alpha = thermal expansion coefficient
    %   delta_T = temperature drop from printing to ambient

    % Material properties for PLA
    E_room = 3500;  % Young's modulus at room temperature (MPa)
    alpha_pla = 68e-6;  % Thermal expansion coefficient (1/K)
    T_print = params.printing.nozzle_temp - 20;  % Approx deposit temp (°C)
    T_ambient = params.environment.ambient_temp;

    % Calculate thermal stress
    % Higher cooling rate → higher stress (less time for stress relaxation)
    cooling_rate_normalized = thermal_data.cooling_rate / max(thermal_data.cooling_rate);
    delta_T = T_print - thermal_data.T_interface;

    % Stress model (MPa)
    quality_data.internal_stress = E_room * alpha_pla * delta_T .* (1 + 0.5 * cooling_rate_normalized);

    %% 3. Porosity
    % Caused by incomplete fusion between layers/extrusions
    % Factors: low temperature, high speed, poor adhesion

    % Empirical model: porosity increases with:
    % - Lower interface temperature
    % - Higher print speed
    % - Poor layer adhesion

    v = sqrt(trajectory_data.vx.^2 + trajectory_data.vy.^2 + trajectory_data.vz.^2);
    v_normalized = v / params.motion.max_velocity;

    T_interface = thermal_data.T_interface;
    T_optimal = params.printing.nozzle_temp * 0.7;  % Optimal interface temp

    % Temperature factor (lower T → higher porosity)
    temp_factor = exp(-0.05 * (T_interface - T_ambient));

    % Speed factor (higher speed → higher porosity)
    speed_factor = v_normalized.^2;

    % Adhesion factor (poor adhesion → higher porosity)
    adhesion_factor = 1 - quality_data.adhesion_ratio;

    % Combined porosity model (0-100%)
    quality_data.porosity = 100 * (0.3 * temp_factor + 0.3 * speed_factor + 0.4 * adhesion_factor);
    quality_data.porosity = min(quality_data.porosity, 20);  # Cap at 20% porosity

    %% 4. Dimensional Accuracy
    % Deviation from intended dimensions due to:
    % - Thermal shrinkage during cooling
    % - Velocity changes (acceleration/deceleration cause over/under-extrusion)
    % - Corner rounding effects

    % Extract velocity magnitude from reference trajectory
    v = sqrt(trajectory_data.vx.^2 + trajectory_data.vy.^2 + trajectory_data.vz.^2);

    % Extract acceleration magnitude
    a = sqrt(trajectory_data.ax.^2 + trajectory_data.ay.^2 + trajectory_data.az.^2);

    % Normalize for empirical model
    v_normalized = v / params.motion.max_velocity;
    a_normalized = a / params.motion.max_accel;

    % Thermal shrinkage component
    thermal_shrinkage = alpha_pla * (T_print - T_ambient) * 10;  % mm

    % Velocity-based dimensional error
    % Higher speed variation → more over/under extrusion → worse dimensional accuracy
    velocity_error = 0.1 * v_normalized.^2;  % mm

    % Acceleration-based dimensional error
    % High acceleration → extrusion lag → dimensional errors
    accel_error = 0.05 * a_normalized;  % mm

    % Combined dimensional accuracy error (mm)
    quality_data.dimensional_accuracy = thermal_shrinkage + velocity_error + accel_error;

    %% 5. Overall Quality Score
    % Composite metric combining all quality aspects
    % Weighted combination: adhesion (40%), stress (20%), porosity (20%), accuracy (20%)

    % Normalize each metric to [0, 1] scale
    adhesion_score = quality_data.adhesion_ratio;  % Already [0, 1]

    % Stress: lower is better, cap at 50 MPa
    stress_score = 1 - min(quality_data.internal_stress / 50, 1.0);

    % Porosity: lower is better
    porosity_score = 1 - min(quality_data.porosity / 20, 1.0);

    # Accuracy: lower error is better, cap at 1mm
    accuracy_score = 1 - min(quality_data.dimensional_accuracy / 1.0, 1.0);

    # Weighted combination
    quality_data.quality_score = (0.4 * adhesion_score + ...
                                  0.2 * stress_score + ...
                                  0.2 * porosity_score + ...
                                  0.2 * accuracy_score);

    fprintf('  Quality metrics calculated:\n');
    fprintf('    Adhesion strength: %.3f ± %.3f (ratio)\n', ...
            mean(quality_data.adhesion_ratio), std(quality_data.adhesion_ratio));
    fprintf('    Internal stress: %.3f ± %.3f (MPa)\n', ...
            mean(quality_data.internal_stress), std(quality_data.internal_stress));
    fprintf('    Porosity: %.2f ± %.2f (%%)\n', ...
            mean(quality_data.porosity), std(quality_data.porosity));
    fprintf('    Dimensional error: %.3f ± %.3f (mm)\n', ...
            mean(quality_data.dimensional_accuracy), std(quality_data.dimensional_accuracy));
    fprintf('    Quality score: %.3f ± %.3f\n', ...
            mean(quality_data.quality_score), std(quality_data.quality_score));

end
