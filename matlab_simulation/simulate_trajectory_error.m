function results = simulate_trajectory_error(trajectory_data, params)
% SIMULATE_TRAJECTORY_ERROR - Simulate trajectory error due to inertia and belt elasticity
%
% This function models the printer dynamics as a second-order mass-spring-damper system:
%   m·x'' + c·x' + k·x = F(t)
%
% Where:
%   m - effective mass of moving system
%   c - damping coefficient (belt and friction)
%   k - stiffness of transmission (belt elasticity)
%   F(t) - forcing function (inertial forces from acceleration)
%
% The system responds to the reference trajectory with:
% 1. Lag due to acceleration (inertia)
% 2. Oscillation due to underdamped response
% 3. Steady-state error (if any)
%
% Inputs:
%   trajectory_data - Structure from parse_gcode.m
%   params          - Physics parameters from physics_parameters.m
%
% Output:
%   results         - Structure containing actual trajectory and error vectors
%
% Reference: Control theory for second-order systems

    fprintf('Simulating trajectory error (second-order system dynamics)...\n');

    %% Extract time series data
    t = trajectory_data.time;
    n_points = length(t);

    % Reference trajectory (from G-code)
    x_ref = trajectory_data.x;
    y_ref = trajectory_data.y;
    z_ref = trajectory_data.z;

    % Reference acceleration (forcing function)
    ax_ref = trajectory_data.ax;
    ay_ref = trajectory_data.ay;

    %% Extract system parameters
    % X-axis dynamics
    mx = params.dynamics.x.mass;
    kx = params.dynamics.x.stiffness;
    cx = params.dynamics.x.damping;
    wn_x = params.dynamics.x.natural_freq;
    zeta_x = params.dynamics.x.damping_ratio;

    % Y-axis dynamics
    my = params.dynamics.y.mass;
    ky = params.dynamics.y.stiffness;
    cy = params.dynamics.y.damping;
    wn_y = params.dynamics.y.natural_freq;
    zeta_y = params.dynamics.y.damping_ratio;

    fprintf('  X-axis: ωn = %.2f rad/s, ζ = %.4f\n', wn_x, zeta_x);
    fprintf('  Y-axis: ωn = %.2f rad/s, ζ = %.4f\n', wn_y, zeta_y);

    %% Simulation time step (use uniform grid)
    dt = params.simulation.time_step;

    % Create uniform time grid
    t_uniform = linspace(t(1), t(end), ceil((t(end) - t(1)) / dt) + 1);
    dt_actual = t_uniform(2) - t_uniform(1);
    n_uniform = length(t_uniform);

    % Interpolate reference to uniform grid
    x_ref_uniform = interp1(t, x_ref, t_uniform, 'linear', 'extrap');
    y_ref_uniform = interp1(t, y_ref, t_uniform, 'linear', 'extrap');
    ax_ref_uniform = interp1(t, ax_ref, t_uniform, 'linear', 'extrap');
    ay_ref_uniform = interp1(t, ay_ref, t_uniform, 'linear', 'extrap');

    %% Simulate X-axis dynamics (second-order system)
    fprintf('  Simulating X-axis dynamics...\n');

    % State space: [x; v] where x is position error, v is velocity error
    % For a second-order system driven by acceleration:
    % x' = v
    % v' = -(c/m)*v - (k/m)*x - a_ref
    %
    % Note: The negative sign on a_ref because the error is defined as:
    % error = actual - reference
    % And the actual position responds to the inertial force F = -m*a_ref

    % Initialize state
    x_state = zeros(2, n_uniform);
    x_state(:, 1) = [0; 0];  % Start with zero error

    % System matrices
    Ax = [0, 1;
          -kx/mx, -cx/mx];
    Bx = [0; -1];  % Input is acceleration

    % Time integration (Euler method)
    for i = 2:n_uniform
        % State update: x(k+1) = x(k) + dt * (Ax*x(k) + Bx*u(k))
        dx = Ax * x_state(:, i-1) + Bx * ax_ref_uniform(i-1);
        x_state(:, i) = x_state(:, i-1) + dt_actual * dx;
    end

    % Extract actual position
    x_act_uniform = x_ref_uniform + x_state(1, :);
    vx_act_uniform = gradient(x_act_uniform, dt_actual);

    %% Simulate Y-axis dynamics
    fprintf('  Simulating Y-axis dynamics...\n');

    y_state = zeros(2, n_uniform);
    y_state(:, 1) = [0; 0];

    Ay = [0, 1;
          -ky/my, -cy/my];
    By = [0; -1];

    for i = 2:n_uniform
        dy = Ay * y_state(:, i-1) + By * ay_ref_uniform(i-1);
        y_state(:, i) = y_state(:, i-1) + dt_actual * dy;
    end

    y_act_uniform = y_ref_uniform + y_state(1, :);
    vy_act_uniform = gradient(y_act_uniform, dt_actual);

    %% Calculate error vectors (as vectors, not just magnitudes)
    fprintf('  Calculating error vectors...\n');

    % Position error vector (components)
    error_x_uniform = x_state(1, :);
    error_y_uniform = y_state(1, :);

    % Position error magnitude
    error_magnitude_uniform = sqrt(error_x_uniform.^2 + error_y_uniform.^2);

    % Error direction angle
    error_direction = atan2(error_y_uniform, error_x_uniform);

    %% Dynamic forces
    fprintf('  Calculating dynamic forces...\n');

    % Inertial forces: F = m * a_ref
    F_inertia_x = mx * ax_ref_uniform;
    F_inertia_y = my * ay_ref_uniform;

    % Elastic forces (belt stretch): F = k * error
    F_elastic_x = kx * error_x_uniform;
    F_elastic_y = ky * error_y_uniform;

    % Damping forces: F = c * v_error
    v_error_x = x_state(2, :);
    v_error_y = y_state(2, :);
    F_damping_x = cx * v_error_x;
    F_damping_y = cy * v_error_y;

    %% Interpolate back to original time grid
    x_act = interp1(t_uniform, x_act_uniform, t, 'linear', 'extrap');
    y_act = interp1(t_uniform, y_act_uniform, t, 'linear', 'extrap');
    vx_act = interp1(t_uniform, vx_act_uniform, t, 'linear', 'extrap');
    vy_act = interp1(t_uniform, vy_act_uniform, t, 'linear', 'extrap');

    error_x = interp1(t_uniform, error_x_uniform, t, 'linear', 'extrap');
    error_y = interp1(t_uniform, error_y_uniform, t, 'linear', 'extrap');
    error_magnitude = interp1(t_uniform, error_magnitude_uniform, t, 'linear', 'extrap');
    error_dir = interp1(t_uniform, error_direction, t, 'linear', 'extrap');

    F_inertia_x = interp1(t_uniform, F_inertia_x, t, 'linear', 'extrap');
    F_inertia_y = interp1(t_uniform, F_inertia_y, t, 'linear', 'extrap');
    F_elastic_x = interp1(t_uniform, F_elastic_x, t, 'linear', 'extrap');
    F_elastic_y = interp1(t_uniform, F_elastic_y, t, 'linear', 'extrap');

    %% Statistical analysis
    fprintf('  Statistical analysis of errors:\n');
    fprintf('    Max X error: %.3f mm\n', max(abs(error_x)));
    fprintf('    Max Y error: %.3f mm\n', max(abs(error_y)));
    fprintf('    Max error magnitude: %.3f mm\n', max(error_magnitude));
    fprintf('    RMS error magnitude: %.3f mm\n', rms(error_magnitude));
    fprintf('    Mean error magnitude: %.3f mm\n', mean(error_magnitude));

    % Error at corners
    corner_mask = trajectory_data.is_corner;
    if sum(corner_mask) > 0
        corner_errors = error_magnitude(corner_mask);
        fprintf('    Max corner error: %.3f mm\n', max(corner_errors));
        fprintf('    Mean corner error: %.3f mm\n', mean(corner_errors));
    end

    %% Frequency analysis (to detect resonance excitation)
    fprintf('  Performing frequency analysis...\n');

    % Power spectral density of error
    [psd_x, freq_x] = pwelch(error_x, [], [], [], 1/dt_actual);
    [psd_y, freq_y] = pwelch(error_y, [], [], [], 1/dt_actual);

    % Find dominant frequencies
    [max_psd_x, idx_x] = max(psd_x);
    [max_psd_y, idx_y] = max(psd_y);

    dominant_freq_x = freq_x(idx_x);
    dominant_freq_y = freq_y(idx_y);

    fprintf('    Dominant error frequency (X): %.2f Hz\n', dominant_freq_x);
    fprintf('    Dominant error frequency (Y): %.2f Hz\n', dominant_freq_y);
    fprintf('    X-axis natural frequency: %.2f Hz\n', wn_x / (2*pi));
    fprintf('    Y-axis natural frequency: %.2f Hz\n', wn_y / (2*pi));

    %% Create output structure
    results.time = t;

    % Reference trajectory
    results.x_ref = x_ref;
    results.y_ref = y_ref;
    results.z_ref = z_ref;

    % Actual trajectory (with dynamics)
    results.x_act = x_act;
    results.y_act = y_act;
    results.z_act = z_ref;  % Z-axis not modeled in dynamics

    % Actual velocity
    results.vx_act = vx_act;
    results.vy_act = vy_act;
    results.vz_act = zeros(size(vy_act));

    % Actual acceleration (numerical)
    results.ax_act = gradient(vx_act, dt);
    results.ay_act = gradient(vy_act, dt);
    results.az_act = zeros(size(vy_act));

    % ERROR VECTORS (not just magnitudes!)
    results.error_x = error_x;           % mm - X component of error vector
    results.error_y = error_y;           % mm - Y component of error vector
    results.error_magnitude = error_magnitude;  % mm - |error vector|
    results.error_direction = error_dir; % rad - Direction of error vector

    % Dynamic forces
    results.F_inertia_x = F_inertia_x;   % N
    results.F_inertia_y = F_inertia_y;   % N
    results.F_elastic_x = F_elastic_x;   % N
    results.F_elastic_y = F_elastic_y;   % N
    results.F_damping_x = F_damping_x;   % N (on uniform grid)
    results.F_damping_y = F_damping_y;   % N (on uniform grid)

    % Belt stretch (displacement)
    results.belt_stretch_x = error_x;    % mm
    results.belt_stretch_y = error_y;    % mm

    % System response metrics
    results.settling_time_x = params.dynamics.x.settling_time;
    results.settling_time_y = params.dynamics.y.settling_time;
    results.overshoot_x = exp(-pi * zeta_x / sqrt(1 - zeta_x^2)) * 100;  % %
    results.overshoot_y = exp(-pi * zeta_y / sqrt(1 - zeta_y^2)) * 100;  % %

    % Frequency analysis
    results.frequency_x = freq_x;
    results.frequency_y = freq_y;
    results.psd_x = psd_x;
    results.psd_y = psd_y;
    results.dominant_freq_x = dominant_freq_x;  % Hz
    results.dominant_freq_y = dominant_freq_y;  % Hz

    % Corner-specific errors
    results.corner_mask = corner_mask;
    results.corner_errors = error_magnitude(corner_mask);
    results.max_corner_error = max(error_magnitude(corner_mask));
    results.mean_corner_error = mean(error_magnitude(corner_mask));

    fprintf('  Trajectory error simulation complete!\n\n');

    %% Optional: Plotting (if debug mode)
    if params.debug.plot_trajectory
        figure('Name', 'Trajectory Error Analysis', 'Position', [100, 100, 1200, 800]);

        % Reference vs Actual trajectory (top view)
        subplot(2, 3, 1);
        plot(x_ref, y_ref, 'b--', 'LineWidth', 1.5); hold on;
        plot(x_act, y_act, 'r-', 'LineWidth', 1);
        plot(x_ref(corner_mask), y_ref(corner_mask), 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'y');
        axis equal;
        grid on;
        xlabel('X (mm)');
        ylabel('Y (mm)');
        title('Reference vs Actual Trajectory');
        legend('Reference', 'Actual', 'Corners', 'Location', 'best');

        % Error magnitude over time
        subplot(2, 3, 2);
        plot(t, error_magnitude, 'r-', 'LineWidth', 1);
        grid on;
        xlabel('Time (s)');
        ylabel('Error Magnitude (mm)');
        title('Position Error Magnitude');

        % Error components (vector visualization)
        subplot(2, 3, 3);
        quiver(x_ref(corner_mask), y_ref(corner_mask), ...
                error_x(corner_mask), error_y(corner_mask), ...
                'AutoScale', 'on', 'MaxHeadSize', 0.5);
        hold on;
        plot(x_ref, y_ref, 'k--', 'LineWidth', 0.5);
        axis equal;
        grid on;
        xlabel('X (mm)');
        ylabel('Y (mm)');
        title('Error Vectors at Corners');

        % X-axis error
        subplot(2, 3, 4);
        plot(t, error_x, 'b-', 'LineWidth', 1);
        grid on;
        xlabel('Time (s)');
        ylabel('X Error (mm)');
        title('X-Axis Position Error');

        % Y-axis error
        subplot(2, 3, 5);
        plot(t, error_y, 'r-', 'LineWidth', 1);
        grid on;
        xlabel('Time (s)');
        ylabel('Y Error (mm)');
        title('Y-Axis Position Error');

        % Power spectral density
        subplot(2, 3, 6);
        semilogy(freq_x, psd_x, 'b-', 'LineWidth', 1.5); hold on;
        semilogy(freq_y, psd_y, 'r-', 'LineWidth', 1.5);
        xline(wn_x / (2*pi), 'b--', 'X \omega_n', 'LineWidth', 1.5);
        xline(wn_y / (2*pi), 'r--', 'Y \omega_n', 'LineWidth', 1.5);
        grid on;
        xlabel('Frequency (Hz)');
        ylabel('PSD (mm²/Hz)');
        title('Error Power Spectral Density');
        legend('X Error', 'Y Error', 'Location', 'best');
        xlim([0, 100]);

        drawnow;
    end

end
