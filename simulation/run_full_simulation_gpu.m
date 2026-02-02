function simulation_data = run_full_simulation_gpu(gcode_file, output_file, parser_options, gpu_id)
% RUN_FULL_SIMULATION_GPU - GPU-accelerated complete FDM simulation
%
% This is the GPU-accelerated version of run_full_simulation.m
% Automatically uses GPU when beneficial, falls back to CPU otherwise.
%
% Inputs:
%   gcode_file     - Path to G-code file (.gcode)
%   output_file    - Path to output .mat file (optional)
%   parser_options - Options for G-code parser (optional)
%   gpu_id         - GPU device ID (optional, default: auto-select)
%                    Use 1 for cuda1, [] for auto-select
%
% Output:
%   simulation_data - Complete simulation results structure
%
% Examples:
%   % Auto-select GPU with most free memory
%   data = run_full_simulation_gpu('print.gcode', 'output.mat', [], []);
%
%   % Use cuda1 specifically
%   opts = struct('layers', 'first', 'use_improved', true);
%   data = run_full_simulation_gpu('print.gcode', 'output.mat', opts, 1);

    fprintf('============================================================\n');
    fprintf('GPU-ACCELERATED FDM 3D PRINTER SIMULATION\n');
    fprintf('============================================================\n');
    fprintf('\n');

    %% Check inputs
    if nargin < 1
        gcode_file = 'Tremendous Hillar_PLA_17m1s.gcode';
    end

    if nargin < 2
        [filepath, name, ~] = fileparts(gcode_file);
        output_file = fullfile(filepath, [name, '_simulation.mat']);
    end

    if nargin < 3
        parser_options = struct();
        parser_options.use_improved = true;
        parser_options.layers = 'first';
        parser_options.include_skirt = false;
    end

    if nargin < 4
        gpu_id = [];  % Auto-select
    end

    fprintf('G-code file: %s\n', gcode_file);
    fprintf('Output file: %s\n', output_file);
    fprintf('\n');

    %% Step 0: Setup GPU
    fprintf('STEP 0: Setting up GPU acceleration...\n');

    % Ensure gpu_utils is in path
    script_dir = fileparts(mfilename('fullpath'));
    addpath(script_dir);

    gpu_info = setup_gpu(gpu_id);
    fprintf('\n');

    %% Step 1: Load physics parameters
    fprintf('STEP 1: Loading physics parameters...\n');
    params = physics_parameters();
    fprintf('\n');

    %% Step 2: Parse G-code
    fprintf('STEP 2: Parsing G-code file...\n');

    if isfield(parser_options, 'use_improved') && parser_options.use_improved
        fprintf('  Using improved parser for 2D trajectory extraction\n');
        trajectory_data = parse_gcode_improved(gcode_file, params, parser_options);
    else
        fprintf('  Using original parser\n');
        trajectory_data = parse_gcode(gcode_file, params);
    end
    fprintf('\n');

    %% Step 3: Simulate thermal field
    fprintf('STEP 3: Simulating thermal field...\n');
    thermal_results = simulate_thermal_field(trajectory_data, params);
    fprintf('\n');

    %% Step 4: Calculate quality metrics (from reference trajectory ONLY)
    fprintf('STEP 4: Calculating implicit quality parameters...\n');
    fprintf('  Note: Quality metrics are computed from reference trajectory + thermal field\n');
    fprintf('  They do NOT depend on simulated trajectory errors\n');
    quality_data = calculate_quality_metrics(trajectory_data, thermal_results, params);
    fprintf('\n');

    %% Step 5: Simulate trajectory error (GPU or CPU)
    fprintf('STEP 5: Simulating trajectory error...\n');

    % Choose GPU or CPU based on data size and GPU availability
    n_points = length(trajectory_data.time);

    if gpu_info.use_gpu && n_points > 500
        fprintf('  Using GPU-accelerated simulation\n');
        trajectory_results = simulate_trajectory_error_gpu(trajectory_data, params, gpu_info);
    else
        if gpu_info.available
            fprintf('  Using CPU simulation (dataset too small: %d points)\n', n_points);
        else
            fprintf('  Using CPU simulation (GPU not available)\n');
        end
        trajectory_results = simulate_trajectory_error(trajectory_data, params);
    end
    fprintf('\n');

    %% Step 6: Combine results
    fprintf('STEP 6: Combining results into unified dataset...\n');

    t = trajectory_data.time;
    n_points = length(t);

    simulation_data = [];

    % === TIME ===
    simulation_data.time = t(:);

    % === REFERENCE TRAJECTORY ===
    simulation_data.x_ref = trajectory_data.x(:);
    simulation_data.y_ref = trajectory_data.y(:);
    simulation_data.z_ref = trajectory_data.z(:);

    % === ACTUAL TRAJECTORY ===
    simulation_data.x_act = trajectory_results.x_act(:);
    simulation_data.y_act = trajectory_results.y_act(:);
    simulation_data.z_act = trajectory_results.z_act(:);

    % === KINEMATICS ===
    simulation_data.vx_ref = trajectory_data.vx(:);
    simulation_data.vy_ref = trajectory_data.vy(:);
    simulation_data.vz_ref = trajectory_data.vz(:);
    simulation_data.vx_act = trajectory_results.vx_act(:);
    simulation_data.vy_act = trajectory_results.vy_act(:);
    simulation_data.vz_act = trajectory_results.vz_act(:);
    simulation_data.v_mag_ref = trajectory_data.v_actual(:);

    simulation_data.ax_ref = trajectory_data.ax(:);
    simulation_data.ay_ref = trajectory_data.ay(:);
    simulation_data.az_ref = trajectory_data.az(:);
    simulation_data.ax_act = trajectory_results.ax_act(:);
    simulation_data.ay_act = trajectory_results.ay_act(:);
    simulation_data.az_act = trajectory_results.az_act(:);
    simulation_data.a_mag_ref = trajectory_data.acceleration(:);

    simulation_data.jx_ref = trajectory_data.jx(:);
    simulation_data.jy_ref = trajectory_data.jy(:);
    simulation_data.jz_ref = trajectory_data.jz(:);
    simulation_data.jerk_mag = trajectory_data.jerk(:);

    % === DYNAMICS ===
    simulation_data.F_inertia_x = trajectory_results.F_inertia_x(:);
    simulation_data.F_inertia_y = trajectory_results.F_inertia_y(:);
    simulation_data.F_elastic_x = trajectory_results.F_elastic_x(:);
    simulation_data.F_elastic_y = trajectory_results.F_elastic_y(:);
    simulation_data.belt_stretch_x = trajectory_results.belt_stretch_x(:);
    simulation_data.belt_stretch_y = trajectory_results.belt_stretch_y(:);

    % === TRAJECTORY ERROR ===
    simulation_data.error_x = trajectory_results.error_x(:);
    simulation_data.error_y = trajectory_results.error_y(:);
    simulation_data.error_magnitude = trajectory_results.error_magnitude(:);
    simulation_data.error_direction = trajectory_results.error_direction(:);

    % === G-CODE FEATURES ===
    simulation_data.is_extruding = trajectory_data.is_extruding(:);
    simulation_data.is_travel = trajectory_data.is_travel(:);
    simulation_data.is_corner = trajectory_data.is_corner(:);
    simulation_data.corner_angle = trajectory_data.corner_angle(:);
    simulation_data.curvature = trajectory_data.curvature(:);
    simulation_data.layer_num = trajectory_data.layer_num(:);
    simulation_data.dist_from_last_corner = trajectory_data.dist_from_last_corner(:);

    % === THERMAL ===
    simulation_data.T_nozzle = thermal_results.T_nozzle_history(:);
    simulation_data.T_interface = thermal_results.T_interface(:);
    simulation_data.T_surface = thermal_results.T_surface(:);
    simulation_data.cooling_rate = thermal_results.cooling_rate(:);
    simulation_data.temp_gradient_z = thermal_results.temp_gradient_z(:);
    simulation_data.interlayer_time = thermal_results.interlayer_time(:);

    % === ADHESION STRENGTH ===
    simulation_data.adhesion_ratio = thermal_results.adhesion_ratio(:);

    % === QUALITY METRICS (Implicit Parameters) ===
    simulation_data.internal_stress = quality_data.internal_stress(:);
    simulation_data.porosity = quality_data.porosity(:);
    simulation_data.dimensional_accuracy = quality_data.dimensional_accuracy(:);
    simulation_data.quality_score = quality_data.quality_score(:);

    % === SYSTEM PARAMETERS ===
    simulation_data.params = params;

    % === GPU INFO ===
    simulation_data.gpu_info = gpu_info;

    fprintf('  Combined dataset: %d time points\n', n_points);
    fprintf('  Number of variables: %d\n', length(fieldnames(simulation_data)));
    fprintf('\n');

    %% Step 7: Save to file
    fprintf('STEP 7: Saving simulation results...\n');
    save(output_file, 'simulation_data', '-v7.3');
    fprintf('  Saved to: %s\n', output_file);
    fprintf('\n');

    %% Step 7: Summary statistics
    fprintf('============================================================\n');
    fprintf('SIMULATION SUMMARY\n');
    fprintf('============================================================\n');
    fprintf('\n');

    fprintf('GPU ACCELERATION:\n');
    if gpu_info.use_gpu
        fprintf('  Used GPU: %s\n', gpu_info.name);
        fprintf('  GPU Memory: %.2f GB free\n', gpu_info.memory);
    else
        fprintf('  Used CPU (GPU not available or not beneficial)\n');
    end
    fprintf('\n');

    fprintf('TRAJECTORY STATISTICS:\n');
    fprintf('  Total print time: %.2f s (%.2f min)\n', max(t), max(t)/60);
    fprintf('  Number of layers: %d\n', max(trajectory_data.layer_num));
    fprintf('\n');

    fprintf('KINEMATICS:\n');
    fprintf('  Max velocity: %.2f mm/s\n', max(simulation_data.v_mag_ref));
    fprintf('  Max acceleration: %.2f mm/s²\n', max(simulation_data.a_mag_ref));
    fprintf('  Max jerk: %.2f mm/s³\n', max(simulation_data.jerk_mag));
    fprintf('\n');

    fprintf('TRAJECTORY ERROR:\n');
    fprintf('  Max error magnitude: %.3f mm\n', max(simulation_data.error_magnitude));
    fprintf('  RMS error magnitude: %.3f mm\n', rms(simulation_data.error_magnitude));
    fprintf('  Mean error magnitude: %.3f mm\n', mean(simulation_data.error_magnitude));
    fprintf('  Max corner error: %.3f mm\n', ...
            max(simulation_data.error_magnitude(simulation_data.is_corner)));
    fprintf('\n');

    fprintf('THERMAL:\n');
    fprintf('  Max nozzle temp: %.2f °C\n', max(simulation_data.T_nozzle));
    fprintf('  Mean interface temp: %.2f °C\n', ...
            mean(simulation_data.T_interface(simulation_data.is_extruding)));
    fprintf('  Max cooling rate: %.2f °C/s\n', ...
            abs(min(simulation_data.cooling_rate)));
    fprintf('\n');

    fprintf('ADHESION:\n');
    valid_adhesion = simulation_data.adhesion_ratio > 0;
    fprintf('  Mean adhesion ratio: %.2f\n', ...
            mean(simulation_data.adhesion_ratio(valid_adhesion)));
    fprintf('  Min adhesion ratio: %.2f\n', ...
            min(simulation_data.adhesion_ratio(valid_adhesion)));
    fprintf('\n');

    fprintf('QUALITY METRICS:\n');
    fprintf('  Internal stress: %.2f ± %.2f MPa\n', ...
            mean(simulation_data.internal_stress), std(simulation_data.internal_stress));
    fprintf('  Porosity: %.2f ± %.2f %%\n', ...
            mean(simulation_data.porosity), std(simulation_data.porosity));
    fprintf('  Dimensional error: %.3f ± %.3f mm\n', ...
            mean(simulation_data.dimensional_accuracy), std(simulation_data.dimensional_accuracy));
    fprintf('  Quality score: %.3f ± %.3f\n', ...
            mean(simulation_data.quality_score), std(simulation_data.quality_score));
    fprintf('\n');

    fprintf('============================================================\n');
    fprintf('SIMULATION COMPLETE!\n');
    fprintf('============================================================\n');
    fprintf('\n');

    %% Optional: Create summary plots
    if params.debug.verbose
        create_summary_plots(simulation_data, output_file);
    end

end

%% Helper function: Create summary plots
function create_summary_plots(data, output_file)
    fprintf('Creating summary plots...\n');

    fig = figure('Name', 'Simulation Summary (GPU)', 'Position', [50, 50, 1600, 1000]);

    % Trajectory error
    subplot(3, 4, 1);
    plot(data.time, data.error_magnitude, 'r-', 'LineWidth', 1.5);
    grid on;
    xlabel('Time (s)');
    ylabel('Error (mm)');
    title('Trajectory Error Magnitude');

    % Error vector components
    subplot(3, 4, 2);
    plot(data.time, data.error_x, 'b-', 'LineWidth', 1); hold on;
    plot(data.time, data.error_y, 'r-', 'LineWidth', 1);
    grid on;
    xlabel('Time (s)');
    ylabel('Error (mm)');
    title('Error Components (X, Y)');
    legend('X', 'Y');

    % Velocity
    subplot(3, 4, 3);
    plot(data.time, data.v_mag_ref, 'b-', 'LineWidth', 1.5);
    grid on;
    xlabel('Time (s)');
    ylabel('Velocity (mm/s)');
    title('Velocity');

    % Acceleration
    subplot(3, 4, 4);
    plot(data.time, data.a_mag_ref, 'r-', 'LineWidth', 1.5);
    grid on;
    xlabel('Time (s)');
    ylabel('Acceleration (mm/s²)');
    title('Acceleration');

    % Jerk
    subplot(3, 4, 5);
    plot(data.time, data.jerk_mag, 'm-', 'LineWidth', 1.5);
    grid on;
    xlabel('Time (s)');
    ylabel('Jerk (mm/s³)');
    title('Jerk');

    % Inertial forces
    subplot(3, 4, 6);
    plot(data.time, data.F_inertia_x, 'b-', 'LineWidth', 1); hold on;
    plot(data.time, data.F_inertia_y, 'r-', 'LineWidth', 1);
    grid on;
    xlabel('Time (s)');
    ylabel('Force (N)');
    title('Inertial Forces');
    legend('X', 'Y');

    % Temperature
    subplot(3, 4, 7);
    plot(data.time, data.T_nozzle, 'r-', 'LineWidth', 1.5); hold on;
    plot(data.time, data.T_interface, 'b-', 'LineWidth', 1);
    yline(data.params.material.glass_transition, 'g--', 'T_g', 'LineWidth', 1);
    grid on;
    xlabel('Time (s)');
    ylabel('Temperature (°C)');
    title('Temperature');
    legend('Nozzle', 'Interface', 'Location', 'best');

    % Cooling rate
    subplot(3, 4, 8);
    plot(data.time, data.cooling_rate, 'b-', 'LineWidth', 1);
    grid on;
    xlabel('Time (s)');
    ylabel('Cooling Rate (°C/s)');
    title('Cooling Rate');

    % Adhesion ratio
    subplot(3, 4, 9);
    valid = data.adhesion_ratio > 0;
    plot(data.time(valid), data.adhesion_ratio(valid), 'g.-', 'LineWidth', 1);
    grid on;
    xlabel('Time (s)');
    ylabel('Adhesion Ratio');
    title('Interlayer Adhesion');
    ylim([0, 1]);

    % Corner angles
    subplot(3, 4, 10);
    corners = data.is_corner;
    plot(data.time(corners), data.corner_angle(corners), 'ro', 'MarkerSize', 4);
    grid on;
    xlabel('Time (s)');
    ylabel('Angle (deg)');
    title('Corner Angles');

    % Layer progression
    subplot(3, 4, 11);
    plot(data.time, data.layer_num, 'b-', 'LineWidth', 1.5);
    grid on;
    xlabel('Time (s)');
    ylabel('Layer Number');
    title('Layer Progression');

    % Reference vs Actual trajectory (X-Y view)
    subplot(3, 4, 12);
    plot(data.x_ref, data.y_ref, 'b--', 'LineWidth', 1.5); hold on;
    plot(data.x_act, data.y_act, 'r-', 'LineWidth', 0.5);
    plot(data.x_ref(corners), data.y_ref(corners), 'ko', 'MarkerSize', 3, 'MarkerFaceColor', 'y');
    axis equal;
    grid on;
    xlabel('X (mm)');
    ylabel('Y (mm)');
    title('Reference vs Actual Trajectory');
    legend('Reference', 'Actual', 'Corners', 'Location', 'best');

    % Save figure
    [filepath, name, ~] = fileparts(output_file);
    fig_file = fullfile(filepath, [name, '_summary.png']);
    saveas(fig, fig_file);
    fprintf('  Summary plots saved to: %s\n', fig_file);

end
