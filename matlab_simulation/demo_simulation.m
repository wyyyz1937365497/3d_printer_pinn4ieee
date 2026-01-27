% DEMO_SIMULATION - Quick demonstration of the simulation system
%
% This script demonstrates the complete workflow:
% 1. Load parameters
% 2. Parse G-code
% 3. Run simulations
% 4. Visualize results
%
% Author: 3D Printer PINN Project
% Date: 2026-01-27

clear; clc;

fprintf('============================================================\n');
fprintf('3D PRINTER SIMULATION DEMO\n');
fprintf('============================================================\n');
fprintf('\n');

%% Configuration
% Add matlab_simulation to path
addpath('matlab_simulation');

% G-code file
gcode_file = 'Tremendous Hillar_PLA_17m1s.gcode';

% Check if file exists
if ~exist(gcode_file, 'file')
    error('G-code file not found: %s\nPlease ensure the file is in the current directory.', gcode_file);
end

%% Step 1: Load physics parameters
fprintf('STEP 1: Loading physics parameters...\n');
params = physics_parameters();

% Display key parameters
fprintf('\nKey parameters:\n');
fprintf('  X-axis natural frequency: %.2f Hz\n', params.dynamics.x.natural_freq / (2*pi));
fprintf('  Y-axis natural frequency: %.2f Hz\n', params.dynamics.y.natural_freq / (2*pi));
fprintf('  PLA thermal diffusivity: %.2e m²/s\n', params.material.thermal_diffusivity);
fprintf('  Convection coefficient (fan): %d W/(m²·K)\n', params.heat_transfer.h_convection_with_fan);
fprintf('\n');

%% Step 2: Parse G-code
fprintf('STEP 2: Parsing G-code...\n');

% Choose parser: use improved parser for 2D trajectory extraction
use_improved_parser = true;  % Set to false to use original parser

if use_improved_parser
    % Use improved parser with options
    parser_opts = struct();
    parser_opts.use_improved = true;
    parser_opts.layers = 'first';  % Extract first layer only
    parser_opts.include_skirt = false;  % Skip skirt

    fprintf('  Using improved parser (2D trajectory extraction)\n');
    fprintf('  Options: layers=%s, include_skirt=%d\n', ...
            mat2str(parser_opts.layers), parser_opts.include_skirt);

    tic;
    trajectory_data = parse_gcode_improved(gcode_file, params, parser_opts);
    parse_time = toc;
else
    % Use original parser (full 3D parsing)
    fprintf('  Using original parser (full 3D)\n');
    tic;
    trajectory_data = parse_gcode(gcode_file, params);
    parse_time = toc;
end

fprintf('  Parse time: %.2f s\n', parse_time);
fprintf('\n');

%% Step 3: Simulate trajectory error
fprintf('STEP 3: Simulating trajectory error...\n');
tic;
trajectory_results = simulate_trajectory_error(trajectory_data, params);
traj_sim_time = toc;
fprintf('  Simulation time: %.2f s\n', traj_sim_time);
fprintf('\n');

%% Step 4: Simulate thermal field
fprintf('STEP 4: Simulating thermal field...\n');
tic;
thermal_results = simulate_thermal_field(trajectory_data, params);
thermal_sim_time = toc;
fprintf('  Simulation time: %.2f s\n', thermal_sim_time);
fprintf('\n');

%% Step 5: Visualize results
fprintf('STEP 5: Visualizing results...\n');

% Enable plotting
params.debug.plot_trajectory = true;
params.debug.plot_temperature = true;

% Create comparison plots
figure('Name', 'Simulation Results Summary', 'Position', [50, 50, 1400, 900]);

% Subplot 1: Trajectory comparison
subplot(2, 3, 1);
plot(trajectory_data.x, trajectory_data.y, 'b--', 'LineWidth', 1.5); hold on;
plot(trajectory_results.x_act, trajectory_results.y_act, 'r-', 'LineWidth', 0.5);
plot(trajectory_data.x(trajectory_data.is_corner), ...
     trajectory_data.y(trajectory_data.is_corner), ...
     'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'y');
axis equal;
grid on;
xlabel('X (mm)');
ylabel('Y (mm)');
title('Reference vs Actual Trajectory');
legend('Reference', 'Actual', 'Corners', 'Location', 'best');

% Subplot 2: Error magnitude
subplot(2, 3, 2);
plot(trajectory_data.time, trajectory_results.error_magnitude, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Error Magnitude (mm)');
title('Position Error Magnitude');
fprintf('  Max error: %.3f mm\n', max(trajectory_results.error_magnitude));

% Subplot 3: Error vectors at corners
subplot(2, 3, 3);
corner_mask = trajectory_data.is_corner;
quiver(trajectory_data.x(corner_mask), trajectory_data.y(corner_mask), ...
       trajectory_results.error_x(corner_mask), ...
       trajectory_results.error_y(corner_mask), ...
       'AutoScale', 'on', 'MaxHeadSize', 0.5);
hold on;
plot(trajectory_data.x, trajectory_data.y, 'k--', 'LineWidth', 0.5);
axis equal;
grid on;
xlabel('X (mm)');
ylabel('Y (mm)');
title('Error Vectors at Corners');

% Subplot 4: Temperature evolution
subplot(2, 3, 4);
plot(trajectory_data.time, thermal_results.T_nozzle, 'r-', 'LineWidth', 1.5); hold on;
plot(trajectory_data.time, thermal_results.T_interface, 'b-', 'LineWidth', 1);
yline(params.material.glass_transition, 'g--', 'T_g', 'LineWidth', 1.5);
yline(params.environment.ambient_temp, 'k--', 'Ambient', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Temperature (°C)');
title('Temperature Evolution');
legend('Nozzle', 'Interface', 'Location', 'best');
ylim([params.environment.ambient_temp - 10, params.printing.nozzle_temp + 10]);

% Subplot 5: Cooling rate
subplot(2, 3, 5);
plot(trajectory_data.time, thermal_results.cooling_rate, 'b-', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Cooling Rate (°C/s)');
title('Cooling Rate');

% Subplot 6: Adhesion strength
subplot(2, 3, 6);
valid_adhesion = thermal_results.adhesion_ratio > 0;
plot(trajectory_data.time(valid_adhesion), thermal_results.adhesion_ratio(valid_adhesion), 'g.-', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Adhesion Ratio');
title('Interlayer Adhesion Strength');
ylim([0, 1.1]);

% Save figure
saveas(gcf, 'simulation_demo_results.png');
fprintf('  Results saved to: simulation_demo_results.png\n');
fprintf('\n');

%% Step 6: Save results
fprintf('STEP 6: Saving simulation results...\n');

% Combine results into single structure
demo_data = struct();
demo_data.trajectory_data = trajectory_data;
demo_data.trajectory_results = trajectory_results;
demo_data.thermal_results = thermal_results;
demo_data.params = params;

% Save to .mat file
output_file = 'demo_simulation_results.mat';
save(output_file, 'demo_data', '-v7.3');
fprintf('  Results saved to: %s\n', output_file);
fprintf('\n');

%% Summary
fprintf('============================================================\n');
fprintf('SIMULATION SUMMARY\n');
fprintf('============================================================\n');
fprintf('\n');

fprintf('Timing:\n');
fprintf('  G-code parsing: %.2f s\n', parse_time);
fprintf('  Trajectory simulation: %.2f s\n', traj_sim_time);
fprintf('  Thermal simulation: %.2f s\n', thermal_sim_time);
fprintf('  Total: %.2f s\n', parse_time + traj_sim_time + thermal_sim_time);
fprintf('\n');

fprintf('Data Statistics:\n');
fprintf('  Time points: %d\n', length(trajectory_data.time));
fprintf('  Layers: %d\n', max(trajectory_data.layer_num));
fprintf('  Corners: %d\n', sum(trajectory_data.is_corner));
fprintf('\n');

fprintf('Trajectory Error:\n');
fprintf('  Max error: %.3f mm\n', max(trajectory_results.error_magnitude));
fprintf('  RMS error: %.3f mm\n', rms(trajectory_results.error_magnitude));
fprintf('  Mean error: %.3f mm\n', mean(trajectory_results.error_magnitude));
fprintf('\n');

fprintf('Thermal:\n');
fprintf('  Max nozzle temp: %.2f °C\n', max(thermal_results.T_nozzle));
fprintf('  Mean interface temp: %.2f °C\n', mean(thermal_results.T_interface));
fprintf('  Mean adhesion ratio: %.2f\n', mean(thermal_results.adhesion_ratio(thermal_results.adhesion_ratio > 0)));
fprintf('\n');

fprintf('============================================================\n');
fprintf('DEMO COMPLETE!\n');
fprintf('============================================================\n');
fprintf('\n');

fprintf('Next steps:\n');
fprintf('  1. View the saved figure: simulation_demo_results.png\n');
fprintf('  2. Convert to Python format:\n');
fprintf('     python matlab_simulation/convert_matlab_to_python.py %s training\n', output_file);
fprintf('  3. Use the data for PINN training\n');
fprintf('\n');
