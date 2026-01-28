% COLLEAT_DATA_GPU - GPU-accelerated data collection script
%
% This script uses GPU acceleration to speed up the MATLAB simulation.
% It uses cuda1 (GPU 1) by default to avoid interfering with
% training tasks on cuda0 (GPU 0).
%
% GPU Selection:
%   - gpu_id = 1: Use cuda1 (default)
%   - gpu_id = []: Auto-select GPU with most free memory
%   - gpu_id = 0: Use cuda0 (if available)
%
% Author: 3D Printer PINN Project
% Date: 2026-01-27

%% Setup
cd('F:\TJ\3d_print\3d_printer_pinn4ieee')
addpath(genpath(fullfile(pwd, 'matlab_simulation')))

%% GPU Configuration
% IMPORTANT: Set gpu_id to use cuda1 (GPU 1)
% This avoids interfering with training on cuda0 (GPU 0)
gpu_id = 1;  % Use cuda1 specifically

% Alternative: Auto-select GPU with most free memory
% gpu_id = [];

% Alternative: Use cuda0 (if you want to use the first GPU)
% gpu_id = 0;

%% Simulation Options
options = struct();
options.layers = 10;                         % Specify layer number (e.g., layer 10)
options.include_type = {'Outer wall'};       % Include only outer wall
options.include_skirt = false;                % Exclude skirt/brim
options.use_improved = true;                  % Use improved parser

%% File Configuration
gcode_file = 'Tremendous Hillar_PLA_17m1s.gcode';
output_file = 'layer10_outer_wall_simulation_gpu.mat';

%% Display Configuration
fprintf('\n');
fprintf('============================================================\n');
fprintf('GPU-ACCELERATED DATA COLLECTION\n');
fprintf('============================================================\n');
fprintf('\n');
fprintf('Configuration:\n');
fprintf('  G-code file: %s\n', gcode_file);
fprintf('  Output file: %s\n', output_file);
fprintf('  Layer: %d\n', options.layers);
fprintf('  Print type: %s\n', options.include_type{:});
fprintf('\n');

%% GPU Selection Information
fprintf('GPU Configuration:\n');
if gpu_id == 1
    fprintf('  Using: cuda1 (GPU 1) - Recommended for this setup\n');
    fprintf('  Reason: Avoids interfering with training on cuda0\n');
elseif gpu_id == 0
    fprintf('  Using: cuda0 (GPU 0)\n');
    fprintf('  WARNING: May interfere with training task\n');
elseif isempty(gpu_id)
    fprintf('  Using: Auto-select\n');
    fprintf('  Strategy: GPU with most free memory\n');
end
fprintf('\n');

%% Run GPU-Accelerated Simulation
fprintf('Starting simulation...\n');
fprintf('\n');

% Start timing
tic;

try
    % Run simulation with GPU acceleration
    simulation_data = run_full_simulation_gpu(...
        gcode_file, ...
        output_file, ...
        options, ...
        gpu_id);

    % Record time
    elapsed_time = toc;

    % Display results
    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('SIMULATION COMPLETED SUCCESSFULLY\n');
    fprintf('============================================================\n');
    fprintf('\n');
    fprintf('Results:\n');
    fprintf('  Total time: %.2f seconds (%.2f minutes)\n', elapsed_time, elapsed_time/60);
    fprintf('  Output file: %s\n', output_file);
    fprintf('  Data points: %d\n', length(simulation_data.time));
    fprintf('\n');

    % Display GPU usage
    if isfield(simulation_data, 'gpu_info')
        if simulation_data.gpu_info.use_gpu
            fprintf('GPU Acceleration:\n');
            fprintf('  Device: %s\n', simulation_data.gpu_info.name);
            fprintf('  Used GPU: Yes ✓\n');
        else
            fprintf('GPU Acceleration:\n');
            fprintf('  Used GPU: No (CPU mode)\n');
            fprintf('  Reason: %s\n', ...
                simulation_data.gpu_info.available ? ...
                'Dataset too small' : ...
                'GPU not available');
        end
        fprintf('\n');
    end

    % Display error statistics
    fprintf('Trajectory Error Statistics:\n');
    fprintf('  Max error: %.3f mm\n', max(simulation_data.error_magnitude));
    fprintf('  RMS error: %.3f mm\n', rms(simulation_data.error_magnitude));
    fprintf('  Mean error: %.3f mm\n', mean(simulation_data.error_magnitude));

    if sum(simulation_data.is_corner) > 0
        corner_errors = simulation_data.error_magnitude(simulation_data.is_corner);
        fprintf('  Max corner error: %.3f mm\n', max(corner_errors));
        fprintf('  Mean corner error: %.3f mm\n', mean(corner_errors));
    end
    fprintf('\n');

    %% Next Steps
    fprintf('Next Steps:\n');
    fprintf('  1. Review the generated plots\n');
    fprintf('  2. Convert to Python format:\n');
    fprintf('     python matlab_simulation/convert_matlab_to_python.py ...\n');
    fprintf('         "%s" training -o training_data\n', output_file);
    fprintf('  3. Use the data for PINN training\n');
    fprintf('\n');

catch ME
    % Handle errors
    elapsed_time = toc;
    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('ERROR: Simulation failed!\n');
    fprintf('============================================================\n');
    fprintf('\n');
    fprintf('Error message:\n');
    fprintf('  %s\n', ME.message);
    fprintf('\n');
    fprintf('Time elapsed before error: %.2f seconds\n', elapsed_time);
    fprintf('\n');

    % Provide troubleshooting tips
    fprintf('Troubleshooting:\n');
    fprintf('  1. Check if GPU is available: gpuDeviceCount\n');
    fprintf('  2. Try CPU version: colleat_data.m\n');
    fprintf('  3. Try different GPU: change gpu_id in this script\n');
    fprintf('  4. Reduce data size: simulate fewer layers\n');
    fprintf('\n');

    % Re-throw error for debugging
    rethrow(ME);
end

%% Optional: Convert to Python format immediately
% Uncomment the following lines to automatically convert after simulation

% fprintf('Converting to Python format...\n');
% system(sprintf('python matlab_simulation/convert_matlab_to_python.py "%s" training -o training_data', output_file));
% fprintf('Conversion complete!\n');
