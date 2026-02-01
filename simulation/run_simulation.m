function simulation_data = run_simulation(gcode_file, varargin)
% RUN_SIMULATION - Unified entry point for FDM 3D printer simulation
%
% This is the main simulation interface that replaces:
%   - run_full_simulation.m
%   - run_full_simulation_gpu.m
%   - regenerate_all_datasets.m
%   - test_firmware_effects*.m
%
% Features:
%   - Automatic GPU/CPU selection
%   - Parameter-based layer control
%   - Firmware effects simulation (optional)
%   - Unified output format
%
% Syntax:
%   data = run_simulation(gcode_file)
%   data = run_simulation(gcode_file, 'Name', Value, ...)
%
% Inputs:
%   gcode_file  - Path to G-code file (.gcode or .gco)
%
% Name-Value Pairs:
%   'OutputFile'     - Output .mat file path (default: auto-generated)
%   'Layers'         - Layer selection (default: 'first')
%                      Options: 'first', 'all', integer, or array [start, step, end]
%   'UseGPU'         - GPU selection (default: true)
%                      Options: true, false, or GPU ID (e.g., 1)
%   'TimeStep'       - Simulation time step in seconds (default: 0.01)
%   'IncludeSkirt'   - Include skirt/brim (default: false)
%   'IncludeType'    - Print types to include (default: {'Outer wall', 'Inner wall'})
%   'Verbose'        - Print progress messages (default: true)
%   'FirmwareEffects'- Enable firmware effects (default: false)
%
% Output:
%   simulation_data - Structure containing all simulation results
%
% Examples:
%   % Basic usage - simulate first layer with auto settings
%   data = run_simulation('print.gcode');
%
%   % Simulate specific layers (25, 50, 75)
%   data = run_simulation('print.gcode', 'Layers', [25, 25, 75]);
%
%   % Simulate layers 10 to 100 with step 5
%   data = run_simulation('print.gcode', 'Layers', [10, 5, 100]);
%
%   % Simulate all layers with custom output
%   data = run_simulation('print.gcode', 'Layers', 'all', ...
%                        'OutputFile', 'results/output.mat');
%
%   % Force CPU usage
%   data = run_simulation('print.gcode', 'UseGPU', false);
%
%   % High-resolution simulation with firmware effects
%   data = run_simulation('print.gcode', 'TimeStep', 0.005, ...
%                        'FirmwareEffects', true);

    %% Parse input arguments
    p = inputParser;
    addRequired(p, 'gcode_file', @ischar);
    addParameter(p, 'OutputFile', '', @ischar);
    addParameter(p, 'Layers', 'first', @(x) ischar(x) || isnumeric(x));
    addParameter(p, 'UseGPU', true, @(x) isnumeric(x) || islogical(x));
    addParameter(p, 'TimeStep', 0.01, @isscalar);
    addParameter(p, 'IncludeSkirt', false, @islogical);
    addParameter(p, 'IncludeType', {'Outer wall', 'Inner wall'}, @iscell);
    addParameter(p, 'Verbose', true, @islogical);
    addParameter(p, 'FirmwareEffects', false, @islogical);

    parse(p, gcode_file, varargin{:});
    opts = p.Results;

    %% Setup paths
    script_dir = fileparts(mfilename('fullpath'));
    addpath(script_dir);

    %% Validate G-code file
    if ~exist(gcode_file, 'file')
        error('G-code file not found: %s', gcode_file);
    end

    %% Generate output filename if not provided
    if isempty(opts.OutputFile)
        [filepath, name, ~] = fileparts(gcode_file);

        % Add layer info to filename
        if ischar(opts.Layers)
            layer_suffix = opts.Layers;
        elseif isnumeric(opts.Layers)
            if numel(opts.Layers) == 1
                layer_suffix = sprintf('layer%d', opts.Layers);
            elseif numel(opts.Layers) == 3
                layer_suffix = sprintf('layers%d-%d-step%d', opts.Layers(1), opts.Layers(3), opts.Layers(2));
            else
                layer_suffix = sprintf('layers_%d-%d', min(opts.Layers), max(opts.Layers));
            end
        end

        opts.OutputFile = fullfile(filepath, sprintf('%s_%s_simulation.mat', name, layer_suffix));
    end

    %% Print header
    if opts.Verbose
        fprintf('============================================================\n');
        fprintf('FDM 3D PRINTER SIMULATION (Unified Entry)\n');
        fprintf('============================================================\n');
        fprintf('\n');
        fprintf('Configuration:\n');
        fprintf('  G-code file: %s\n', gcode_file);
        fprintf('  Output file: %s\n', opts.OutputFile);

        if ischar(opts.Layers)
            fprintf('  Layers: %s\n', opts.Layers);
        elseif isnumeric(opts.Layers)
            if numel(opts.Layers) == 1
                fprintf('  Layers: %d (single layer)\n', opts.Layers);
            elseif numel(opts.Layers) == 3
                fprintf('  Layers: %d to %d (step %d)\n', opts.Layers(1), opts.Layers(3), opts.Layers(2));
            else
                fprintf('  Layers: [%s] (custom)\n', num2str(opts.Layers));
            end
        end

        fprintf('  Time step: %.3f ms (%.1f Hz)\n', opts.TimeStep * 1000, 1/opts.TimeStep);
        fprintf('  GPU: %s\n', mat2str(opts.UseGPU));
        fprintf('  Firmware effects: %s\n', mat2str(opts.FirmwareEffects));
        fprintf('\n');
    end

    %% Step 0: Setup GPU (if requested)
    if opts.Verbose
        fprintf('STEP 0: Initializing computation backend...\n');
    end

    gpu_info = struct('use_gpu', false, 'available', false);

    if opts.UseGPU
        try
            gpu_id = [];
            if isnumeric(opts.UseGPU) && opts.UseGPU > 0
                gpu_id = opts.UseGPU;
            end
            gpu_info = setup_gpu(gpu_id);
        catch ME
            if opts.Verbose
                fprintf('  Warning: GPU initialization failed (%s)\n', ME.message);
                fprintf('  Falling back to CPU simulation\n');
            end
        end
    end

    if opts.Verbose
        if gpu_info.use_gpu
            fprintf('  Using GPU: %s (%.2f GB free)\n', gpu_info.name, gpu_info.memory);
        else
            fprintf('  Using CPU\n');
        end
        fprintf('\n');
    end

    %% Step 1: Load physics parameters
    if opts.Verbose
        fprintf('STEP 1: Loading physics parameters...\n');
    end

    params = physics_parameters();
    params.simulation.time_step = opts.TimeStep;

    if opts.Verbose
        fprintf('\n');
    end

    %% Step 2: Parse G-code and extract trajectory
    if opts.Verbose
        fprintf('STEP 2: Parsing G-code file...\n');
    end

    % Build parser options
    parser_options = struct();
    parser_options.use_improved = true;
    parser_options.layers = opts.Layers;
    parser_options.include_skirt = opts.IncludeSkirt;
    parser_options.include_type = opts.IncludeType;

    trajectory_data = parse_gcode_improved(gcode_file, params, parser_options);

    if opts.Verbose
        fprintf('  Parsed %d time points\n', length(trajectory_data.time));
        fprintf('  Layers: %d to %d\n', min(trajectory_data.layer_num), max(trajectory_data.layer_num));
        fprintf('\n');
    end

    %% Step 3: Simulate thermal field
    if opts.Verbose
        fprintf('STEP 3: Simulating thermal field...\n');
    end

    thermal_results = simulate_thermal_field(trajectory_data, params);

    if opts.Verbose
        fprintf('  Max interface temp: %.1f°C\n', max(thermal_results.T_interface));
        fprintf('\n');
    end

    %% Step 4: Calculate quality metrics
    if opts.Verbose
        fprintf('STEP 4: Calculating quality metrics...\n');
    end

    quality_data = calculate_quality_metrics(trajectory_data, thermal_results, params);

    if opts.Verbose
        fprintf('  Average quality score: %.3f\n', mean(quality_data.quality_score));
        fprintf('\n');
    end

    %% Step 5: Simulate trajectory error
    if opts.Verbose
        fprintf('STEP 5: Simulating trajectory error...\n');
    end

    n_points = length(trajectory_data.time);

    % Choose simulation method
    if opts.FirmwareEffects
        % Use firmware effects model
        if opts.Verbose
            fprintf('  Using firmware effects model (junction deviation, resonance, etc.)\n');
        end
        trajectory_results = simulate_trajectory_error_with_firmware_effects(...
            trajectory_data, params, gpu_info);
    elseif gpu_info.use_gpu && n_points > 500
        % Use GPU acceleration
        if opts.Verbose
            fprintf('  Using GPU-accelerated dynamics simulation\n');
        end
        trajectory_results = simulate_trajectory_error_gpu(trajectory_data, params, gpu_info);
    else
        % Use CPU
        if opts.Verbose
            if gpu_info.available
                fprintf('  Using CPU simulation (dataset: %d points)\n', n_points);
            else
                fprintf('  Using CPU simulation\n');
            end
        end
        trajectory_results = simulate_trajectory_error(trajectory_data, params);
    end

    if opts.Verbose
        fprintf('  Max error: %.3f mm\n', max(trajectory_results.error_magnitude));
        fprintf('  Mean error: %.3f mm\n', mean(trajectory_results.error_magnitude));
        fprintf('\n');
    end

    %% Step 6: Combine all results
    if opts.Verbose
        fprintf('STEP 6: Combining results...\n');
    end

    simulation_data = combine_results(trajectory_data, trajectory_results, ...
                                      thermal_results, quality_data, gpu_info);

    if opts.Verbose
        fprintf('  Total variables: %d\n', length(fieldnames(simulation_data)));
        fprintf('  Time points: %d\n', length(simulation_data.time));
        fprintf('\n');
    end

    %% Step 7: Save to file
    if opts.Verbose
        fprintf('STEP 7: Saving simulation results...\n');
    end

    % Ensure output directory exists
    [output_dir, ~, ~] = fileparts(opts.OutputFile);
    if ~isempty(output_dir) && ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    save(opts.OutputFile, 'simulation_data', '-v7.3');

    if opts.Verbose
        fprintf('  Saved to: %s\n', opts.OutputFile);
        fprintf('\n');
    end

    %% Step 8: Print summary
    if opts.Verbose
        print_simulation_summary(simulation_data, gpu_info);
    end

end

%% Helper function: Combine all simulation results
function data = combine_results(trajectory, trajectory_err, thermal, quality, gpu_info)

    t = trajectory.time;

    data = [];

    % Time
    data.time = t(:);

    % Reference trajectory
    data.x_ref = trajectory.x(:);
    data.y_ref = trajectory.y(:);
    data.z_ref = trajectory.z(:);

    % Actual trajectory (with errors)
    data.x_act = trajectory_err.x_act(:);
    data.y_act = trajectory_err.y_act(:);
    data.z_act = trajectory_err.z_act(:);

    % Kinematics
    data.vx_ref = trajectory.vx(:);
    data.vy_ref = trajectory.vy(:);
    data.vz_ref = trajectory.vz(:);
    data.vx_act = trajectory_err.vx_act(:);
    data.vy_act = trajectory_err.vy_act(:);
    data.vz_act = trajectory_err.vz_act(:);
    data.v_mag_ref = trajectory.v_actual(:);

    data.ax_ref = trajectory.ax(:);
    data.ay_ref = trajectory.ay(:);
    data.az_ref = trajectory.az(:);
    data.ax_act = trajectory_err.ax_act(:);
    data.ay_act = trajectory_err.ay_act(:);
    data.az_act = trajectory_err.az_act(:);
    data.a_mag_ref = trajectory.acceleration(:);

    data.jx_ref = trajectory.jx(:);
    data.jy_ref = trajectory.jy(:);
    data.jz_ref = trajectory.jz(:);
    data.jerk_mag = trajectory.jerk(:);

    % Dynamics
    data.F_inertia_x = trajectory_err.F_inertia_x(:);
    data.F_inertia_y = trajectory_err.F_inertia_y(:);
    data.F_elastic_x = trajectory_err.F_elastic_x(:);
    data.F_elastic_y = trajectory_err.F_elastic_y(:);
    data.belt_stretch_x = trajectory_err.belt_stretch_x(:);
    data.belt_stretch_y = trajectory_err.belt_stretch_y(:);

    % Errors
    data.error_x = trajectory_err.error_x(:);
    data.error_y = trajectory_err.error_y(:);
    data.error_magnitude = trajectory_err.error_magnitude(:);
    data.error_direction = trajectory_err.error_direction(:);

    % G-code features
    data.is_extruding = trajectory.is_extruding(:);
    data.is_travel = trajectory.is_travel(:);
    data.is_corner = trajectory.is_corner(:);
    data.corner_angle = trajectory.corner_angle(:);
    data.curvature = trajectory.curvature(:);
    data.layer_num = trajectory.layer_num(:);
    data.dist_from_last_corner = trajectory.dist_from_last_corner(:);

    % Thermal
    data.T_nozzle = thermal.T_nozzle_history(:);
    data.T_interface = thermal.T_interface(:);
    data.T_surface = thermal.T_surface(:);
    data.cooling_rate = thermal.cooling_rate(:);
    data.temp_gradient_z = thermal.temp_gradient_z(:);
    data.interlayer_time = thermal.interlayer_time(:);

    % Adhesion
    data.adhesion_ratio = thermal.adhesion_ratio(:);

    % Quality metrics
    data.internal_stress = quality.internal_stress(:);
    data.porosity = quality.porosity(:);
    data.dimensional_accuracy = quality.dimensional_accuracy(:);
    data.quality_score = quality.quality_score(:);

    % System info
    data.params = trajectory.params;
    data.gpu_info = gpu_info;

end

%% Helper function: Print simulation summary
function print_simulation_summary(data, gpu_info)

    fprintf('============================================================\n');
    fprintf('SIMULATION SUMMARY\n');
    fprintf('============================================================\n');
    fprintf('\n');

    % Computation info
    fprintf('COMPUTATION:\n');
    if gpu_info.use_gpu
        fprintf('  Backend: GPU (%s)\n', gpu_info.name);
        fprintf('  GPU Memory: %.2f GB free\n', gpu_info.memory);
    else
        fprintf('  Backend: CPU\n');
    end
    fprintf('\n');

    % Trajectory statistics
    fprintf('TRAJECTORY:\n');
    fprintf('  Print time: %.2f s (%.2f min)\n', max(data.time), max(data.time)/60);
    fprintf('  Layers: %d\n', max(data.layer_num));
    fprintf('  Time points: %d\n', length(data.time));
    fprintf('\n');

    % Kinematics
    fprintf('KINEMATICS:\n');
    fprintf('  Max velocity: %.1f mm/s\n', max(data.v_mag_ref));
    fprintf('  Max acceleration: %.1f mm/s²\n', max(data.a_mag_ref));
    fprintf('  Max jerk: %.1f mm/s³\n', max(data.jerk_mag));
    fprintf('\n');

    % Errors
    fprintf('TRAJECTORY ERROR:\n');
    fprintf('  Max: %.3f mm\n', max(data.error_magnitude));
    fprintf('  Mean: %.3f mm\n', mean(data.error_magnitude));
    fprintf('  RMS: %.3f mm\n', rms(data.error_magnitude));
    fprintf('\n');

    % Thermal
    fprintf('THERMAL:\n');
    fprintf('  Max nozzle temp: %.1f°C\n', max(data.T_nozzle));
    fprintf('  Mean interface temp: %.1f°C\n', ...
            mean(data.T_interface(data.is_extruding)));
    fprintf('  Max cooling rate: %.1f°C/s\n', abs(min(data.cooling_rate)));
    fprintf('\n');

    % Quality
    fprintf('QUALITY:\n');
    fprintf('  Internal stress: %.2f ± %.2f MPa\n', ...
            mean(data.internal_stress), std(data.internal_stress));
    fprintf('  Porosity: %.2f ± %.2f %%\n', ...
            mean(data.porosity), std(data.porosity));
    fprintf('  Quality score: %.3f ± %.3f\n', ...
            mean(data.quality_score), std(data.quality_score));
    fprintf('\n');

    fprintf('============================================================\n');
    fprintf('SIMULATION COMPLETE\n');
    fprintf('============================================================\n');
    fprintf('\n');

end
