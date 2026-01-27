function trajectory_data = parse_gcode_improved(gcode_file, params, options)
% PARSE_GCODE_IMPROVED - Improved G-code parser for 2D trajectory extraction
%
% This parser is specifically designed to extract 2D printing trajectories
% from G-code files, handling:
% - Preparation moves (skirt, brim, travel)
% - Multi-layer prints
% - Different print types (perimeter, infill, etc.)
% - Only extrusion moves (filters out travel moves)
%
% The original parse_gcode.m is preserved for backward compatibility.
%
% Inputs:
%   gcode_file - Path to G-code file
%   params     - Physics parameters (from physics_parameters.m)
%   options    - Optional structure with fields:
%     .layers       - 'first', 'all', or specific layer number (default: 'first')
%     .include_skirt - Include skirt/brim (default: false)
%     .include_type - Cell array of types to include (default: {'Inner wall', 'Outer wall', ...})
%     .min_segment  - Minimum segment length in mm (default: 0.1)
%
% Output:
%   trajectory_data - Structure with 2D trajectory information
%
% Example:
%   % Extract first layer only
%   data = parse_gcode_improved('print.gcode', params, struct('layers', 'first'));
%
%   % Extract specific layers (1, 3, 5)
%   data = parse_gcode_improved('print.gcode', params, struct('layers', [1,3,5]));
%
%   % Extract all layers
%   data = parse_gcode_improved('print.gcode', params, struct('layers', 'all'));

    fprintf('Parsing G-code with improved parser: %s\n', gcode_file);

    %% Default options
    if nargin < 3
        options = struct();
    end

    if ~isfield(options, 'layers')
        options.layers = 'first';  % Default: first layer only
    end

    if ~isfield(options, 'include_skirt')
        options.include_skirt = false;  % Default: skip skirt
    end

    if ~isfield(options, 'include_type')
        % Default: include all actual printing types
        options.include_type = {'Inner wall', 'Outer wall', 'Sparse infill', ...
                                'Internal solid infill', 'Top surface', 'Bottom surface'};
    end

    if ~isfield(options, 'min_segment')
        options.min_segment = params.gcode.min_segment_length;
    end

    %% Read G-code file
    fid = fopen(gcode_file, 'r');
    if fid == -1
        error('Cannot open G-code file: %s', gcode_file);
    end

    gcode_lines = textscan(fid, '%s', 'Delimiter', '\n');
    gcode_lines = gcode_lines{1};
    fclose(fid);

    fprintf('  Total lines: %d\n', length(gcode_lines));

    %% Parse state
    state = struct();
    state.x = 0;
    state.y = 0;
    state.z = 0;
    state.e = 0;
    state.f = 1500;
    state.current_layer = 0;
    state.current_type = '';
    state.is_extruding = false;

    %% First pass: Find layer information
    fprintf('  Scanning for layer information...\n');

    layer_indices = [];
    for i = 1:length(gcode_lines)
        line = strtrim(gcode_lines{i});
        if startsWith(line, ';LAYER_CHANGE')
            layer_indices = [layer_indices; i];
        end
    end

    n_layers = length(layer_indices);
    fprintf('  Found %d layers\n', n_layers);

    %% Determine which layers to process
    if ischar(options.layers) || isstring(options.layers)
        switch options.layers
            case 'first'
                layers_to_process = 1;
            case 'all'
                layers_to_process = 1:n_layers;
            otherwise
                error('Unknown layer option: %s', options.layers);
        end
    elseif isnumeric(options.layers)
        layers_to_process = options.layers(options.layers <= n_layers & options.layers > 0);
        if isempty(layers_to_process)
            error('No valid layers found');
        end
    else
        error('Invalid layers specification');
    end

    fprintf('  Processing layers: %s\n', mat2str(layers_to_process));

    %% Second pass: Extract trajectory
    fprintf('  Extracting trajectory data...\n');

    % Pre-allocate storage
    max_points = 100000;  % Estimate
    data.x = zeros(max_points, 1);
    data.y = zeros(max_points, 1);
    data.z = zeros(max_points, 1);
    data.e = zeros(max_points, 1);
    data.f = zeros(max_points, 1);
    data.is_extruding = false(max_points, 1);
    data.print_type = cell(max_points, 1);
    data.layer_num = zeros(max_points, 1);
    data.segment_idx = zeros(max_points, 1);
    data.time = zeros(max_points, 1);

    point_idx = 0;
    current_segment = 0;
    current_time = 0;
    prev_e = 0;

    % Parse each line
    for i = 1:length(gcode_lines)
        line = strtrim(gcode_lines{i});

        % Skip empty lines and comments
        if isempty(line) || startsWith(line, ';') && ~startsWith(line, ';LAYER') && ~startsWith(line, ';TYPE')
            continue;
        end

        % Check for layer change
        if startsWith(line, ';LAYER_CHANGE')
            state.current_layer = state.current_layer + 1;

            % Check if we should process this layer
            if ~ismember(state.current_layer, layers_to_process)
                state.is_extruding = false;  % Stop processing until next target layer
                continue;
            end

            fprintf('    Processing layer %d at line %d\n', state.current_layer, i);
            state.is_extruding = true;
            continue;
        end

        % Check for type change
        if startsWith(line, ';TYPE:')
            type_name = strtrim(line(7:end));
            state.current_type = type_name;

            % Check if we should include this type
            if ~options.include_skirt && strcmp(type_name, 'Skirt')
                state.is_extruding = false;
            elseif ~ismember(type_name, options.include_type)
                state.is_extruding = false;
            elseif ismember(state.current_layer, layers_to_process)
                state.is_extruding = true;
            end
            continue;
        end

        % Skip if not in extrusion mode
        if ~state.is_extruding || ~ismember(state.current_layer, layers_to_process)
            % Still parse to update state
            if startsWith(line, 'G0') || startsWith(line, 'G1')
                [new_x, new_y, new_z, new_e, new_f] = parse_g1_line(line, state);
                if ~isnan(new_x), state.x = new_x; end
                if ~isnan(new_y), state.y = new_y; end
                if ~isnan(new_z), state.z = new_z; end
                if ~isnan(new_e), state.e = new_e; end
                if ~isnan(new_f), state.f = new_f; end
            end
            continue;
        end

        % Parse movement commands
        if startsWith(line, 'G0') || startsWith(line, 'G1')
            [new_x, new_y, new_z, new_e, new_f] = parse_g1_line(line, state);

            % Check if there's actual extrusion
            e_delta = new_e - state.e;
            is_extrusion_move = abs(e_delta) > 1e-6;

            % Calculate movement
            if ~isnan(new_x) || ~isnan(new_y)
                x_new = new_x;
                if isnan(new_x)
                    x_new = state.x;
                end
                y_new = new_y;
                if isnan(new_y)
                    y_new = state.y;
                end
                z_new = new_z;
                if isnan(new_z)
                    z_new = state.z;
                end

                dx = x_new - state.x;
                dy = y_new - state.y;
                dz = z_new - state.z;

                segment_length = sqrt(dx^2 + dy^2 + dz^2);

                % Only record if there's significant movement and extrusion
                if segment_length >= options.min_segment && is_extrusion_move
                    point_idx = point_idx + 1;
                    current_segment = current_segment + 1;

                    % Calculate time
                    if isnan(new_f)
                        velocity = state.f / 60;  % mm/s
                    else
                        velocity = new_f / 60;  % mm/s
                    end
                    segment_time = segment_length / velocity;
                    current_time = current_time + segment_time;

                    % Store data
                    data.x(point_idx) = x_new;
                    data.y(point_idx) = y_new;
                    data.z(point_idx) = z_new;
                    data.e(point_idx) = new_e;
                    if isnan(new_f)
                        data.f(point_idx) = state.f;
                    else
                        data.f(point_idx) = new_f;
                    end
                    data.is_extruding(point_idx) = true;
                    data.print_type{point_idx} = state.current_type;
                    data.layer_num(point_idx) = state.current_layer;
                    data.segment_idx(point_idx) = current_segment;
                    data.time(point_idx) = current_time;

                    % Update state
                    state.x = x_new;
                    state.y = y_new;
                    state.z = z_new;
                    state.e = new_e;
                    if isnan(new_f)
                        state.f = state.f;
                    else
                        state.f = new_f;
                    end
                end
            end
        end
    end

    %% Trim arrays to actual size
    n_points = point_idx;

    if n_points == 0
        error('No trajectory points found. Check G-code file and options.');
    end

    data.x = data.x(1:n_points);
    data.y = data.y(1:n_points);
    data.z = data.z(1:n_points);
    data.e = data.e(1:n_points);
    data.f = data.f(1:n_points);
    data.is_extruding = data.is_extruding(1:n_points);
    data.print_type = data.print_type(1:n_points);
    data.layer_num = data.layer_num(1:n_points);
    data.segment_idx = data.segment_idx(1:n_points);
    data.time = data.time(1:n_points);

    fprintf('  Extracted %d trajectory points\n', n_points);

    %% Calculate kinematics
    fprintf('  Calculating kinematics...\n');

    % Initialize kinematic variables
    vx = zeros(n_points, 1);
    vy = zeros(n_points, 1);
    vz = zeros(n_points, 1);
    ax = zeros(n_points, 1);
    ay = zeros(n_points, 1);
    az = zeros(n_points, 1);
    jx = zeros(n_points, 1);
    jy = zeros(n_points, 1);
    jz = zeros(n_points, 1);

    % Calculate velocities (using central differences for interior points)
    dt = [data.time(1); diff(data.time)];  % Time intervals
    dt(dt == 0) = NaN;  % Avoid division by zero

    % Velocities
    for i = 1:n_points
        if i == 1
            % Forward difference for first point
            if n_points > 1
                vx(i) = (data.x(2) - data.x(1)) / dt(2);
                vy(i) = (data.y(2) - data.y(1)) / dt(2);
                vz(i) = (data.z(2) - data.z(1)) / dt(2);
            else
                vx(i) = 0; vy(i) = 0; vz(i) = 0;
            end
        elseif i == n_points
            % Backward difference for last point
            vx(i) = (data.x(end) - data.x(end-1)) / dt(end);
            vy(i) = (data.y(end) - data.y(end-1)) / dt(end);
            vz(i) = (data.z(end) - data.z(end-1)) / dt(end);
        else
            % Central difference for interior points
            vx(i) = (data.x(i+1) - data.x(i-1)) / (dt(i+1) + dt(i));
            vy(i) = (data.y(i+1) - data.y(i-1)) / (dt(i+1) + dt(i));
            vz(i) = (data.z(i+1) - data.z(i-1)) / (dt(i+1) + dt(i));
        end
    end

    % Calculate accelerations
    for i = 1:n_points
        if i == 1
            % Forward difference for first point
            if n_points > 1
                ax(i) = (vx(2) - vx(1)) / dt(2);
                ay(i) = (vy(2) - vy(1)) / dt(2);
                az(i) = (vz(2) - vz(1)) / dt(2);
            else
                ax(i) = 0; ay(i) = 0; az(i) = 0;
            end
        elseif i == n_points
            % Backward difference for last point
            ax(i) = (vx(end) - vx(end-1)) / dt(end);
            ay(i) = (vy(end) - vy(end-1)) / dt(end);
            az(i) = (vz(end) - vz(end-1)) / dt(end);
        else
            % Central difference for interior points
            ax(i) = (vx(i+1) - vx(i-1)) / (dt(i+1) + dt(i));
            ay(i) = (vy(i+1) - vy(i-1)) / (dt(i+1) + dt(i));
            az(i) = (vz(i+1) - vz(i-1)) / (dt(i+1) + dt(i));
        end
    end

    % Calculate jerks
    for i = 1:n_points
        if i == 1
            % Forward difference for first point
            if n_points > 1
                jx(i) = (ax(2) - ax(1)) / dt(2);
                jy(i) = (ay(2) - ay(1)) / dt(2);
                jz(i) = (az(2) - az(1)) / dt(2);
            else
                jx(i) = 0; jy(i) = 0; jz(i) = 0;
            end
        elseif i == n_points
            % Backward difference for last point
            jx(i) = (ax(end) - ax(end-1)) / dt(end);
            jy(i) = (ay(end) - ay(end-1)) / dt(end);
            jz(i) = (az(end) - az(end-1)) / dt(end);
        else
            % Central difference for interior points
            jx(i) = (ax(i+1) - ax(i-1)) / (dt(i+1) + dt(i));
            jy(i) = (ay(i+1) - ay(i-1)) / (dt(i+1) + dt(i));
            jz(i) = (az(i+1) - az(i-1)) / (dt(i+1) + dt(i));
        end
    end

    % Calculate magnitudes
    v_actual = sqrt(vx.^2 + vy.^2 + vz.^2);
    acceleration = sqrt(ax.^2 + ay.^2 + az.^2);
    jerk = sqrt(jx.^2 + jy.^2 + jz.^2);

    fprintf('  Kinematics calculated\n');

    %% Detect corners and features
    fprintf('  Detecting corners and geometric features...\n');

    % Initialize feature arrays
    is_corner = false(n_points, 1);
    corner_angle = zeros(n_points, 1);
    curvature = zeros(n_points, 1);
    dist_from_last_corner = zeros(n_points, 1);

    % Calculate corner detection (based on angle change in velocity vector)
    vel_vectors = [vx, vy, vz];
    vel_magnitudes = sqrt(sum(vel_vectors.^2, 2));
    
    % Normalize velocity vectors
    valid_idx = vel_magnitudes > 1e-6;  % Avoid division by zero
    vel_unit = vel_vectors;
    vel_unit(valid_idx, :) = vel_unit(valid_idx, :) ./ vel_magnitudes(valid_idx, 1);

    % Detect corners based on velocity direction changes
    for i = 2:n_points-1
        if valid_idx(i-1) && valid_idx(i) && valid_idx(i+1)
            % Dot product of velocity directions before and after point i
            dot_product = dot(vel_unit(i-1, :), vel_unit(i+1, :));
            % Clamp to [-1, 1] to avoid numerical errors in acos
            dot_product = max(-1, min(1, dot_product));
            
            angle_change = acos(dot_product);  % in radians
            corner_angle_degrees = rad2deg(angle_change);
            
            % Mark as corner if angle change is significant (>30 degrees)
            if corner_angle_degrees > 30
                is_corner(i) = true;
                corner_angle(i) = corner_angle_degrees;
            end
        end
    end

    % Calculate curvature (how sharply the path turns at each point)
    for i = 2:n_points-1
        if valid_idx(i-1) && valid_idx(i) && valid_idx(i+1)
            % Cross product to determine turning direction
            cross_prod = cross(vel_unit(i-1, :), vel_unit(i+1, :));
            curvature(i) = norm(cross_prod);
        end
    end

    % Calculate distance from last corner
    last_corner_idx = 0;
    for i = 1:n_points
        if is_corner(i)
            last_corner_idx = i;
            dist_from_last_corner(i) = 0;
        elseif last_corner_idx > 0
            % Calculate cumulative distance since last corner
            dist_from_last_corner(i) = sum(sqrt(...
                diff(data.x(last_corner_idx:i)).^2 + ...
                diff(data.y(last_corner_idx:i)).^2 + ...
                diff(data.z(last_corner_idx:i)).^2));
        else
            % If no corner yet, calculate from beginning
            if i > 1
                dist_from_last_corner(i) = sum(sqrt(...
                    diff(data.x(1:i)).^2 + ...
                    diff(data.y(1:i)).^2 + ...
                    diff(data.z(1:i)).^2));
            end
        end
    end

    fprintf('  Features detected\n');

    %% Create output structure
    trajectory_data = struct();

    % Coordinates
    trajectory_data.x = data.x;
    trajectory_data.y = data.y;
    trajectory_data.z = data.z;

    % Kinematics
    trajectory_data.vx = vx;
    trajectory_data.vy = vy;
    trajectory_data.vz = vz;
    trajectory_data.v_actual = v_actual;
    trajectory_data.ax = ax;
    trajectory_data.ay = ay;
    trajectory_data.az = az;
    trajectory_data.acceleration = acceleration;
    trajectory_data.jx = jx;
    trajectory_data.jy = jy;
    trajectory_data.jz = jz;
    trajectory_data.jerk = jerk;

    % Features
    trajectory_data.is_extruding = data.is_extruding;
    trajectory_data.is_travel = ~data.is_extruding;
    trajectory_data.is_corner = is_corner;
    trajectory_data.corner_angle = corner_angle;
    trajectory_data.curvature = curvature;
    trajectory_data.layer_num = data.layer_num;
    trajectory_data.dist_from_last_corner = dist_from_last_corner;

    % Time and segments
    trajectory_data.time = data.time;
    trajectory_data.segment_idx = data.segment_idx;

    fprintf('  Trajectory extraction completed\n');
    fprintf('  Points: %d\n', n_points);
    fprintf('  Duration: %.2f s\n', data.time(end));
    fprintf('  Estimated print time: %.2f min\n', data.time(end)/60);

end

% Helper function to parse G1/G0 line
function [x, y, z, e, f] = parse_g1_line(line, state)
% PARSE_G1_LINE - Parse a G-code line to extract coordinates and feedrate
    x = NaN; y = NaN; z = NaN; e = NaN; f = NaN;

    % Match coordinate values using regular expressions
    x_match = regexp(line, 'X\s*([+-]?\d*\.?\d+)', 'match');
    y_match = regexp(line, 'Y\s*([+-]?\d*\.?\d+)', 'match');
    z_match = regexp(line, 'Z\s*([+-]?\d*\.?\d+)', 'match');
    e_match = regexp(line, 'E\s*([+-]?\d*\.?\d+)', 'match');
    f_match = regexp(line, 'F\s*([+-]?\d*\.?\d+)', 'match');

    if ~isempty(x_match)
        x = str2double(extract_numbers(x_match{1}));
    end
    if ~isempty(y_match)
        y = str2double(extract_numbers(y_match{1}));
    end
    if ~isempty(z_match)
        z = str2double(extract_numbers(z_match{1}));
    end
    if ~isempty(e_match)
        e = str2double(extract_numbers(e_match{1}));
    end
    if ~isempty(f_match)
        f = str2double(extract_numbers(f_match{1}));
    end
end

% Helper function to extract numbers from matched strings
function num_str = extract_numbers(match_str)
    % Extract numeric part from string like 'X123.45' or 'F1500'
    num_pos = regexp(match_str, '[0-9+\-\.]');
    if ~isempty(num_pos)
        start_pos = num_pos(1);
        num_str = match_str(start_pos:end);
    else
        num_str = '0';
    end
end
