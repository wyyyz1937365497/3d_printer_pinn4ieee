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
# The original parse_gcode.m is preserved for backward compatibility.
#
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
                x_new = isnan(new_x) ? state.x : new_x;
                y_new = isnan(new_y) ? state.y : new_y;
                z_new = isnan(new_z) ? state.z : new_z;

                dx = x_new - state.x;
                dy = y_new - state.y;
                dz = z_new - state.z;

                segment_length = sqrt(dx^2 + dy^2 + dz^2);

                % Only record if there's significant movement and extrusion
                if segment_length >= options.min_segment && is_extrusion_move
                    point_idx = point_idx + 1;
                    current_segment = current_segment + 1;

                    % Calculate time
                    velocity = (isnan(new_f) ? state.f : new_f) / 60;  % mm/s
                    segment_time = segment_length / velocity;
                    current_time = current_time + segment_time;

                    % Store data
                    data.x(point_idx) = x_new;
                    data.y(point_idx) = y_new;
                    data.z(point_idx) = z_new;
                    data.e(point_idx) = new_e;
                    data.f(point_idx) = isnan(new_f) ? state.f : new_f;
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
                    state.f = isnan(new_f) ? state.f : new_f;
                end
            end
        end
    end

    %% Trim arrays to actual size
    n_points = point_idx;

    if n_points == 0
        error('No trajectory points found. Check G-code file and options.');
    end

    fprintf('  Extracted %d trajectory points\n', n_points);

    % Trim all arrays
    fields_to_trim = {'x', 'y', 'z', 'e', 'f', 'is_extruding', 'layer_num', ...
                     'segment_idx', 'time'};
    for i = 1:length(fields_to_trim)
        field = fields_to_trim{i};
        data.(field) = data.(field)(1:n_points);
    end
    data.print_type = data.print_type(1:n_points);

    %% Calculate kinematic derivatives
    fprintf('  Calculating kinematic derivatives...\n');

    % Position differences
    data.dx = zeros(n_points, 1);
    data.dy = zeros(n_points, 1);
    data.dz = zeros(n_points, 1);
    data.dt = zeros(n_points, 1);

    data.dx(2:end) = diff(data.x);
    data.dy(2:end) = diff(data.y);
    data.dz(2:end) = diff(data.z);
    data.dt(2:end) = diff(data.time);

    % Velocity
    valid_dt = data.dt > 0;
    data.vx = zeros(n_points, 1);
    data.vy = zeros(n_points, 1);
    data.vz = zeros(n_points, 1);
    data.v_mag = zeros(n_points, 1);

    data.vx(valid_dt) = data.dx(valid_dt) ./ data.dt(valid_dt);
    data.vy(valid_dt) = data.dy(valid_dt) ./ data.dt(valid_dt);
    data.vz(valid_dt) = data.dz(valid_dt) ./ data.dt(valid_dt);
    data.v_mag(valid_dt) = sqrt(data.vx(valid_dt).^2 + data.vy(valid_dt).^2 + data.vz(valid_dt).^2);

    % Acceleration
    data.ax = zeros(n_points, 1);
    data.ay = zeros(n_points, 1);
    data.az = zeros(n_points, 1);
    data.a_mag = zeros(n_points, 1);

    valid_accel = valid_dt(2:end) & data.dt(1:end-1) > 0;
    data.ax(2:end) = diff(data.vx) ./ data.dt(2:end);
    data.ay(2:end) = diff(data.vy) ./ data.dt(2:end);
    data.az(2:end) = diff(data.vz) ./ data.dt(2:end);
    data.a_mag(2:end) = sqrt(data.ax(2:end).^2 + data.ay(2:end).^2 + data.az(2:end).^2);

    % Jerk
    data.jx = zeros(n_points, 1);
    data.jy = zeros(n_points, 1);
    data.jz = zeros(n_points, 1);
    data.jerk = zeros(n_points, 1);

    valid_jerk = valid_accel(2:end) & valid_dt(3:end);
    data.jx(3:end) = diff(data.ax) ./ data.dt(3:end);
    data.jy(3:end) = diff(data.ay) ./ data.dt(3:end);
    data.jz(3:end) = diff(data.az) ./ data.dt(3:end);
    data.jerk(3:end) = sqrt(data.jx(3:end).^2 + data.jy(3:end).^2 + data.jz(3:end).^2);

    %% Detect corners
    fprintf('  Detecting corners...\n');

    data.is_corner = false(n_points, 1);
    data.corner_angle = zeros(n_points, 1);
    data.curvature = zeros(n_points, 1);

    for i = 3:n_points
        % Use three points to detect corners
        p1 = [data.x(i-2), data.y(i-2)];
        p2 = [data.x(i-1), data.y(i-1)];
        p3 = [data.x(i),   data.y(i)];

        % Vectors
        v1 = p2 - p1;
        v2 = p3 - p2;

        % Skip if segments are too short
        if norm(v1) < 1e-6 || norm(v2) < 1e-6
            continue;
        end

        % Calculate angle
        cos_angle = dot(v1, v2) / (norm(v1) * norm(v2));
        cos_angle = max(-1, min(1, cos_angle));
        angle_rad = acos(cos_angle);
        angle_deg = rad2deg(angle_rad);

        data.corner_angle(i) = angle_deg;

        % Mark as corner
        if angle_deg > params.gcode.corner_angle_threshold
            data.is_corner(i) = true;
        end

        % Calculate curvature
        segment_length = norm(v2);
        data.curvature(i) = angle_rad / (segment_length + 1e-6);
    end

    n_corners = sum(data.is_corner);
    fprintf('  Detected %d corners\n', n_corners);

    %% Movement direction
    data.direction_angle = atan2(data.dy, data.dx);  % radians

    %% Distance from last corner
    data.dist_from_last_corner = inf(n_points, 1);
    last_corner_idx = 0;
    for i = 1:n_points
        if data.is_corner(i)
            last_corner_idx = i;
        end
        if last_corner_idx > 0
            data.dist_from_last_corner(i) = sqrt(...
                (data.x(i) - data.x(last_corner_idx))^2 + ...
                (data.y(i) - data.y(last_corner_idx))^2);
        end
    end

    %% Create output structure (compatible with original parser)
    trajectory_data = [];

    % Time
    trajectory_data.time = data.time;

    % Position
    trajectory_data.x = data.x;
    trajectory_data.y = data.y;
    trajectory_data.z = data.z;

    % Extrusion
    trajectory_data.e = data.e;

    % Flags (for compatibility)
    trajectory_data.is_extruding = data.is_extruding;
    trajectory_data.is_travel = ~data.is_extruding;
    trajectory_data.is_corner = data.is_corner;

    % Kinematics
    trajectory_data.vx = data.vx;
    trajectory_data.vy = data.vy;
    trajectory_data.vz = data.vz;
    trajectory_data.v_actual = data.v_mag;

    trajectory_data.ax = data.ax;
    trajectory_data.ay = data.ay;
    trajectory_data.az = data.az;
    trajectory_data.acceleration = data.a_mag;

    trajectory_data.jx = data.jx;
    trajectory_data.jy = data.jy;
    trajectory_data.jz = data.jz;
    trajectory_data.jerk = data.jerk;

    % Features
    trajectory_data.corner_angle = data.corner_angle;
    trajectory_data.curvature = data.curvature;
    trajectory_data.layer_num = data.layer_num;
    trajectory_data.dist_from_last_corner = data.dist_from_last_corner;

    % Feed rate
    trajectory_data.f = data.f;

    % Additional fields specific to improved parser
    trajectory_data.print_type = data.print_type;
    trajectory_data.segment_idx = data.segment_idx;
    trajectory_data.dx = data.dx;
    trajectory_data.dy = data.dy;
    trajectory_data.dz = data.dz;

    %% Print summary
    fprintf('  G-code parsing complete!\n');
    fprintf('\n');
    fprintf('  Summary:\n');
    fprintf('    Time range: %.2f - %.2f s\n', min(data.time), max(data.time));
    fprintf('    X range: %.2f - %.2f mm\n', min(data.x), max(data.x));
    fprintf('    Y range: %.2f - %.2f mm\n', min(data.y), max(data.y));
    fprintf('    Z range: %.2f - %.2f mm\n', min(data.z), max(data.z));
    fprintf('    Layers extracted: %d\n', length(unique(data.layer_num)));
    fprintf('    Total points: %d\n', n_points);
    fprintf('    Corners: %d\n', n_corners);
    fprintf('    Max velocity: %.2f mm/s\n', max(data.v_mag));
    fprintf('    Max acceleration: %.2f mm/s²\n', max(data.a_mag));
    fprintf('    Max jerk: %.2f mm/s³\n', max(data.jerk));
    fprintf('\n');

end

%% Helper function: Parse G1 line
function [x, y, z, e, f] = parse_g1_line(line, state)
    % Parse a G0/G1 line and extract coordinates
    x = NaN;
    y = NaN;
    z = NaN;
    e = NaN;
    f = NaN;

    % Tokenize
    tokens = regexp(line, '[A-Z][-+]?[0-9.]*', 'match');

    for i = 1:length(tokens)
        token = tokens{i};
        cmd = token(1);
        val = str2double(token(2:end));

        if isnan(val)
            continue;
        end

        switch cmd
            case 'X'
                x = val;
            case 'Y'
                y = val;
            case 'Z'
                z = val;
            case 'E'
                e = val;
            case 'F'
                f = val;
        end
    end
end
