function trajectory_data = reconstruct_trajectory(gcode_file, params, options)
% RECONSTRUCT_TRAJECTORY - Reconstruct print head motion from G-code
%
% This function parses G-code and reconstructs the actual print head motion
% by simulating Ender-3 V2's motion planning (jerk-limited trajectory).
%
% Inputs:
%   gcode_file - Path to G-code file
%   params     - Physics parameters structure
%   options    - Parser options
%     .layers          - Layer(s) to extract ('first', 'all', or [1,5,10])
%     .include_type    - Print types to include (e.g., {'Outer wall', 'Inner wall'})
%     .include_skirt   - Include skirt/brim (default: false)
%     .time_step       - Interpolation time step in seconds (default: 0.01)
%     .min_segment     - Minimum segment length to keep (default: 0.1 mm)
%
% Output:
%   trajectory_data - Structure containing time-series data
%
% Key features:
%   - Parses G-code waypoints
%   - Simulates motion planning with S-curve velocity profile
%   - Considers jerk, acceleration, and velocity limits
%   - Interpolates to dense time grid
%   - Calculates position, velocity, acceleration, jerk at each time step

    fprintf('Reconstructing print head trajectory from G-code...\n');
    fprintf('  File: %s\n', gcode_file);

    %% Default options
    if nargin < 3
        options = struct();
    end

    default_options = struct();
    default_options.layers = 'first';
    default_options.include_type = {'Outer wall', 'Inner wall', 'Skirt', 'Support'};
    default_options.include_skirt = false;
    default_options.time_step = 0.01;  % 10ms time step
    default_options.min_segment = 0.1;  % 0.1mm minimum segment

    options = set_default_options(options, default_options);

    %% Read G-code file
    fid = fopen(gcode_file, 'r');
    if fid == -1
        error('Cannot open G-code file: %s', gcode_file);
    end

    gcode_lines = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    gcode_lines = gcode_lines{1};

    fprintf('  Total lines: %d\n', length(gcode_lines));

    %% Parse G-code to waypoints
    fprintf('  Parsing G-code waypoints...\n');

    waypoints = parse_waypoints(gcode_lines, options);

    if isempty(waypoints)
        error('No waypoints extracted from G-code');
    end

    fprintf('  Extracted %d waypoints\n', length(waypoints.x));
    fprintf('  Layer numbers in waypoints: %s\n', mat2str(unique(waypoints.layer_num)));

    %% Filter by layer
    if ischar(options.layers)
        if strcmpi(options.layers, 'first')
            target_layers = 1;
        elseif strcmpi(options.layers, 'all')
            target_layers = unique(waypoints.layer_num);
        else
            target_layers = str2num(options.layers);
        end
    else
        target_layers = options.layers;
    end

    mask = ismember(waypoints.layer_num, target_layers);
    waypoints = filter_structure(waypoints, mask);

    if isempty(waypoints)
        error('No data for selected layers');
    end

    fprintf('  Selected layers: %s\n', mat2str(unique(waypoints.layer_num)));

    %% Reconstruct motion for each layer
    fprintf('  Reconstructing motion with jerk-limited trajectory...\n');

    trajectory_data = reconstruct_motion(waypoints, params, options);

    fprintf('  Reconstruction complete!\n');
    fprintf('    Total time points: %d\n', length(trajectory_data.time));
    fprintf('    Duration: %.2f s (%.2f min)\n', ...
            trajectory_data.time(end), trajectory_data.time(end)/60);
    fprintf('    Average sampling rate: %.2f Hz\n', ...
            1/mean(diff(trajectory_data.time)));

end

%% Helper: Set default options
function opts = set_default_options(opts, defaults)
    fields = fieldnames(defaults);
    for i = 1:length(fields)
        if ~isfield(opts, fields{i})
            opts.(fields{i}) = defaults.(fields{i});
        end
    end
end

%% Helper: Parse G-code to waypoints
function waypoints = parse_waypoints(gcode_lines, options)
    % Initialize state
    state = struct();
    state.x = NaN;
    state.y = NaN;
    state.z = NaN;
    state.e = 0;
    state.f = 1500;  % Default feedrate mm/min
    state.current_layer = 0;
    state.current_type = 'Unknown';
    state.is_extruding = false;

    % Pre-allocate
    max_waypoints = 50000;
    waypoints.x = zeros(max_waypoints, 1);
    waypoints.y = zeros(max_waypoints, 1);
    waypoints.z = zeros(max_waypoints, 1);
    waypoints.e = zeros(max_waypoints, 1);
    waypoints.f = zeros(max_waypoints, 1);
    waypoints.type = cell(max_waypoints, 1);
    waypoints.layer_num = zeros(max_waypoints, 1);
    waypoints.is_extruding = false(max_waypoints, 1);

    wp_idx = 0;

    % Parse each line
    for i = 1:length(gcode_lines)
        line = strtrim(gcode_lines{i});

        % Skip empty and comments (except layer/type markers)
        if isempty(line) || (startsWith(line, ';') && ~startsWith(line, ';LAYER') && ~startsWith(line, ';TYPE') && ~startsWith(line, ';HEIGHT'))
            continue;
        end

        % Check for layer change (format: ;LAYER_CHANGE)
        if startsWith(line, ';LAYER_CHANGE')
            % Layer change detected, but layer number is not specified
            % We'll increment when we see HEIGHT or Z movement
            continue;
        end

        % Check for Z height information (format: ;:5 for layer 25 at Z=5.0mm)
        if startsWith(line, ';:')
            % Extract everything after ;: to get the Z height
            height_str = line(3:end);  % Remove first 2 characters (';:')
            height_val = str2double(height_str);

            % Calculate layer number from Z height (assuming 0.2mm layer height)
            if ~isnan(height_val) && height_val > 0
                state.current_layer = round(height_val / 0.2);  % Assuming 0.2mm layer height
                fprintf('    Parsed Z=%s mm → Layer %d\n', height_str, state.current_layer);
            end
            continue;
        end

        % Check for type change
        if startsWith(line, ';TYPE:')
            % Extract everything after ;TYPE: (may contain spaces)
            type_str = strtrim(line(7:end));  % Remove ';TYPE:' prefix
            state.current_type = type_str;
            fprintf('    Type changed to: %s\n', state.current_type);
            continue;
        end

        % Check for skirt
        if contains(line, 'SKIRT')
            state.current_type = 'Skirt';
            continue;
        end

        % Parse G1/G0 commands
        if startsWith(line, 'G1') || startsWith(line, 'G0')
            is_G1 = startsWith(line, 'G1');

            % Extract coordinates
            new_x = NaN;
            new_y = NaN;
            new_z = NaN;
            new_e = NaN;
            new_f = NaN;

            % Parse X
            if contains(line, 'X')
                tokens = regexp(line, 'X([0-9.-]+)', 'tokens');
                if ~isempty(tokens)
                    new_x = str2double(tokens{1}{1});
                end
            end

            % Parse Y
            if contains(line, 'Y')
                tokens = regexp(line, 'Y([0-9.-]+)', 'tokens');
                if ~isempty(tokens)
                    new_y = str2double(tokens{1}{1});
                end
            end

            % Parse Z
            if contains(line, 'Z')
                tokens = regexp(line, 'Z([0-9.-]+)', 'tokens');
                if ~isempty(tokens)
                    new_z = str2double(tokens{1}{1});
                end
            end

            % Parse E (extrusion)
            if contains(line, 'E')
                tokens = regexp(line, 'E([0-9.-]+)', 'tokens');
                if ~isempty(tokens)
                    new_e = str2double(tokens{1}{1});
                end
            end

            % Parse F (feedrate)
            if contains(line, 'F')
                tokens = regexp(line, 'F([0-9.-]+)', 'tokens');
                if ~isempty(tokens)
                    new_f = str2double(tokens{1}{1});
                end
            end

            % Update state
            if ~isnan(new_x), state.x = new_x; end
            if ~isnan(new_y), state.y = new_y; end
            if ~isnan(new_z)
                state.z = new_z;
                % Update layer number from Z height (fallback)
                if new_z > 0
                    state.current_layer = round(new_z / 0.2);  % Assuming 0.2mm layer height
                end
            end

            % Check if extruding
            if ~isnan(new_e)
                state.is_extruding = (new_e > state.e);
                state.e = new_e;
            end

            if ~isnan(new_f), state.f = new_f; end

            % Record waypoint if movement and extrusion
            if ~isnan(state.x) && ~isnan(state.y)
                % Filter by type - Skirt
                if strcmp(state.current_type, 'Skirt')
                    if isfield(options, 'include_skirt') && ~options.include_skirt
                        continue;  % Skip skirt
                    end
                end

                % Filter by type - include_type list
                if isfield(options, 'include_type') && ~isempty(options.include_type)
                    if ~ismember(state.current_type, options.include_type)
                        continue;
                    end
                end

                wp_idx = wp_idx + 1;
                waypoints.x(wp_idx) = state.x;
                waypoints.y(wp_idx) = state.y;
                waypoints.z(wp_idx) = state.z;
                waypoints.e(wp_idx) = state.e;
                waypoints.f(wp_idx) = state.f;
                waypoints.type{wp_idx} = state.current_type;
                waypoints.layer_num(wp_idx) = state.current_layer;
                waypoints.is_extruding(wp_idx) = state.is_extruding;

                % Debug: record first waypoint of each layer
                if wp_idx == 1 || waypoints.layer_num(wp_idx) ~= waypoints.layer_num(wp_idx-1)
                    type_str = char(state.current_type);
                    fprintf('    Layer %d: First waypoint recorded at (%.2f, %.2f, Z=%.2f), Type=%s\n', ...
                            state.current_layer, state.x, state.y, state.z, type_str);
                end
            end
        end
    end

    % Trim unused pre-allocated space
    waypoints.x = waypoints.x(1:wp_idx);
    waypoints.y = waypoints.y(1:wp_idx);
    waypoints.z = waypoints.z(1:wp_idx);
    waypoints.e = waypoints.e(1:wp_idx);
    waypoints.f = waypoints.f(1:wp_idx);
    waypoints.type = waypoints.type(1:wp_idx);
    waypoints.layer_num = waypoints.layer_num(1:wp_idx);
    waypoints.is_extruding = waypoints.is_extruding(1:wp_idx);
end

%% Helper: Reconstruct motion with jerk-limited trajectory
function trajectory_data = reconstruct_motion(waypoints, params, options)
    fprintf('    Simulating motion planning...\n');

    % Get motion limits
    v_max = params.motion.max_velocity;       % mm/s
    a_max = params.motion.max_accel;          % mm/s²
    j_max = params.motion.max_jerk;          % mm/s³

    % Convert feedrate from mm/min to mm/s
    target_velocities = waypoints.f / 60;  % mm/s

    % Limit by max velocity
    target_velocities = min(target_velocities, v_max);

    % Initialize arrays
    n_segments = length(waypoints.x) - 1;

    % Pre-allocate for dense trajectory
    estimated_points = n_segments * 100;
    trajectory_data.x = zeros(estimated_points, 1);
    trajectory_data.y = zeros(estimated_points, 1);
    trajectory_data.z = zeros(estimated_points, 1);
    trajectory_data.vx = zeros(estimated_points, 1);
    trajectory_data.vy = zeros(estimated_points, 1);
    trajectory_data.vz = zeros(estimated_points, 1);
    trajectory_data.ax = zeros(estimated_points, 1);
    trajectory_data.ay = zeros(estimated_points, 1);
    trajectory_data.az = zeros(estimated_points, 1);
    trajectory_data.jx = zeros(estimated_points, 1);
    trajectory_data.jy = zeros(estimated_points, 1);
    trajectory_data.jz = zeros(estimated_points, 1);
    trajectory_data.time = zeros(estimated_points, 1);
    trajectory_data.is_extruding = false(estimated_points, 1);
    trajectory_data.print_type = cell(estimated_points, 1);
    trajectory_data.layer_num = zeros(estimated_points, 1);

    idx = 0;
    current_time = 0;
    current_layer = waypoints.layer_num(1);

    % Process each segment
    for i = 1:n_segments
        % Segment endpoints
        x0 = waypoints.x(i);
        y0 = waypoints.y(i);
        z0 = waypoints.z(i);
        x1 = waypoints.x(i+1);
        y1 = waypoints.y(i+1);
        z1 = waypoints.z(i+1);

        % Segment vector
        dx = x1 - x0;
        dy = y1 - y0;
        dz = z1 - z0;
        segment_length = sqrt(dx^2 + dy^2 + dz^2);

        if segment_length < options.min_segment
            continue;
        end

        % Target velocity
        v_target = target_velocities(i+1);

        % Check if extruding
        is_extruding = waypoints.is_extruding(i+1);
        print_type = waypoints.type{i+1};
        layer_num = waypoints.layer_num(i+1);

        % Simulate S-curve velocity profile for this segment
        [t_seg, x_seg, y_seg, z_seg, ...
         vx_seg, vy_seg, vz_seg, ...
         ax_seg, ay_seg, az_seg, ...
         jx_seg, jy_seg, jz_seg] = ...
            simulate_scurve_segment(x0, y0, z0, x1, y1, z1, ...
                                   v_target, a_max, j_max, ...
                                   options.time_step);

        % Store data
        n_points = length(t_seg);
        new_idx = idx + n_points;

        if new_idx > estimated_points
            % Expand arrays
            trajectory_data.x = [trajectory_data.x; zeros(estimated_points, 1)];
            trajectory_data.y = [trajectory_data.y; zeros(estimated_points, 1)];
            trajectory_data.z = [trajectory_data.z; zeros(estimated_points, 1)];
            trajectory_data.vx = [trajectory_data.vx; zeros(estimated_points, 1)];
            trajectory_data.vy = [trajectory_data.vy; zeros(estimated_points, 1)];
            trajectory_data.vz = [trajectory_data.vz; zeros(estimated_points, 1)];
            trajectory_data.ax = [trajectory_data.ax; zeros(estimated_points, 1)];
            trajectory_data.ay = [trajectory_data.ay; zeros(estimated_points, 1)];
            trajectory_data.az = [trajectory_data.az; zeros(estimated_points, 1)];
            trajectory_data.jx = [trajectory_data.jx; zeros(estimated_points, 1)];
            trajectory_data.jy = [trajectory_data.jy; zeros(estimated_points, 1)];
            trajectory_data.jz = [trajectory_data.jz; zeros(estimated_points, 1)];
            trajectory_data.time = [trajectory_data.time; zeros(estimated_points, 1)];
            trajectory_data.is_extruding = [trajectory_data.is_extruding; false(estimated_points, 1)];
            trajectory_data.print_type = [trajectory_data.print_type; cell(estimated_points, 1)];
            trajectory_data.layer_num = [trajectory_data.layer_num; zeros(estimated_points, 1)];
            estimated_points = estimated_points * 2;
        end

        trajectory_data.x(idx+1:new_idx) = x_seg;
        trajectory_data.y(idx+1:new_idx) = y_seg;
        trajectory_data.z(idx+1:new_idx) = z_seg;
        trajectory_data.vx(idx+1:new_idx) = vx_seg;
        trajectory_data.vy(idx+1:new_idx) = vy_seg;
        trajectory_data.vz(idx+1:new_idx) = vz_seg;
        trajectory_data.ax(idx+1:new_idx) = ax_seg;
        trajectory_data.ay(idx+1:new_idx) = ay_seg;
        trajectory_data.az(idx+1:new_idx) = az_seg;
        trajectory_data.jx(idx+1:new_idx) = jx_seg;
        trajectory_data.jy(idx+1:new_idx) = jy_seg;
        trajectory_data.jz(idx+1:new_idx) = jz_seg;
        trajectory_data.time(idx+1:new_idx) = t_seg + current_time;
        trajectory_data.is_extruding(idx+1:new_idx) = is_extruding;
        trajectory_data.print_type(idx+1:new_idx) = repmat({print_type}, n_points, 1);
        trajectory_data.layer_num(idx+1:new_idx) = layer_num;

        idx = new_idx;
        current_time = t_seg(end) + current_time;
    end

    % Trim unused space
    trajectory_data.x = trajectory_data.x(1:idx);
    trajectory_data.y = trajectory_data.y(1:idx);
    trajectory_data.z = trajectory_data.z(1:idx);
    trajectory_data.vx = trajectory_data.vx(1:idx);
    trajectory_data.vy = trajectory_data.vy(1:idx);
    trajectory_data.vz = trajectory_data.vz(1:idx);
    trajectory_data.ax = trajectory_data.ax(1:idx);
    trajectory_data.ay = trajectory_data.ay(1:idx);
    trajectory_data.az = trajectory_data.az(1:idx);
    trajectory_data.jx = trajectory_data.jx(1:idx);
    trajectory_data.jy = trajectory_data.jy(1:idx);
    trajectory_data.jz = trajectory_data.jz(1:idx);
    trajectory_data.time = trajectory_data.time(1:idx);
    trajectory_data.is_extruding = trajectory_data.is_extruding(1:idx);
    trajectory_data.print_type = trajectory_data.print_type(1:idx);
    trajectory_data.layer_num = trajectory_data.layer_num(1:idx);

    % Ensure time vector is strictly increasing
    if any(diff(trajectory_data.time) <= 0)
        fprintf('    Warning: Found non-increasing time values, fixing...\n');
        % Find and fix duplicates
        valid_idx = [true; diff(trajectory_data.time) > 0];
        n_invalid = sum(~valid_idx);
        fprintf('    Removing %d duplicate time points...\n', n_invalid);

        trajectory_data.x = trajectory_data.x(valid_idx);
        trajectory_data.y = trajectory_data.y(valid_idx);
        trajectory_data.z = trajectory_data.z(valid_idx);
        trajectory_data.vx = trajectory_data.vx(valid_idx);
        trajectory_data.vy = trajectory_data.vy(valid_idx);
        trajectory_data.vz = trajectory_data.vz(valid_idx);
        trajectory_data.ax = trajectory_data.ax(valid_idx);
        trajectory_data.ay = trajectory_data.ay(valid_idx);
        trajectory_data.az = trajectory_data.az(valid_idx);
        trajectory_data.jx = trajectory_data.jx(valid_idx);
        trajectory_data.jy = trajectory_data.jy(valid_idx);
        trajectory_data.jz = trajectory_data.jz(valid_idx);
        trajectory_data.time = trajectory_data.time(valid_idx);
        trajectory_data.is_extruding = trajectory_data.is_extruding(valid_idx);
        trajectory_data.print_type = trajectory_data.print_type(valid_idx);
        trajectory_data.layer_num = trajectory_data.layer_num(valid_idx);
    end

    fprintf('    Motion planning complete!\n');
end

%% Helper: Simulate S-curve segment
function [t, x, y, z, vx, vy, vz, ax, ay, az, jx, jy, jz] = ...
    simulate_scurve_segment(x0, y0, z0, x1, y1, z1, v_target, a_max, j_max, dt)
    % Segment vector
    dx = x1 - x0;
    dy = y1 - y0;
    dz = z1 - z0;
    L = sqrt(dx^2 + dy^2 + dz^2);

    if L < 1e-6
        % No movement
        t = 0;
        x = x0;
        y = y0;
        z = z0;
        vx = 0;
        vy = 0;
        vz = 0;
        ax = 0;
        ay = 0;
        az = 0;
        jx = 0;
        jy = 0;
        jz = 0;
        return;
    end

    % Unit direction vector
    ux = dx / L;
    uy = dy / L;
    uz = dz / L;

    % S-curve velocity profile (7 phases)
    % 1. Acceleration increase (jerk positive)
    % 2. Constant acceleration
    % 3. Acceleration decrease (jerk negative)
    % 4. Constant velocity
    % 5. Deceleration increase (jerk negative)
    % 6. Constant deceleration
    % 7. Deceleration decrease (jerk positive)

    % Simplified: Assume we can reach target velocity
    % Time to accelerate: t_acc = v_target / a_max
    % Distance during acceleration: d_acc = 0.5 * v_target * t_acc

    % For simplicity, use trapezoidal velocity profile
    t_acc = min(v_target / a_max, sqrt(L / a_max));  % Limited by distance
    v_peak = min(v_target, a_max * t_acc);

    d_acc = 0.5 * v_peak * t_acc;
    d_const = L - 2 * d_acc;

    if d_const < 0
        % Triangle profile (can't reach full speed)
        v_peak = sqrt(a_max * L / 2);
        t_acc = v_peak / a_max;
        t_const = 0;
    else
        t_const = d_const / v_peak;
    end

    t_total = 2 * t_acc + t_const;

    % Time vector
    t = 0:dt:t_total;

    % Velocity profile (magnitude)
    v = zeros(size(t));
    for i = 1:length(t)
        if t(i) < t_acc
            % Acceleration phase
            v(i) = v_peak * (t(i) / t_acc);
        elseif t(i) < t_acc + t_const
            % Constant velocity phase
            v(i) = v_peak;
        else
            % Deceleration phase
            t_dec = t(i) - t_acc - t_const;
            v(i) = v_peak * (1 - t_dec / t_acc);
        end
    end

    % Position along path
    s = cumtrapz(t, v);  % Integrate velocity

    % Cartesian coordinates
    x = x0 + ux * s;
    y = y0 + uy * s;
    z = z0 + uz * s;

    % Velocity components
    vx = ux * v;
    vy = uy * v;
    vz = uz * v;

    % Acceleration (derivative of velocity)
    a = gradient(v, dt);
    ax = ux * a;
    ay = uy * a;
    az = uz * a;

    % Jerk (derivative of acceleration)
    j = gradient(a, dt);
    jx = ux * j;
    jy = uy * j;
    jz = uz * j;
end

%% Helper: Filter structure array by mask
function data = filter_structure(data, mask)
    fields = fieldnames(data);
    for i = 1:length(fields)
        if iscell(data.(fields{i}))
            data.(fields{i}) = data.(fields{i})(mask);
        else
            data.(fields{i}) = data.(fields{i})(mask);
        end
    end
end
