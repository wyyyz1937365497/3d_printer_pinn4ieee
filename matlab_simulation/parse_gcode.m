function trajectory_data = parse_gcode(gcode_file, params)
% PARSE_GCODE - Parse G-code file and extract trajectory information
%
% This function reads a G-code file and extracts:
% - Position (X, Y, Z)
% - Extrusion (E)
% - Feed rate (F)
% - Calculated velocity, acceleration, jerk
% - Corner detection
% - Segment classification (travel vs extrusion)
%
% Inputs:
%   gcode_file - Path to G-code file
%   params     - Physics parameters structure (from physics_parameters.m)
%
% Output:
%   trajectory_data - Structure containing all trajectory information
%
% Reference: G-code format (RepRap/Marlin dialect)

    fprintf('Parsing G-code file: %s\n', gcode_file);

    %% Read G-code file
    fid = fopen(gcode_file, 'r');
    if fid == -1
        error('Cannot open G-code file: %s', gcode_file);
    end

    % Read entire file
    gcode_lines = textscan(fid, '%s', 'Delimiter', '\n');
    gcode_lines = gcode_lines{1};
    fclose(fid);

    fprintf('  Total lines in G-code: %d\n', length(gcode_lines));

    %% Initialize parsing state
    state.x = 0;      % Current X position (mm)
    state.y = 0;      % Current Y position (mm)
    state.z = 0;      % Current Z position (mm)
    state.e = 0;      % Current extruder position (mm)
    state.f = 1500;   % Current feed rate (mm/min)
    state.is_relative = false;   % G90 (absolute) by default
    state.e_relative = true;     % E is relative by default (M83)

    %% Storage arrays
    % Pre-allocate for performance (estimate size)
    max_segments = length(gcode_lines);
    data.time = zeros(max_segments, 1);
    data.x = zeros(max_segments, 1);
    data.y = zeros(max_segments, 1);
    data.z = zeros(max_segments, 1);
    data.e = zeros(max_segments, 1);
    data.f = zeros(max_segments, 1);
    data.is_extruding = false(max_segments, 1);
    data.is_travel = false(max_segments, 1);
    data.is_corner = false(max_segments, 1);
    data.corner_angle = zeros(max_segments, 1);
    data.curvature = zeros(max_segments, 1);
    data.layer_num = zeros(max_segments, 1);
    data.comment = cell(max_segments, 1);

    %% Parse G-code line by line
    segment_idx = 0;
    current_layer = 0;
    current_time = 0;
    prev_direction = [0; 0];  % Initial movement direction

    for i = 1:length(gcode_lines)
        line = strtrim(gcode_lines{i});

        % Skip empty lines and comments
        if isempty(line) || line(1) == ';' || ~any(isstrprop(line, 'alphanum'))
            continue;
        end

        % Extract comment if present
        comment_start = strfind(line, ';');
        if ~isempty(comment_start)
            comment = line(comment_start:end);
            line = line(1:comment_start-1);
        else
            comment = '';
        end

        % Parse G-code commands
        tokens = regexp(line, '[A-Z][-+]?[0-9.]*', 'match');

        % Process each token
        has_g1 = false;
        new_x = state.x;
        new_y = state.y;
        new_z = state.z;
        new_e = state.e;
        new_f = state.f;
        is_layer_change = false;

        for j = 1:length(tokens)
            token = tokens{j};
            cmd = token(1);
            val = str2double(token(2:end));

            switch cmd
                case 'G'
                    switch round(val)
                        case 0  % Rapid move
                            has_g1 = true;
                        case 1  % Linear move
                            has_g1 = true;
                        case 90  % Absolute positioning
                            state.is_relative = false;
                        case 91  % Relative positioning
                            state.is_relative = true;
                        case 92  % Set position
                            % Reset position - handled in E/M processing
                    end

                case 'M'
                    switch round(val)
                        case 82  % Absolute extrusion
                            state.e_relative = false;
                        case 83  % Relative extrusion
                            state.e_relative = true;
                    end

                case 'X'
                    if state.is_relative
                        new_x = state.x + val;
                    else
                        new_x = val;
                    end

                case 'Y'
                    if state.is_relative
                        new_y = state.y + val;
                    else
                        new_y = val;
                    end

                case 'Z'
                    if state.is_relative
                        new_z = state.z + val;
                    else
                        new_z = val;
                    end
                    % Detect layer change
                    if new_z > state.z
                        current_layer = current_layer + 1;
                        is_layer_change = true;
                    end

                case 'E'
                    if state.e_relative
                        new_e = state.e + val;
                    else
                        new_e = val;
                    end

                case 'F'
                    % Feed rate in mm/min
                    new_f = val;
            end
        end

        % If this is a movement command (G0 or G1)
        if has_g1
            % Calculate movement
            dx = new_x - state.x;
            dy = new_y - state.y;
            dz = new_z - state.z;
            de = new_e - state.e;

            segment_length = sqrt(dx^2 + dy^2 + dz^2);

            % Skip if no movement
            if segment_length < params.gcode.min_segment_length
                continue;
            end

            segment_idx = segment_idx + 1;

            % Calculate time for this segment
            velocity = new_f / 60;  % Convert mm/min to mm/s
            segment_time = segment_length / velocity;
            current_time = current_time + segment_time;

            % Store data
            data.time(segment_idx) = current_time;
            data.x(segment_idx) = new_x;
            data.y(segment_idx) = new_y;
            data.z(segment_idx) = new_z;
            data.e(segment_idx) = new_e;
            data.f(segment_idx) = new_f;
            data.layer_num(segment_idx) = current_layer;
            data.comment{segment_idx} = comment;

            % Classify segment
            if abs(de) > params.gcode.extrusion_threshold
                data.is_extruding(segment_idx) = true;
            else
                data.is_travel(segment_idx) = true;
            end

            % Detect corners (change in direction)
            if segment_idx > 1
                prev_dx = data.x(segment_idx-1) - data.x(segment_idx-2);
                prev_dy = data.y(segment_idx-1) - data.y(segment_idx-2);

                if abs(prev_dx) > 1e-6 || abs(prev_dy) > 1e-6
                    % Calculate angle change
                    prev_dir = [prev_dx; prev_dy];
                    prev_dir = prev_dir / norm(prev_dir);
                    curr_dir = [dx; dy];
                    curr_dir = curr_dir / norm(curr_dir);

                    % Angle between vectors
                    cos_angle = max(-1, min(1, dot(prev_dir, curr_dir)));
                    angle_rad = acos(cos_angle);
                    angle_deg = rad2deg(angle_rad);

                    % Store corner information
                    data.corner_angle(segment_idx) = angle_deg;

                    % Mark as corner if angle exceeds threshold
                    if angle_deg > params.gcode.corner_angle_threshold
                        data.is_corner(segment_idx) = true;
                    end

                    % Calculate curvature (1/radius of turn)
                    % For polyline: κ = |Δθ| / segment_length
                    data.curvature(segment_idx) = angle_rad / (segment_length + 1e-6);
                end
            end

            % Update state
            state.x = new_x;
            state.y = new_y;
            state.z = new_z;
            state.e = new_e;
            state.f = new_f;
        end
    end

    %% Trim arrays to actual size
    n_segments = segment_idx;
    data.time = data.time(1:n_segments);
    data.x = data.x(1:n_segments);
    data.y = data.y(1:n_segments);
    data.z = data.z(1:n_segments);
    data.e = data.e(1:n_segments);
    data.f = data.f(1:n_segments);
    data.is_extruding = data.is_extruding(1:n_segments);
    data.is_travel = data.is_travel(1:n_segments);
    data.is_corner = data.is_corner(1:n_segments);
    data.corner_angle = data.corner_angle(1:n_segments);
    data.curvature = data.curvature(1:n_segments);
    data.layer_num = data.layer_num(1:n_segments);
    data.comment = data.comment(1:n_segments);

    fprintf('  Parsed %d movement segments\n', n_segments);
    fprintf('  Layers: %d\n', max(data.layer_num));
    fprintf('  Corners detected: %d\n', sum(data.is_corner));

    %% Calculate kinematic derivatives
    fprintf('  Calculating kinematic derivatives...\n');

    % Velocity (from G-code F commands, verified with position difference)
    data.velocity_mm_s = data.f / 60;  % mm/s

    % Calculate actual velocity from position (for validation)
    data.dx = zeros(n_segments, 1);
    data.dy = zeros(n_segments, 1);
    data.dz = zeros(n_segments, 1);
    data.dt = zeros(n_segments, 1);

    data.dx(2:end) = diff(data.x);
    data.dy(2:end) = diff(data.y);
    data.dz(2:end) = diff(data.z);
    data.dt(2:end) = diff(data.time);

    % Actual velocity (magnitude)
    data.v_actual = zeros(n_segments, 1);
    valid_dt = data.dt > 0;
    data.v_actual(valid_dt) = sqrt(data.dx(valid_dt).^2 + ...
                                     data.dy(valid_dt).^2 + ...
                                     data.dz(valid_dt).^2) ./ data.dt(valid_dt);

    % Velocity components
    data.vx = zeros(n_segments, 1);
    data.vy = zeros(n_segments, 1);
    data.vz = zeros(n_segments, 1);
    data.vx(valid_dt) = data.dx(valid_dt) ./ data.dt(valid_dt);
    data.vy(valid_dt) = data.dy(valid_dt) ./ data.dt(valid_dt);
    data.vz(valid_dt) = data.dz(valid_dt) ./ data.dt(valid_dt);

    % Acceleration (finite difference)
    data.ax = zeros(n_segments, 1);
    data.ay = zeros(n_segments, 1);
    data.az = zeros(n_segments, 1);
    data.acceleration = zeros(n_segments, 1);

    valid_accel = valid_dt(2:end) & data.dt(1:end-1) > 0;
    data.ax(2:end) = diff(data.vx) ./ data.dt(2:end);
    data.ay(2:end) = diff(data.vy) ./ data.dt(2:end);
    data.az(2:end) = diff(data.vz) ./ data.dt(2:end);
    data.acceleration(2:end) = sqrt(data.ax(2:end).^2 + ...
                                     data.ay(2:end).^2 + ...
                                     data.az(2:end).^2);

    % Jerk (derivative of acceleration)
    data.jx = zeros(n_segments, 1);
    data.jy = zeros(n_segments, 1);
    data.jz = zeros(n_segments, 1);
    data.jerk = zeros(n_segments, 1);

    valid_jerk = valid_accel(2:end) & valid_dt(3:end);
    data.jx(3:end) = diff(data.ax) ./ data.dt(3:end);
    data.jy(3:end) = diff(data.ay) ./ data.dt(3:end);
    data.jz(3:end) = diff(data.az) ./ data.dt(3:end);
    data.jerk(3:end) = sqrt(data.jx(3:end).^2 + ...
                             data.jy(3:end).^2 + ...
                             data.jz(3:end).^2);

    %% Movement direction angle
    data.direction_angle = atan2(data.dy, data.dx);  % radians

    %% Additional G-code features
    % Distance from last corner
    data.dist_from_last_corner = inf(n_segments, 1);
    last_corner_idx = 0;
    for i = 1:n_segments
        if data.is_corner(i)
            last_corner_idx = i;
        end
        if last_corner_idx > 0
            data.dist_from_last_corner(i) = sqrt(...
                (data.x(i) - data.x(last_corner_idx))^2 + ...
                (data.y(i) - data.y(last_corner_idx))^2);
        end
    end

    %% Create output structure
    trajectory_data = data;

    fprintf('  G-code parsing complete!\n');
    fprintf('\n');
    fprintf('  Trajectory Statistics:\n');
    fprintf('    Time range: %.2f - %.2f s\n', min(data.time), max(data.time));
    fprintf('    X range: %.2f - %.2f mm\n', min(data.x), max(data.x));
    fprintf('    Y range: %.2f - %.2f mm\n', min(data.y), max(data.y));
    fprintf('    Z range: %.2f - %.2f mm\n', min(data.z), max(data.z));
    fprintf('    Max velocity: %.2f mm/s\n', max(data.v_actual));
    fprintf('    Max acceleration: %.2f mm/s²\n', max(data.acceleration));
    fprintf('    Max jerk: %.2f mm/s³\n', max(data.jerk));
    fprintf('    Extruding segments: %d (%.1f%%)\n', ...
            sum(data.is_extruding), 100*sum(data.is_extruding)/n_segments);
    fprintf('    Travel segments: %d (%.1f%%)\n', ...
            sum(data.is_travel), 100*sum(data.is_travel)/n_segments);
    fprintf('\n');

end
