function thermal_results = simulate_thermal_field(trajectory_data, params)
% SIMULATE_THERMAL_FIELD - Simulate temperature field during FDM printing
%
% This function models the thermal evolution of the printed part using:
% 1. Moving heat source model (the nozzle)
% 2. 3D transient heat conduction equation
% 3. Convective and radiative cooling
% 4. Interlayer heat transfer
%
% Heat equation:
%   ∂T/∂t = α·∇²T + Q_source - Q_cooling
%
% where:
%   α - thermal diffusivity
%   Q_source - heat input from extrusion
%   Q_cooling - convective + radiative cooling
%
% Inputs:
%   trajectory_data - Structure from parse_gcode.m
%   params          - Physics parameters from physics_parameters.m
%
% Output:
%   thermal_results - Structure containing temperature field data
%
% Reference: Heat transfer in additive manufacturing literature

    fprintf('Simulating thermal field (moving heat source model)...\n');

    %% Extract trajectory data
    t = trajectory_data.time;
    x_nozzle = trajectory_data.x;  % Use reference trajectory initially
    y_nozzle = trajectory_data.y;
    z_nozzle = trajectory_data.z;
    v_extrude = trajectory_data.v_actual;  % mm/s

    n_points = length(t);

    %% Simulation domain
    fprintf('  Setting up simulation domain...\n');

    % Determine bounds from trajectory
    x_min = min(x_nozzle) - 10;  % 10mm margin
    x_max = max(x_nozzle) + 10;
    y_min = min(y_nozzle) - 10;
    y_max = max(y_nozzle) + 10;
    z_min = 0;
    z_max = max(z_nozzle) + 2;

    % Spatial resolution
    dx = params.simulation.dx;    % mm
    dy = params.simulation.dy;    % mm
    dz = params.simulation.dz;    % mm (layer resolution)

    % Create grid
    x_grid = x_min:dx:x_max;
    y_grid = y_min:dy:y_max;
    z_grid = z_min:dz:z_max;

    nx = length(x_grid);
    ny = length(y_grid);
    nz = length(z_grid);

    fprintf('    Grid size: %d × %d × %d = %.1f million points\n', ...
            nx, ny, nz, (nx*ny*nz)/1e6);
    fprintf('    X range: %.1f - %.1f mm\n', x_min, x_max);
    fprintf('    Y range: %.1f - %.1f mm\n', y_min, y_max);
    fprintf('    Z range: %.1f - %.1f mm\n', z_min, z_max);

    %% Time stepping
    dt_thermal = params.simulation.dt_thermal;  % s
    fprintf('    Time step: %.4f s\n', dt_thermal);

    %% Initialize temperature field
    fprintf('  Initializing temperature field...\n');

    T = ones(ny, nx, nz) * params.environment.ambient_temp;  % °C

    % Set bed temperature at bottom
    T(:,:,1) = params.printing.bed_temp;

    % Temperature field storage (at selected points and times)
    % We'll track temperature at nozzle position and layer interfaces

    T_interface = zeros(n_points, 1);  % Temperature at layer interface
    T_surface = zeros(n_points, 1);    % Surface temperature
    cooling_rate = zeros(n_points, 1); % dT/dt

    %% Material properties
    alpha = params.material.thermal_diffusivity;      % m²/s
    k_thermal = params.material.thermal_conductivity; % W/(m·K)
    rho = params.material.density;                    % kg/m³
    cp = params.material.specific_heat;               % J/(kg·K)

    %% Heat transfer coefficients
    h_conv = params.heat_transfer.h_convection_with_fan;  % W/(m²·K)
    h_rad = params.heat_transfer.h_radiation;            % W/(m²·K) (linearized)
    h_total = h_conv + h_rad;

    %% Moving heat source parameters
    T_nozzle = params.printing.nozzle_temp;  % °C
    nozzle_dia = params.nozzle.diameter;     % mm

    % Heat source radius (Gaussian distribution)
    r_source = nozzle_dia;  % mm

    %% Simplified thermal model (point tracking)
    % Full 3D finite difference is too slow for Python integration
    % Instead, we track temperature at key locations using analytical solutions

    fprintf('  Using analytical moving heat source model...\n');

    % Initialize layer temperature tracking
    current_layer = 0;
    layer_deposition_time = [];
    layer_center_temp = [];

    T_nozzle_history = zeros(n_points, 1);  % Temperature at nozzle position

    % Previous temperature for cooling rate calculation
    T_prev = params.environment.ambient_temp;

    %% Main time loop
    fprintf('  Running thermal simulation...\n');

    for i = 1:n_points
        % Current nozzle position
        xi = x_nozzle(i);
        yi = y_nozzle(i);
        zi = z_nozzle(i);

        % Check if we're extruding
        is_extruding = trajectory_data.is_extruding(i);

        % Current layer
        layer_i = trajectory_data.layer_num(i);

        % Layer change detection
        if layer_i > current_layer
            current_layer = layer_i;
            layer_deposition_time(current_layer) = t(i);
            fprintf('    Layer %d at t=%.2f s\n', current_layer, t(i));
        end

        %% Temperature at nozzle position (simplified)
        % This is a simplified model that accounts for:
        % 1. Heat input from nozzle
        % 2. Cooling due to environment
        % 3. Thermal inertia of the material

        if is_extruding
            % Extruding: material is hot
            T_local = T_nozzle;
        else
            % Travel: material cools
            % Use Newton's law of cooling: dT/dt = -h*A/(m*cp) * (T - T_amb)
            time_since_extrusion = 0;
            for j = i-1:-1:1
                if trajectory_data.is_extruding(j)
                    time_since_extrusion = t(i) - t(j);
                    break;
                end
            end

            % Cooling model with protection against division by zero
            thickness_m = max(dz*1e-3, 1e-6); % Ensure minimum thickness
            cooling_constant = h_total / (rho * cp * thickness_m + 1e-10);  % 1/s, avoid division by zero
            T_local = (T_nozzle - params.environment.ambient_temp) * ...
                      exp(-max(cooling_constant, 1e-6) * time_since_extrusion) + ...
                      params.environment.ambient_temp;
        end

        T_nozzle_history(i) = T_local;

        %% Layer interface temperature
        % Temperature at the interface between current layer and previous layer
        if layer_i > 1
            % Time since previous layer was deposited at this location
            % Find when nozzle was near this (x,y) on previous layer
            prev_layer_mask = trajectory_data.layer_num == layer_i - 1;

            if sum(prev_layer_mask) > 0
                % Find closest point on previous layer
                dist_prev = sqrt((trajectory_data.x(prev_layer_mask) - xi).^2 + ...
                                 (trajectory_data.y(prev_layer_mask) - yi).^2);

                [min_dist, idx_min] = min(dist_prev);

                if min_dist < 5  % Within 5mm
                    prev_layer_indices = find(prev_layer_mask);
                    prev_idx = prev_layer_indices(idx_min);
                    time_diff = t(i) - t(prev_idx);

                    % Temperature of previous layer at this location
                    % Avoid NaN in exponential calculation
                    valid_time_diff = max(time_diff, 0);
                    decay_factor = exp(-max(cooling_constant, 1e-6) * valid_time_diff);
                    T_prev_layer = T_nozzle_history(prev_idx) * decay_factor + ...
                                   params.environment.ambient_temp * (1 - decay_factor);

                    T_interface(i) = (T_local + T_prev_layer) / 2;  % Average
                    % Ensure finite values
                    if ~isfinite(T_interface(i))
                        T_interface(i) = params.environment.ambient_temp;
                    end
                else
                    T_interface(i) = params.environment.ambient_temp;
                end
            else
                T_interface(i) = params.environment.ambient_temp;
            end
        else
            T_interface(i) = T_local;  % First layer - no interface below
        end

        %% Surface temperature
        T_surface(i) = T_local;

        %% Cooling rate
        if i > 1 && (t(i) - t(i-1)) > 1e-6
            dt_step = t(i) - t(i-1);
            cooling_rate(i) = (T_local - T_prev) / dt_step;
            % Protect against extreme values
            if abs(cooling_rate(i)) > 1e6
                cooling_rate(i) = sign(cooling_rate(i)) * 1e6;
            end
        else
            cooling_rate(i) = 0;
        end

        T_prev = T_local;
        
        % Ensure all temperatures are finite
        if ~isfinite(T_local)
            T_local = params.environment.ambient_temp;
        end
    end

    %% Calculate thermal metrics
    fprintf('  Calculating thermal metrics...\n');

    % Temperature above melting point (for molecular diffusion)
    time_above_melting = sum(T_nozzle_history > params.material.melting_point) * mean(diff(t));

    % Temperature above glass transition
    time_above_tg = sum(T_nozzle_history > params.material.glass_transition) * mean(diff(t));

    % Maximum cooling rate
    max_cooling_rate = max(cooling_rate(T_nozzle_history < params.printing.nozzle_temp - 50));

    % Mean interface temperature during printing
    mean_interface_temp = mean(T_interface(trajectory_data.is_extruding));

    fprintf('    Time above Tm: %.2f s\n', time_above_melting);
    fprintf('    Time above Tg: %.2f s\n', time_above_tg);
    fprintf('    Max cooling rate: %.2f °C/s\n', abs(max_cooling_rate));
    fprintf('    Mean interface temp: %.2f °C\n', mean_interface_temp);

    %% Calculate adhesion strength (simplified model)
    fprintf('  Estimating interlayer adhesion strength...\n');

    % Wool-O'Connor healing model (simplified)
    % Bond strength depends on:
    % 1. Interface temperature
    % 2. Time above critical temperature
    % 3. Cooling rate

    adhesion_ratio = zeros(n_points, 1);

    for i = 1:n_points
        if trajectory_data.layer_num(i) > 1 && trajectory_data.is_extruding(i)
            T_int = T_interface(i);

            % Healing ratio based on temperature
            if T_int < params.material.glass_transition || ~isfinite(T_int)
                healing_ratio = 0;
            elseif T_int >= params.material.melting_point
                healing_ratio = 1.0;
            else
                % Linear ramp from Tg to Tm
                delta_T = params.material.melting_point - params.material.glass_transition;
                if abs(delta_T) < 1e-6
                    healing_ratio = 0.5; % Avoid division by zero
                else
                    healing_ratio = (T_int - params.material.glass_transition) / delta_T;
                end
            end

            % Adjust for cooling rate (fast cooling = poor healing)
            cooling_val = abs(cooling_rate(i));
            if ~isfinite(cooling_val) || cooling_val > 1e6
                cooling_factor = 0.1; % Assume very fast cooling
            else
                cooling_factor = min(1.0, 10 / (cooling_val + 1));
            end

            adhesion_ratio(i) = healing_ratio * cooling_factor;
            
            % Final check for finite values
            if ~isfinite(adhesion_ratio(i)) || adhesion_ratio(i) < 0
                adhesion_ratio(i) = 0;
            end
            if adhesion_ratio(i) > 1
                adhesion_ratio(i) = 1;
            end
        else
            adhesion_ratio(i) = 0; % Default value
        end
    end

    mean_adhesion = mean(adhesion_ratio(adhesion_ratio > 0));
    min_adhesion = min(adhesion_ratio(adhesion_ratio > 0));

    fprintf('    Mean adhesion ratio: %.2f\n', mean_adhesion);
    fprintf('    Min adhesion ratio: %.2f\n', min_adhesion);

    %% Interlayer time interval
    fprintf('  Calculating interlayer time intervals...\n');

    interlayer_time = zeros(n_points, 1);

    for i = 2:n_points
        if trajectory_data.layer_num(i) > trajectory_data.layer_num(i-1)
            % This is the start of a new layer
            % Calculate time since previous layer at this location
            current_layer = trajectory_data.layer_num(i);

            % Find when we were at similar (x,y) on previous layer
            prev_layer_mask = trajectory_data.layer_num == current_layer - 1;

            if sum(prev_layer_mask) > 0
                xi = x_nozzle(i);
                yi = y_nozzle(i);

                dist_prev = sqrt((trajectory_data.x(prev_layer_mask) - xi).^2 + ...
                                 (trajectory_data.y(prev_layer_mask) - yi).^2);

                [min_dist, idx_min] = min(dist_prev);

                if min_dist < 10  % Within 10mm
                    prev_layer_indices = find(prev_layer_mask);
                    prev_idx = prev_layer_indices(idx_min);
                    interlayer_time(i) = t(i) - t(prev_idx);
                end
            end
        end
    end

    mean_interlayer_time = mean(interlayer_time(interlayer_time > 0));

    fprintf('    Mean interlayer time: %.2f s\n', mean_interlayer_time);

    %% Temperature gradient (simplified)
    fprintf('  Estimating temperature gradients...\n');

    % Vertical gradient (between layers)
    temp_gradient_z = zeros(n_points, 1);

    for i = 2:n_points
        if trajectory_data.is_extruding(i) && trajectory_data.layer_num(i) > 1
            % Estimate temperature difference with previous layer
            T_current = T_nozzle_history(i);

            % Find corresponding point on previous layer
            prev_layer_mask = trajectory_data.layer_num == trajectory_data.layer_num(i) - 1;

            if sum(prev_layer_mask) > 0
                xi = x_nozzle(i);
                yi = y_nozzle(i);

                dist_prev = sqrt((trajectory_data.x(prev_layer_mask) - xi).^2 + ...
                                 (trajectory_data.y(prev_layer_mask) - yi).^2);

                [min_dist, idx_min] = min(dist_prev);

                if min_dist < 10
                    prev_layer_indices = find(prev_layer_mask);
                    prev_idx = prev_layer_indices(idx_min);
                    T_prev_layer_local = T_nozzle_history(prev_idx);

                    % Vertical gradient (°C/mm)
                    temp_gradient_z(i) = abs(T_current - T_prev_layer_local) / params.extrusion.height;
                end
            end
        end
    end

    %% Create output structure
    thermal_results.time = t;

    % Nozzle position (actual trajectory)
    thermal_results.x_nozzle = x_nozzle;
    thermal_results.y_nozzle = y_nozzle;
    thermal_results.z_nozzle = z_nozzle;

    % Temperature field
    thermal_results.T_nozzle_history = T_nozzle_history;  % °C
    thermal_results.T_interface = T_interface;            % °C
    thermal_results.T_surface = T_surface;                % °C

    % Temperature gradients
    thermal_results.temp_gradient_z = temp_gradient_z;    % °C/mm

    % Cooling
    thermal_results.cooling_rate = cooling_rate;          % °C/s
    thermal_results.time_above_melting = time_above_melting;  % s
    thermal_results.time_above_tg = time_above_tg;        % s

    % Interlayer
    % Clean up interlayer time data
    interlayer_time(~isfinite(interlayer_time)) = 0;
    thermal_results.interlayer_time = interlayer_time;    % s
    thermal_results.mean_interlayer_time = mean_interlayer_time;  % s

    % Adhesion strength
    % Clean up adhesion ratio data
    adhesion_ratio(~isfinite(adhesion_ratio)) = 0;
    adhesion_ratio(adhesion_ratio < 0) = 0;
    adhesion_ratio(adhesion_ratio > 1) = 1;
    thermal_results.adhesion_ratio = adhesion_ratio;      % -
    thermal_results.mean_adhesion = mean_adhesion;        % -
    thermal_results.min_adhesion = min_adhesion;          % -

    % Environmental
    thermal_results.T_ambient = params.environment.ambient_temp;
    thermal_results.T_bed = params.printing.bed_temp;
    thermal_results.T_nozzle_setpoint = params.printing.nozzle_temp;

    % Grid information
    thermal_results.x_grid = x_grid;
    thermal_results.y_grid = y_grid;
    thermal_results.z_grid = z_grid;
    thermal_results.dx = dx;
    thermal_results.dy = dy;
    thermal_results.dz = dz;

    fprintf('  Thermal field simulation complete!\n\n');

    %% Optional: Plotting
    if params.debug.plot_temperature
        figure('Name', 'Thermal Field Analysis', 'Position', [100, 100, 1200, 800]);

        % Temperature at nozzle position
        subplot(2, 3, 1);
        plot(t, T_nozzle_history, 'r-', 'LineWidth', 1.5);
        yline(params.material.melting_point, 'b--', 'Melting Point', 'LineWidth', 1.5);
        yline(params.material.glass_transition, 'g--', 'T_g', 'LineWidth', 1.5);
        yline(params.environment.ambient_temp, 'k--', 'Ambient', 'LineWidth', 1);
        grid on;
        xlabel('Time (s)');
        ylabel('Temperature (°C)');
        title('Temperature at Nozzle Position');
        ylim([params.environment.ambient_temp - 10, T_nozzle + 10]);

        % Interface temperature
        subplot(2, 3, 2);
        plot(t, T_interface, 'b-', 'LineWidth', 1.5);
        yline(params.material.glass_transition, 'g--', 'T_g', 'LineWidth', 1.5);
        grid on;
        xlabel('Time (s)');
        ylabel('Interface Temperature (°C)');
        title('Layer Interface Temperature');

        % Cooling rate
        subplot(2, 3, 3);
        plot(t, cooling_rate, 'r-', 'LineWidth', 1);
        grid on;
        xlabel('Time (s)');
        ylabel('Cooling Rate (°C/s)');
        title('Cooling Rate');
        ylim([min(cooling_rate)*1.1, max(abs(cooling_rate))*1.1]);

        % Adhesion ratio
        subplot(2, 3, 4);
        valid_adhesion = adhesion_ratio > 0;
        plot(t(valid_adhesion), adhesion_ratio(valid_adhesion), 'b.-', 'LineWidth', 1);
        grid on;
        xlabel('Time (s)');
        ylabel('Adhesion Ratio');
        title('Interlayer Adhesion Strength');
        ylim([0, 1.1]);

        % Interlayer time
        subplot(2, 3, 5);
        valid_interlayer = interlayer_time > 0;
        plot(t(valid_interlayer), interlayer_time(valid_interlayer), 'g.-', 'LineWidth', 1);
        grid on;
        xlabel('Time (s)');
        ylabel('Time (s)');
        title('Interlayer Time Interval');

        % Temperature gradient
        subplot(2, 3, 6);
        valid_grad = temp_gradient_z > 0;
        plot(t(valid_grad), temp_gradient_z(valid_grad), 'm-', 'LineWidth', 1);
        grid on;
        xlabel('Time (s)');
        ylabel('Gradient (°C/mm)');
        title('Vertical Temperature Gradient');

        drawnow;
    end

end
