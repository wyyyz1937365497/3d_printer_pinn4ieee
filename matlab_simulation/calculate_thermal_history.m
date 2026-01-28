function T_initial = calculate_thermal_history(layer_num, print_times, layer_intervals, params)
% CALCULATE_THERMAL_HISTORY - Calculate initial temperature considering thermal accumulation
%
% This function models the thermal history of the print by considering:
% 1. Heat input from each layer (nozzle heating)
% 2. Heat loss during layer intervals (convection + radiation)
% 3. Thermal diffusion within the material
% 4. Accumulation effect as layers build up
%
% Inputs:
%   layer_num       - Current layer number (1-based)
%   print_times     - Array of printing times for each layer [s]
%   layer_intervals - Array of time intervals between layers [s]
%   params          - Physics parameters structure
%
% Output:
%   T_initial       - Estimated initial temperature for this layer [°C]
%
% Reference:
%   Based on heat transfer model for FDM 3D printing
%   Considering semi-infinite solid with moving heat source

    %% Extract parameters
    T_amb = params.environment.ambient_temp;           % Ambient temperature [°C]
    T_nozzle = params.printing.nozzle_temp;            % Nozzle temperature [°C]
    h_conv = params.heat_transfer.h_convection_with_fan;  % Convection coefficient [W/(m²·K)]
    alpha = params.material.thermal_diffusivity;       % Thermal diffusivity [m²/s]

    %% Material properties for PLA
    rho = params.material.density;                     % Density [kg/m³]
    cp = params.material.specific_heat;                % Specific heat [J/(kg·K)]
    k = params.material.thermal_conductivity;          % Thermal conductivity [W/(m·K)]

    %% Geometry
    % Extract layer height from params or use default (0.2mm for most PLA prints)
    if isfield(params, 'gcode') && isfield(params.gcode, 'layer_height')
        layer_height = params.gcode.layer_height;      % Layer height [mm]
    else
        layer_height = 0.2;                            % Default layer height [mm]
    end
    layer_thickness = layer_height / 1000;             % Convert to m

    %% Initialize temperature history array
    % T_history(i) = temperature at the top of layer i after it's printed
    T_history = zeros(layer_num, 1);
    T_history(1) = T_amb;  % First layer starts at ambient temperature

    %% If this is the first layer
    if layer_num == 1
        T_initial = T_amb;
        fprintf('    Layer 1: Initial temperature = ambient (%.1f°C)\n', T_amb);
        return;
    end

    %% Model thermal accumulation for each previous layer
    for layer = 2:layer_num
        % Previous layer's ending temperature
        T_prev_end = T_history(layer - 1);

        % Time interval since previous layer
        if layer <= length(layer_intervals)
            dt = layer_intervals(layer - 1);
        else
            % Assume typical layer interval if not available
            dt = 10;  % Default 10 seconds
        end

        %% Phase 1: Nozzle heating during layer printing
        % Approximate heat input from nozzle
        % The nozzle deposits hot material, heating the substrate

        % Characteristic heating time (layer print time)
        if layer <= length(print_times)
            t_print = print_times(layer);
        else
            t_print = 30;  % Default 30 seconds per layer
        end

        % Heat penetration depth during printing
        delta_heat = 2 * sqrt(alpha * t_print);  % Thermal diffusion length [m]

        % Average temperature rise during printing
        % Simplified: Assume exponential approach to nozzle temperature
        % T_avg = T_prev_end + (T_nozzle - T_prev_end) * (1 - exp(-t_print/tau_heating))
        % where tau_heating is characteristic heating time

        % Characteristic time for heating
        tau_heating = (rho * cp * layer_thickness) / h_conv;  % [s]

        % Temperature rise during this layer's printing
        delta_T_heating = (T_nozzle - T_prev_end) * ...
                          (1 - exp(-t_print / tau_heating)) * ...
                          exp(-layer_num / 20);  % Diminishing effect with layers

        T_after_printing = T_prev_end + delta_T_heating;

        %% Phase 2: Cooling during layer interval
        % Newton's law of cooling: dT/dt = -h*A/(m*cp) * (T - T_amb)
        % Solution: T(t) = T_amb + (T_initial - T_amb) * exp(-t/tau_cooling)

        % Characteristic cooling time
        % Consider convection from top surface
        area_to_volume_ratio = 1 / layer_thickness;  % [1/m]
        tau_cooling = (rho * cp) / (h_conv * area_to_volume_ratio);  % [s]

        % Apply cooling
        T_after_cooling = T_amb + (T_after_printing - T_amb) * ...
                          exp(-dt / tau_cooling);

        %% Phase 3: Thermal diffusion from deeper layers
        % Heat diffuses from below (warmer layers)
        % Approximate as weighted average of recent layers
        if layer > 3
            % Consider thermal memory of last 3 layers
            weights = [0.5, 0.3, 0.2];  % Recent layers have more influence
            T_from_below = weights(1) * T_history(layer-1) + ...
                          weights(2) * T_history(layer-2) + ...
                          weights(3) * T_history(layer-3);
            % Blend with current temperature
            T_after_cooling = 0.7 * T_after_cooling + 0.3 * T_from_below;
        end

        % Store this layer's temperature
        T_history(layer) = T_after_cooling;
    end

    %% Return the initial temperature for the current layer
    T_initial = T_history(layer_num);

    %% Debug output
    if layer_num <= 5 || mod(layer_num, 10) == 0
        fprintf('    Layer %d: Initial temperature = %.1f°C (ambient: %.1f°C, rise: %.1f°C)\n', ...
                layer_num, T_initial, T_amb, T_initial - T_amb);
    end

    %% Sanity check
    if T_initial > T_nozzle - 10
        warning('Initial temperature (%.1f°C) too close to nozzle temp (%.1f°C). May indicate model issue.', ...
                T_initial, T_nozzle);
    end

    if T_initial < T_amb
        warning('Initial temperature (%.1f°C) below ambient (%.1f°C). This should not happen.', ...
                T_initial, T_amb);
        T_initial = T_amb;
    end

    %% Visualization (optional)
    if params.debug.plot_temperature
        figure('Name', 'Thermal History', 'Position', [100, 100, 800, 400]);
        plot(1:layer_num, T_history, 'b-o', 'LineWidth', 1.5);
        hold on;
        yline(T_amb, 'r--', 'Ambient', 'LineWidth', 1.5);
        yline(T_nozzle, 'g--', 'Nozzle', 'LineWidth', 1.5);
        xline(layer_num, 'k:', 'Current Layer', 'LineWidth', 1.5);
        grid on;
        xlabel('Layer Number');
        ylabel('Temperature (°C)');
        title('Thermal History: Initial Temperature by Layer');
        legend('Layer Temperature', 'Ambient', 'Nozzle', 'Current Layer', 'Location', 'best');
        ylim([T_amb - 5, T_nozzle + 5]);
    end
end
