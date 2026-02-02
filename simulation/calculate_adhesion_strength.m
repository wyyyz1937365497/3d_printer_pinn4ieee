function adhesion_data = calculate_adhesion_strength(thermal_data, params)
% CALCULATE_ADHESION_STRENGTH - Calculate interlayer adhesion strength
%
% This function computes the strength of bond between adjacent layers
% based on the Wool-O'Connor polymer healing model.
%
% Inputs:
%   thermal_data - Structure from simulate_thermal_field.m
%                  Must contain: T_interface, interlayer_time, cooling_rate
%   params       - Physics parameters from physics_parameters.m
%
% Output:
%   adhesion_data - Structure containing:
%     .adhesion_ratio    - Adhesion strength ratio (0-1, relative to bulk)
%     .adhesion_strength - Absolute adhesion strength (MPa)
%
% Reference:
%   - Wool, R. P., & O'Connor, K. M. (1981)
%   - A model for crack healing in polymers
%
% Physics:
%   Adhesion strength depends on:
%   1. Interface temperature (higher → better diffusion)
%   2. Time above glass transition (longer → more healing)
%   3. Cooling rate (slower → less residual stress)

    fprintf('Calculating interlayer adhesion strength...\n');

    %% Check if adhesion_ratio is already calculated in thermal_data
    if isfield(thermal_data, 'adhesion_ratio')
        fprintf('  Using pre-calculated adhesion from thermal simulation\n');
        adhesion_data.adhesion_ratio = thermal_data.adhesion_ratio;

        % Calculate absolute adhesion strength (MPa)
        bulk_strength_pla = 60;  % MPa (typical tensile strength of PLA)
        adhesion_data.adhesion_strength = adhesion_data.adhesion_ratio * bulk_strength_pla;

        % Statistics
        fprintf('  Adhesion strength statistics:\n');
        fprintf('    Mean adhesion ratio: %.3f ± %.3f\n', ...
                mean(adhesion_data.adhesion_ratio, 'omitnan'), ...
                std(adhesion_data.adhesion_ratio, 'omitnan'));
        fprintf('    Min adhesion ratio: %.3f\n', min(adhesion_data.adhesion_ratio));
        fprintf('    Max adhesion ratio: %.3f\n', max(adhesion_data.adhesion_ratio));
        fprintf('    Mean adhesion strength: %.2f ± %.2f MPa\n', ...
                mean(adhesion_data.adhesion_strength, 'omitnan'), ...
                std(adhesion_data.adhesion_strength, 'omitnan'));

        fprintf('Interlayer adhesion calculation complete!\n');
        return;
    end

    n_points = length(thermal_data.T_interface);

    %% Extract thermal data
    T_interface = thermal_data.T_interface;
    T_glass = params.material.glass_transition;  % PLA: ~60°C
    T_print = params.printing.nozzle_temp;       % Usually 200-220°C

    % Get interlayer time (time between printing adjacent layers)
    if isfield(thermal_data, 'interlayer_time')
        interlayer_time = thermal_data.interlayer_time;
    else
        % Estimate from cooling rate
        % Assume we need to cool from T_print to T_interface
        interlayer_time = (T_print - T_interface) ./ thermal_data.cooling_rate;
    end

    %% Wool-O'Connor Polymer Healing Model
    % The healing ratio (degree of strength recovery) is:
    %
    %   H(t, T) = (X(t)/X_inf)^2
    %
    % where X(t) is the diffusion depth at time t:
    %
    %   X(t) ∝ sqrt(D(T) * t)
    %
    % and D(T) is the diffusion coefficient following Arrhenius law:
    %
    %   D(T) = D0 * exp(-Ea / (R * T))
    %
    % Simplified model:
    %   adhesion_ratio = f(T_interface, time_above_Tg)

    %% 1. Temperature factor
    % Higher interface temperature → more polymer chain diffusion
    % Model: exponential increase above Tg
    temp_factor = 1 - exp(-0.15 * (T_interface - T_glass));
    temp_factor(T_interface < T_glass) = 0;  % No healing below Tg

    %% 2. Time factor
    % Longer time above Tg → more diffusion and healing
    % Typical time scale: 10-100 seconds
    time_constant = 30;  % seconds (characteristic healing time)
    time_factor = 1 - exp(-interlayer_time / time_constant);

    %% 3. Cooling rate factor
    % Slower cooling → less residual stress → better adhesion
    if isfield(thermal_data, 'cooling_rate')
        cooling_rate = thermal_data.cooling_rate;
        max_cooling_rate = 10;  % °C/s (very fast cooling)
        cooling_factor = exp(-cooling_rate / max_cooling_rate);
    else
        cooling_factor = 1.0;  % Default: no penalty
    end

    %% Combined adhesion ratio (0-1)
    % This is the strength relative to bulk material strength
    adhesion_data.adhesion_ratio = temp_factor .* time_factor .* cooling_factor;

    % Ensure bounds [0, 1]
    adhesion_data.adhesion_ratio = max(0, min(adhesion_data.adhesion_ratio, 1.0));

    %% Calculate absolute adhesion strength (MPa)
    % Bulk PLA strength: ~50-70 MPa
    % Actual adhesion is a fraction of bulk strength
    bulk_strength_pla = 60;  % MPa (typical tensile strength of PLA)
    adhesion_data.adhesion_strength = adhesion_data.adhesion_ratio * bulk_strength_pla;

    %% Statistics
    fprintf('  Adhesion strength statistics:\n');
    fprintf('    Mean adhesion ratio: %.3f ± %.3f\n', ...
            mean(adhesion_data.adhesion_ratio, 'omitnan'), ...
            std(adhesion_data.adhesion_ratio, 'omitnan'));
    fprintf('    Min adhesion ratio: %.3f\n', min(adhesion_data.adhesion_ratio));
    fprintf('    Max adhesion ratio: %.3f\n', max(adhesion_data.adhesion_ratio));
    fprintf('    Mean adhesion strength: %.2f ± %.2f MPa\n', ...
            mean(adhesion_data.adhesion_strength, 'omitnan'), ...
            std(adhesion_data.adhesion_strength, 'omitnan'));

    fprintf('Interlayer adhesion calculation complete!\n');

end
