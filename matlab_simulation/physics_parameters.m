function params = physics_parameters()
% PHYSICS_PARAMETERS - Physical parameters for Ender-3 V2 FDM 3D Printer Simulation
%
% This file contains all physical parameters needed for accurate simulation of:
% 1. Trajectory error due to inertia and belt elasticity (second-order system)
% 2. Thermal field and interlayer adhesion (moving heat source model)
%
% Sources:
% - Ender-3 V2 technical specifications
% - PLA material properties (thermal and mechanical)
% - GT2 belt characteristics
% - NEMA 17 42-34 stepper motor specifications
% - FDM heat transfer coefficients from literature
%
% Output: params structure with all parameters

%% ========================================================================
% 1. PRINTER MECHANICAL PARAMETERS (Ender-3 V2)
% ===========================================================================

% --- Moving Mass ---
% The effective mass that moves during printing
params.mass.extruder_assy = 0.350;      % kg - Extruder assembly (hotend + duct + mount)
params.mass.x_carriage = 0.120;          % kg - X-carriage
params.mass.y_belt = 0.015;             % kg - Moving belt mass (estimated)
params.mass.total_x = 0.485;            % kg - Total moving mass in X direction
params.mass.total_y = 0.650;            % kg - Total moving mass in Y direction (includes X)

% --- GT2 Belt Specifications ---
% GT2 timing belt characteristics (affects elasticity)
params.belt.pitch = 2.0;                % mm - Tooth pitch
params.belt.width = 6.0;                % mm - Belt width
params.belt.length_x = 0.420;           % m - X-axis belt length
params.belt.length_y = 0.520;           % m - Y-axis belt length
params.belt.stiffness = 150000;         % N/m - Effective belt stiffness (experimental)
params.belt.damping = 25.0;             % N·s/m - Belt damping coefficient

% Belt stiffness calculation reference:
% For rubber timing belt: k ≈ EA/L where E≈2GPa, A≈width*pitch
% k ≈ (2e9 × 6e-3 × 2e-3) / 0.45 ≈ 53,000 N/m (single strand)
% With preload and tensioning, effective stiffness is higher
% Sources: GT2 belt specifications, mechanical engineering handbooks

% --- Stepper Motor Specifications (NEMA 17 42-34) ---
% Used for X and Y axes
params.motor.step_angle = 1.8;          % degrees - Step angle
params.motor.steps_per_rev = 200;       % steps/rev
params.motor.holding_torque = 0.40;     % N·m - Holding torque (rated)
params.motor.current = 1.5;             % A - Rated current
params.motor.voltage = 12.0;            % V - Supply voltage
params.motor.inductance = 3.0;          % mH - Phase inductance
params.motor.resistance = 1.5;          % Ohm - Phase resistance

% Motor rotor inertia
params.motor.rotor_inertia = 54e-6;     % kg·m² - Rotor inertia (typical for NEMA 17)
params.motor.detent_torque = 0.02;      % N·m - Detent torque

% --- Transmission System ---
% Gear reduction and pulley system
params.pulley.teeth = 20;               % teeth - GT2 pulley
params.pulley.diameter = 12.13;         % mm - Effective diameter (20 teeth × 2mm / π)
params.pulley.radius = 6.065e-3;        % m - Effective radius
params.microstepping = 16;              % - - Microstepping setting (1/16)

% --- Jerk and Acceleration Limits ---
% From Marlin firmware configuration (M205, M201)
params.motion.max_accel = 500;          % mm/s² - Maximum acceleration
params.motion.max_accel_x = 500;        % mm/s² - X-axis max acceleration
params.motion.max_accel_y = 500;        % mm/s² - Y-axis max acceleration
params.motion.max_jerk = 10.0;          % mm/s - Instantaneous velocity change (jerk)
params.motion.max_jerk_x = 8.0;         % mm/s - X-axis jerk limit
params.motion.max_jerk_y = 8.0;         % mm/s - Y-axis jerk limit
params.motion.max_velocity = 500;       % mm/s - Maximum velocity (from M203)

% --- System Dynamics Parameters ---
% Second-order system parameters for trajectory error modeling
% m·x'' + c·x' + k·x = F(t)

% X-axis dynamics
params.dynamics.x.mass = 0.485;         % kg - Effective mass
params.dynamics.x.stiffness = 150000;   % N/m - Belt stiffness
params.dynamics.x.damping = 25.0;       % N·s/m - Damping coefficient
params.dynamics.x.natural_freq = sqrt(params.dynamics.x.stiffness / params.dynamics.x.mass);  % rad/s
params.dynamics.x.damping_ratio = params.dynamics.x.damping / (2 * sqrt(params.dynamics.x.mass * params.dynamics.x.stiffness));
params.dynamics.x.settling_time = 4 / (params.dynamics.x.damping_ratio * params.dynamics.x.natural_freq);  % s

% Y-axis dynamics
params.dynamics.y.mass = 0.650;         % kg - Effective mass
params.dynamics.y.stiffness = 150000;   % N/m - Belt stiffness
params.dynamics.y.damping = 25.0;       % N·s/m - Damping coefficient
params.dynamics.y.natural_freq = sqrt(params.dynamics.y.stiffness / params.dynamics.y.mass);  % rad/s
params.dynamics.y.damping_ratio = params.dynamics.y.damping / (2 * sqrt(params.dynamics.y.mass * params.dynamics.y.stiffness));
params.dynamics.y.settling_time = 4 / (params.dynamics.y.damping_ratio * params.dynamics.y.natural_freq);  % s

%% ========================================================================
% 2. PLA MATERIAL PROPERTIES
% ===========================================================================

% --- Mechanical Properties ---
params.material.name = 'PLA';
params.material.density = 1240;          % kg/m³ - Density (from G-code)
params.material.elastic_modulus = 3.5e9; % Pa - Young's modulus (3.5 GPa)
params.material.poisson_ratio = 0.36;   % - - Poisson's ratio
params.material.yield_strength = 60e6;   % Pa - Yield strength (60 MPa)
params.material.tensile_strength = 70e6; % Pa - Ultimate tensile strength (70 MPa)

% --- Thermal Properties ---
% Sources: PLA technical data sheets and research papers
params.material.thermal_conductivity = 0.13;  % W/(m·K) - Thermal conductivity
params.material.specific_heat = 1200;         % J/(kg·K) - Specific heat capacity
params.material.thermal_diffusivity = params.material.thermal_conductivity / ...
                                      (params.material.density * params.material.specific_heat);  % m²/s

% PLA thermal diffusivity ≈ 0.13 / (1240 × 1200) ≈ 8.7e-8 m²/s

% --- Phase Transition Temperatures ---
params.material.glass_transition = 60;   % °C - Glass transition temperature (Tg)
params.material.melting_point = 171;     % °C - Melting temperature (Tm)
params.material.cold_crystallization = 107;  % °C - Cold crystallization

% --- Printing Temperatures ---
params.printing.nozzle_temp = 220;       % °C - Nozzle temperature (from G-code)
params.printing.bed_temp = 60;           % °C - Heated bed temperature (from G-code)
params.printing.min_fan_temp = 220;      % °C - Temperature at which fan starts
params.printing.chamber_temp = 25;       % °C - Ambient chamber temperature (typical)

% --- Filament Specifications ---
params.filament.diameter = 1.75;         % mm - Nominal filament diameter
params.filament.density = 1.24;          % g/cm³ - Filament density (from G-code)

%% ========================================================================
% 3. EXTRUSION AND FLOW PARAMETERS
% ===========================================================================

% --- Nozzle Specifications ---
params.nozzle.diameter = 0.4;           % mm - Nozzle diameter
params.nozzle.material = 'Brass';       % - - Nozzle material

% --- Extrusion Parameters ---
params.extrusion.width = 0.45;          % mm - Extrusion width (from G-code)
params.extrusion.height = 0.2;          % mm - Layer height (from G-code)
params.extrusion.length_ratio = 1.0;    % - - Extrusion multiplier

% --- Extrusion Geometry ---
params.extrusion.cross_section_area = params.extrusion.width * params.extrusion.height;  % mm²
params.extrusion.volume_flow_max = 15;  % mm³/s - Maximum volumetric flow rate

% --- Heat Input Model ---
% Heat input from extrusion: Q = ṁ × c × ΔT
% where ṁ = ρ × A × v (mass flow rate)
params.extrusion.heat_capacity_flow = params.material.density * ...
                                      params.extrusion.cross_section_area * 1e-9 * ...
                                      params.material.specific_heat;  % J/(m·K)

%% ========================================================================
% 4. THERMAL MODEL PARAMETERS (Moving Heat Source)
% ===========================================================================

% --- Heat Transfer Coefficients ---
% Sources: FDM heat transfer literature, experimental measurements
params.heat_transfer.h_convection_no_fan = 10;    % W/(m²·K) - Natural convection (no fan)
params.heat_transfer.h_convection_with_fan = 44;  % W/(m²·K) - Forced convection (fan on)
params.heat_transfer.h_conduction_bed = 150;      % W/(m²·K) - Contact with heated bed
params.heat_transfer.h_radiation = 10;            % W/(m²·K) - Effective radiation (linearized)

% Combined heat transfer coefficient (typical printing condition)
params.heat_transfer.h_combined = params.heat_transfer.h_convection_with_fan + ...
                                   params.heat_transfer.h_radiation;  % W/(m²·K)

% Stefan-Boltzmann constant for radiation
params.heat_transfer.sigma = 5.67e-8;     % W/(m²·K⁴) - Stefan-Boltzmann constant

% --- Emissivity ---
params.heat_transfer.emissivity_pla = 0.92;  % - - PLA emissivity
params.heat_transfer.emissivity_bed = 0.95;  % - - Build surface emissivity

% --- Cooling Fan Specifications ---
params.fan.max_speed = 255;               % - - PWM maximum value (0-255)
params.fan.typical_speed = 255;           % - - Typical operating speed
params.fan.diameter = 40;                 % mm - Fan diameter
params.fan.flow_rate = 5.5;               % CFM - Air flow (typical 40mm fan)

% --- Environmental Conditions ---
params.environment.ambient_temp = 25;     % °C - Ambient temperature
params.environment.humidity = 50;         % % - Relative humidity (affects cooling)
params.environment.chamber_temp = 25;     % °C - Chamber temperature (Ender-3 V2 is open-frame)

%% ========================================================================
% 5. NUMERICAL SIMULATION PARAMETERS
% ===========================================================================

% --- Time Stepping ---
params.simulation.time_step = 0.001;     % s - Simulation time step (1 ms)
params.simulation.max_time = 1000;       % s - Maximum simulation time

% --- Spatial Discretization (for thermal model) ---
params.simulation.dx = 1.0;              % mm - Spatial resolution in X
params.simulation.dy = 1.0;              % mm - Spatial resolution in Y
params.simulation.dz = 0.1;              % mm - Spatial resolution in Z (layer resolution)

% --- Stability Criteria ---
% Explicit finite difference: Δt ≤ Δx² / (4α)
params.simulation.dt_stability_limit = (params.simulation.dx * 1e-3)^2 / ...
                                      (4 * params.material.thermal_diffusivity);  % s

% Adaptive time step for thermal simulation
params.simulation.dt_thermal = min(params.simulation.time_step, ...
                                   params.simulation.dt_stability_limit * 0.9);  % s

% --- Output Configuration ---
params.output.save_interval = 100;       % - - Save every N steps
params.output.interpolate = true;        % - - Interpolate to uniform time grid

%% ========================================================================
% 6. INTERLAYER ADHESION MODEL PARAMETERS
% ===========================================================================

% Wool-O'Connor Polymer Healing Model
% Sources: Research on FDM interlayer bonding
%
% Healing model: H = H∞ × exp(-Ea/RT) × t^n
% where:
%   H - Healing ratio (bond strength development)
%   H∞ - Maximum healing
%   Ea - Activation energy
%   R - Gas constant
%   T - Temperature (K)
%   t - Time
%   n - Time exponent

params.adhesion.activation_energy = 50e3;    % J/mol - Activation energy for PLA diffusion
params.adhesion.gas_constant = 8.314;        % J/(mol·K) - Universal gas constant
params.adhesion.time_exponent = 0.5;         % - - Typically 0.5 for Fickian diffusion
params.adhesion.max_healing = 1.0;           % - - Maximum healing ratio
params.adhesion.reference_temp = 220;        % °C - Reference temperature

% Simplified adhesion strength model (temperature-dependent)
% σ_adhesion = σ_bulk × [1 - exp(-t/τ(T))]
% where τ(T) = τ₀ × exp(Ea/RT)

params.adhesion.bulk_strength = 70e6;        % Pa - Bulk material strength
params.adhesion.pre_exponential = 1e-3;      % s - Pre-exponential factor

% Critical temperature for molecular diffusion
params.adhesion.min_diffusion_temp = params.material.glass_transition + 10;  % °C
params.adhesion.optimal_temp = params.material.melting_point - 20;  % °C

% --- Healing Time Threshold ---
params.adhesion.min_healing_time = 0.5;      % s - Minimum time for any bonding
params.adhesion.optimal_healing_time = 2.0;  % s - Optimal time for maximum strength

%% ========================================================================
% 7. G-CODE PROCESSING PARAMETERS
% ===========================================================================

% G-code parsing configuration
params.gcode.coordinate_system = 'absolute';  % - - G90 (absolute) or G91 (relative)
params.gcode.extrusion_mode = 'relative';     % - - E values (typically relative)

% Corner detection parameters
params.gcode.corner_angle_threshold = 15;     % degrees - Minimum angle to detect corner
params.gcode.min_segment_length = 0.1;        % mm - Minimum segment to process

% Travel vs extrusion classification
params.gcode.extrusion_threshold = 0.01;     % mm - Minimum E change to be extrusion

%% ========================================================================
% 8. DIAGNOSTIC AND DEBUG PARAMETERS
% ===========================================================================

params.debug.plot_trajectory = false;        % - - Plot reference vs actual trajectory
params.debug.plot_temperature = false;       % - - Plot temperature field evolution
params.debug.plot_forces = false;            % - - Plot inertial and elastic forces
params.debug.verbose = false;                % - - Print progress messages (also controls plot generation)

params.debug.save_intermediate = false;      % - - Save intermediate results
params.debug.check_stability = true;         % - - Check numerical stability

%% ========================================================================
% 9. VALIDATION AND REFERENCE DATA
% ===========================================================================

% Experimental validation data (from literature)
% Sources: Research papers on Ender-3 V2 performance

params.validation.typical_corner_error = 0.3;  % mm - Typical corner rounding error
params.validation.resonance_freq_x = 45;       % Hz - X-axis resonance frequency
params.validation.resonance_freq_y = 35;       % Hz - Y-axis resonance frequency

% Print quality metrics
params.quality.max_allowable_error = 0.5;     % mm - Maximum acceptable dimensional error
params.quality.min_adhesion_ratio = 0.7;       % - - Minimum interlayer strength ratio

%% ========================================================================
% END OF PARAMETER DEFINITION
% ===========================================================================

% Display summary if verbose
if params.debug.verbose
    fprintf('========================================\n');
    fprintf('Physics Parameters Loaded Successfully\n');
    fprintf('========================================\n');
    fprintf('Printer: Ender-3 V2\n');
    fprintf('Material: %s\n', params.material.name);
    fprintf('\n');
    fprintf('X-axis Dynamics:\n');
    fprintf('  Mass: %.3f kg\n', params.dynamics.x.mass);
    fprintf('  Natural freq: %.2f rad/s (%.2f Hz)\n', ...
            params.dynamics.x.natural_freq, ...
            params.dynamics.x.natural_freq / (2*pi));
    fprintf('  Damping ratio: %.4f\n', params.dynamics.x.damping_ratio);
    fprintf('\n');
    fprintf('Y-axis Dynamics:\n');
    fprintf('  Mass: %.3f kg\n', params.dynamics.y.mass);
    fprintf('  Natural freq: %.2f rad/s (%.2f Hz)\n', ...
            params.dynamics.y.natural_freq, ...
            params.dynamics.y.natural_freq / (2*pi));
    fprintf('  Damping ratio: %.4f\n', params.dynamics.y.damping_ratio);
    fprintf('\n');
    fprintf('Thermal Properties:\n');
    fprintf('  Conductivity: %.2f W/(m·K)\n', params.material.thermal_conductivity);
    fprintf('  Specific heat: %.0f J/(kg·K)\n', params.material.specific_heat);
    fprintf('  Diffusivity: %.2e m²/s\n', params.material.thermal_diffusivity);
    fprintf('\n');
    fprintf('Heat Transfer:\n');
    fprintf('  Convection (fan): %d W/(m²·K)\n', params.heat_transfer.h_convection_with_fan);
    fprintf('  Convection (no fan): %d W/(m²·K)\n', params.heat_transfer.h_convection_no_fan);
    fprintf('========================================\n');
end

end
