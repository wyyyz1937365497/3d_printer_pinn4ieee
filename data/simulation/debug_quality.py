"""
Debug quality model calculations
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from data.simulation.simulation_pipeline import (
    PrintConfiguration,
    CompletePrintSimulation,
)

def debug_quality_calculation():
    """Debug why quality metrics are at min/max values"""

    print("="*80)
    print("DEBUGGING QUALITY CALCULATION")
    print("="*80)

    sim = CompletePrintSimulation()

    # Test with different temperatures
    for temp in [190, 200, 210, 220, 230, 240]:
        config = PrintConfiguration(
            sample_id=f'debug_{temp}',
            nozzle_temp=float(temp),
            bed_temp=60.0,
            print_speed=50.0,
            print_duration=10.0,
        )

        result = sim.simulate_print(config)
        sensor_data = result['sensor_data']
        quality = result['quality_metrics']

        # Debug adhesion calculation
        thermal_data = {
            'part_temp': sensor_data['part_temp'],
            'cooling_rate': sensor_data['cooling_rate'],
        }

        # Manually call the adhesion model with detailed output
        from data.simulation.quality_formation_model import InterlayerAdhesionModel
        adhesion_model = InterlayerAdhesionModel()

        temp_history = thermal_data['part_temp']
        cooling_rate = thermal_data['cooling_rate']

        # Print key intermediate values
        avg_temp = np.mean(temp_history)
        max_temp = np.max(temp_history)
        time_above_melt = np.sum(temp_history > 170.0)  # PLA melting point
        total_time = len(temp_history) * 0.1
        melt_fraction = time_above_melt / total_time

        T_kelvin = avg_temp + 273.15
        activation_energy = 50000.0  # J/mol
        gas_constant = 8.314
        diffusion_factor = np.exp(-activation_energy / (gas_constant * T_kelvin))

        temp_optimal = 220.0
        temp_deviation = np.abs(avg_temp - temp_optimal)
        temp_factor = np.exp(-(temp_deviation / 20.0) ** 2)

        avg_cooling_rate = np.mean(np.abs(cooling_rate))
        cooling_factor = 1.0 / (1.0 + avg_cooling_rate / 5.0)

        base_strength = 30.0
        pressure_factor = 1.25
        time_factor = 1.0 - np.exp(-melt_fraction * 5)

        # Calculate step by step
        print(f"\n🌡️ Temperature: {temp}°C")
        print(f"   Avg temp: {avg_temp:.2f}°C")
        print(f"   Max temp: {max_temp:.2f}°C")
        print(f"   Time above melt: {time_above_melt}/{total_time:.1f}s ({melt_fraction:.2%})")
        print(f"\n📊 Factors:")
        print(f"   Diffusion factor: {diffusion_factor:.6f} × 1000 = {diffusion_factor*1000:.6f}")
        print(f"   Temp factor: {temp_factor:.6f}")
        print(f"   Cooling factor: {cooling_factor:.6f}")
        print(f"   Pressure factor: {pressure_factor:.6f}")
        print(f"   Time factor: {time_factor:.6f}")
        print(f"\n🔢 Calculation:")
        calculated = base_strength * diffusion_factor * 1000 * temp_factor * cooling_factor * pressure_factor * time_factor
        print(f"   Adhesion = {base_strength} × {diffusion_factor*1000:.3f} × {temp_factor:.3f} × {cooling_factor:.3f} × {pressure_factor:.3f} × {time_factor:.3f}")
        print(f"   = {calculated:.2f} MPa")
        print(f"   After clipping [5.0, 40.0]: {max(5.0, min(calculated, 40.0)):.2f} MPa")
        print(f"   Actual output: {quality['adhesion_strength']:.2f} MPa")

if __name__ == '__main__':
    debug_quality_calculation()
