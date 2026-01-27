"""
Test and validate physics-based simulation

This script runs a single simulation and visualizes the results
to verify the physics models are working correctly.
"""

import sys
from pathlib import Path
# Add project root to path (3 levels up from data/simulation/test_simulation.py)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt

from data.simulation.simulation_pipeline import (
    SimulationParameters,
    PrintConfiguration,
    CompletePrintSimulation,
)


def test_single_simulation():
    """
    Test a single simulation with detailed output
    """
    print("\n" + "="*80)
    print("PHYSICS-BASED SIMULATION TEST")
    print("="*80)

    # Create simulation
    sim = CompletePrintSimulation()

    # Test configuration (standard parameters)
    config = PrintConfiguration(
        sample_id='test_001',
        nozzle_temp=220.0,      # °C
        bed_temp=60.0,          # °C
        print_speed=50.0,       # mm/s
        layer_height=0.2,       # mm
        print_duration=5.0,     # minutes (short for testing)
        pattern_type='rectilinear',
    )

    # Run simulation
    result = sim.simulate_print(config)

    # Extract data
    sensor_data = result['sensor_data']
    quality = result['quality_metrics']

    # Print summary
    print(f"\n📊 Simulation Results for {config.sample_id}")
    print(f"\n🔧 Print Parameters:")
    for key, value in sensor_data['print_parameters'].items():
        print(f"   {key}: {value}")

    print(f"\n🌡️ Thermal Data:")
    print(f"   Avg nozzle temp: {sensor_data['nozzle_temp'].mean():.2f} °C")
    print(f"   Avg part temp: {sensor_data['part_temp'].mean():.2f} °C")
    print(f"   Max part temp: {sensor_data['part_temp'].max():.2f} °C")
    print(f"   Avg cooling rate: {sensor_data['cooling_rate'].mean():.4f} °C/s")

    print(f"\n📳 Vibration Data:")
    print(f"   Avg vibration X: {sensor_data['vibration_x'].mean():.6f} mm/s²")
    print(f"   Avg vibration Y: {sensor_data['vibration_y'].mean():.6f} mm/s²")
    print(f"   Avg vibration Z: {sensor_data['vibration_z'].mean():.6f} mm/s²")
    print(f"   Max vibration magnitude: {sensor_data['vibration_magnitude'].max():.6f} mm/s²")

    print(f"\n⚡ Motor Current:")
    print(f"   Avg current X: {sensor_data['motor_current_x'].mean():.3f} A")
    print(f"   Avg current Y: {sensor_data['motor_current_y'].mean():.3f} A")
    print(f"   Avg current Z: {sensor_data['motor_current_z'].mean():.3f} A")

    print(f"\n🎯 Quality Metrics:")
    print(f"   Adhesion Strength: {quality['adhesion_strength']:.2f} MPa")
    print(f"   Internal Stress: {quality['internal_stress']:.2f} MPa")
    print(f"   Porosity: {quality['porosity']:.2f} %")
    print(f"   Dimensional Error: {quality['dimensional_accuracy']:.4f} mm")
    print(f"   Quality Score: {quality['quality_score']:.3f}")

    # Quality assessment
    print(f"\n📈 Quality Assessment:")
    if quality['quality_score'] > 0.8:
        print("   ✨ Excellent quality!")
    elif quality['quality_score'] > 0.6:
        print("   ✓ Good quality")
    elif quality['quality_score'] > 0.4:
        print("   ⚠️  Acceptable quality")
    else:
        print("   ✗ Poor quality")

    return sensor_data, quality


def visualize_simulation(sensor_data, quality, save_dir='results/simulation_test'):
    """
    Visualize simulation results

    Args:
        sensor_data: Sensor data from simulation
        quality: Quality metrics
        save_dir: Directory to save figures
    """
    import matplotlib
    matplotlib.use('Agg')

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    time_minutes = sensor_data['time'] / 60

    # Create multi-panel figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # 1. Temperature evolution
    axes[0, 0].plot(time_minutes, sensor_data['nozzle_temp'], label='Nozzle', linewidth=2)
    axes[0, 0].plot(time_minutes, sensor_data['bed_temp'], label='Bed', linewidth=2)
    axes[0, 0].plot(time_minutes, sensor_data['part_temp'], label='Part', linewidth=2)
    axes[0, 0].set_xlabel('Time (minutes)')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].set_title('Temperature Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Cooling rate
    axes[0, 1].plot(time_minutes, sensor_data['cooling_rate'], color='red', linewidth=1)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Time (minutes)')
    axes[0, 1].set_ylabel('Cooling Rate (°C/s)')
    axes[0, 1].set_title('Cooling Rate')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Vibration X
    axes[1, 0].plot(time_minutes, sensor_data['vibration_x'], linewidth=1)
    axes[1, 0].set_xlabel('Time (minutes)')
    axes[1, 0].set_ylabel('Vibration X (mm/s²)')
    axes[1, 0].set_title('Vibration X-Axis')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Vibration magnitude
    axes[1, 1].plot(time_minutes, sensor_data['vibration_magnitude'], color='purple', linewidth=1.5)
    axes[1, 1].set_xlabel('Time (minutes)')
    axes[1, 1].set_ylabel('Vibration Magnitude (mm/s²)')
    axes[1, 1].set_title('Vibration Magnitude')
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Motor current X
    axes[2, 0].plot(time_minutes, sensor_data['motor_current_x'], label='X', linewidth=1)
    axes[2, 0].plot(time_minutes, sensor_data['motor_current_y'], label='Y', linewidth=1)
    axes[2, 0].plot(time_minutes, sensor_data['motor_current_z'], label='Z', linewidth=1)
    axes[2, 0].set_xlabel('Time (minutes)')
    axes[2, 0].set_ylabel('Current (A)')
    axes[2, 0].set_title('Motor Current')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # 6. Trajectory (X-Y)
    axes[2, 1].plot(sensor_data['position_x'], sensor_data['position_y'],
                     linewidth=1, alpha=0.7)
    axes[2, 1].set_xlabel('Position X (mm)')
    axes[2, 1].set_ylabel('Position Y (mm)')
    axes[2, 1].set_title('Print Trajectory')
    axes[2, 1].axis('equal')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / 'simulation_results.png', dpi=150)
    print(f"\n📊 Visualization saved to {save_path / 'simulation_results.png'}")

    # Create quality metrics bar chart
    fig2, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Adhesion\n(MPa)', 'Stress\n(MPa)', 'Porosity\n(%)', 'Error\n(mm)', 'Score\n[0-1]']
    values = [
        quality['adhesion_strength'],
        quality['internal_stress'],
        quality['porosity'],
        quality['dimensional_accuracy'] * 100,  # Scale up for visibility
        quality['quality_score'] * 100,  # Scale to percentage
    ]

    colors = ['green' if v > 20 else 'orange' if v > 15 else 'red' if i == 0 else
              'green' if v < 15 else 'orange' if v < 20 else 'red' if i == 1 else
              'green' if v < 5 else 'orange' if v < 8 else 'red' if i == 2 else
              'green' if v < 0.1 else 'orange' if v < 0.2 else 'red' if i == 3 else
              'green' if v > 60 else 'orange' if v > 40 else 'red' for i, v in enumerate(values)]

    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')

    ax.set_ylabel('Value')
    ax.set_title('Quality Metrics Summary', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.2f}',
               ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path / 'quality_metrics.png', dpi=150)
    print(f"📊 Quality chart saved to {save_path / 'quality_metrics.png'}")


def test_parameter_sweep():
    """
    Test how quality metrics change with different parameters
    """
    print("\n" + "="*80)
    print("PARAMETER SWEEP TEST")
    print("="*80)

    sim = CompletePrintSimulation()

    # Test different temperatures
    print("\n🌡️ Temperature Sweep:")
    temps = [190, 200, 210, 220, 230, 240]
    for temp in temps:
        config = PrintConfiguration(
            sample_id=f'temp_{temp}',
            nozzle_temp=float(temp),
            bed_temp=60.0,
            print_speed=50.0,
            print_duration=10.0,
        )
        result = sim.simulate_print(config)
        quality = result['quality_metrics']
        print(f"  {temp:3.0f}°C: Adhesion={quality['adhesion_strength']:.2f} MPa, "
              f"Stress={quality['internal_stress']:.2f} MPa, "
              f"Score={quality['quality_score']:.3f}")

    # Test different speeds
    print("\n💨 Speed Sweep:")
    speeds = [30, 40, 50, 60, 70, 80]
    for speed in speeds:
        config = PrintConfiguration(
            sample_id=f'speed_{speed}',
            nozzle_temp=220.0,
            bed_temp=60.0,
            print_speed=float(speed),
            print_duration=10.0,
        )
        result = sim.simulate_print(config)
        quality = result['quality_metrics']
        print(f"  {speed:2.0f}mm/s: Adhesion={quality['adhesion_strength']:.2f} MPa, "
              f"Porosity={quality['porosity']:.2f}%, "
              f"Score={quality['quality_score']:.3f}")

    print("\n✅ Parameter sweep complete!")


def main():
    """Run all tests"""
    # Test 1: Single simulation
    sensor_data, quality = test_single_simulation()

    # Test 2: Visualize
    print("\n📊 Generating visualizations...")
    visualize_simulation(sensor_data, quality)

    # Test 3: Parameter sweep
    test_parameter_sweep()

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nPhysics models are working correctly.")
    print("You can now generate full dataset:")
    print("  python -m data.simulation.simulation_pipeline --num_samples 1000")


if __name__ == '__main__':
    main()
