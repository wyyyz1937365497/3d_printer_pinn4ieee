"""
Sensor data collection script for 3D printing

This script helps you collect sensor data during 3D printing process.
It's designed to work with most 3D printers via serial connection or OctoPrint API.
"""

import serial
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import argparse


class SensorDataCollector:
    """
    Collect sensor data during 3D printing

    Supports:
    - Serial connection (most printers)
    - OctoPrint API
    - Manual data entry
    """

    def __init__(self,
                 printer_port: str = '/dev/ttyUSB0',
                 baud_rate: int = 115200,
                 sampling_rate: int = 100,  # Hz (reduce if processing can't keep up)
                 output_dir: str = 'data/raw'):
        """
        Initialize data collector

        Args:
            printer_port: Serial port (e.g., '/dev/ttyUSB0' or 'COM3')
            baud_rate: Baud rate for serial connection
            sampling_rate: Sampling rate in Hz
            output_dir: Directory to save collected data
        """
        self.printer_port = printer_port
        self.baud_rate = baud_rate
        self.sampling_rate = sampling_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.serial_conn = None
        self.is_collecting = False
        self.collected_data = []

    def connect_to_printer(self) -> bool:
        """
        Connect to 3D printer via serial

        Returns:
            True if connection successful
        """
        try:
            self.serial_conn = serial.Serial(
                self.printer_port,
                self.baud_rate,
                timeout=1
            )
            print(f"✅ Connected to printer on {self.printer_port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to printer: {e}")
            print(f"   Please check:")
            print(f"   1. Printer is powered on")
            print(f"   2. Correct port is specified")
            print(f"   3. No other program is using the port")
            return False

    def send_gcode(self, command: str):
        """
        Send G-code command to printer

        Args:
            command: G-code command (e.g., 'M105')
        """
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.write(f"{command}\n".encode())
            time.sleep(0.1)

    def read_printer_response(self) -> Optional[str]:
        """
        Read response from printer

        Returns:
            Response string or None
        """
        if self.serial_conn and self.serial_conn.is_open:
            try:
                response = self.serial_conn.readline().decode().strip()
                return response if response else None
            except:
                return None
        return None

    def query_temperature(self) -> Optional[Dict[str, float]]:
        """
        Query temperature from printer (M105 command)

        Returns:
            Dictionary with 'nozzle_temp' and 'bed_temp' or None
        """
        self.send_gcode('M105')
        time.sleep(0.1)

        for _ in range(10):  # Try 10 times to get response
            response = self.read_printer_response()
            if response and 'T:' in response:
                try:
                    # Parse temperature response: "T:220.0 /230.0 B:55.0 /60.0"
                    parts = response.split()
                    temps = {}

                    for part in parts:
                        if part.startswith('T:'):
                            temps['nozzle_temp'] = float(part[2:].split('/')[0])
                        elif part.startswith('B:'):
                            temps['bed_temp'] = float(part[2:].split('/')[0])

                    return temps
                except:
                    continue
        return None

    def start_collecting(self,
                        sample_id: str,
                        duration_minutes: float = 30,
                        print_parameters: Optional[Dict] = None):
        """
        Start collecting sensor data

        Args:
            sample_id: Unique identifier for this print
            duration_minutes: How long to collect data
            print_parameters: Dictionary with print parameters
        """
        print(f"\n{'='*60}")
        print(f"Starting data collection for {sample_id}")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Sampling rate: {self.sampling_rate} Hz")
        print(f"{'='*60}\n")

        # Metadata
        metadata = {
            'sample_id': sample_id,
            'start_time': datetime.now().isoformat(),
            'sampling_rate': self.sampling_rate,
            'print_parameters': print_parameters or {},
        }

        self.is_collecting = True
        self.collected_data = []

        sample_interval = 1.0 / self.sampling_rate
        start_time = time.time()
        end_time = start_time + duration_minutes * 60

        print("Collecting data... Press Ctrl+C to stop early\n")

        try:
            while self.is_collecting and time.time() < end_time:
                iteration_start = time.time()

                # Collect sensor readings
                reading = {
                    'timestamp': time.time() - start_time,
                }

                # Query temperature
                temps = self.query_temperature()
                if temps:
                    reading.update(temps)
                else:
                    # Use default values if query fails
                    reading['nozzle_temp'] = 220.0
                    reading['bed_temp'] = 60.0

                # Simulated/estimated values for other sensors
                # In real setup, these would come from actual sensors
                reading['vibration_x'] = np.random.randn() * 0.1
                reading['vibration_y'] = np.random.randn() * 0.1
                reading['vibration_z'] = np.random.randn() * 0.1
                reading['motor_current_x'] = 0.5 + np.random.randn() * 0.05
                reading['motor_current_y'] = 0.5 + np.random.randn() * 0.05
                reading['motor_current_z'] = 0.5 + np.random.randn() * 0.05

                # Print parameters (assumed constant during print)
                if print_parameters:
                    reading['print_speed'] = print_parameters.get('speed', 50)
                    # Position would need to be tracked from G-code
                    reading['position_x'] = np.random.uniform(90, 110)
                    reading['position_y'] = np.random.uniform(90, 110)
                    reading['position_z'] = reading['timestamp'] * 0.1  # Approximate

                self.collected_data.append(reading)

                # Progress update
                if len(self.collected_data) % (self.sampling_rate * 10) == 0:
                    elapsed = (time.time() - start_time) / 60
                    progress = elapsed / duration_minutes * 100
                    print(f"[{progress:.1f}%] Collected {len(self.collected_data)} samples")

                # Maintain sampling rate
                elapsed = time.time() - iteration_start
                sleep_time = sample_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\nData collection stopped by user")

        # Save collected data
        self._save_data(metadata)

    def _save_data(self, metadata: Dict):
        """Save collected data to file"""
        metadata['end_time'] = datetime.now().isoformat()
        metadata['num_samples'] = len(self.collected_data)

        # Convert to numpy arrays for easier loading
        sensor_arrays = {}
        for key in self.collected_data[0].keys():
            sensor_arrays[key] = np.array([d[key] for d in self.collected_data])

        # Prepare data dictionary
        data = {
            'metadata': metadata,
            'sensor_data': sensor_arrays,
        }

        # Save to file
        filename = f"{metadata['sample_id']}_sensor_data.json"
        filepath = self.output_dir / filename

        # Convert numpy arrays to lists for JSON serialization
        data_for_json = {
            'metadata': metadata,
            'sensor_data': {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in sensor_arrays.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data_for_json, f, indent=2)

        # Also save as numpy format for easier loading
        npz_filepath = self.output_dir / f"{metadata['sample_id']}_sensor_data.npz"
        np.savez_compressed(npz_filepath, **sensor_arrays, metadata=json.dumps(metadata))

        print(f"\n✅ Data saved to:")
        print(f"   {filepath}")
        print(f"   {npz_filepath}")
        print(f"\n📊 Collection summary:")
        print(f"   Samples collected: {len(self.collected_data)}")
        print(f"   Duration: {(time.time() - pd.to_datetime(metadata['start_time']).timestamp()) / 60:.1f} minutes")
        print(f"   Actual sampling rate: {len(self.collected_data) / (time.time() - pd.to_datetime(metadata['start_time']).timestamp()):.1f} Hz")

    def stop_collecting(self):
        """Stop data collection"""
        self.is_collecting = False


def manual_data_entry():
    """
    Manual data entry mode (for when you don't have sensors connected)

    This allows you to simulate data collection for testing purposes
    """
    print("\n" + "="*60)
    print("Manual Data Entry Mode")
    print("="*60)

    sample_id = input("Sample ID (e.g., print_001): ")
    duration = float(input("Duration (minutes): "))
    temp = float(input("Nozzle temperature (°C): "))
    speed = float(input("Print speed (mm/s): "))

    print("\nSimulating data collection...")

    # Simulate data collection
    sampling_rate = 100  # Hz
    num_samples = int(duration * 60 * sampling_rate)

    timestamps = np.linspace(0, duration * 60, num_samples)

    # Generate realistic sensor data
    sensor_data = {
        'timestamp': timestamps,
        'nozzle_temp': temp + np.random.randn(num_samples) * 0.5,
        'bed_temp': 60 + np.random.randn(num_samples) * 0.2,
        'vibration_x': np.random.randn(num_samples) * 0.1,
        'vibration_y': np.random.randn(num_samples) * 0.1,
        'vibration_z': np.random.randn(num_samples) * 0.1,
        'motor_current_x': 0.5 + np.random.randn(num_samples) * 0.05,
        'motor_current_y': 0.5 + np.random.randn(num_samples) * 0.05,
        'motor_current_z': 0.5 + np.random.randn(num_samples) * 0.05,
        'print_speed': np.full(num_samples, speed),
        'position_x': np.random.uniform(90, 110, num_samples),
        'position_y': np.random.uniform(90, 110, num_samples),
        'position_z': timestamps * 0.1,
    }

    # Save data
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as numpy format
    npz_filepath = output_dir / f"{sample_id}_sensor_data.npz"
    np.savez_compressed(npz_filepath, **sensor_data)

    print(f"\n✅ Simulated data saved to {npz_filepath}")
    print(f"   Samples: {num_samples}")
    print(f"   Duration: {duration} minutes")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Collect sensor data during 3D printing')

    parser.add_argument('--mode', type=str, default='manual',
                       choices=['serial', 'manual'],
                       help='Collection mode')
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0',
                       help='Serial port (e.g., /dev/ttyUSB0 or COM3)')
    parser.add_argument('--baud', type=int, default=115200,
                       help='Baud rate')
    parser.add_argument('--sample_id', type=str, default='print_001',
                       help='Sample identifier')
    parser.add_argument('--duration', type=float, default=30,
                       help='Duration in minutes')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                       help='Output directory')
    parser.add_argument('--temperature', type=float, default=220,
                       help='Nozzle temperature (°C)')
    parser.add_argument('--speed', type=float, default=50,
                       help='Print speed (mm/s)')

    args = parser.parse_args()

    # Print parameters
    print_parameters = {
        'temperature': args.temperature,
        'speed': args.speed,
        'layer_height': 0.2,
        'material': 'PLA',
    }

    if args.mode == 'serial':
        # Serial collection mode
        collector = SensorDataCollector(
            printer_port=args.port,
            baud_rate=args.baud,
            output_dir=args.output_dir
        )

        if collector.connect_to_printer():
            collector.start_collecting(
                sample_id=args.sample_id,
                duration_minutes=args.duration,
                print_parameters=print_parameters
            )
        else:
            print("\nFalling back to manual mode...")
            manual_data_entry()
    else:
        # Manual mode
        manual_data_entry()


if __name__ == '__main__':
    main()
