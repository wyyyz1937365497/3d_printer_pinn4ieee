"""
Pair sensor data with quality test results and preprocess for training

This script:
1. Pairs sensor data with corresponding quality test results
2. Converts data to the format required for training
3. Splits into train/val/test sets
4. Normalizes features
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
import argparse


class DataPreprocessor:
    """
    Preprocess data for training
    """

    def __init__(self,
                 raw_sensor_dir: str = 'data/raw',
                 raw_quality_dir: str = 'data/raw/quality_data',
                 output_dir: str = 'data/processed'):
        """
        Initialize preprocessor

        Args:
            raw_sensor_dir: Directory with raw sensor data
            raw_quality_dir: Directory with quality test results
            output_dir: Directory to save processed data
        """
        self.raw_sensor_dir = Path(raw_sensor_dir)
        self.raw_quality_dir = Path(raw_quality_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_sensor_data(self) -> Dict[str, np.ndarray]:
        """
        Load all sensor data from directory

        Returns:
            Dictionary mapping sample_id to sensor data
        """
        sensor_data = {}

        # Find all .npz files
        npz_files = list(self.raw_sensor_dir.glob('*_sensor_data.npz'))

        print(f"Found {len(npz_files)} sensor data files")

        for npz_file in npz_files:
            sample_id = npz_file.stem.replace('_sensor_data', '')
            try:
                data = np.load(npz_file)
                sensor_data[sample_id] = data
                print(f"  ✅ Loaded {sample_id}")
            except Exception as e:
                print(f"  ❌ Failed to load {sample_id}: {e}")

        return sensor_data

    def load_quality_data(self) -> Dict[str, Dict]:
        """
        Load all quality test results

        Returns:
            Dictionary mapping sample_id to quality data
        """
        quality_data = {}

        # Find all JSON files
        json_files = list(self.raw_quality_dir.glob('*_quality_data.json'))

        print(f"\nFound {len(json_files)} quality data files")

        for json_file in json_files:
            sample_id = json_file.stem.replace('_quality_data', '')
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    quality_data[sample_id] = data
                    print(f"  ✅ Loaded {sample_id}")
            except Exception as e:
                print(f"  ❌ Failed to load {sample_id}: {e}")

        return quality_data

    def pair_data(self,
                 sensor_data: Dict,
                 quality_data: Dict) -> List[Dict]:
        """
        Pair sensor data with quality data

        Args:
            sensor_data: Dictionary of sensor data
            quality_data: Dictionary of quality data

        Returns:
            List of paired data dictionaries
        """
        paired_data = []

        print("\nPairing sensor and quality data...")

        # Find common sample IDs
        sensor_ids = set(sensor_data.keys())
        quality_ids = set(quality_data.keys())
        common_ids = sensor_ids.intersection(quality_ids)

        print(f"Common samples: {len(common_ids)}")
        print(f"Only sensor data: {len(sensor_ids - quality_ids)}")
        print(f"Only quality data: {len(quality_ids - sensor_ids)}")

        for sample_id in common_ids:
            try:
                sensor = sensor_data[sample_id]
                quality = quality_data[sample_id]

                # Extract features
                features = self._extract_features(sensor)

                # Extract targets
                targets = quality['quality_metrics']

                paired_data.append({
                    'sample_id': sample_id,
                    'features': features,
                    'targets': targets,
                })

                print(f"  ✅ Paired {sample_id}")

            except Exception as e:
                print(f"  ❌ Failed to pair {sample_id}: {e}")

        return paired_data

    def _extract_features(self, sensor_data: np.ndarray) -> np.ndarray:
        """
        Extract features from sensor data

        Creates a [seq_len, num_features] array

        Args:
            sensor_data: Sensor data from npz file

        Returns:
            Feature array
        """
        # Required features
        feature_keys = [
            'nozzle_temp',
            'bed_temp',
            'vibration_x',
            'vibration_y',
            'vibration_z',
            'motor_current_x',
            'motor_current_y',
            'motor_current_z',
            'print_speed',
            'position_x',
            'position_y',
            'position_z',
        ]

        # Stack features
        features_list = []
        for key in feature_keys:
            if key in sensor_data:
                features_list.append(sensor_data[key])
            else:
                # Use default value if missing
                seq_len = len(sensor_data['timestamp'])
                if 'temp' in key:
                    features_list.append(np.full(seq_len, 220.0))
                else:
                    features_list.append(np.zeros(seq_len))

        # Stack along feature dimension
        features = np.stack(features_list, axis=1)  # [seq_len, num_features]

        return features

    def normalize_features(self,
                          paired_data: List[Dict],
                          fit_on_train: bool = True) -> Tuple[List[Dict], StandardScaler]:
        """
        Normalize features using StandardScaler

        Args:
            paired_data: List of paired data
            fit_on_train: Whether to fit scaler on all data (True) or return unfitted (False)

        Returns:
            Normalized data and fitted scaler
        """
        print("\nNormalizing features...")

        # Collect all features for fitting scaler
        all_features = []
        for data in paired_data:
            all_features.append(data['features'])

        # Concatenate all features
        concatenated = np.concatenate(all_features, axis=0)  # [total_samples, num_features]

        # Fit scaler
        scaler = StandardScaler()
        scaler.fit(concatenated)

        print(f"  Feature means: {scaler.mean_}")
        print(f"  Feature stds: {scaler.scale_}")

        # Normalize each sample
        normalized_data = []
        for data in paired_data:
            normalized_features = scaler.transform(data['features'])
            normalized_data.append({
                'sample_id': data['sample_id'],
                'features': normalized_features,
                'targets': data['targets'],
            })

        print("  ✅ Features normalized")

        return normalized_data, scaler

    def split_data(self,
                  paired_data: List[Dict],
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  shuffle: bool = True,
                  seed: int = 42) -> Tuple[List, List, List]:
        """
        Split data into train/val/test sets

        Args:
            paired_data: List of paired data
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            shuffle: Whether to shuffle data
            seed: Random seed

        Returns:
            train, val, test data lists
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        np.random.seed(seed)

        n_samples = len(paired_data)
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        train_data = [paired_data[i] for i in train_indices]
        val_data = [paired_data[i] for i in val_indices]
        test_data = [paired_data[i] for i in test_indices]

        print(f"\nData split:")
        print(f"  Train: {len(train_data)} samples ({len(train_data)/n_samples*100:.1f}%)")
        print(f"  Val: {len(val_data)} samples ({len(val_data)/n_samples*100:.1f}%)")
        print(f"  Test: {len(test_data)} samples ({len(test_data)/n_samples*100:.1f}%)")

        return train_data, val_data, test_data

    def save_processed_data(self,
                           train_data: List,
                           val_data: List,
                           test_data: List,
                           scaler: StandardScaler):
        """
        Save processed data to files

        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            scaler: Fitted StandardScaler
        """
        print("\nSaving processed data...")

        # Save train data
        train_path = self.output_dir / 'train_data.pkl'
        with open(train_path, 'wb') as f:
            pickle.dump(train_data, f)
        print(f"  ✅ Train data saved to {train_path}")

        # Save val data
        val_path = self.output_dir / 'val_data.pkl'
        with open(val_path, 'wb') as f:
            pickle.dump(val_data, f)
        print(f"  ✅ Val data saved to {val_path}")

        # Save test data
        test_path = self.output_dir / 'test_data.pkl'
        with open(test_path, 'wb') as f:
            pickle.dump(test_data, f)
        print(f"  ✅ Test data saved to {test_path}")

        # Save scaler
        scaler_path = self.output_dir / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"  ✅ Scaler saved to {scaler_path}")

        # Save metadata
        metadata = {
            'num_train_samples': len(train_data),
            'num_val_samples': len(val_data),
            'num_test_samples': len(test_data),
            'num_features': train_data[0]['features'].shape[1],
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
        }

        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✅ Metadata saved to {metadata_path}")

    def process_pipeline(self,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15):
        """
        Run complete preprocessing pipeline

        Args:
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            test_ratio: Test data ratio
        """
        print("="*60)
        print("Data Preprocessing Pipeline")
        print("="*60)

        # Step 1: Load data
        sensor_data = self.load_sensor_data()
        quality_data = self.load_quality_data()

        if not sensor_data or not quality_data:
            print("\n❌ Error: No data found!")
            print("   Please ensure you have:")
            print("   1. Sensor data in data/raw/")
            print("   2. Quality data in data/raw/quality_data/")
            return

        # Step 2: Pair data
        paired_data = self.pair_data(sensor_data, quality_data)

        if not paired_data:
            print("\n❌ Error: No paired data found!")
            print("   Ensure sample IDs match between sensor and quality data")
            return

        # Step 3: Normalize
        normalized_data, scaler = self.normalize_features(paired_data)

        # Step 4: Split
        train_data, val_data, test_data = self.split_data(
            normalized_data,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )

        # Step 5: Save
        self.save_processed_data(train_data, val_data, test_data, scaler)

        print("\n" + "="*60)
        print("✅ Preprocessing completed successfully!")
        print("="*60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Pair and preprocess data for training')

    parser.add_argument('--sensor_dir', type=str, default='data/raw',
                       help='Directory with sensor data')
    parser.add_argument('--quality_dir', type=str, default='data/raw/quality_data',
                       help='Directory with quality data')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test data ratio')

    args = parser.parse_args()

    # Create preprocessor
    preprocessor = DataPreprocessor(
        raw_sensor_dir=args.sensor_dir,
        raw_quality_dir=args.quality_dir,
        output_dir=args.output_dir
    )

    # Run pipeline
    preprocessor.process_pipeline(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )


if __name__ == '__main__':
    main()
