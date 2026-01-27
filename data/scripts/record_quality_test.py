"""
Record quality test results

This script helps record quality test results after printing
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import argparse


class QualityTestRecorder:
    """
    Record quality test results

    Stores results for:
    - Adhesion strength (tensile test)
    - Internal stress
    - Porosity
    - Dimensional accuracy
    """

    def __init__(self, output_dir: str = 'data/raw/quality_data'):
        """
        Initialize recorder

        Args:
            output_dir: Directory to save quality test results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def record_test(self,
                   sample_id: str,
                   adhesion_strength: Optional[float] = None,
                   internal_stress: Optional[float] = None,
                   porosity: Optional[float] = None,
                   dimensional_accuracy: Optional[float] = None,
                   quality_score: Optional[float] = None,
                   test_notes: str = "") -> Dict:
        """
        Record quality test results

        Args:
            sample_id: Sample identifier (should match sensor data sample_id)
            adhesion_strength: Interlayer adhesion strength (MPa)
            internal_stress: Internal/residual stress (MPa)
            porosity: Porosity percentage (%)
            dimensional_accuracy: Dimensional error (mm)
            quality_score: Overall quality score [0-1]
            test_notes: Additional notes

        Returns:
            Dictionary with recorded data
        """
        # Prepare data
        test_data = {
            'sample_id': sample_id,
            'test_date': datetime.now().isoformat(),
            'quality_metrics': {},
            'test_info': {
                'notes': test_notes,
            }
        }

        # Add quality metrics if provided
        if adhesion_strength is not None:
            test_data['quality_metrics']['adhesion_strength'] = adhesion_strength

        if internal_stress is not None:
            test_data['quality_metrics']['internal_stress'] = internal_stress

        if porosity is not None:
            test_data['quality_metrics']['porosity'] = porosity

        if dimensional_accuracy is not None:
            test_data['quality_metrics']['dimensional_accuracy'] = dimensional_accuracy

        if quality_score is not None:
            test_data['quality_metrics']['quality_score'] = quality_score
        else:
            # Calculate quality score if not provided
            test_data['quality_metrics']['quality_score'] = self._calculate_quality_score(
                adhesion_strength, internal_stress, porosity, dimensional_accuracy
            )

        # Save to file
        filename = f"{sample_id}_quality_data.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(test_data, f, indent=2)

        print(f"✅ Quality data saved to {filepath}")

        return test_data

    def _calculate_quality_score(self,
                                adhesion: Optional[float],
                                stress: Optional[float],
                                porosity: Optional[float],
                                accuracy: Optional[float]) -> float:
        """
        Calculate overall quality score from individual metrics

        Uses sigmoid functions to normalize each metric to [0, 1]

        Args:
            adhesion: Adhesion strength (MPa)
            stress: Internal stress (MPa)
            porosity: Porosity (%)
            accuracy: Dimensional error (mm)

        Returns:
            Quality score [0, 1]
        """
        score = 0.5  # Default score
        count = 0

        # Adhesion contribution (35%)
        if adhesion is not None:
            adhesion_score = 1.0 / (1.0 + np.exp(-(adhesion - 20) / 5))
            score += 0.35 * adhesion_score
            count += 1

        # Stress contribution (25%) - lower is better
        if stress is not None:
            stress_score = 1.0 / (1.0 + np.exp(-(15 - stress) / 5))
            score += 0.25 * stress_score
            count += 1

        # Porosity contribution (20%) - lower is better
        if porosity is not None:
            porosity_score = 1.0 / (1.0 + np.exp(-(5 - porosity) / 2))
            score += 0.20 * porosity_score
            count += 1

        # Accuracy contribution (20%) - lower is better
        if accuracy is not None:
            accuracy_score = 1.0 / (1.0 + np.exp(-(0.1 - accuracy) / 0.03))
            score += 0.20 * accuracy_score
            count += 1

        # Normalize if partial data
        if count > 0 and count < 4:
            score = score / (0.35 + 0.25 + 0.20 + 0.20) * count

        return np.clip(score, 0.0, 1.0)

    def batch_record(self, test_results: list):
        """
        Record multiple test results at once

        Args:
            test_results: List of dictionaries, each containing test data
        """
        for result in test_results:
            self.record_test(**result)

        print(f"\n✅ Batch recorded {len(test_results)} test results")


def interactive_recording():
    """
    Interactive mode for recording quality test results
    """
    print("\n" + "="*60)
    print("Quality Test Recording")
    print("="*60 + "\n")

    recorder = QualityTestRecorder()

    sample_id = input("Sample ID (e.g., print_001): ")

    print("\nQuality Test Results (press Enter to skip):")

    try:
        adhesion = input("Adhesion strength (MPa): ")
        adhesion = float(adhesion) if adhesion else None

        stress = input("Internal stress (MPa): ")
        stress = float(stress) if stress else None

        porosity = input("Porosity (%): ")
        porosity = float(porosity) if porosity else None

        accuracy = input("Dimensional accuracy error (mm): ")
        accuracy = float(accuracy) if accuracy else None

        notes = input("Additional notes: ")

        # Record test
        test_data = recorder.record_test(
            sample_id=sample_id,
            adhesion_strength=adhesion,
            internal_stress=stress,
            porosity=porosity,
            dimensional_accuracy=accuracy,
            test_notes=notes
        )

        # Display calculated score
        quality_score = test_data['quality_metrics']['quality_score']
        print(f"\n📊 Calculated Quality Score: {quality_score:.3f}")

        if quality_score > 0.8:
            print("   ✨ Excellent quality!")
        elif quality_score > 0.6:
            print("   ✓ Good quality")
        elif quality_score > 0.4:
            print("   ⚠️  Acceptable quality")
        else:
            print("   ✗ Poor quality")

    except ValueError as e:
        print(f"\n❌ Invalid input: {e}")
        print("   Please enter numeric values for measurements")


def generate_test_template():
    """
    Generate a CSV template for batch recording
    """
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)

    template_path = output_dir / 'quality_test_template.csv'

    header = "sample_id,adhesion_strength(MPa),internal_stress(MPa),porosity(%),dimensional_accuracy(mm),notes\n"
    example_row = "print_001,25.5,12.3,3.2,0.05,Test with standard parameters\n"

    with open(template_path, 'w') as f:
        f.write(header)
        f.write(example_row)

    print(f"\n✅ Template created: {template_path}")
    print("\nYou can now:")
    print("1. Open the CSV file in Excel or similar")
    print("2. Fill in your test results")
    print("3. Save the file")
    print("4. Run: python data/scripts/record_quality_test.py --csv quality_test_template.csv")


def import_from_csv(csv_path: str):
    """
    Import quality test results from CSV file

    Args:
        csv_path: Path to CSV file
    """
    import csv

    recorder = QualityTestRecorder()

    test_results = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_result = {
                'sample_id': row['sample_id'],
            }

            if row.get('adhesion_strength(MPa)'):
                try:
                    test_result['adhesion_strength'] = float(row['adhesion_strength(MPa)'])
                except:
                    pass

            if row.get('internal_stress(MPa)'):
                try:
                    test_result['internal_stress'] = float(row['internal_stress(MPa)'])
                except:
                    pass

            if row.get('porosity(%)'):
                try:
                    test_result['porosity'] = float(row['porosity(%)'])
                except:
                    pass

            if row.get('dimensional_accuracy(mm)'):
                try:
                    test_result['dimensional_accuracy'] = float(row['dimensional_accuracy(mm)'])
                except:
                    pass

            test_result['test_notes'] = row.get('notes', '')

            test_results.append(test_result)

    # Batch record
    recorder.batch_record(test_results)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Record quality test results')

    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'template', 'csv'],
                       help='Recording mode')
    parser.add_argument('--csv', type=str,
                       help='CSV file to import')
    parser.add_argument('--sample_id', type=str,
                       help='Sample ID')
    parser.add_argument('--adhesion', type=float,
                       help='Adhesion strength (MPa)')
    parser.add_argument('--stress', type=float,
                       help='Internal stress (MPa)')
    parser.add_argument('--porosity', type=float,
                       help='Porosity (%)')
    parser.add_argument('--accuracy', type=float,
                       help='Dimensional accuracy error (mm)')
    parser.add_argument('--notes', type=str,
                       help='Additional notes')

    args = parser.parse_args()

    if args.mode == 'interactive':
        interactive_recording()
    elif args.mode == 'template':
        generate_test_template()
    elif args.mode == 'csv':
        if args.csv:
            import_from_csv(args.csv)
        else:
            print("Error: Please specify --csv argument")


if __name__ == '__main__':
    main()
