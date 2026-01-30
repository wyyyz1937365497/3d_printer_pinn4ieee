"""
Data Quality and Feature-Target Correlation Analysis

This script analyzes:
1. Feature statistics and distributions
2. Feature-target correlations
3. Which quality targets are predictable from input features
4. Recommended tasks to focus on
"""

import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import get_config
from data.simulation import PrinterSimulationDataset


def load_sample_data(data_dir, max_samples=50000):
    """Load sample data for analysis"""
    import glob
    import random

    mat_files = glob.glob(os.path.join(data_dir, "*.mat"))
    if not mat_files:
        # Try wildcard expansion
        import glob
        expanded = glob.glob(data_dir)
        mat_files = []
        for p in expanded:
            if os.path.isdir(p):
                mat_files.extend(glob.glob(os.path.join(p, "*.mat")))

    random.Random(42).shuffle(mat_files)

    # Use subset of files for faster analysis
    analysis_files = mat_files[:10]  # Use first 10 files

    config = get_config(preset='unified')
    dataset = PrinterSimulationDataset(
        analysis_files,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        stride=config.data.stride,  # Use full stride for more samples
        mode='train',
        scaler=None,
        fit_scaler=False
    )

    print(f"Loaded {len(dataset)} sequences from {len(analysis_files)} files")

    # Sample sequences
    n_samples = min(max_samples, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    all_features = []
    all_targets = []

    for idx in indices:
        sample = dataset[idx]
        # Aggregate sequence: use mean over time
        features = sample['input_features'].mean(axis=0)  # [12]
        targets = sample['quality_targets']  # [5]

        all_features.append(features)
        all_targets.append(targets)

    features = np.stack(all_features)  # [N, 12]
    targets = np.stack(all_targets)    # [N, 5]

    return features, targets, dataset.INPUT_FEATURES, dataset.OUTPUT_QUALITY


def analyze_correlations(features, targets, feature_names, target_names):
    """Analyze feature-target correlations"""

    # Create DataFrame
    feature_df = pd.DataFrame(features, columns=feature_names)
    target_df = pd.DataFrame(targets, columns=target_names)

    # Compute correlations
    correlations = {}
    for target in target_names:
        corrs = {}
        for feature in feature_names:
            corr, p_value = stats.pearsonr(feature_df[feature], target_df[target])
            corrs[feature] = {'correlation': corr, 'p_value': p_value}
        correlations[target] = corrs

    return correlations, feature_df, target_df


def print_diagnostics(correlations, feature_names, target_names):
    """Print diagnostic results"""

    print("=" * 80)
    print("DATA DIAGNOSTIC REPORT")
    print("=" * 80)

    # For each target, find top predictive features
    print("\nFEATURE-TARGET CORRELATIONS (sorted by absolute correlation)")
    print("-" * 80)

    for target in target_names:
        print(f"\n{target.upper()}:")
        print("-" * 40)

        corrs = correlations[target]
        sorted_features = sorted(
            corrs.items(),
            key=lambda x: abs(x[1]['correlation']),
            reverse=True
        )

        for i, (feat, stats) in enumerate(sorted_features[:5]):
            corr = stats['correlation']
            p_val = stats['p_value']
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

            print(f"  {i+1}. {feat:20s}: r={corr:7.4f}  {significance}")

    # Predictability assessment
    print("\n" + "=" * 80)
    print("PREDICTABILITY ASSESSMENT")
    print("-" * 80)

    predictability = {}
    for target in target_names:
        corrs = correlations[target]
        max_abs_corr = max(abs(c['correlation']) for c in corrs.values())
        strong_features = sum(1 for c in corrs.values() if abs(c['correlation']) > 0.5)
        moderate_features = sum(1 for c in corrs.values() if 0.3 < abs(c['correlation']) <= 0.5)

        if max_abs_corr > 0.7:
            rating = "✓ Highly Predictable"
            score = 5
        elif max_abs_corr > 0.5:
            rating = "✓ Moderately Predictable"
            score = 4
        elif max_abs_corr > 0.3:
            rating = "~ Weakly Predictable"
            score = 3
        elif max_abs_corr > 0.1:
            rating = "✗ Poorly Predictable"
            score = 2
        else:
            rating = "✗ Not Predictable"
            score = 1

        predictability[target] = {
            'max_corr': max_abs_corr,
            'strong_features': strong_features,
            'moderate_features': moderate_features,
            'score': score,
            'rating': rating
        }

        print(f"\n{target:30s}: {rating}")
        print(f"  Max correlation: {max_abs_corr:.4f}")
        print(f"  Strong features (|r|>0.5): {strong_features}")
        print(f"  Moderate features (0.3<|r|<=0.5): {moderate_features}")

    # Recommend tasks
    print("\n" + "=" * 80)
    print("RECOMMENDED TASKS FOR SIMPLIFIED MODEL")
    print("-" * 80)

    sorted_targets = sorted(
        predictability.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )

    recommended = sorted_targets[:3]

    print(f"\nTop {len(recommended)} recommended tasks (based on predictability):")
    for i, (target, stats) in enumerate(recommended):
        print(f"\n  {i+1}. {target}")
        print(f"     Rating: {stats['rating']}")
        print(f"     Max correlation: {stats['max_corr']:.4f}")

    return predictability, recommended


def visualize_correlations(correlations, feature_names, target_names, save_dir='diagnostics'):
    """Create correlation heatmaps"""

    os.makedirs(save_dir, exist_ok=True)

    # Create correlation matrix
    corr_matrix = np.zeros((len(target_names), len(feature_names)))

    for i, target in enumerate(target_names):
        for j, feature in enumerate(feature_names):
            corr_matrix[i, j] = correlations[target][feature]['correlation']

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 6))

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(target_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(target_names)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pearson Correlation', rotation=270, labelpad=15)

    # Add correlation values
    for i in range(len(target_names)):
        for j in range(len(feature_names)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title('Feature-Target Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Create per-target bar plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, target in enumerate(target_names):
        ax = axes[i]
        corrs = correlations[target]
        values = [corrs[f]['correlation'] for f in feature_names]

        colors = ['green' if abs(v) > 0.5 else 'orange' if abs(v) > 0.3 else 'red' for v in values]

        ax.barh(feature_names, values, color=colors)
        ax.set_xlabel('Correlation')
        ax.set_title(f'{target}')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0.5, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(x=-0.5, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)

    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_barplots.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualizations saved to: {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Diagnose data quality and feature correlations')
    parser.add_argument('--data_dir', type=str, default='data_simulation_*/')
    parser.add_argument('--max_samples', type=int, default=50000)
    parser.add_argument('--save_dir', type=str, default='diagnostics')
    args = parser.parse_args()

    print("=" * 80)
    print("DATA QUALITY AND FEATURE CORRELATION ANALYSIS")
    print("=" * 80)
    print(f"\nData directory: {args.data_dir}")
    print(f"Max samples: {args.max_samples}")
    print()

    # Load data
    features, targets, feature_names, target_names = load_sample_data(
        args.data_dir,
        max_samples=args.max_samples
    )

    print(f"\nFeature shape: {features.shape}")
    print(f"Target shape:  {targets.shape}")

    # Analyze correlations
    correlations, feature_df, target_df = analyze_correlations(
        features, targets, feature_names, target_names
    )

    # Print diagnostics
    predictability, recommended = print_diagnostics(
        correlations, feature_names, target_names
    )

    # Visualize
    visualize_correlations(correlations, feature_names, target_names, args.save_dir)

    # Save results
    results = {
        'recommended_tasks': [t[0] for t in recommended],
        'predictability_scores': {t: p['score'] for t, p in predictability.items()},
        'max_correlations': {t: p['max_corr'] for t, p in predictability.items()}
    }

    import json
    with open(os.path.join(args.save_dir, 'diagnostics_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.save_dir}/diagnostics_results.json")
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
