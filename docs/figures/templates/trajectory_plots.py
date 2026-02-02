"""
Standardized plotting functions for trajectory correction IEEE papers.

All functions generate publication-quality figures with:
- Consistent styling (fonts, colors, line widths)
- IEEE-recommended dimensions
- Vector output (PDF/EPS) support
- Grayscale-compatible colormaps
- Times New Roman font (IEEE standard)

Author: 3D Printer PINN Project Team
Date: 2026-02-01
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# IEEE style configuration
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.linewidth'] = 0.5
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['grid.linestyle'] = '--'
matplotlib.rcParams['grid.alpha'] = 0.3


def plot_trajectory_2d(x_ref, y_ref, x_act=None, y_act=None,
                        errors=None, save_path=None, figsize=(6, 6)):
    """
    Create 2D trajectory plot with optional actual trajectory and error heat map.

    Parameters
    ----------
    x_ref, y_ref : array-like
        Reference trajectory coordinates [mm]
    x_act, y_act : array-like, optional
        Actual trajectory coordinates [mm]
    errors : array-like, optional
        Error magnitude for color mapping [mm]
    save_path : str or Path, optional
        Path to save figure (PDF format recommended)
    figsize : tuple, optional
        Figure size in inches (default: (6, 6))

    Returns
    -------
    fig, ax : matplotlib figure and axis objects

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 2*np.pi, 100)
    >>> x_ref = np.cos(t)
    >>> y_ref = np.sin(t)
    >>> fig, ax = plot_trajectory_2d(x_ref, y_ref)
    >>> plt.savefig('trajectory.pdf', format='pdf', dpi=300)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot reference trajectory
    ax.plot(x_ref, y_ref, 'b--', linewidth=1.5, label='Reference', alpha=0.7)

    # Plot actual trajectory if provided
    if x_act is not None and y_act is not None:
        ax.plot(x_act, y_act, 'r-', linewidth=0.8, label='Actual', alpha=0.5)

    # Plot error heat map if provided
    if errors is not None and x_act is not None and y_act is not None:
        sc = ax.scatter(x_act, y_act, c=errors, cmap='gray_r',
                       s=2, alpha=0.6, vmin=0, vmax=np.max(errors))
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Error Magnitude [mm]', fontsize=10)

    # Formatting
    ax.set_xlabel('X Position [mm]', fontsize=11)
    ax.set_ylabel('Y Position [mm]', fontsize=11)
    ax.set_title('2D Trajectory Comparison', fontsize=12, fontweight='bold')
    ax.grid(True)
    ax.axis('equal')
    ax.legend(fontsize=10, loc='best')

    # Save if path provided
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    return fig, ax


def plot_error_time_series(time, error_x, error_y, error_mag,
                            save_path=None, figsize=(10, 4)):
    """
    Plot error components over time.

    Parameters
    ----------
    time : array-like
        Time vector [s]
    error_x, error_y : array-like
        X and Y error components [mm]
    error_mag : array-like
        Error magnitude [mm]
    save_path : str or Path, optional
        Path to save figure
    figsize : tuple, optional
        Figure size (default: (10, 4))

    Returns
    -------
    fig, axes : matplotlib figure and axis objects
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # X and Y components
    axes[0].plot(time, error_x, 'b-', label='X Error', linewidth=1, alpha=0.7)
    axes[0].plot(time, error_y, 'r-', label='Y Error', linewidth=1, alpha=0.7)
    axes[0].set_xlabel('Time [s]', fontsize=11)
    axes[0].set_ylabel('Error [mm]', fontsize=11)
    axes[0].set_title('Error Components', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # Magnitude
    axes[1].plot(time, error_mag, 'k-', linewidth=1.5)
    axes[1].set_xlabel('Time [s]', fontsize=11)
    axes[1].set_ylabel('Error Magnitude [mm]', fontsize=11)
    axes[1].set_title('Error Magnitude', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    return fig, axes


def plot_error_distribution(error_x, error_y, error_mag,
                            save_path=None, figsize=(10, 3)):
    """
    Plot error distribution histograms.

    Parameters
    ----------
    error_x, error_y : array-like
        X and Y error components [mm]
    error_mag : array-like
        Error magnitude [mm]
    save_path : str or Path, optional
        Path to save figure
    figsize : tuple, optional
        Figure size (default: (10, 3))

    Returns
    -------
    fig, axes : matplotlib figure and axis objects
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # X error
    axes[0].hist(error_x, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('X Error [mm]', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title(f'X Error: μ={np.mean(error_x):.4f}, σ={np.std(error_x):.4f}',
                     fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Y error
    axes[1].hist(error_y, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Y Error [mm]', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title(f'Y Error: μ={np.mean(error_y):.4f}, σ={np.std(error_y):.4f}',
                     fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Magnitude
    axes[2].hist(error_mag, bins=50, color='gray', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Error Magnitude [mm]', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title(f'Magnitude: μ={np.mean(error_mag):.4f}, σ={np.std(error_mag):.4f}',
                     fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    return fig, axes


def plot_correction_comparison(error_before, error_after,
                               labels=['Before', 'After'],
                               save_path=None, figsize=(8, 5)):
    """
    Compare error distributions before and after correction.

    Parameters
    ----------
    error_before, error_after : array-like
        Error magnitude before and after correction [mm]
    labels : list of str, optional
        Labels for legend
    save_path : str or Path, optional
        Path to save figure
    figsize : tuple, optional
        Figure size (default: (8, 5))

    Returns
    -------
    fig, axes : matplotlib figure and axis objects
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram comparison
    axes[0].hist(error_before, bins=50, alpha=0.6, label=labels[0], color='red')
    axes[0].hist(error_after, bins=50, alpha=0.6, label=labels[1], color='green')
    axes[0].set_xlabel('Error Magnitude [mm]', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Error Distribution Comparison', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Box plot
    axes[1].boxplot([error_before, error_after], labels=labels)
    axes[1].set_ylabel('Error Magnitude [mm]', fontsize=11)
    axes[1].set_title('Error Statistics', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add statistics text
    stats_text = f"Before: μ={np.mean(error_before):.4f}, σ={np.std(error_before):.4f}\n"
    stats_text += f"After:  μ={np.mean(error_after):.4f}, σ={np.std(error_after):.4f}\n"
    stats_text += f"Improvement: {(1 - np.mean(error_after)/np.mean(error_before))*100:.1f}%"
    axes[1].text(0.5, 0.97, stats_text, transform=axes[1].transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    return fig, axes


def plot_prediction_vs_actual(actual, predicted, title="Prediction Accuracy",
                               save_path=None, figsize=(6, 6)):
    """
    Plot predicted vs actual values with R² score.

    Parameters
    ----------
    actual, predicted : array-like
        Actual and predicted values
    title : str, optional
        Plot title
    save_path : str or Path, optional
        Path to save figure
    figsize : tuple, optional
        Figure size (default: (6, 6))

    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """
    from sklearn.metrics import r2_score

    fig, ax = plt.subplots(figsize=figsize)

    # Calculate R²
    r2 = r2_score(actual, predicted)

    # Scatter plot
    ax.scatter(actual, predicted, alpha=0.3, s=1)

    # Perfect prediction line
    min_val = min(np.min(actual), np.min(predicted))
    max_val = max(np.max(actual), np.max(predicted))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--',
            linewidth=2, label='Perfect Prediction')

    # Formatting
    ax.set_xlabel('Actual Value', fontsize=11)
    ax.set_ylabel('Predicted Value', fontsize=11)
    ax.set_title(f'{title}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.axis('equal')

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    return fig, ax


def plot_training_curves(train_loss, val_loss, learning_rate=None,
                         save_path=None, figsize=(12, 4)):
    """
    Plot training curves (loss and learning rate).

    Parameters
    ----------
    train_loss, val_loss : array-like
        Training and validation loss per epoch
    learning_rate : array-like, optional
        Learning rate per epoch
    save_path : str or Path, optional
        Path to save figure
    figsize : tuple, optional
        Figure size (default: (12, 4))

    Returns
    -------
    fig, axes : matplotlib figure and axis objects
    """
    if learning_rate is not None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes = [axes[0], axes[1]]

    epochs = np.arange(1, len(train_loss) + 1)

    # Loss curves
    axes[0].plot(epochs, train_loss, label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss (MAE)', fontsize=11)
    axes[0].set_title('Training Progress', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # Learning rate
    if learning_rate is not None:
        axes[1].plot(epochs, learning_rate, color='green', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Learning Rate', fontsize=11)
        axes[1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')

        # Additional plot for loss zoom
        best_epoch = np.argmin(val_loss)
        axes[2].plot(epochs, train_loss, label='Train', linewidth=2, alpha=0.7)
        axes[2].plot(epochs, val_loss, label='Val', linewidth=2, alpha=0.7)
        axes[2].axvline(best_epoch + 1, color='red', linestyle='--',
                    linewidth=2, label=f'Best (Epoch {best_epoch + 1})')
        axes[2].set_xlabel('Epoch', fontsize=11)
        axes[2].set_ylabel('Loss (MAE)', fontsize=11)
        axes[2].set_title('Loss (Last 50 Epochs Zoom)', fontsize=12, fontweight='bold')
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(max(1, len(epochs) - 50), len(epochs))
        axes[2].set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    return fig, axes


# Main function for testing
if __name__ == '__main__':
    # Generate test data
    print("Generating test plots...")

    # Test 1: Trajectory plot
    t = np.linspace(0, 2*np.pi, 200)
    x_ref = np.cos(t) * 50
    y_ref = np.sin(t) * 50
    x_act = x_ref + np.random.normal(0, 0.5, len(t))
    y_act = y_ref + np.random.normal(0, 0.5, len(t))
    errors = np.sqrt((x_act - x_ref)**2 + (y_act - y_ref)**2)

    fig, ax = plot_trajectory_2d(x_ref, y_ref, x_act, y_act, errors)
    plt.savefig('test_trajectory.pdf', format='pdf', dpi=300)
    plt.close()
    print("✓ Test trajectory plot saved")

    # Test 2: Error time series
    fig, axes = plot_error_time_series(t, x_act - x_ref, y_act - y_ref, errors)
    plt.savefig('test_error_timeseries.pdf', format='pdf', dpi=300)
    plt.close()
    print("✓ Test error time series saved")

    # Test 3: Error distribution
    fig, axes = plot_error_distribution(x_act - x_ref, y_act - y_ref, errors)
    plt.savefig('test_error_distribution.pdf', format='pdf', dpi=300)
    plt.close()
    print("✓ Test error distribution saved")

    print("\n✓ All test plots generated successfully!")
    print("  - test_trajectory.pdf")
    print("  - test_error_timeseries.pdf")
    print("  - test_error_distribution.pdf")
