"""
Visualization Utilities for Lab Assignment 1
Neural Networks and Deep Learning

This module contains functions for creating all required plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

# NEON COLOR PALETTE
NEON_COLORS = {
    'pink': '#FF10F0',      # Neon Pink/Magenta
    'cyan': '#00FFFF',      # Neon Cyan
    'green': '#39FF14',     # Neon Green
    'yellow': '#FFFF00',    # Neon Yellow
    'orange': '#FF6600',    # Neon Orange
    'purple': '#BC13FE',    # Neon Purple
    'red': '#FF073A',       # Neon Red
    'blue': '#0DFFFF',      # Neon Blue
    'lime': '#CCFF00',      # Neon Lime
}


def plot_dataset(X, y, title="Dataset", save_path=None):
    """
    Create a scatter plot of 2D dataset with color-coded labels.

    Args:
        X: Input data (N, 2)
        y: Labels (N, 1) or (N,)
        title: Plot title
        save_path: Path to save figure (optional)
    """
    # Convert to numpy if needed
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    y = y.flatten()

    # Create figure with dark background for neon effect
    plt.figure(figsize=(8, 6), facecolor='#0a0a0a')
    ax = plt.gca()
    ax.set_facecolor('#0a0a0a')

    # Plot points with different colors based on labels
    mask_0 = y == 0
    mask_1 = y == 1

    plt.scatter(X[mask_0, 0], X[mask_0, 1], c=NEON_COLORS['pink'], label='Label 0',
                s=100, alpha=0.9, edgecolors=NEON_COLORS['red'], linewidths=2)
    plt.scatter(X[mask_1, 0], X[mask_1, 1], c=NEON_COLORS['cyan'], label='Label 1',
                s=100, alpha=0.9, edgecolors=NEON_COLORS['blue'], linewidths=2)

    plt.xlabel('x₁', fontsize=12, color='white')
    plt.ylabel('x₂', fontsize=12, color='white')
    plt.title(title, fontsize=14, fontweight='bold', color='white')

    # Style axes and legend for neon theme
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color(NEON_COLORS['cyan'])
    ax.spines['left'].set_color(NEON_COLORS['cyan'])
    ax.spines['top'].set_color(NEON_COLORS['cyan'])
    ax.spines['right'].set_color(NEON_COLORS['cyan'])

    legend = plt.legend(fontsize=10, facecolor='#1a1a1a', edgecolor=NEON_COLORS['purple'])
    plt.setp(legend.get_texts(), color='white')
    plt.grid(True, alpha=0.2, color=NEON_COLORS['purple'], linestyle='--')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_training_curve(accuracies, angles=None, title="Training Progress", save_path=None):
    """
    Plot training accuracy and optionally decision boundary angle.

    Args:
        accuracies: List of accuracy values
        angles: List of boundary angles (optional)
        title: Plot title
        save_path: Path to save figure
    """
    epochs = range(1, len(accuracies) + 1)

    if angles is not None:
        # Two y-axes with neon theme
        fig, ax1 = plt.subplots(figsize=(10, 6), facecolor='#0a0a0a')
        ax1.set_facecolor('#0a0a0a')

        color1 = NEON_COLORS['cyan']
        ax1.set_xlabel('Epoch', fontsize=12, color='white')
        ax1.set_ylabel('Accuracy', color=color1, fontsize=12)
        ax1.plot(epochs, accuracies, color=color1, linewidth=3, label='Accuracy',
                marker='o', markersize=2, markevery=max(1, len(epochs)//20))
        ax1.tick_params(axis='y', labelcolor=color1, colors='white')
        ax1.tick_params(axis='x', colors='white')
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.2, color=NEON_COLORS['purple'], linestyle='--')
        ax1.spines['bottom'].set_color(NEON_COLORS['green'])
        ax1.spines['top'].set_color(NEON_COLORS['green'])
        ax1.spines['left'].set_color(color1)
        ax1.spines['right'].set_color(NEON_COLORS['orange'])

        ax2 = ax1.twinx()
        color2 = NEON_COLORS['orange']
        ax2.set_ylabel('Boundary Angle (radians)', color=color2, fontsize=12)
        ax2.plot(epochs, angles, color=color2, linewidth=3, label='Boundary Angle',
                marker='s', markersize=2, markevery=max(1, len(epochs)//20))
        ax2.tick_params(axis='y', labelcolor=color2, colors='white')
        ax2.spines['right'].set_color(color2)

        # Add legends with neon styling
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10,
                          facecolor='#1a1a1a', edgecolor=NEON_COLORS['purple'])
        plt.setp(legend.get_texts(), color='white')

        plt.title(title, fontsize=14, fontweight='bold', color='white')
        fig.tight_layout()
    else:
        # Single y-axis with neon theme
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a0a0a')
        ax.set_facecolor('#0a0a0a')

        ax.plot(epochs, accuracies, linewidth=3, color=NEON_COLORS['cyan'],
               marker='o', markersize=3, markevery=max(1, len(epochs)//20))
        ax.set_xlabel('Epoch', fontsize=12, color='white')
        ax.set_ylabel('Accuracy', fontsize=12, color=NEON_COLORS['cyan'])
        ax.set_title(title, fontsize=14, fontweight='bold', color='white')
        ax.set_ylim(-0.05, 1.05)

        # Style axes for neon theme
        ax.tick_params(colors='white')
        ax.tick_params(axis='y', labelcolor=NEON_COLORS['cyan'])
        ax.spines['bottom'].set_color(NEON_COLORS['green'])
        ax.spines['left'].set_color(NEON_COLORS['cyan'])
        ax.spines['top'].set_color(NEON_COLORS['green'])
        ax.spines['right'].set_color(NEON_COLORS['cyan'])
        ax.grid(True, alpha=0.2, color=NEON_COLORS['purple'], linestyle='--')

        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_decision_boundary_perceptron(X, y, model, title="Decision Boundary", save_path=None):
    """
    Plot dataset with perceptron decision boundary.

    Args:
        X: Input data (N, 2)
        y: Labels (N, 1)
        model: Trained perceptron model
        title: Plot title
        save_path: Path to save figure
    """
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    y = y.flatten()

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    # Plot data points with neon colors
    mask_0 = y == 0
    mask_1 = y == 1
    plt.scatter(X[mask_0, 0], X[mask_0, 1], c=NEON_COLORS['pink'], label='Label 0',
                s=120, alpha=0.9, edgecolors=NEON_COLORS['red'], linewidths=2)
    plt.scatter(X[mask_1, 0], X[mask_1, 1], c=NEON_COLORS['cyan'], label='Label 1',
                s=120, alpha=0.9, edgecolors=NEON_COLORS['blue'], linewidths=2)

    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    try:
        x1_line, x2_line = model.get_boundary_line((x_min, x_max))
        plt.plot(x1_line, x2_line, color=NEON_COLORS['green'], linestyle='--',
                linewidth=3, label='Decision Boundary')
    except:
        print("Could not plot decision boundary")

    plt.xlabel('x₁', fontsize=12, color='white')
    plt.ylabel('x₂', fontsize=12, color='white')
    plt.title(title, fontsize=14, fontweight='bold', color='white')

    # Style axes and legend
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color(NEON_COLORS['cyan'])
    ax.spines['left'].set_color(NEON_COLORS['cyan'])
    ax.spines['top'].set_color(NEON_COLORS['cyan'])
    ax.spines['right'].set_color(NEON_COLORS['cyan'])

    legend = plt.legend(fontsize=10, facecolor='#1a1a1a', edgecolor=NEON_COLORS['purple'])
    plt.setp(legend.get_texts(), color='white')
    plt.grid(True, alpha=0.2, color=NEON_COLORS['purple'], linestyle='--')
    plt.xlim(x_min, x_max)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_decision_boundary_nn(X, y, model, title="Decision Boundary", save_path=None):
    """
    Plot dataset with neural network decision boundary (via grid sampling).

    Args:
        X: Input data (N, 2)
        y: Labels (N, 1)
        model: Trained TwoLayerNN model
        title: Plot title
        save_path: Path to save figure
    """
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    y = y.flatten()

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    # Create a grid for decision boundary visualization
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    x1_grid, x2_grid, predictions = model.grid(
        (x1_min, x1_max), (x2_min, x2_max), res=20
    )

    # Plot grid predictions as background with neon colors
    mask_pred_0 = predictions == 0
    mask_pred_1 = predictions == 1

    plt.scatter(x1_grid[mask_pred_0], x2_grid[mask_pred_0],
                c=NEON_COLORS['pink'], s=50, alpha=0.15, marker='s')
    plt.scatter(x1_grid[mask_pred_1], x2_grid[mask_pred_1],
                c=NEON_COLORS['cyan'], s=50, alpha=0.15, marker='s')

    # Plot actual data points on top with bright neon colors
    mask_0 = y == 0
    mask_1 = y == 1
    plt.scatter(X[mask_0, 0], X[mask_0, 1], c=NEON_COLORS['pink'], label='Label 0',
                s=120, alpha=0.95, edgecolors=NEON_COLORS['red'], linewidths=2.5)
    plt.scatter(X[mask_1, 0], X[mask_1, 1], c=NEON_COLORS['cyan'], label='Label 1',
                s=120, alpha=0.95, edgecolors=NEON_COLORS['blue'], linewidths=2.5)

    plt.xlabel('x₁', fontsize=12, color='white')
    plt.ylabel('x₂', fontsize=12, color='white')
    plt.title(title, fontsize=14, fontweight='bold', color='white')

    # Style axes and legend
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color(NEON_COLORS['green'])
    ax.spines['left'].set_color(NEON_COLORS['green'])
    ax.spines['top'].set_color(NEON_COLORS['green'])
    ax.spines['right'].set_color(NEON_COLORS['green'])

    legend = plt.legend(fontsize=10, facecolor='#1a1a1a', edgecolor=NEON_COLORS['purple'])
    plt.setp(legend.get_texts(), color='white')
    plt.grid(True, alpha=0.2, color=NEON_COLORS['purple'], linestyle='--')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_multiple_curves(curves_dict, title="Training Comparison", ylabel="Accuracy",
                        save_path=None):
    """
    Plot multiple training curves on the same axes.

    Args:
        curves_dict: Dictionary mapping labels to accuracy lists
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    # Use different neon colors for each curve
    neon_palette = [NEON_COLORS['cyan'], NEON_COLORS['pink'], NEON_COLORS['green'],
                    NEON_COLORS['orange'], NEON_COLORS['purple'], NEON_COLORS['yellow']]

    for idx, (label, curve) in enumerate(curves_dict.items()):
        epochs = range(1, len(curve) + 1)
        color = neon_palette[idx % len(neon_palette)]
        ax.plot(epochs, curve, linewidth=3, label=label, color=color,
               marker='o', markersize=2, markevery=max(1, len(epochs)//20))

    ax.set_xlabel('Epoch', fontsize=12, color='white')
    ax.set_ylabel(ylabel, fontsize=12, color='white')
    ax.set_title(title, fontsize=14, fontweight='bold', color='white')

    # Style axes
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color(NEON_COLORS['green'])
    ax.spines['left'].set_color(NEON_COLORS['green'])
    ax.spines['top'].set_color(NEON_COLORS['green'])
    ax.spines['right'].set_color(NEON_COLORS['green'])

    legend = ax.legend(fontsize=10, facecolor='#1a1a1a', edgecolor=NEON_COLORS['purple'])
    plt.setp(legend.get_texts(), color='white')
    ax.grid(True, alpha=0.2, color=NEON_COLORS['purple'], linestyle='--')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_boxplot(data_dict, title="Distribution Comparison", ylabel="Value",
                save_path=None):
    """
    Create a box-and-whisker plot.

    Args:
        data_dict: Dictionary mapping labels to lists of values
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    labels = list(data_dict.keys())
    data = [data_dict[label] for label in labels]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    # Customize with neon colors
    neon_box_colors = [NEON_COLORS['cyan'], NEON_COLORS['pink'],
                       NEON_COLORS['green'], NEON_COLORS['orange']]

    for idx, patch in enumerate(bp['boxes']):
        color = neon_box_colors[idx % len(neon_box_colors)]
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor(color)
        patch.set_linewidth(2)

    # Style whiskers, caps, medians, and fliers with neon colors
    for whisker in bp['whiskers']:
        whisker.set_color(NEON_COLORS['purple'])
        whisker.set_linewidth(2)
    for cap in bp['caps']:
        cap.set_color(NEON_COLORS['purple'])
        cap.set_linewidth(2)
    for median in bp['medians']:
        median.set_color(NEON_COLORS['yellow'])
        median.set_linewidth(3)
    for flier in bp['fliers']:
        flier.set_markeredgecolor(NEON_COLORS['red'])
        flier.set_markerfacecolor(NEON_COLORS['red'])

    ax.set_ylabel(ylabel, fontsize=12, color='white')
    ax.set_title(title, fontsize=14, fontweight='bold', color='white')

    # Style axes
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color(NEON_COLORS['cyan'])
    ax.spines['left'].set_color(NEON_COLORS['cyan'])
    ax.spines['top'].set_color(NEON_COLORS['cyan'])
    ax.spines['right'].set_color(NEON_COLORS['cyan'])

    ax.grid(True, alpha=0.2, axis='y', color=NEON_COLORS['purple'], linestyle='--')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
