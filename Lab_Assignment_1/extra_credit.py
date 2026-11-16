"""
Extra Credit: The Importance of Initialization
Lab Assignment 1 - Neural Networks and Deep Learning

This script investigates how different initialization schemes affect
convergence speed of two-layer neural networks.
"""

import torch
import numpy as np
import os

from two_layer_nn import TwoLayerNN
from datasets import create_linear_dataset
from training_utils import train_two_layer_nn, get_convergence_time
from visualization import plot_boxplot

print("=" * 80)
print("EXTRA CREDIT: THE IMPORTANCE OF INITIALIZATION")
print("=" * 80)
print()

# Create results directory
os.makedirs('results', exist_ok=True)

# Load dataset
X_linear, y_linear = create_linear_dataset()
print(f"Dataset: {len(X_linear)} samples, {X_linear.shape[1]}D")
print()

# Hyperparameters
input_dim = 2
hidden_dim = 2
epochs = 500
lr = 0.1
num_runs = 100

print("=" * 80)
print("HYPOTHESIS")
print("=" * 80)
print()
print("I hypothesize that the baseline Gaussian initialization will converge fastest")
print("because it provides appropriate scale (variance) for gradient flow. The small")
print("uniform initialization will be slower due to smaller initial gradients. The")
print("constant initialization (all 1.0) will fail completely because it breaks")
print("symmetry - all hidden units will learn identical features, effectively")
print("reducing the network to a single hidden unit.")
print()

# =============================================================================
# Scheme 1: Baseline Gaussian Initialization
# =============================================================================
print("=" * 80)
print("SCHEME 1: Baseline Gaussian Initialization")
print("=" * 80)
print(f"w1 ~ N(0, 1/√{input_dim}), w2 ~ N(0, 1/√{hidden_dim})")
print(f"Training {num_runs} instances...\n")

convergence_times_gaussian = []

for i in range(num_runs):
    # Standard initialization (already in TwoLayerNN.__init__)
    model = TwoLayerNN(input_dim, hidden_dim)

    # Train
    accuracies = train_two_layer_nn(model, X_linear, y_linear, epochs, lr, verbose=False)

    # Get convergence time
    conv_time = get_convergence_time(accuracies, threshold=1.0)
    if conv_time is not None:
        convergence_times_gaussian.append(conv_time)
    else:
        convergence_times_gaussian.append(epochs + 1)  # Didn't converge

    if (i + 1) % 20 == 0:
        print(f"  Completed {i + 1}/{num_runs} runs")

# Statistics
mean_gauss = np.mean(convergence_times_gaussian)
std_gauss = np.std(convergence_times_gaussian)
converged_gauss = sum(1 for t in convergence_times_gaussian if t <= epochs)

print(f"\nResults:")
print(f"  Converged: {converged_gauss}/{num_runs}")
print(f"  Mean convergence time: {mean_gauss:.2f} epochs")
print(f"  Std convergence time: {std_gauss:.2f} epochs")
print()

# =============================================================================
# Scheme 2: Small Uniform Initialization
# =============================================================================
print("=" * 80)
print("SCHEME 2: Small Uniform Initialization")
print("=" * 80)
print(f"w1 ~ U(-1/(10√{input_dim}), 1/(10√{input_dim}))")
print(f"w2 ~ U(-1/(10√{hidden_dim}), 1/(10√{hidden_dim}))")
print(f"Training {num_runs} instances...\n")

convergence_times_uniform = []

for i in range(num_runs):
    model = TwoLayerNN(input_dim, hidden_dim)

    # Override with uniform initialization
    scale_w1 = 1.0 / (10 * np.sqrt(input_dim))
    scale_w2 = 1.0 / (10 * np.sqrt(hidden_dim))

    model.w1 = (torch.rand(hidden_dim, input_dim) * 2 - 1) * scale_w1
    model.w2 = (torch.rand(hidden_dim, 1) * 2 - 1) * scale_w2

    # Train
    accuracies = train_two_layer_nn(model, X_linear, y_linear, epochs, lr, verbose=False)

    # Get convergence time
    conv_time = get_convergence_time(accuracies, threshold=1.0)
    if conv_time is not None:
        convergence_times_uniform.append(conv_time)
    else:
        convergence_times_uniform.append(epochs + 1)

    if (i + 1) % 20 == 0:
        print(f"  Completed {i + 1}/{num_runs} runs")

# Statistics
mean_uniform = np.mean(convergence_times_uniform)
std_uniform = np.std(convergence_times_uniform)
converged_uniform = sum(1 for t in convergence_times_uniform if t <= epochs)

print(f"\nResults:")
print(f"  Converged: {converged_uniform}/{num_runs}")
print(f"  Mean convergence time: {mean_uniform:.2f} epochs")
print(f"  Std convergence time: {std_uniform:.2f} epochs")
print()

# =============================================================================
# Scheme 3: Constant Initialization (all 1.0)
# =============================================================================
print("=" * 80)
print("SCHEME 3: Constant Initialization (all 1.0)")
print("=" * 80)
print("All weights = 1.0, biases = 0.0")
print("Training 1 instance (deterministic)...\n")

model = TwoLayerNN(input_dim, hidden_dim)

# Override with constant initialization
model.w1 = torch.ones(hidden_dim, input_dim)
model.w2 = torch.ones(hidden_dim, 1)

# Train
accuracies = train_two_layer_nn(model, X_linear, y_linear, epochs, lr, verbose=False)

# Get convergence time
conv_time_constant = get_convergence_time(accuracies, threshold=1.0)
if conv_time_constant is None:
    conv_time_constant = epochs + 1

print(f"Results:")
if conv_time_constant <= epochs:
    print(f"  Converged at epoch: {conv_time_constant}")
else:
    print(f"  Did NOT converge within {epochs} epochs")
    print(f"  Final accuracy: {accuracies[-1]:.4f}")
print()

# =============================================================================
# Visualization and Analysis
# =============================================================================
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print()

# Filter out non-converged runs for fair comparison
convergence_times_gaussian_filtered = [t for t in convergence_times_gaussian if t <= epochs]
convergence_times_uniform_filtered = [t for t in convergence_times_uniform if t <= epochs]

# Create box plot (only for random initialization schemes)
if convergence_times_gaussian_filtered and convergence_times_uniform_filtered:
    plot_boxplot(
        {
            'Gaussian': convergence_times_gaussian_filtered,
            'Uniform': convergence_times_uniform_filtered
        },
        title="Convergence Time Comparison: Initialization Schemes",
        ylabel="Convergence Time (epochs)",
        save_path='results/extra_credit_boxplot.png'
    )

# Print summary table
print("Summary Statistics (converged runs only):")
print("-" * 60)
print(f"{'Scheme':<20} {'Converged':<15} {'Mean':<10} {'Std':<10}")
print("-" * 60)
print(f"{'Gaussian':<20} {converged_gauss}/{num_runs:<10} {mean_gauss:<10.2f} {std_gauss:<10.2f}")
print(f"{'Uniform':<20} {converged_uniform}/{num_runs:<10} {mean_uniform:<10.2f} {std_uniform:<10.2f}")
print(f"{'Constant':<20} {'0/1' if conv_time_constant > epochs else '1/1':<15} {conv_time_constant:<10} {'-':<10}")
print("-" * 60)
print()

print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print()

# Determine which scheme was fastest
if mean_gauss < mean_uniform:
    fastest = "Gaussian"
else:
    fastest = "Uniform"

print(f"1. FASTEST CONVERGENCE: {fastest}")
if fastest == "Gaussian":
    print("   The hypothesis is SUPPORTED. Gaussian initialization provides the optimal")
    print("   initial scale for gradients, allowing efficient learning from the start.")
else:
    print("   The hypothesis is PARTIALLY REFUTED. Uniform initialization performed")
    print("   better than expected, possibly due to the specific problem characteristics.")
print()

# Analyze variance
if std_gauss < std_uniform:
    most_consistent = "Gaussian"
else:
    most_consistent = "Uniform"

print(f"2. MOST CONSISTENT: {most_consistent}")
print(f"   Gaussian std: {std_gauss:.2f}, Uniform std: {std_uniform:.2f}")
if most_consistent == "Gaussian":
    print("   Gaussian initialization provides more predictable convergence behavior")
    print("   across different random seeds, likely due to its concentration around")
    print("   the mean and well-studied properties for neural network training.")
else:
    print("   Uniform initialization shows less variance in convergence times,")
    print("   suggesting more stable gradient dynamics despite slower average speed.")
print()

print("3. CONSTANT INITIALIZATION FAILURE:")
if conv_time_constant > epochs:
    print("   As predicted, constant initialization FAILED to converge!")
    print("   This demonstrates the critical importance of symmetry breaking:")
    print("   - All hidden units start with identical weights")
    print("   - They receive identical gradients during backpropagation")
    print("   - They remain identical throughout training")
    print("   - The network effectively has only ONE hidden unit")
    print("   This is known as the 'symmetry problem' in neural network initialization.")
else:
    print("   Surprisingly, constant initialization converged! This may occur because:")
    print("   - The bias terms (initialized to 0) provide some asymmetry")
    print("   - Numerical precision differences break symmetry slightly")
    print("   - The specific problem is simple enough that reduced capacity suffices")
print()

# =============================================================================
# Handcrafted Initialization (Challenge)
# =============================================================================
print("=" * 80)
print("CHALLENGE: Handcrafted Initialization to MAXIMIZE Convergence Time")
print("=" * 80)
print()

print("Strategy: Initialize weights to maximize loss while staying in valid range.")
print("Approach: Set weights such that initial predictions are far from true labels.")
print()

# Analyze the dataset
print("Dataset analysis:")
print(f"  Mean x1: {X_linear[:, 0].mean():.3f}, Mean x2: {X_linear[:, 1].mean():.3f}")
print(f"  Label distribution: {y_linear.sum().item():.0f} ones, {len(y_linear) - y_linear.sum().item():.0f} zeros")
print()

# Strategy: Make initial predictions very wrong
# For linear boundary: points above y=x should be 1, below should be 0
# To maximize convergence time, we want initial predictions to be opposite

print("Handcrafted weights:")
print("  Strategy: Initialize to predict OPPOSITE of true labels")
print("  w1 = [[0.5, -0.5], [-0.5, 0.5]]  (creates wrong boundary)")
print("  w2 = [[1.0], [-1.0]]  (amplifies wrong predictions)")
print()

model_handcrafted = TwoLayerNN(input_dim, hidden_dim)

# Handcrafted initialization
model_handcrafted.w1 = torch.tensor([[0.5, -0.5],
                                     [-0.5, 0.5]], dtype=torch.float32)
model_handcrafted.w2 = torch.tensor([[1.0],
                                     [-1.0]], dtype=torch.float32)

# Train
accuracies_handcrafted = train_two_layer_nn(
    model_handcrafted, X_linear, y_linear, epochs, lr, verbose=False
)

conv_time_handcrafted = get_convergence_time(accuracies_handcrafted, threshold=1.0)
if conv_time_handcrafted is None:
    conv_time_handcrafted = epochs + 1

print("Results:")
print(f"  Initial accuracy: {accuracies_handcrafted[0]:.4f}")
if conv_time_handcrafted <= epochs:
    print(f"  Converged at epoch: {conv_time_handcrafted}")
else:
    print(f"  Did NOT converge within {epochs} epochs")
    print(f"  Final accuracy: {accuracies_handcrafted[-1]:.4f}")
print()

print("Comparison with baseline:")
print(f"  Baseline (Gaussian) mean: {mean_gauss:.2f} epochs")
print(f"  Handcrafted: {conv_time_handcrafted if conv_time_handcrafted <= epochs else f'>{epochs}'} epochs")
if conv_time_handcrafted > mean_gauss:
    print(f"  SUCCESS! Handcrafted initialization slowed convergence by "
          f"{conv_time_handcrafted - mean_gauss:.2f} epochs")
else:
    print(f"  Baseline was actually slower (or handcrafted converged quickly)")
print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()
print("This experiment demonstrates that initialization is CRITICAL for neural")
print("network training:")
print()
print("1. Good initialization (Gaussian) enables fast, reliable convergence")
print("2. Poor scale (too small) slows learning but eventually succeeds")
print("3. Broken symmetry (constant) can completely prevent learning")
print("4. Adversarial initialization (handcrafted) can maximize convergence time")
print()
print("Modern deep learning uses sophisticated initialization schemes like:")
print("- Xavier/Glorot initialization (for sigmoid/tanh)")
print("- He initialization (for ReLU)")
print("- Layer normalization and batch normalization to reduce initialization sensitivity")
print()
print("All results saved to 'results/' directory")
