"""
Dataset Generation for Lab Assignment 1
Neural Networks and Deep Learning

This module contains functions to generate all required datasets.
"""

import torch
import numpy as np
from itertools import product


def create_linear_dataset():
    """
    Create linearly separable dataset with boundary y = x.

    Points: All combinations from S = {-1.0, -0.5, 0.0, 0.5, 1.0}
    Labels: 1 if x2 > x1, else 0

    Returns:
        X: Tensor of shape (25, 2)
        y: Tensor of shape (25, 1)
    """
    S = [-1.0, -0.5, 0.0, 0.5, 1.0]
    points = [(x1, x2) for x1 in S for x2 in S]

    X = torch.tensor(points, dtype=torch.float32)

    # Label: 1 if x2 > x1, else 0
    y = torch.tensor([[1.0 if x2 > x1 else 0.0] for x1, x2 in points],
                     dtype=torch.float32)

    return X, y


def create_xor_dataset():
    """
    Create XOR dataset.

    Points: {(1,1), (1,-1), (-1,-1), (-1,1)}
    Labels: XOR pattern (1 if exactly one coordinate is 1)

    Returns:
        X: Tensor of shape (4, 2)
        y: Tensor of shape (4, 1)
    """
    X = torch.tensor([[1.0, 1.0],
                      [1.0, -1.0],
                      [-1.0, -1.0],
                      [-1.0, 1.0]], dtype=torch.float32)

    y = torch.tensor([[0.0],
                      [1.0],
                      [0.0],
                      [1.0]], dtype=torch.float32)

    return X, y


def create_noisy_xor_dataset(noise_std):
    """
    Create noisy XOR dataset by replicating and adding Gaussian noise.

    Each original XOR point is replicated 10 times, then Gaussian noise
    is added to the inputs.

    Args:
        noise_std: Standard deviation of Gaussian noise

    Returns:
        X_noisy: Tensor of shape (40, 2)
        y: Tensor of shape (40, 1)
    """
    X, y = create_xor_dataset()

    # Replicate each sample 10 times
    X_expanded = X.repeat(10, 1)  # (40, 2)
    y_expanded = y.repeat(10, 1)  # (40, 1)

    # Add Gaussian noise: X_noisy = X + N(0, r)
    noise = torch.randn_like(X_expanded) * noise_std
    X_noisy = X_expanded + noise

    return X_noisy, y_expanded


def create_parity_dataset(dim):
    """
    Create parity dataset for given dimensionality.

    The parity labeling rule: Label a point with 1 if it contains an
    odd number of -1 coordinate entries, else 0.

    Args:
        dim: Input dimensionality (4, 6, or 8)

    Returns:
        X: Tensor of shape (2^dim, dim) - all hypercube vertices
        y: Tensor of shape (2^dim, 1) - parity labels
    """
    # Generate all vertices of hypercube: coordinates are +1 or -1
    vertices = list(product([-1.0, 1.0], repeat=dim))
    X = torch.tensor(vertices, dtype=torch.float32)

    # Parity labeling: 1 if odd number of -1s, else 0
    labels = []
    for vertex in vertices:
        num_negative = sum(1 for v in vertex if v == -1.0)
        label = 1.0 if num_negative % 2 == 1 else 0.0
        labels.append([label])

    y = torch.tensor(labels, dtype=torch.float32)

    return X, y


def get_dataset_info(X, y):
    """
    Print information about a dataset.

    Args:
        X: Input data
        y: Labels
    """
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Number of samples: {len(X)}")
    print(f"Input dimension: {X.shape[1]}")
    print(f"Label distribution: {y.sum().item():.0f} ones, {len(y) - y.sum().item():.0f} zeros")
    print(f"X range: [{X.min().item():.2f}, {X.max().item():.2f}]")
    print()


if __name__ == "__main__":
    # Test dataset creation
    print("=" * 60)
    print("Testing Dataset Creation")
    print("=" * 60)
    print()

    print("Linear Dataset:")
    X_linear, y_linear = create_linear_dataset()
    get_dataset_info(X_linear, y_linear)

    print("XOR Dataset:")
    X_xor, y_xor = create_xor_dataset()
    get_dataset_info(X_xor, y_xor)

    print("Noisy XOR Dataset (r=0.5):")
    X_noisy, y_noisy = create_noisy_xor_dataset(0.5)
    get_dataset_info(X_noisy, y_noisy)

    print("Parity Dataset (dim=4):")
    X_parity, y_parity = create_parity_dataset(4)
    get_dataset_info(X_parity, y_parity)
