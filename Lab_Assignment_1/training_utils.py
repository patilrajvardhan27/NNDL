"""
Training Utilities for Lab Assignment 1
Neural Networks and Deep Learning

This module contains training loops and helper functions.
"""

import torch
import numpy as np
from tqdm import tqdm


def train_perceptron(model, X, y, epochs, lr, verbose=False):
    """
    Train a perceptron model.

    Args:
        model: Perceptron instance
        X: Training data (N, input_dim)
        y: Training labels (N, 1)
        epochs: Number of training epochs
        lr: Learning rate
        verbose: Whether to print progress

    Returns:
        accuracies: List of accuracy values per epoch
        angles: List of decision boundary angles per epoch
    """
    accuracies = []
    angles = []

    iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)

    for epoch in iterator:
        # Train on all samples
        for i in range(len(X)):
            x_i = X[i]
            y_i = y[i]

            # Forward pass
            y_pred = model.forward(x_i)

            # Backward pass
            model.backward(x_i, y_pred, y_i, lr)

        accuracy = model.acc(X, y)
        accuracies.append(accuracy)

        if model.d == 2:
            angle = model.get_angle()
            angles.append(angle)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")

    return accuracies, angles


def train_two_layer_nn(model, X, y, epochs, lr, verbose=False):
    """
    Train a two-layer neural network.

    Args:
        model: TwoLayerNN instance
        X: Training data (N, input_dim)
        y: Training labels (N, 1)
        epochs: Number of training epochs
        lr: Learning rate
        verbose: Whether to print progress

    Returns:
        accuracies: List of accuracy values per epoch
    """
    accuracies = []

    iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)

    for epoch in iterator:
        # Train on all samples
        for i in range(len(X)):
            x_i = X[i]
            y_i = y[i]

            # Forward pass
            model.forward(x_i)

            # Backward pass
            model.backward(y_i, lr)

        accuracy = model.acc(X, y)
        accuracies.append(accuracy)

        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")

    return accuracies


def train_multiple_instances(ModelClass, X, y, num_instances, epochs, lr,
                             input_dim, hidden_dim=None, verbose=False):
    """
    Train multiple instances with different random initializations.

    Args:
        ModelClass: Perceptron or TwoLayerNN class
        X: Training data
        y: Training labels
        num_instances: Number of instances to train
        epochs: Number of epochs per instance
        lr: Learning rate
        input_dim: Input dimension
        hidden_dim: Hidden dimension (for TwoLayerNN only)
        verbose: Whether to print progress

    Returns:
        models: List of trained models
        all_accuracies: List of accuracy curves
        final_accuracies: List of final accuracy values
    """
    models = []
    all_accuracies = []
    final_accuracies = []

    iterator = tqdm(range(num_instances), desc="Training instances") if verbose else range(num_instances)

    for i in iterator:
        # Create new model instance
        if hidden_dim is not None:
            model = ModelClass(input_dim, hidden_dim)
        else:
            model = ModelClass(input_dim)

        # Train the model
        if ModelClass.__name__ == 'Perceptron':
            accuracies, _ = train_perceptron(model, X, y, epochs, lr, verbose=False)
        else:
            accuracies = train_two_layer_nn(model, X, y, epochs, lr, verbose=False)

        models.append(model)
        all_accuracies.append(accuracies)
        final_accuracies.append(accuracies[-1])

    if verbose:
        print(f"\nFinal accuracies - Mean: {np.mean(final_accuracies):.4f}, "
              f"Std: {np.std(final_accuracies):.4f}")
        print(f"Perfect accuracy (100%): {sum(acc == 1.0 for acc in final_accuracies)}/{num_instances}")

    return models, all_accuracies, final_accuracies


def get_convergence_time(accuracies, threshold=1.0):
    """
    Get the epoch at which model first achieves threshold accuracy.

    Args:
        accuracies: List of accuracy values
        threshold: Accuracy threshold (default 1.0 for 100%)

    Returns:
        Convergence epoch (or None if never converged)
    """
    for epoch, acc in enumerate(accuracies):
        if acc >= threshold:
            return epoch + 1
    return None


def find_best_model(models, all_accuracies):
    """
    Find the best performing model based on convergence speed.

    Args:
        models: List of trained models
        all_accuracies: List of accuracy curves

    Returns:
        best_model: The best performing model
        best_accuracies: Accuracy curve of best model
        best_idx: Index of best model
    """
    convergence_times = []

    for accuracies in all_accuracies:
        conv_time = get_convergence_time(accuracies)
        if conv_time is not None:
            convergence_times.append(conv_time)
        else:
            # If didn't converge, use large number
            convergence_times.append(float('inf'))

    # Best model is the one that converged fastest
    best_idx = np.argmin(convergence_times)
    best_model = models[best_idx]
    best_accuracies = all_accuracies[best_idx]

    return best_model, best_accuracies, best_idx


def compute_statistics(values):
    """
    Compute statistics for a list of values.

    Args:
        values: List or array of numerical values

    Returns:
        Dictionary with statistics
    """
    values = np.array(values)
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values),
        'q25': np.percentile(values, 25),
        'q75': np.percentile(values, 75)
    }
