"""
Test Setup Script for Lab Assignment 1
Neural Networks and Deep Learning

Run this script to verify your environment is set up correctly
before running the full experiments.
"""

import sys

print("=" * 80)
print("TESTING ENVIRONMENT SETUP")
print("=" * 80)
print()

# Test 1: Python version
print("Test 1: Python Version")
print("-" * 40)
print(f"Python version: {sys.version}")
if sys.version_info >= (3, 8):
    print("✅ PASS: Python version is sufficient (3.8+)")
else:
    print("❌ FAIL: Python version too old. Need 3.8+")
    sys.exit(1)
print()

# Test 2: PyTorch
print("Test 2: PyTorch")
print("-" * 40)
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print("✅ PASS: PyTorch is installed")
except ImportError:
    print("❌ FAIL: PyTorch not found")
    print("Install with: pip install torch")
    sys.exit(1)
print()

# Test 3: NumPy
print("Test 3: NumPy")
print("-" * 40)
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    print("✅ PASS: NumPy is installed")
except ImportError:
    print("❌ FAIL: NumPy not found")
    print("Install with: pip install numpy")
    sys.exit(1)
print()

# Test 4: Matplotlib
print("Test 4: Matplotlib")
print("-" * 40)
try:
    import matplotlib
    import matplotlib.pyplot as plt
    print(f"Matplotlib version: {matplotlib.__version__}")
    print("✅ PASS: Matplotlib is installed")
except ImportError:
    print("❌ FAIL: Matplotlib not found")
    print("Install with: pip install matplotlib")
    sys.exit(1)
print()

# Test 5: Project files
print("Test 5: Project Files")
print("-" * 40)
required_files = [
    'perceptron.py',
    'two_layer_nn.py',
    'datasets.py',
    'training_utils.py',
    'visualization.py',
    'main_experiments.py'
]

import os
all_files_present = True
for file in required_files:
    if os.path.exists(file):
        print(f"✅ Found: {file}")
    else:
        print(f"❌ Missing: {file}")
        all_files_present = False

if all_files_present:
    print("\n✅ PASS: All required files present")
else:
    print("\n❌ FAIL: Some files are missing")
    sys.exit(1)
print()

# Test 6: Import custom modules
print("Test 6: Import Custom Modules")
print("-" * 40)
try:
    from perceptron import Perceptron
    print("✅ Imported: Perceptron")

    from two_layer_nn import TwoLayerNN
    print("✅ Imported: TwoLayerNN")

    from datasets import create_linear_dataset, create_xor_dataset
    print("✅ Imported: datasets")

    from training_utils import train_perceptron
    print("✅ Imported: training_utils")

    from visualization import plot_dataset
    print("✅ Imported: visualization")

    print("\n✅ PASS: All modules import successfully")
except ImportError as e:
    print(f"\n❌ FAIL: Import error - {e}")
    sys.exit(1)
print()

# Test 7: Basic tensor operations
print("Test 7: Basic Tensor Operations")
print("-" * 40)
try:
    # Create tensors
    x = torch.tensor([1.0, 2.0, 3.0])
    w = torch.randn(3, 1)

    # Matrix multiplication
    result = torch.matmul(w.T, x.unsqueeze(1))

    # Element-wise operations
    a = torch.sigmoid(result)

    print(f"Created tensor x: {x.shape}")
    print(f"Created tensor w: {w.shape}")
    print(f"Result shape: {result.shape}")
    print("✅ PASS: Basic tensor operations work")
except Exception as e:
    print(f"❌ FAIL: Tensor operation error - {e}")
    sys.exit(1)
print()

# Test 8: Quick model test
print("Test 8: Quick Model Test")
print("-" * 40)
try:
    # Test Perceptron
    perceptron = Perceptron(input_dim=2)
    x_test = torch.tensor([1.0, -1.0])
    y_pred = perceptron.forward(x_test)
    print(f"Perceptron prediction: {y_pred.item()}")
    print("✅ Perceptron forward pass works")

    # Test TwoLayerNN
    nn = TwoLayerNN(input_dim=2, hidden_dim=2)
    y_pred = nn.forward(x_test)
    print(f"TwoLayerNN prediction: {y_pred.item():.4f}")
    print("✅ TwoLayerNN forward pass works")

    print("\n✅ PASS: Models can make predictions")
except Exception as e:
    print(f"❌ FAIL: Model test error - {e}")
    sys.exit(1)
print()

# Test 9: Dataset creation
print("Test 9: Dataset Creation")
print("-" * 40)
try:
    X_linear, y_linear = create_linear_dataset()
    print(f"Linear dataset: X={X_linear.shape}, y={y_linear.shape}")
    print(f"  Samples: {len(X_linear)}")

    X_xor, y_xor = create_xor_dataset()
    print(f"XOR dataset: X={X_xor.shape}, y={y_xor.shape}")
    print(f"  Samples: {len(X_xor)}")

    print("✅ PASS: Datasets can be created")
except Exception as e:
    print(f"❌ FAIL: Dataset creation error - {e}")
    sys.exit(1)
print()

# Test 10: Results directory
print("Test 10: Results Directory")
print("-" * 40)
try:
    os.makedirs('results', exist_ok=True)
    print("✅ Created/verified 'results/' directory")
    print("✅ PASS: Can create output directories")
except Exception as e:
    print(f"❌ FAIL: Directory creation error - {e}")
    sys.exit(1)
print()

# Optional: tqdm
print("Optional: Progress Bars (tqdm)")
print("-" * 40)
try:
    from tqdm import tqdm
    print(f"tqdm version: {tqdm.__version__ if hasattr(tqdm, '__version__') else 'unknown'}")
    print("✅ tqdm is installed (progress bars will show)")
except ImportError:
    print("⚠️  tqdm not installed (progress bars won't show)")
    print("   Optional: pip install tqdm")
print()

# Summary
print("=" * 80)
print("SETUP VERIFICATION COMPLETE")
print("=" * 80)
print()
print("✅ All tests passed! Your environment is ready.")
print()
print("Next steps:")
print("  1. Read EXPLANATION.md for detailed guide")
print("  2. Read README.md for quick start")
print("  3. Run: python main_experiments.py")
print()
print("Good luck! 🚀")
