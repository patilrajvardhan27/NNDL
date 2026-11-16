# Lab Assignment 1 - Running Instructions

This guide shows how to run each part of the lab independently.

## Setup

1. Activate your virtual environment:
```bash
source venv/bin/activate
```

2. Install required packages:
```bash
pip install torch numpy matplotlib tqdm
```

## Running Individual Parts

### Part 1: Limits of the Perceptron
Trains perceptron models on linear and XOR datasets to demonstrate perceptron limitations.

```bash
python3 perceptron.py
```

**What it does:**
- Creates linear and XOR datasets
- Trains perceptron on both datasets
- Generates visualizations showing decision boundaries
- Saves results to `results/part1_*.png`

---

### Part 2: Multi-Layer Models
Trains two-layer neural networks to solve both linear and XOR problems.

```bash
python3 two_layer_nn.py
```

**What it does:**
- Trains 10 instances of two-layer NN on linear dataset
- Trains 10 instances of two-layer NN on XOR dataset
- Shows that multi-layer networks can solve XOR
- Saves results to `results/part2_*.png`

---

### Part 3: Harder Datasets
Tests neural networks on noisy XOR and higher-dimensional parity problems.

```bash
python3 part3_harder_datasets.py
```

**What it does:**
- Tests noisy XOR with different noise levels (r=0.25, 0.5, 0.75)
- Tests parity datasets in dimensions 4, 6, and 8
- Compares different hidden layer sizes
- Saves results to `results/part3_*.png`

---

### Extra Credit: Initialization
Investigates how different weight initialization schemes affect convergence.

```bash
python3 extra_credit.py
```

---

### Test Datasets Only
To just test dataset creation without running experiments:

```bash
python3 datasets.py
```

---

### Run All Experiments (Original Main Script)
If you want to run everything at once:

```bash
python3 main_experiments.py
```

## Results

All visualizations and plots are saved to the `results/` directory:
- `part1_*.png` - Perceptron experiments
- `part2_*.png` - Two-layer network experiments
- `part3_*.png` - Harder datasets experiments

## File Structure

```
.
├── perceptron.py              # Part 1 (standalone)
├── two_layer_nn.py            # Part 2 (standalone)
├── part3_harder_datasets.py   # Part 3 (standalone)
├── extra_credit.py            # Extra credit (standalone)
├── datasets.py                # Dataset generation utilities
├── training_utils.py          # Training helper functions
├── visualization.py           # Plotting utilities
├── main_experiments.py        # Original full experiment script
└── results/                   # Output directory for plots
```
