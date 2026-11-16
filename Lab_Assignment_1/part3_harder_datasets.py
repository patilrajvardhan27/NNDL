import torch
import numpy as np
import os

from two_layer_nn import TwoLayerNN
from datasets import create_noisy_xor_dataset, create_parity_dataset
from training_utils import train_two_layer_nn, train_multiple_instances, find_best_model
from visualization import plot_dataset, plot_decision_boundary_nn, plot_boxplot, plot_multiple_curves

os.makedirs('results', exist_ok=True)

noise_levels = [0.25, 0.5, 0.75]
noisy_res = {}

for r in noise_levels:
    Xn, yn = create_noisy_xor_dataset(r)
    plot_dataset(Xn, yn, f"Noisy XOR (r={r})", save_path=f'results/part3_noisy_xor_r{r}_dataset.png')

    mods, accs, faccs = train_multiple_instances(TwoLayerNN, Xn, yn, num_instances=100, epochs=1000, lr=0.1, input_dim=2, hidden_dim=2, verbose=False)
    noisy_res[f"r={r}"] = faccs

    bm, ba, _ = find_best_model(mods, accs)
    plot_decision_boundary_nn(Xn, yn, bm, f"Best (r={r})", save_path=f'results/part3_noisy_xor_r{r}_boundary.png')

plot_boxplot(noisy_res, "Noisy XOR Accuracy", ylabel="Accuracy", save_path='results/part3_noisy_xor_boxplot.png')

for r, accs in noisy_res.items():
    print(f"{r}: μ={np.mean(accs):.4f}, σ={np.std(accs):.4f}")

dims = [4, 6, 8]
par_eq = {}
par_2 = {}

for d in dims:
    Xp, yp = create_parity_dataset(d)

    m1 = TwoLayerNN(input_dim=d, hidden_dim=d)
    a1 = train_two_layer_nn(m1, Xp, yp, epochs=1000, lr=0.1, verbose=False)
    par_eq[f"dim={d}"] = a1

    m2 = TwoLayerNN(input_dim=d, hidden_dim=2)
    a2 = train_two_layer_nn(m2, Xp, yp, epochs=1000, lr=0.1, verbose=False)
    par_2[f"dim={d}"] = a2

    plot_multiple_curves({f"h={d}": a1, "h=2": a2}, title=f"Parity (dim={d})", save_path=f'results/part3_parity_dim{d}_comparison.png')

plot_multiple_curves(par_eq, title="Parity: h=d", save_path='results/part3_parity_hidden_eq_dim.png')
plot_multiple_curves(par_2, title="Parity: h=2", save_path='results/part3_parity_hidden_2.png')

print("\nh=d:")
for lbl, accs in par_eq.items():
    print(f"  {lbl}: {accs[-1]:.4f}")

print("\nh=2:")
for lbl, accs in par_2.items():
    print(f"  {lbl}: {accs[-1]:.4f}")
