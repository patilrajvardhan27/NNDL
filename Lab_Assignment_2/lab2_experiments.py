import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from collections import defaultdict
import time
import os
import sys

torch.manual_seed(42)
np.random.seed(42)

NUM_EPOCHS = 10
LR = 0.01
NUM_CLASSES = 10

print(f"Config: epochs={NUM_EPOCHS}, lr={LR}")

# Loading and Preprocess CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

print("Download CIFAR-10...")
train_ds = datasets.CIFAR10('./data', train=True,  download=True, transform=transform)
test_ds  = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

# Load all into tensors for fast access during training
train_imgs = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
train_lbls = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
test_imgs  = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
test_lbls  = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])

print(f"Train: {train_imgs.shape}, Test: {test_imgs.shape}")
print(f"Classes: {train_ds.classes}")


# Activation Functions  (all implemented from scratch)
def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x.clamp(-50, 50)))

def leaky_relu(x):
    return torch.where(x > 0, x, 0.1 * x)

def silu(x):
    return x * sigmoid(x)

# Cross-Entropy Loss  
def cross_entropy(logits, targets):
    b = logits.shape[0]
    shifted = logits - logits.max(dim=1, keepdim=True)[0]
    log_softmax = shifted - torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))
    return -log_softmax[torch.arange(b), targets].mean()


# Optimizer
def zero_grads(params):
    for p in params:
        p.grad = None

def sgd_step(params, lr):
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                p.data -= lr * p.grad

def momentum_step(params, vel, lr, alpha):
    with torch.no_grad():
        for i, p in enumerate(params):
            if p.grad is not None:
                vel[i] = alpha * vel[i] + p.grad
                p.data -= lr * vel[i]

def momentum_wd_step(params, vel, lr, alpha, beta):
    with torch.no_grad():
        for i, p in enumerate(params):
            if p.grad is not None:
                vel[i] = alpha * vel[i] + p.grad + beta * p.data
                p.data -= lr * vel[i]


# Evaluation code
def precision_recall(preds, targets, nc=10):
    precs, recs = [], []
    for c in range(nc):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        precs.append(tp / (tp + fp) if tp + fp > 0 else 0.0)
        recs.append(tp / (tp + fn) if tp + fn > 0 else 0.0)
    return np.mean(precs), np.mean(recs)

def evaluate(model, imgs, lbls, bs=256):
    preds_all = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            preds_all.append(model(imgs[i:i+bs]).argmax(1))
    preds = torch.cat(preds_all)
    acc = (preds == lbls).float().mean().item()
    p, r = precision_recall(preds, lbls)
    return acc, p, r


# Unified Training Loop
def train_model(model, epochs, lr, bs=1, alpha=0.0, beta=0.0,
                track_grads=False, name="Model"):
    print(f"\n{'='*55}")
    print(f" {name}")
    print(f" bs={bs}  alpha={alpha}  beta={beta}  lr={lr}")
    print(f"{'='*55}")

    params = [p for p in model.parameters() if p.requires_grad]
    vel = [torch.zeros_like(p) for p in params] if alpha > 0 else None

    hist = {'loss': [], 'acc': [], 'prec': [], 'rec': [], 'grad_norms': {}}
    n = len(train_imgs)
    t0 = time.time()

    for ep in range(epochs):
        perm = torch.randperm(n)
        running_loss, steps = 0.0, 0
        gn = defaultdict(list) if (track_grads and ep == 0) else None

        for s in range(0, n, bs):
            idx = perm[s:min(s + bs, n)]
            x, y = train_imgs[idx], train_lbls[idx]

            zero_grads(params)
            loss = cross_entropy(model(x), y)
            loss.backward()

            if gn is not None:
                for j, p in enumerate(params):
                    if p.grad is not None:
                        gn[j].append(p.grad.abs().sum().item())

            if vel is not None:
                (momentum_wd_step if beta > 0 else momentum_step)(
                    params, vel, lr, alpha, *([beta] if beta > 0 else []))
            else:
                sgd_step(params, lr)

            running_loss += loss.item()
            steps += 1

        avg_loss = running_loss / steps
        hist['loss'].append(avg_loss)
        acc, p, r = evaluate(model, test_imgs, test_lbls)
        hist['acc'].append(acc)
        hist['prec'].append(p)
        hist['rec'].append(r)

        if gn:
            for j in gn:
                hist['grad_norms'][j] = np.mean(gn[j])

        elapsed = time.time() - t0
        print(f"  Ep {ep+1:>2}/{epochs} | Loss {avg_loss:.4f} | "
              f"Acc {acc:.4f} | P {p:.4f} | R {r:.4f} | {elapsed:.0f}s",
              flush=True)

    return hist


#  MODELS
class ShallowNet(nn.Module):
    def __init__(self, in_dim=3072, hid=256, out=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, out)

    def forward(self, x):
        x = sigmoid(self.fc1(x.flatten(1)))
        return self.fc2(x)


class DeepCNN(nn.Module):
    def __init__(self, act_fn=sigmoid):
        super().__init__()
        self.act = act_fn
        self.c1 = nn.Conv2d(3,  16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.f1 = nn.Linear(64*4*4, 128)
        self.f2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.act(self.c1(x)))    
        x = self.pool(self.act(self.c2(x)))    
        x = self.pool(self.act(self.c3(x)))    
        x = self.act(self.f1(x.flatten(1)))    
        return self.f2(x)                      


class ExtendedCNN(nn.Module):
    def __init__(self, act_fn=sigmoid):
        super().__init__()
        self.act = act_fn
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.extras = nn.ModuleList(
            [nn.Conv2d(16, 16, 3, padding=1) for _ in range(10)]
        )
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.f1 = nn.Linear(64*4*4, 128)
        self.f2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.act(self.c1(x)))
        for conv in self.extras:
            x = self.act(conv(x))
        x = self.pool(self.act(self.c2(x)))
        x = self.pool(self.act(self.c3(x)))
        x = self.act(self.f1(x.flatten(1)))
        return self.f2(x)


class SkipCNN_A(nn.Module):
    def __init__(self, act_fn=sigmoid):
        super().__init__()
        self.act = act_fn
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.extras = nn.ModuleList(
            [nn.Conv2d(16, 16, 3, padding=1) for _ in range(10)]
        )
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.f1 = nn.Linear(64*4*4, 128)
        self.f2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.act(self.c1(x)))

        # Skip 1: skip over layers 0-1
        residual = x
        x = self.act(self.extras[0](x))
        x = self.act(self.extras[1](x)) + residual

        x = self.act(self.extras[2](x))

        # Skip 2: skip over layers 3-4
        residual = x
        x = self.act(self.extras[3](x))
        x = self.act(self.extras[4](x)) + residual

        x = self.act(self.extras[5](x))

        # Skip 3: skip over layers 6-7
        residual = x
        x = self.act(self.extras[6](x))
        x = self.act(self.extras[7](x)) + residual

        x = self.act(self.extras[8](x))
        x = self.act(self.extras[9](x))

        x = self.pool(self.act(self.c2(x)))
        x = self.pool(self.act(self.c3(x)))
        x = self.act(self.f1(x.flatten(1)))
        return self.f2(x)


class SkipCNN_B(nn.Module):
    def __init__(self, act_fn=sigmoid):
        super().__init__()
        self.act = act_fn
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.extras = nn.ModuleList(
            [nn.Conv2d(16, 16, 3, padding=1) for _ in range(10)]
        )
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.f1 = nn.Linear(64*4*4, 128)
        self.f2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.act(self.c1(x)))

        residual = x
        x = self.act(self.extras[0](x))
        x = self.act(self.extras[1](x))
        x = self.act(self.extras[2](x)) + residual
        residual = x
        x = self.act(self.extras[3](x))
        x = self.act(self.extras[4](x))
        x = self.act(self.extras[5](x)) + residual
        residual = x
        x = self.act(self.extras[6](x))
        x = self.act(self.extras[7](x))
        x = self.act(self.extras[8](x)) + residual
        x = self.act(self.extras[9](x))
        x = self.pool(self.act(self.c2(x)))
        x = self.pool(self.act(self.c3(x)))
        x = self.act(self.f1(x.flatten(1)))
        return self.f2(x)


print("PART 1: BASELINE MODELS")

shallow = ShallowNet()
h_shallow = train_model(shallow, NUM_EPOCHS, LR, bs=1,
                        name="Shallow 2-Layer (Sigmoid, bs=1)")

baseline = DeepCNN(act_fn=sigmoid)
h_baseline = train_model(baseline, NUM_EPOCHS, LR, bs=1,
                         name="Deep CNN Baseline (Sigmoid, bs=1)")

print(f"\n Shallow  final acc: {h_shallow['acc'][-1]:.4f}")
print(f" Baseline final acc: {h_baseline['acc'][-1]:.4f}")


print("PART 2: ACTIVATION FUNCTIONS & OPTIMIZERS")

act_experiments = {}

m_leaky = DeepCNN(act_fn=leaky_relu)
h_leaky = train_model(m_leaky, NUM_EPOCHS, LR, bs=1,
                      name="Deep CNN (Leaky ReLU, bs=1)")
act_experiments['Leaky ReLU'] = {'hist': h_leaky, 'fn': leaky_relu}

m_silu = DeepCNN(act_fn=silu)
h_silu = train_model(m_silu, NUM_EPOCHS, LR, bs=1,
                     name="Deep CNN (SiLU, bs=1)")
act_experiments['SiLU'] = {'hist': h_silu, 'fn': silu}

best_act_name = max(act_experiments,
                    key=lambda k: act_experiments[k]['hist']['acc'][-1])
best_act_fn = act_experiments[best_act_name]['fn']
print(f"\n Best activation: {best_act_name} "
      f"(acc={act_experiments[best_act_name]['hist']['acc'][-1]:.4f})")
 
batch_experiments = {}
for bs in [16, 32, 64]:
    m = DeepCNN(act_fn=best_act_fn)
    h = train_model(m, NUM_EPOCHS, LR, bs=bs,
                    name=f"Deep CNN ({best_act_name}, bs={bs})")
    batch_experiments[bs] = h

best_bs = max(batch_experiments,
              key=lambda k: batch_experiments[k]['acc'][-1])
print(f"\nBest batch size: {best_bs} "
      f"(acc={batch_experiments[best_bs]['acc'][-1]:.4f})")

mom_experiments = {}
for alpha in [0.5, 0.9, 0.99]:
    m = DeepCNN(act_fn=best_act_fn)
    h = train_model(m, NUM_EPOCHS, LR, bs=best_bs, alpha=alpha,
                    name=f"Deep CNN ({best_act_name}, bs={best_bs}, mom={alpha})")
    mom_experiments[alpha] = h

best_alpha = max(mom_experiments,
                 key=lambda k: mom_experiments[k]['acc'][-1])
print(f"\nBest momentum: {best_alpha} "
      f"(acc={mom_experiments[best_alpha]['acc'][-1]:.4f})")

print("#  PART 3: SKIP CONNECTIONS")

ext_model = ExtendedCNN(act_fn=best_act_fn)
h_ext = train_model(ext_model, NUM_EPOCHS, LR, bs=best_bs,
                    alpha=best_alpha, track_grads=True,
                    name=f"Extended CNN (no skips, {best_act_name})")

skip_a = SkipCNN_A(act_fn=best_act_fn)
h_skip_a = train_model(skip_a, NUM_EPOCHS, LR, bs=best_bs,
                       alpha=best_alpha, track_grads=True,
                       name=f"Skip Config A (len-2, {best_act_name})")

skip_b = SkipCNN_B(act_fn=best_act_fn)
h_skip_b = train_model(skip_b, NUM_EPOCHS, LR, bs=best_bs,
                       alpha=best_alpha, track_grads=True,
                       name=f"Skip Config B (len-3, {best_act_name})")

print(f"\n Extended  acc: {h_ext['acc'][-1]:.4f}")
print(f" Skip-A    acc: {h_skip_a['acc'][-1]:.4f}")
print(f" Skip-B    acc: {h_skip_b['acc'][-1]:.4f}")

print("EXTRA CREDIT:")

wd_no_skip = {}
for beta in [0.0001, 0.001, 0.01]:
    m = DeepCNN(act_fn=best_act_fn)
    h = train_model(m, NUM_EPOCHS, LR, bs=best_bs,
                    alpha=best_alpha, beta=beta,
                    name=f"DeepCNN + WD beta={beta}")
    wd_no_skip[beta] = h

best_skip_cls = SkipCNN_A if h_skip_a['acc'][-1] >= h_skip_b['acc'][-1] else SkipCNN_B
best_skip_label = "Skip-A" if best_skip_cls == SkipCNN_A else "Skip-B"

wd_skip = {}
for beta in [0.0001, 0.001, 0.01]:
    m = best_skip_cls(act_fn=best_act_fn)
    h = train_model(m, NUM_EPOCHS, LR, bs=best_bs,
                    alpha=best_alpha, beta=beta,
                    name=f"{best_skip_label} + WD beta={beta}")
    wd_skip[beta] = h
print("  RESULTS SUMMARY")

all_results = {
    'Shallow (Sigmoid)':       h_shallow,
    'Baseline (Sigmoid)':      h_baseline,
    'Leaky ReLU':              h_leaky,
    'SiLU':                    h_silu,
}
for bs, h in batch_experiments.items():
    all_results[f'{best_act_name} bs={bs}'] = h
for a, h in mom_experiments.items():
    all_results[f'{best_act_name} mom={a}'] = h
all_results['Extended (no skip)'] = h_ext
all_results['Skip Config A']     = h_skip_a
all_results['Skip Config B']     = h_skip_b
for beta, h in wd_no_skip.items():
    all_results[f'DeepCNN WD={beta}'] = h
for beta, h in wd_skip.items():
    all_results[f'{best_skip_label} WD={beta}'] = h

print(f"{'Model':<35} {'Acc':>7} {'Prec':>7} {'Rec':>7}")
print("-"*60)
for name, h in all_results.items():
    print(f"{name:<35} {h['acc'][-1]:>7.4f} {h['prec'][-1]:>7.4f} {h['rec'][-1]:>7.4f}")

os.makedirs('plots', exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))
for label, h in [('Shallow (Sigmoid)', h_shallow),
                  ('Baseline (Sigmoid)', h_baseline),
                  ('Leaky ReLU', h_leaky),
                  ('SiLU', h_silu)]:
    ax.plot(range(1, NUM_EPOCHS+1), h['loss'], marker='o', label=label)
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
ax.set_title('Training Loss: Baseline vs Activation Functions')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/fig1_loss_activations.png', dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
for label, h in [('Shallow (Sigmoid)', h_shallow),
                  ('Baseline (Sigmoid)', h_baseline),
                  ('Leaky ReLU', h_leaky),
                  ('SiLU', h_silu)]:
    ax.plot(range(1, NUM_EPOCHS+1), h['acc'], marker='o', label=label)
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Accuracy')
ax.set_title('Test Accuracy: Baseline vs Activation Functions')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/fig2_acc_activations.png', dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
for bs, h in batch_experiments.items():
    ax.plot(range(1, NUM_EPOCHS+1), h['acc'], marker='o', label=f'bs={bs}')
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Accuracy')
ax.set_title(f'Test Accuracy vs Batch Size ({best_act_name})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/fig3_batch_sizes.png', dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
for a, h in mom_experiments.items():
    ax.plot(range(1, NUM_EPOCHS+1), h['acc'], marker='o', label=f'alpha={a}')
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Accuracy')
ax.set_title(f'Test Accuracy vs Momentum ({best_act_name}, bs={best_bs})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/fig4_momentum.png', dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
for label, h in [('Extended (no skip)', h_ext),
                  ('Skip Config A (len-2)', h_skip_a),
                  ('Skip Config B (len-3)', h_skip_b)]:
    ax.plot(range(1, NUM_EPOCHS+1), h['acc'], marker='o', label=label)
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Accuracy')
ax.set_title('Test Accuracy: Extended Models & Skip Connections')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/fig5_skip_connections.png', dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(12, 6))
layer_indices = sorted(h_ext['grad_norms'].keys())
if layer_indices:
    width = 0.25
    x_pos = np.arange(len(layer_indices))
    norms_ext  = [h_ext['grad_norms'].get(i, 0) for i in layer_indices]
    norms_a    = [h_skip_a['grad_norms'].get(i, 0) for i in layer_indices]
    norms_b    = [h_skip_b['grad_norms'].get(i, 0) for i in layer_indices]

    ax.bar(x_pos - width, norms_ext, width, label='No Skips')
    ax.bar(x_pos,         norms_a,   width, label='Skip Config A')
    ax.bar(x_pos + width, norms_b,   width, label='Skip Config B')
    ax.set_xlabel('Parameter Index')
    ax.set_ylabel('Average L1 Gradient Norm (Epoch 1)')
    ax.set_title('Gradient Magnitudes Across Layers')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('plots/fig6_gradient_norms.png', dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for beta, h in wd_no_skip.items():
    axes[0].plot(range(1, NUM_EPOCHS+1), h['acc'], marker='o', label=f'beta={beta}')
axes[0].set_title('Weight Decay: DeepCNN (no skip)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Test Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

for beta, h in wd_skip.items():
    axes[1].plot(range(1, NUM_EPOCHS+1), h['acc'], marker='o', label=f'beta={beta}')
axes[1].set_title(f'Weight Decay: {best_skip_label}')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Test Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/fig7_weight_decay.png', dpi=150)
plt.close()
 
fig, ax = plt.subplots(figsize=(16, 7))
names = list(all_results.keys())
accs  = [all_results[n]['acc'][-1] for n in names]
bars = ax.barh(range(len(names)), accs, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(names))))
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('Test Accuracy')
for i, v in enumerate(accs):
    ax.text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=7)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('plots/fig8_summary.png', dpi=150)
plt.close()

