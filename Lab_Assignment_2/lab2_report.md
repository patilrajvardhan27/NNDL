# Lab Assignment 2: Training Deep Neural Networks

---

## Methods

### Dataset and Preprocessing

I chose CIFAR-10, which contains 60,000 color images (32x32x3) across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), split into 50,000 training and 10,000 test samples. This dataset suits our needs because natural-image classification with small networks is genuinely hard — pixel-level features vary with lighting, pose, and background, making it unlikely that a shallow model would exceed 50% accuracy. For preprocessing, each image was converted to a float tensor and channel-wise normalized to zero mean and unit variance using the known CIFAR-10 statistics (means: 0.4914, 0.4822, 0.4465; standard deviations: 0.2470, 0.2435, 0.2616). This normalization is especially important when using sigmoid activations, since unnormalized pixel values push pre-activations into the sigmoid's saturating tails from the start.

To verify dataset difficulty, I built a two-layer fully-connected network: the flattened 3072-dimensional input feeds into a 256-unit hidden layer with sigmoid activation, followed by a 10-way softmax output. It was trained for 10 epochs with vanilla SGD (lr=0.01, batch size=1).

### Baseline Deep Network

The baseline is a 5-layer convolutional network tailored to CIFAR-10's image modality. Layer 1 is a Conv2d (3->16 channels, 3x3, padding 1) followed by sigmoid and MaxPool(2x2), producing 16x16x16 feature maps. Layer 2 is Conv2d (16->32, 3x3, padding 1) with sigmoid and MaxPool(2x2), giving 32x8x8. Layer 3 is Conv2d (32->64, 3x3, padding 1) with sigmoid and MaxPool(2x2), giving 64x4x4. The features are flattened to 1024 dimensions and passed through Layer 4, a Linear(1024->128) with sigmoid, and finally Layer 5, a Linear(128->10) producing logits. Softmax is applied inside the cross-entropy loss computation using the log-sum-exp trick for numerical stability. Training used vanilla SGD (lr=0.01, batch size=1) for 10 epochs, with the optimizer implemented from scratch — no torch.optim was used. Gradients were computed via autograd and parameters updated manually.

### Activation Functions

I selected Leaky ReLU and SiLU as the two replacement activations. Leaky ReLU, f(x) = max(x, 0.1x), is the standard fix for dying neurons and fully avoids sigmoid's saturation — its derivative is always either 1.0 or 0.1, so gradients never vanish. SiLU (Swish), f(x) = x * sigmoid(x), is a smoother alternative that behaves like ReLU for large positive inputs but has a slight non-monotonic dip for negative inputs. Both were implemented from scratch using basic tensor operations (torch.where for Leaky ReLU, element-wise multiply with custom sigmoid for SiLU). The modified models have the identical architecture as the baseline — only the activation function is swapped.

**Hypothesis:** I expected both activations to substantially outperform sigmoid, with Leaky ReLU having an edge because its gradient floor of 0.1 never saturates. Sigmoid's derivative maxes out at 0.25 and approaches zero in the tails, so after five layers the gradient signal gets multiplied by roughly 0.25^4 ~ 0.004. SiLU uses sigmoid internally, so I anticipated it might show some instability under noisy batch-size-1 training.

### Mini-Batch SGD

After identifying the best activation function, I retrained that model with mini-batch SGD at batch sizes of 16, 32, and 64. The implementation averages the per-sample losses within each batch before calling backward — since the gradient of the average equals the average of the gradients (linearity of differentiation), this is mathematically equivalent to averaging per-sample gradients. The learning rate (0.01) and epoch count (10) were held constant so batch size is the only variable.

### Mini-Batch SGD with Momentum

Momentum maintains a running average of past gradients: m^{i+1} = alpha * m^i + g^i, then theta^{i+1} = theta^i - lr * m^{i+1}, where g^i is the current mini-batch gradient and m is initialized to zero. I tested alpha = 0.5, 0.9, and 0.99 using the best batch size from the previous experiment. All other hyperparameters stayed fixed.

### Extended Architecture

To set up the skip-connection experiments, I added 10 Conv2d(16, 16, 3x3, padding=1) layers immediately after the first conv+pool block. These layers all operate on 16-channel, 16x16 feature maps and do not change tensor shape. Each is followed by the best activation function (Leaky ReLU). This produces a 15-layer network total. Training used mini-batch SGD with momentum using the best batch size (16) and best momentum (alpha=0.5) for 10 epochs. During the first epoch, I recorded the L1-norm of every parameter's gradient at each step, then averaged across steps to get one gradient magnitude per parameter.

**Hypothesis:** I expected the extended model to perform worse than the 5-layer baseline. Ten extra layers mean the gradient passes through ten additional nonlinearities, compounding any attenuation even with Leaky ReLU.

### Skip Connection Configurations

I tested two skip-connection layouts, each with exactly 3 additive skip connections among the 10 extra layers:

- **Config A (length-2 skips):** Shortcuts across layers 0-1, 3-4, and 6-7. This creates frequent short-range gradient highways through every other pair of layers.
- **Config B (length-3 skips):** Shortcuts across layers 0-2, 3-5, and 6-8. Fewer but longer skip paths that provide broader gradient coverage.

Both use element-wise addition (tensors are the same shape). All other training details are identical to the extended model.

---

## Results

### Performance Summary

| Model | Test Accuracy | Avg Precision | Avg Recall |
|-------|:---:|:---:|:---:|
| Shallow 2-Layer (Sigmoid) | 0.3958 | 0.4409 | 0.3958 |
| Deep Baseline (Sigmoid, bs=1) | 0.5451 | 0.5453 | 0.5451 |
| Deep CNN (Leaky ReLU, bs=1) | 0.6529 | 0.6636 | 0.6529 |
| Deep CNN (SiLU, bs=1) | 0.3974 | 0.4656 | 0.3974 |
| Leaky ReLU, bs=16 | 0.7283 | 0.7385 | 0.7283 |
| Leaky ReLU, bs=32 | 0.6977 | 0.7000 | 0.6977 |
| Leaky ReLU, bs=64 | 0.5792 | 0.6381 | 0.5792 |
| Leaky ReLU, bs=16, mom=0.5 | 0.7285 | 0.7317 | 0.7285 |
| Leaky ReLU, bs=16, mom=0.9 | 0.6875 | 0.7017 | 0.6875 |
| Leaky ReLU, bs=16, mom=0.99 | 0.1000 | 0.0100 | 0.1000 |
| Extended 15-layer (no skips) | 0.1000 | 0.0100 | 0.1000 |
| Skip Config A (len-2 skips) | 0.7099 | 0.7094 | 0.7099 |
| Skip Config B (len-3 skips) | 0.7324 | 0.7378 | 0.7324 |

[Insert fig1_loss_activations.png — Training loss curves: sigmoid baseline vs Leaky ReLU vs SiLU]

[Insert fig2_acc_activations.png — Test accuracy over epochs: sigmoid baseline vs Leaky ReLU vs SiLU]

[Insert fig3_batch_sizes.png — Test accuracy comparison for batch sizes 16, 32, 64]

[Insert fig4_momentum.png — Test accuracy comparison for momentum alpha = 0.5, 0.9, 0.99]

[Insert fig5_skip_connections.png — Test accuracy: extended (no skips) vs Config A vs Config B]

[Insert fig6_gradient_norms.png — Per-layer L1 gradient norms for extended, Skip-A, and Skip-B models]

[Insert fig8_summary.png — Final accuracy bar chart across all models]

---

## Analysis

### Trend 1: Activation function choice is the single biggest factor in deep-network training

The deep sigmoid baseline was stuck at 10% accuracy (random guessing) for the first 5 epochs before finally learning, reaching 54.5% by epoch 10. Leaky ReLU hit 56.9% after just one epoch and finished at 65.3%. This is the vanishing gradient problem in action — sigmoid's derivative maxes out at 0.25 and drops toward zero in the tails, so after five layers the gradient signal to the early conv layers was essentially zero for the first half of training. Leaky ReLU avoids this entirely since its derivative is always at least 0.1. SiLU started strong (60.9% by epoch 2) but then diverged — its loss increased from epoch 4 onward and accuracy collapsed to 39.7%. The likely cause is SiLU's non-monotonic region combining with noisy batch-size-1 gradients to create unstable updates. This showed that a theoretically fancier activation function is not always better; stability under noisy conditions matters just as much.

### Trend 2: Mini-batching had a bigger practical effect than momentum

Switching from batch size 1 to 16 jumped accuracy from 65.3% to 72.8% — a massive gain from just averaging 16 gradients together. Batch size 16 outperformed 32 (69.8%) and 64 (57.9%) because, at a fixed learning rate of 0.01, more frequent updates (3125 per epoch for bs=16 vs 781 for bs=64) mattered more than cleaner gradients. The bs=64 model's loss was still dropping at epoch 10, suggesting it was simply under-trained. **Best batch size: 16.** Momentum was surprisingly less impactful: alpha=0.5 gave 72.9% (nearly identical to no momentum at 72.8%), alpha=0.9 dropped to 68.8%, and alpha=0.99 completely diverged (NaN loss, 10% accuracy). With lr=0.01, high momentum builds up too much velocity and overshoots. **Best alpha: 0.5.**

### Trend 3: Skip connections revived a completely dead deep network

The extended 15-layer model without skips scored 10% — it never learned anything. Even with Leaky ReLU, passing gradients through 14 nonlinearities means worst-case scaling of 0.1^14 ~ 10^-14 for early layers. Adding just three skip connections brought the model back to life: Config A (length-2 skips) reached 71.0% and Config B (length-3 skips) reached 73.2%. The gradient norm plots confirm this — in the no-skip model, early-layer gradients were orders of magnitude smaller than later layers, while skip connections made gradient magnitudes more uniform across layers. Config B outperformed Config A because its three longer skips cover layers 0-8 (broader coverage), while Config A only covers layers 0-1, 3-4, 6-7 leaving layers 2, 5, 8, 9 outside any skip path. **Best skip configuration: Config B (length-3).**

The shallow 2-layer model's accuracy of 39.6% (well under 50%) confirms CIFAR-10 is too hard for shallow models, validating the dataset choice.

---

## Extra Credit

### Weight Decay

| Model | Test Accuracy | Avg Precision | Avg Recall |
|-------|:---:|:---:|:---:|
| DeepCNN + WD beta=0.0001 | 0.7363 | 0.7402 | 0.7363 |
| DeepCNN + WD beta=0.001 | 0.7539 | 0.7606 | 0.7539 |
| DeepCNN + WD beta=0.01 | 0.6947 | 0.7111 | 0.6947 |
| Skip-B + WD beta=0.0001 | 0.7371 | 0.7371 | 0.7371 |
| Skip-B + WD beta=0.001 | 0.7232 | 0.7416 | 0.7232 |
| Skip-B + WD beta=0.01 | 0.6066 | 0.6798 | 0.6066 |

[Insert fig7_weight_decay.png — Weight decay comparison for DeepCNN and Skip-B at different beta values]

Weight decay adds a penalty term (beta * theta) to the momentum update, pushing weights toward zero to discourage memorization. For the DeepCNN (no skips), beta=0.001 produced 75.4% — the highest accuracy of any model in this study and a clear improvement over the same model without decay (72.9%). Beta=0.0001 gave 73.6% (marginal help), while beta=0.01 degraded to 69.5% (too aggressive). For Skip-B, weight decay was less helpful: beta=0.0001 gave 73.7% (roughly matching the no-decay 73.2%), beta=0.001 dropped to 72.3%, and beta=0.01 fell to 60.7%. The skip-connection model was more sensitive because the additive identity paths amplify the effect of the weight penalty — shrinking weights also weakens the skip connections' ability to propagate signal. **Best beta: 0.001 for DeepCNN, 0.0001 for Skip-B.** Moderate weight decay acts as an effective regularizer for simpler architectures, but models with skip connections already have a built-in regularizing effect from the identity paths, so additional decay can hurt.
