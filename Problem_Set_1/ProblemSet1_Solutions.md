# Problem Set 1 - Solutions
## Neural Networks and Deep Learning

---

## Question 1: Analysis of the Cross-Entropy Loss Function [10 points]

### Part (a): Best and Worst Case Loss Values for 10 Classes

With one-hot encoding, the ground truth vector y has exactly one entry equal to 1 at the true class position k, with zeros elsewhere. This simplifies cross-entropy to **L(y, ŷ) = -ln(ŷₖ)**, where we only evaluate the predicted probability for the correct class.

#### i. Best case when predicted class is correct: **L = 0**

When the model correctly identifies the right class, the optimal scenario occurs with complete confidence. If ŷₖ = 1.0 for the true class, we get L = -ln(1.0) = 0. This represents the theoretical minimum, indicating perfect prediction.

#### ii. Worst case when predicted class is incorrect: **L → ∞**

When the prediction is wrong and the model assigns near-zero probability to the true class, the loss explodes. As ŷₖ → 0, we have L = -ln(0) → ∞. This unbounded penalty reflects how cross-entropy severely punishes confident misclassifications.

#### iii. Best case when predicted class is incorrect: **L ≈ 0.693 (ln(2))**

Even with an incorrect argmax, we can minimize loss by making the true class a close second. The limiting case occurs when two classes essentially tie at 50% probability each, with the wrong class barely winning. Here ŷₖ approaches 0.5, giving L = -ln(0.5) = ln(2) ≈ 0.693.

#### iv. Worst case when predicted class is correct: **L ≈ 2.303 (ln(10))**

The prediction is technically correct but with minimal confidence. This happens when ŷ approaches uniform distribution where each of the 10 classes gets equal probability. With ŷₖ = 0.1, we get L = -ln(0.1) = ln(10) ≈ 2.303. The model got lucky with its argmax but demonstrates no real learning.

---

### Part (b): Comparison with 1000 Classes

**Only quantity (iv) changes when scaling to 1000 classes.**

The perfect prediction case (i) remains at **L = 0** since -ln(1) = 0 regardless of class count. The catastrophic incorrect case (ii) stays at **L → ∞** because -ln(0) is unbounded in any context. The barely-incorrect scenario (iii) maintains **L = ln(2) ≈ 0.693** since it only involves two classes competing for maximum probability.

However, the barely-correct worst case (iv) definitely changes. With 1000 classes, uniform distribution spreads probability much thinner at ŷₖ = 1/1000 = 0.001. This yields **L = ln(1000) ≈ 6.908**, substantially higher than the 10-class scenario. Random guessing becomes increasingly penalized as options multiply, which cross-entropy captures through this logarithmic growth.

---

### Part (c): Formula for Uniform Prediction

**L(n) = ln(n)**

For n classes with uniform prediction, each class receives probability ŷᵢ = 1/n. Against a one-hot encoded label where yₖ = 1 for the true class:

L(y, ŷ) = -Σᵢ₌₁ⁿ yᵢ ln(ŷᵢ)
        = -yₖ ln(ŷₖ)    [all other yᵢ = 0]
        = -ln(1/n)
        = **ln(n)**

This elegant formula shows that random guessing incurs logarithmically increasing penalties. A uniform predictor on binary classification suffers ln(2) ≈ 0.693 loss, while the same strategy on 1000 classes yields ln(1000) ≈ 6.908 loss.

---

## Question 2: Backpropagation on Computational Graphs [15 points]

### Part (a): Scalar Valued Computational Graph

#### Forward Pass

Starting with X₁ = 1 and X₂ = -1:

**H₁ = X₁ × X₂** = 1 × (-1) = **-1**

**H₂ = min(X₁, H₁)** = min(1, -1) = **-1**

**H₃ = W₁H₁ - W₂H₂** = (-0.5)(-1) - (1.0)(-1) = 0.5 + 1.0 = **1.5**

**Ŷ = σ(H₃ + H₂)** = σ(1.5 - 1) = σ(0.5) = 1/(1 + e⁻⁰·⁵) ≈ **0.6225**

#### Loss Computation

For binary classification with Y = 0:

**L = -ln(1 - Ŷ)** = -ln(0.3775) ≈ **0.974**

#### Backward Pass

Let S = H₃ + H₂ = 0.5 (input to sigmoid)

**dL/dŜ** = 1/(1-Ŷ) = 1/0.3775 ≈ 2.649

The sigmoid derivative: σ'(S) = σ(S)(1-σ(S)) = 0.6225 × 0.3775 ≈ 0.2350

**dL/dS** = 2.649 × 0.2350 ≈ **0.6225**

Since S = H₃ + H₂, both receive this gradient initially. However, H₂ appears in two computational paths:
- Direct path from S → +0.6225
- Through H₃ (since H₃ = W₁H₁ - W₂H₂) → -W₂ × 0.6225 = -0.6225

These cancel perfectly: **∇H₂ = 0**

**∇H₃ = 0.6225**

**∇W₂** = ∇H₃ × ∂H₃/∂W₂ = 0.6225 × (-H₂) = 0.6225 × 1 = **0.6225**

**∇W₁** = ∇H₃ × ∂H₃/∂W₁ = 0.6225 × H₁ = 0.6225 × (-1) = **-0.6225**

For H₁, since ∇H₂ = 0 and the min function selected H₁ (meaning ∂H₂/∂H₁ = 1, ∂H₂/∂X₁ = 0):

**∇H₁** = ∇H₃ × W₁ = 0.6225 × (-0.5) = **-0.3113**

For H₁ = X₁X₂:

**∇X₂** = ∇H₁ × X₁ = -0.3113 × 1 = **-0.3113**

**∇X₁** = ∇H₁ × X₂ = -0.3113 × (-1) = **0.3113**

#### Summary Table

| Ŷ | L | ∇W₂ | ∇W₁ | ∇H₃ | ∇H₂ | ∇H₁ | ∇X₂ | ∇X₁ |
|-------|-------|--------|---------|--------|------|---------|---------|--------|
| 0.6225 | 0.974 | 0.6225 | -0.6225 | 0.6225 | 0 | -0.3113 | -0.3113 | 0.3113 |

---

### Part (b): Tensor Valued Computational Graph

#### Forward Pass

**Step 1: Convolution W₁ ⊛ X**

With zero padding, X becomes [0, 1.0, -1.0, 0.5, 0]. Using kernel W₁ = [-0.5, 0.0, 0.5] with stride 1:

- Position 0: (-0.5)(0) + 0(1.0) + (0.5)(-1.0) = **-0.5**
- Position 1: (-0.5)(1.0) + 0(-1.0) + (0.5)(0.5) = **-0.25**
- Position 2: (-0.5)(-1.0) + 0(0.5) + (0.5)(0) = **0.5**

**W₁ ⊛ X = [-0.5, -0.25, 0.5]ᵀ**

**Step 2: Apply Leaky ReLU**

f(x) = max(x, 0.1x)

- f(-0.5) = 0.1(-0.5) = -0.05
- f(-0.25) = 0.1(-0.25) = -0.025
- f(0.5) = 0.5

**f(W₁ ⊛ X) = [-0.05, -0.025, 0.5]ᵀ**

**Step 3: Add Skip Connection**

**H₁ = [-0.05, -0.025, 0.5]ᵀ + [1.0, -1.0, 0.5]ᵀ = [0.95, -1.025, 1.0]ᵀ**

**Step 4: Matrix Multiplication H₂ = W₂ × H₁**

```
H₂[0] = 1(0.95) + 2(-1.025) + (-1)(1.0) = 0.95 - 2.05 - 1.0 = -2.1
H₂[1] = -1(0.95) + 0(-1.025) + 2(1.0) = -0.95 + 0 + 2.0 = 1.05
H₂[2] = 0(0.95) + (-2)(-1.025) + 1(1.0) = 0 + 2.05 + 1.0 = 3.05
```

**H₂ = [-2.1, 1.05, 3.05]ᵀ**

**Step 5: Softmax**

exp(H₂) = [e⁻²·¹, e¹·⁰⁵, e³·⁰⁵] = [0.1225, 2.8577, 21.1153]

Sum = 24.0955

**Ŷ = [0.00508, 0.11859, 0.87633]ᵀ**

#### Loss Computation

Y = 2 → one-hot encoding: [0, 0, 1]ᵀ

**L = -ln(ŷ₂) = -ln(0.87633) ≈ 0.132**

#### Backward Pass

Using the softmax-cross-entropy identity dL/dp = ŷ - y:

**∇H₂ = [0.00508, 0.11859, 0.87633]ᵀ - [0, 0, 1]ᵀ = [0.00508, 0.11859, -0.12367]ᵀ**

For H₂ = W₂H₁:

**∇H₁ = W₂ᵀ × ∇H₂**

```
∇H₁[0] = 1(0.00508) + (-1)(0.11859) + 0(-0.12367) = -0.11351
∇H₁[1] = 2(0.00508) + 0(0.11859) + (-2)(-0.12367) = 0.25750
∇H₁[2] = -1(0.00508) + 2(0.11859) + 1(-0.12367) = 0.11351
```

**∇H₁ = [-0.11351, 0.25750, 0.11351]ᵀ**

**∇W₂ = ∇H₂ × H₁ᵀ** (outer product)

```
∇W₂ = [[0.00483, -0.00521, 0.00508],
       [0.11266, -0.12156, 0.11859],
       [-0.11751, 0.12676, -0.12367]]
```

For H₁ = f(W₁ ⊛ X) + X:

The gradient ∇H₁ splits into two paths - through the skip connection and through the activation.

**Through Leaky ReLU:**

Leaky ReLU derivative: f'(x) = 1 if x ≥ 0, else 0.1

```
∇[W₁ ⊛ X][0] = -0.11351 × 0.1 = -0.01135  (input -0.5 < 0)
∇[W₁ ⊛ X][1] = 0.25750 × 0.1 = 0.02575   (input -0.25 < 0)
∇[W₁ ⊛ X][2] = 0.11351 × 1 = 0.11351     (input 0.5 ≥ 0)
```

**Backprop through convolution for ∇W₁:**

```
∇W₁[0] = -0.01135(0) + 0.02575(1.0) + 0.11351(-1.0) = -0.08776
∇W₁[1] = -0.01135(1.0) + 0.02575(-1.0) + 0.11351(0.5) = 0.01101
∇W₁[2] = -0.01135(-1.0) + 0.02575(0.5) + 0.11351(0) = 0.02423
```

**∇W₁ = [-0.08776, 0.01101, 0.02423]ᵀ**

**For ∇X:** (gradient from both skip connection and convolution)

Transposed convolution of ∇[W₁⊛X] with flipped kernel:

```
From convolution:
  ∇X_conv[0] = -0.01135(0) + 0.02575(-0.5) + 0.11351(0) = -0.01288
  ∇X_conv[1] = -0.01135(0.5) + 0.02575(0) + 0.11351(0.5) = 0.05108
  ∇X_conv[2] = -0.01135(-0.5) + 0.02575(0.5) + 0.11351(0) = 0.01855

From skip connection: ∇H₁ directly

Total: ∇X = ∇X_conv + ∇H₁
```

**∇X = [-0.12639, 0.30858, 0.13206]ᵀ**

#### Summary Table

| Ŷ | L | ∇H₂ | ∇H₁ |
|---|---|-----|-----|
| [0.00508, 0.11859, 0.87633]ᵀ | 0.132 | [0.00508, 0.11859, -0.12367]ᵀ | [-0.11351, 0.25750, 0.11351]ᵀ |

| ∇W₂ | ∇W₁ | ∇X |
|-----|-----|-----|
| [[0.00483, -0.00521, 0.00508], [0.11266, -0.12156, 0.11859], [-0.11751, 0.12676, -0.12367]] | [-0.08776, 0.01101, 0.02423]ᵀ | [-0.12639, 0.30858, 0.13206]ᵀ |

---

### Part (c): Analysis of Skip Connections

#### i. Gradient ∇X Without Skip Connection

Removing H₁ = f(W₁ ⊛ X) + X and making it H₁ = f(W₁ ⊛ X) only eliminates the direct gradient path. The gradient now flows solely through the convolution and activation layers:

**∇X (no skip) = [-0.01288, 0.05108, 0.01855]ᵀ**

#### ii. Comparison of Gradients

The original gradients with skip connection are **roughly 10 times larger** in magnitude:

- With skip: ∇X = [-0.12639, 0.30858, 0.13206]ᵀ
- Without skip: ∇X = [-0.01288, 0.05108, 0.01855]ᵀ

The skip connection preserves gradient flow by providing a highway that bypasses the convolution and activation. Without it, gradients must traverse the Leaky ReLU (which attenuates negative inputs by 90%) and the convolution operation, causing substantial gradient diminishment.

#### iii. Usefulness for Training

**The gradients from part (b) with skip connection are far superior for training.**

If X represented an intermediate output from a preceding network, the larger gradients translate to stronger learning signals propagating backward. This directly addresses the vanishing gradient problem that plagued early deep architectures - as gradients backpropagate through many layers of activations and transformations, they can diminish exponentially.

Skip connections revolutionized deep learning by allowing gradients to flow unimpeded through shortcuts. The direct path ensures even early layers receive meaningful updates rather than having their gradients fade to near-zero. This enabled the construction of networks with hundreds of layers (like ResNet-152), whereas previous architectures struggled beyond a few dozen layers due to gradient vanishing.

---

## Question 3: Extra Credit [2.5 points]

### Part (a): Determining Accuracy from Loss (10 Classes)

From Question 1, we established the loss ranges for 10 classes:

**Correct predictions:**
- Best case: L = 0 (perfect confidence)
- Worst case: L = ln(10) ≈ 2.303 (uniform distribution)
- Range: **[0, 2.303]**

**Incorrect predictions:**
- Best case: L = ln(2) ≈ 0.693 (near tie)
- Worst case: L → ∞ (zero confidence in truth)
- Range: **[0.693, ∞)**

**Determinable regions:**

- **L < 0.693:** Definitely correct (accuracy = 100%)
  Even the best incorrect prediction can't achieve loss below ln(2).

- **0.693 ≤ L ≤ 2.303:** Ambiguous
  Could be either correct (low confidence) or incorrect (high confidence in wrong answer).

- **L > 2.303:** Definitely incorrect (accuracy = 0%)
  Even the worst correct prediction (uniform) can't exceed ln(10).

We can safely deduce accuracy only when **L < ln(2) ≈ 0.693** (correct) or **L > ln(10) ≈ 2.303** (incorrect).

---

### Part (b): Number of Classes for Complete Inference

For any classification problem with n ≥ 2 classes:

- Correct prediction range: [0, ln(n)]
- Incorrect prediction range: [ln(2), ∞)

**Overlap region:** [ln(2), ln(n)]

For no overlap to exist, we would need ln(n) < ln(2), implying n < 2. Since meaningful classification requires at least 2 classes, and with 2 classes we have ln(2) = ln(2) (exact overlap at the boundary):

**No number of classes allows complete accuracy inference for all loss values.**

There will always be an ambiguous range [ln(2), ln(n)] where the loss alone cannot determine whether the prediction was correct. As the number of classes increases, this ambiguous region expands, making accuracy inference even more difficult.

---

### Part (c): Label Smoothing Analysis

With label smoothing: y_smooth = 0.9 × y + 0.01

For 10 classes:
- Correct class coefficient: 0.91
- Each incorrect class coefficient: 0.01

The loss becomes:
**L = -0.91·ln(ŷₖ) - Σⱼ≠ₖ 0.01·ln(ŷⱼ)**

Using the convention that a·ln(0) = 0 when a > 0:

**Correct predictions:**
- Best: ŷₖ = 1, others = 0 → L = -0.91·ln(1) - 9(0) = **0**
- Worst: uniform ŷ = 0.1 → L = -0.91·ln(0.1) - 0.09·ln(0.1) = **2.303**
- Range: **[0, 2.303]** (unchanged from standard encoding)

**Incorrect predictions:**
- Best: ŷₖ ≈ 0.5, ŷⱼ ≈ 0.5 (near tie) → L = -0.91·ln(0.5) - 0.01·ln(0.5) ≈ **0.638**
- Worst: ŷₖ → 0 → L → **∞**
- Range: **[0.638, ∞)**

**Determinable regions with label smoothing:**

- **L < 0.638:** Definitely correct
- **0.638 ≤ L ≤ 2.303:** Ambiguous
- **L > 2.303:** Definitely incorrect

**Comparison with part (a):**

Without smoothing: Determinable correct range was [0, 0.693)
With smoothing: Determinable correct range is [0, 0.638)

Label smoothing actually **reduces the determinable range** slightly. The ambiguous region expands because the best incorrect case now has higher loss (0.638 vs 0.693) due to the non-zero coefficients on all classes. While label smoothing improves model generalization during training by preventing overconfidence, it paradoxically makes accuracy inference from loss values slightly more difficult.

---

## End of Solutions
