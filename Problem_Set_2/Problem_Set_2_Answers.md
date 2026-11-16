# Problem Set 2

> **Handwritten vs Typed Guide:**
> The PDF allows inserting pictures of handwritten work for mathematical derivations and plots. Here's what should be handwritten vs typed:
>
> - **Q1 (a, b, c)** -- TYPE these. They are written analysis/discussion responses.
> - **Q2 (a)** -- HANDWRITE this. It's a by-hand batch-norm computation with matrices and fractions.
> - **Q2 (b)** -- HANDWRITE this. It's a by-hand distance vector computation with Euclidean norms.
> - **Q2 (c)** -- TYPE this. It's a written discussion (1-2 paragraphs).
> - **Q3 (a, b, c)** -- HANDWRITE these. They involve parameter counting arithmetic, layer-by-layer size derivations, and filling in the summary table.

## 1. Diagnosing Model Training [6 points]

### (a) Analysis of the two training curves

**Setting #1:** The model is not learning effectively. Both train and test loss decrease very slowly from ~2.3 (which is ln(10), consistent with random guessing on 10 classes). Train and test accuracy barely climb above ~10%, which is essentially random chance. The train and test curves overlap almost perfectly, which is a classic sign of underfitting. The model simply isn't making meaningful progress over 25 epochs.

**Setting #2:** The model is learning much better. Train loss drops quickly and train accuracy climbs to around 50-60%. However, the test loss and test accuracy curves are quite noisy and oscillate a lot. Around epoch 10-12, you can see the test loss starting to creep up while train loss keeps going down, which indicates the beginning of overfitting. So the model does learn, but the optimization is unstable.

### (b) Which hyperparameter was changed?

The most likely change is the **learning rate** -- it was increased from Setting #1 to Setting #2. In Setting #1, the learning rate is too small, so the model barely updates its weights over 25 epochs and stays near random performance. In Setting #2, a higher learning rate lets the model converge much faster, but the oscillations in test loss/accuracy are a telltale sign that the learning rate is a bit too high, causing the optimizer to overshoot minima repeatedly.

### (c) Has the model reached its best performance?

No, I don't think Setting #2 has reached its best performance. The noisy oscillations tell us the learning rate is slightly too aggressive, and the growing train-test gap suggests overfitting is starting. To improve:

- **Reduce the learning rate slightly** -- find a middle ground between Settings #1 and #2, or use a learning rate scheduler that starts at the Setting #2 rate and decays over time. This would smooth out those oscillations while still learning fast.
- **Add regularization** (like dropout or weight decay) to close the gap between train and test performance.

---

## 2. Normalization Layers [8 points]

### (a) Compute batch-norm output y
> *HANDWRITE this section -- show the mean, variance, and normalized matrix computations on paper.*

Shape is (2, 3, 3, 3), so N=2, H=3, W=3, NHW=18. With gamma=1 and beta=0, the formula simplifies to y = (x - E[x]) / sqrt(Var[x]).

**Channel 1:**

Values from x(1,1): {1,0,0,0,1,0,0,0,1} and x(2,1): {0,1,0,0,1,0,1,0,1}

Sum = 7, so E[x](1) = 7/18

Count of 0s: 11, count of 1s: 7

Var = (1/18)[11(7/18)^2 + 7(11/18)^2] = (11 x 49 + 7 x 121)/(18 x 324) = 1386/5832 = 77/324

sqrt(Var) = sqrt(77)/18

For x=0: y = -7/sqrt(77) ≈ -0.798
For x=1: y = 11/sqrt(77) ≈ 1.254

y(1,1,:,:) =

|  1.254 | -0.798 | -0.798 |
|--------|--------|--------|
| -0.798 |  1.254 | -0.798 |
| -0.798 | -0.798 |  1.254 |

y(2,1,:,:) =

| -0.798 |  1.254 | -0.798 |
|--------|--------|--------|
| -0.798 |  1.254 | -0.798 |
|  1.254 | -0.798 |  1.254 |

**Channel 2:**

Values from x(1,2): {0,2,0,0,1,0,2,1,0} and x(2,2): {0,0,2,0,0,2,1,1,1}

Sum = 13, so E[x](2) = 13/18

Count of 0s: 8, 1s: 5, 2s: 5

Var = (1/18)[8(13/18)^2 + 5(5/18)^2 + 5(23/18)^2] = (8 x 169 + 5 x 25 + 5 x 529)/(18 x 324) = 4122/5832 = 229/324

sqrt(Var) = sqrt(229)/18

For x=0: y = -13/sqrt(229) ≈ -0.859
For x=1: y = 5/sqrt(229) ≈ 0.330
For x=2: y = 23/sqrt(229) ≈ 1.520

y(1,2,:,:) =

| -0.859 |  1.520 | -0.859 |
|--------|--------|--------|
| -0.859 |  0.330 | -0.859 |
|  1.520 |  0.330 | -0.859 |

y(2,2,:,:) =

| -0.859 | -0.859 |  1.520 |
|--------|--------|--------|
| -0.859 | -0.859 |  1.520 |
|  0.330 |  0.330 |  0.330 |

**Channel 3:**

Values from x(1,3): {0,1,3,3,0,0,1,1,2} and x(2,3): {0,3,1,1,0,0,0,3,0}

Sum = 19, so E[x](3) = 19/18

Count of 0s: 8, 1s: 5, 2s: 1, 3s: 4

Var = (1/18)[8(19/18)^2 + 5(1/18)^2 + 1(17/18)^2 + 4(35/18)^2] = (8 x 361 + 5 x 1 + 289 + 4 x 1225)/(18 x 324) = 8082/5832 = 449/324

sqrt(Var) = sqrt(449)/18

For x=0: y = -19/sqrt(449) ≈ -0.897
For x=1: y = -1/sqrt(449) ≈ -0.047
For x=2: y = 17/sqrt(449) ≈ 0.802
For x=3: y = 35/sqrt(449) ≈ 1.652

y(1,3,:,:) =

| -0.897 | -0.047 |  1.652 |
|--------|--------|--------|
|  1.652 | -0.897 | -0.897 |
| -0.047 | -0.047 |  0.802 |

y(2,3,:,:) =

| -0.897 |  1.652 | -0.047 |
|--------|--------|--------|
| -0.047 | -0.897 | -0.897 |
| -0.897 |  1.652 | -0.897 |

---

### (b) Channel-wise distance vectors
> *HANDWRITE this section -- show the element-wise differences and Euclidean norm calculations on paper.*

**For x (original):**

Channel 1 diff: x(1,1)-x(2,1) = [[1,-1,0],[0,0,0],[-1,0,0]], ||d|| = sqrt(1+1+1) = **sqrt(3)**

Channel 2 diff: x(1,2)-x(2,2) = [[0,2,-2],[0,1,-2],[1,0,-1]], ||d|| = sqrt(4+4+1+4+1+1) = **sqrt(15)**

Channel 3 diff: x(1,3)-x(2,3) = [[0,-2,2],[2,0,0],[1,-2,2]], ||d|| = sqrt(4+4+4+1+4+4) = **sqrt(21)**

**d_x = [sqrt(3), sqrt(15), sqrt(21)] ≈ [1.732, 3.873, 4.583]**

**For x_act = LeakyReLU(x):**

All values in x are 0, 1, 2, or 3 -- all non-negative. So LeakyReLU doesn't change any of them.

**d_xact = [sqrt(3), sqrt(15), sqrt(21)] -- same as d_x**

**For y_act = LeakyReLU(y):**

Batch-norm created negative values. LeakyReLU scales negatives by 0.1, so negative y values get multiplied by 0.1. Working through each channel:

Channel 1: After LeakyReLU, -7/sqrt(77) becomes -0.7/sqrt(77). Computing the element-wise differences and Euclidean distance gives ||d|| = 11.7 x sqrt(3)/sqrt(77) ≈ **2.309**

Channel 2: Similarly, -13/sqrt(229) becomes -1.3/sqrt(229). Working through gives ||d|| ≈ **3.082**

Channel 3: Negative values (-19/sqrt(449) and -1/sqrt(449)) get scaled by 0.1. Working through gives ||d|| ≈ **3.432**

**d_yact ≈ [2.309, 3.082, 3.432]**

---

### (c) Discussion
> *TYPE this section -- it's a written paragraph response.*

**(i)** Only y_act produced a different channel-wise distance vector compared to x, while x_act produced the same vector as x. This is because all original input values are non-negative (0, 1, 2, 3), so LeakyReLU acts as the identity on all of them. However, batch-norm subtracts the channel mean, which creates negative values. When LeakyReLU is then applied to these centered values, it scales the negative parts by 0.1, which is a nonlinear distortion that changes the relative spacing between samples. This is why d_yact differs from d_x.

**(ii)** Yes, it is possible to have y_act = x_act. If we set gamma = sqrt(Var[x]) and beta = E[x] for each channel, then batch-norm becomes y = sqrt(Var) x (x - E[x])/sqrt(Var) + E[x] = x. In that case, y = x, so LeakyReLU(y) = LeakyReLU(x), meaning y_act = x_act. Since gamma and beta are learnable, the network can learn these values.

**(iii)** Including batch-norm strictly increases the model's expressive power because the model with batch-norm can represent everything the model without it can, plus more. As shown above, batch-norm can always learn to be the identity (just recover the original input), so it never reduces what the model can express. But beyond that, gamma and beta give the network independent control over the scale and shift of each channel's activations, separate from what the preceding layer's weights and biases produce. Without batch-norm, the distribution of features is entirely determined by the learned weights. With batch-norm, the network gets to first normalize features to a standard distribution and then re-scale/shift them optimally. This extra degree of freedom allows it to represent functions it couldn't before.

---

## 3. Computing Model Sizes [11 points]

> *HANDWRITE all three model computations (a, b, c) below -- show the layer-by-layer parameter counts, feature sizes, and memory arithmetic on paper. Include the final summary table.*

### Model (a): Fully Connected

- Flatten input: 3 x 64 x 64 = 12,288
- Hidden layer: 2 x 12,288 = 24,576 neurons
- Output: 100

**Parameters:**
- FC1: 12,288 x 24,576 + 24,576 = 302,014,464
- FC2: 24,576 x 100 + 100 = 2,457,700
- **Total: 304,472,164**

**Feature cost:**
- Input: 12,288
- Hidden output: 24,576
- Final output: 100
- **Total: 36,964**

**Memory cost:** (304,472,164 + 36,964) x 4 = **1,218,036,512 bytes (≈ 1.218 GB)**

---

### Model (b): Recurrent

- Patches: 64/4 = 16 per dimension, 16 x 16 = 256 patches
- Each patch flattened: 3 x 4 x 4 = 48
- Sequence length T = 256, input dim = 48, hidden dim = 512
- Concatenated output: 256 x 512 = 131,072

**Parameters:**
- RNN: W_xh (48 x 512) + W_hh (512 x 512) + b (512) = 24,576 + 262,144 + 512 = 287,232
- FC: 131,072 x 100 + 100 = 13,107,300
- **Total: 13,394,532**

**Feature cost:**
- Input patches: 256 x 48 = 12,288
- All hidden states: 256 x 512 = 131,072
- FC output: 100
- **Total: 143,460**

**Memory cost:** (13,394,532 + 143,460) x 4 = **54,151,968 bytes (≈ 54.15 MB)**

---

### Model (c): Convolutional

7 conv+BN layers until spatial dimension reaches 8x8:

| Layer | Channels | Spatial | Output Shape |
|-------|----------|---------|--------------|
| Conv1 | 3 -> 6   | 64 -> 56 | 6 x 56 x 56 |
| Conv2 | 6 -> 12  | 56 -> 48 | 12 x 48 x 48 |
| Conv3 | 12 -> 24 | 48 -> 40 | 24 x 40 x 40 |
| Conv4 | 24 -> 48 | 40 -> 32 | 48 x 32 x 32 |
| Conv5 | 48 -> 96 | 32 -> 24 | 96 x 24 x 24 |
| Conv6 | 96 -> 192 | 24 -> 16 | 192 x 16 x 16 |
| Conv7 | 192 -> 384 | 16 -> 8 | 384 x 8 x 8 |

Then flatten (384 x 8 x 8 = 24,576) and FC to 100.

**Parameters:**

Conv layers (C_out x C_in x 9 x 9 + C_out):
1,464 + 5,844 + 23,352 + 93,360 + 373,344 + 1,493,184 + 5,972,352 = 7,962,900

BN layers (2 x C for gamma and beta):
12 + 24 + 48 + 96 + 192 + 384 + 768 = 1,524

FC: 24,576 x 100 + 100 = 2,457,700

**Total params: 10,422,124**

**Feature cost** (storing input + output of each conv and BN + FC output):
12,288 + 18,816 + 18,816 + 27,648 + 27,648 + 38,400 + 38,400 + 49,152 + 49,152 + 55,296 + 55,296 + 49,152 + 49,152 + 24,576 + 24,576 + 100 = **538,468**

**Memory cost:** (10,422,124 + 538,468) x 4 = **43,842,368 bytes (≈ 43.84 MB)**

---

### Summary Table

|           | # Params      | Feature Cost | Memory Cost  |
|-----------|---------------|--------------|--------------|
| Model (a) | 304,472,164   | 36,964       | ≈ 1.218 GB   |
| Model (b) | 13,394,532    | 143,460      | ≈ 54.15 MB   |
| Model (c) | 10,422,124    | 538,468      | ≈ 43.84 MB   |

The fully connected model is absurdly expensive because of the huge weight matrix in the first layer. The convolutional model uses the fewest total bytes despite having more intermediate features, thanks to weight sharing in the conv filters.
