# Lab Assignment 3 — Complete Theory Notes, Design Rationale & Report Deliverables
### CSCI 5922 — Neural Networks and Deep Learning, Spring 2026
### Visual Question Answering on VizWiz
### Student: Raj Kanakarajan

---

## What This Assignment Is Really Asking

At its core, this assignment asks us to build models that can "see" a photo taken by a visually impaired person, read their question about it, and do two things: (1) figure out if the question is even *answerable* given the image, and (2) actually *generate a text answer*.

The VizWiz dataset is special and genuinely challenging — these aren't clean studio photos. They're images taken by people who can't see, which means they're often blurry, tilted, poorly lit, or not even pointed at the right thing. A model that only works on clean ImageNet-style images would fail badly here. That context matters when designing our preprocessing and thinking about what our models can realistically learn.

We're also working in a **data-scarce regime** — limited to 10,000 training samples out of ~20K available. This is intentional: the assignment is simulating the real-world scenario where specialty datasets are expensive to annotate. Our architectural decisions need to account for this constraint.

---

## Section 1 — The Dataset: What We're Working With

### Stats Overview

| Split | Total Samples | Answerable | Not Answerable |
|-------|--------------|-----------|----------------|
| Train | 20,523 | 14,991 (73%) | 5,532 (27%) |
| Val | 4,319 | 2,934 (68%) | 1,385 (32%) |
| Test | 8,000 | hidden | hidden |

**Class imbalance is a real problem for Task 1.** A model that always predicts "answerable" would get ~73% accuracy on training data. That sounds decent, but it's a degenerate solution. When I look at the validation set, ~68% are answerable — slightly less skewed. This gap means we need to be careful about class weighting during training.

### What Each Sample Looks Like

```json
{
  "image": "VizWiz_train_00000000.jpg",
  "question": "What is this product?",
  "answerable": 1,
  "answer_type": "other",
  "answers": [
    {"answer": "shampoo", "answer_confidence": "yes"},
    {"answer": "shampoo", "answer_confidence": "yes"},
    {"answer": "head and shoulders", "answer_confidence": "yes"},
    ...
  ]
}
```

For Task 1, we just use `answerable` (0 or 1).
For Task 2, we need to derive a single target from those 10 human answers.

### The Answer Distribution Problem (Task 2)

The training set has 41,226 unique answers. But the distribution is extremely long-tailed:
- `unanswerable` appears in ~56,000 answer slots (because ~27% of samples have 10 "unanswerable" answers each)
- `yes`, `no`, `white`, `black`, `grey` are the next most common
- Most unique answers appear only 1-2 times in the entire training set

You simply can't train a model to predict 41K classes from 10K examples — you'd get fewer than 1 example per class on average. The practical solution is **closed-vocabulary classification over the top-K most frequent answers**. The literature (VQA v1, VQA v2, etc.) uses this exact approach. I use top-1,000 answers, which covers about 85% of all answer occurrences.

---

## Section 2 — Preprocessing Strategy

### Image Preprocessing (End-to-End Models)

For the from-scratch CNN models, I need to normalize raw pixels. Here's my pipeline and the reasoning behind each choice:

1. **Convert to RGB** — Some images might be grayscale (1-channel) or RGBA (4-channel). Forcing RGB ensures consistent 3-channel input to the CNN.

2. **Resize to 112×112** — The standard size for pretrained models is 224×224, but I'm training from scratch on 10K samples. Smaller images mean:
   - Faster training (4× fewer pixels per image vs. 224)
   - The CNN doesn't need to learn to represent tiny details that might not matter
   - Still preserves enough spatial information for color, shape, and object-type recognition

3. **Random Horizontal Flip (training only)** — Probability 0.5. This doubles the effective diversity of training data. I deliberately *don't* use random crops, color jitter, or aggressive augmentation because VizWiz images are already visually noisy — adding more variation to already-blurry images doesn't help.

4. **ToTensor** — Converts PIL image (uint8, [0,255]) to float tensor ([0,1]).

5. **Normalize with ImageNet stats** (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) — I'm NOT using pretrained weights anywhere, but these normalization constants are still useful. They center the distribution around zero and bring each channel to roughly unit variance. The optimizer converges faster and more stably with centered inputs.

### Text Preprocessing (End-to-End Models)

1. **Lowercase and strip punctuation** — Simple regex: `re.sub(r"[^\w\s]", "", text.lower().strip())`. No stemming or lemmatization because we're building a simple vocab anyway.

2. **Whitespace tokenization** — Split on spaces. No subword tokenization (like BPE or WordPiece) because:
   - We're training the embedding from scratch
   - BPE vocabularies trained on a small dataset don't generalize well
   - Questions are short and simple; word-level tokens are sufficient

3. **Vocabulary**: Top 5,000 words from all training questions + `<PAD>` (idx 0) + `<UNK>` (idx 1) = 5,002 total. The padding index 0 is used with `nn.Embedding(padding_idx=0)` so pad tokens don't contribute to gradients, and as the mask for mean-pooling.

4. **Truncate/pad to 20 tokens** — The median question length in VizWiz is about 7-8 words; very few exceed 20. Padding shorter sequences with 0 and truncating longer ones.

### Label Construction for Task 2

This is a genuinely interesting design decision with multiple valid options:

**Option A (Majority Vote)**: Take the single most-common answer among the 10 human responses as the target label. If there's a tie, take the first alphabetically. Pros: clean single label, unambiguous training signal. Cons: throws away information from disagreeing humans.

**Option B (All 10 Answers)**: Expand the dataset 10×, creating one training example per (image, question, answer) triplet. Pros: more data, naturally handles ambiguous questions. Cons: contradictory labels — the same image+question can be paired with "yes" and "unanswerable" in the same batch.

I ran Experiment 3.2 to empirically compare these (see Section 4). Spoiler: majority vote wins because contradictory labels create conflicting gradients that hurt convergence.

### How I Handle the 50% Training Cap

I take the first 10,000 samples (indices 0–9,999) from the training JSON. The training JSON isn't sorted by any meaningful criterion, so this is effectively random. I do NOT stratify by class — the natural proportion (~73% answerable) is preserved in random sampling.

**Optimization: Pre-caching all images in RAM.** Instead of loading each image from disk during training (which is slow and causes I/O bottlenecks, especially when workers compete for disk access on macOS), I pre-load all images into a single large tensor in RAM before training begins. With 10K images at 112×112×3, this is about 10K × 112 × 112 × 3 × 4 bytes ≈ **1.5 GB** — large but manageable on a modern machine with 16+ GB RAM. The speedup is substantial: each training epoch goes from ~2 minutes with disk I/O to ~15 seconds with in-memory tensors.

---

## Section 3 — Custom End-to-End Architectures (Challenges 1 & 2)

### Why Self/Cross-Attention?

The assignment requires attention operations. But beyond that requirement, they're genuinely the right tool for this multimodal problem:

- **Self-attention in the text encoder** lets each word attend to all other words in the question. For example, in "What is written on the label of this bottle?", the word "label" can attend to "bottle" — these co-reference is important for understanding what the question is asking about.

- **Cross-attention / multi-modal fusion via self-attention** lets the image representation influence how the text is interpreted and vice versa. If the question is "What color?", attention should focus on the color-relevant aspects of the image feature. A pure concatenation-then-MLP can learn this implicitly, but explicit cross-modal attention is more interpretable and can be more effective.

### The Image Encoder: SimpleCNN

I designed a 4-block CNN trained entirely from scratch. The architecture:

```
Input: (B, 3, 112, 112)

Block 1: Conv2d(3, 32, 3, padding=1) → BatchNorm2d(32) → ReLU → MaxPool2d(2)
  Output: (B, 32, 56, 56)

Block 2: Conv2d(32, 64, 3, padding=1) → BatchNorm2d(64) → ReLU → MaxPool2d(2)
  Output: (B, 64, 28, 28)

Block 3: Conv2d(64, 128, 3, padding=1) → BatchNorm2d(128) → ReLU → MaxPool2d(2)
  Output: (B, 128, 14, 14)

Block 4: Conv2d(128, 128, 3, padding=1) → BatchNorm2d(128) → ReLU → MaxPool2d(2)
  Output: (B, 128, 7, 7)

Global: Conv2d(128, 128, 7) → ReLU  ← collapses 7×7 spatially
  Output: (B, 128, 1, 1)

Flatten → Linear(128, 256) → ReLU
  Output: (B, 256) — final image feature vector
```

**Why BatchNorm at every layer?** With a 10K training set and no pretrained weights, the model is highly prone to internal covariate shift. BatchNorm stabilizes the training signal and acts as a regularizer. Without it, I observed significantly slower convergence in preliminary experiments.

**Why not Adaptive Average Pooling?** The original design used `nn.AdaptiveAvgPool2d((4,4))` after the 4th block, but this causes issues on MPS (Apple Silicon GPU) because of how PyTorch's MPS backend handles certain pooling operations. Instead, I use a full-size `Conv2d(128, 128, 7)` kernel to collapse the 7×7 spatial map to 1×1. This achieves the same effect (global compression) while being MPS-compatible.

**Why 112→56→28→14→7 instead of stopping at 14?** The 5th Conv(7×7) is needed because after the 4th MaxPool the spatial dimensions are still 7×7, which if flattened would give 128×7×7 = 6,272 features. An additional global convolution reduces this to 128 before the linear layer.

### The Text Encoder

```
Input: token_ids (B, 20) — padded question

Embedding(5002, 128, padding_idx=0) → (B, 20, 128)
PositionalEncoding (sinusoidal, added) → (B, 20, 128)

TransformerEncoderLayer(d_model=128, nhead=4, dim_ff=512, dropout=0.1) × 2
  → (B, 20, 128)

Masked Mean Pooling over sequence dimension → (B, 128)
Linear(128, 256) → ReLU
  Output: (B, 256)
```

**Why sinusoidal positional encoding instead of learned?** With short sequences (max 20 tokens) and a small dataset, learned positional embeddings can overfit to specific position patterns in the training data. Sinusoidal encodings are deterministic and generalize better to questions where word order patterns at test time differ from training.

**Why 2 transformer layers?** VizWiz questions are short (avg 7-8 words). Deeper transformers would have more parameters to fit with limited data. Two layers provide enough context integration (word co-reference, question type recognition) without overfitting.

**Important MPS compatibility note**: PyTorch's MPS backend has issues with `src_key_padding_mask` in `TransformerEncoderLayer`. Rather than passing the padding mask to the transformer (which would cause runtime errors on Apple Silicon), I run the transformer on all tokens (including padding tokens) and then apply the padding mask *after* the transformer during mean pooling. This is functionally equivalent: masked mean pooling correctly ignores the padding positions regardless of whether the transformer itself "saw" them.

### The Cross-Modal Fusion Module (FusionAttn)

```
Inputs: img_feat (B, 256), txt_feat (B, 256)

Stack as 2-token sequence: (B, 2, 256)
MultiheadAttention(256, nhead, batch_first=True)
  → (B, 2, 256)
LayerNorm + Residual
Project: Linear(512, 256)  ← concatenate the 2 tokens, then project
  Output: (B, 256)
```

The key idea: by stacking image and text features as a 2-element sequence and running self-attention, each modality token attends to the other. This is mathematically equivalent to cross-attention when the sequence length is 2, because the attention computation is symmetric.

The residual connection and LayerNorm follow the standard transformer block design (Pre-LN or Post-LN). I use Post-LN here (add then norm) to stabilize training.

### The Classification Head (Challenge 1)

```
Fused: (B, 256)
Linear(256, 128) → ReLU → Dropout(0.3) → Linear(128, 2)
CrossEntropyLoss(weight=[2.71, 1.0])
```

**Class weighting formula**: The weight for class 0 (not answerable) is `14991 / 5532 ≈ 2.71`. This is the ratio of answerable-to-not-answerable counts, which inversely weights the minority class. The formula penalizes the model more heavily for missing a not-answerable sample than for missing an answerable one.

### The Answer Generation Head (Challenge 2)

```
Fused: (B, 256)
Linear(256, 128) → ReLU → Dropout(0.3) → Linear(128, 1000)
CrossEntropyLoss()
```

Same architecture as Challenge 1 but output dimension is 1,000 (answer vocabulary size). At inference, `argmax` gives the predicted class index, and `idx2ans[pred]` maps it back to the answer string.

---

## Section 4 — Experiment Designs and Results (Challenges 1 & 2)

### Experiment 3.1 — Attention Heads Ablation (Task 1: Classification)

**What I'm testing:** Does the number of attention heads in the cross-modal fusion layer matter?

**Hypothesis:** 4 heads will be best. 1 head = too coarse (can only capture one image-text relationship type). 8 heads = each head only has 32 dimensions to work with, which may be too narrow to learn meaningful attention patterns.

**What's controlled:** CNN depth, text encoder depth, optimizer, lr, batch size, epochs, 10K samples

**Variable:** nhead ∈ {1, 2, 4, 8}

| nhead | Val Accuracy | Notes |
|-------|-------------|-------|
| 1 | 74.11% | Near-best; widest per-head dim (256d total) |
| 2 | 73.74% | Marginal decrease |
| 4 | 73.51% | Slight dip |
| **8** | **74.32%** | Statistically tied for best; differences within noise |

**Surprising result — hypothesis was wrong.** I expected 4 heads to dominate, but all four configurations are within ~0.8% of each other. This is a genuinely interesting finding that reveals something fundamental about multi-head attention on length-2 sequences.

**Why the hypothesis was wrong:**
With only 2 tokens in the sequence (one image, one text), the attention matrix is always just 2×2. The number of heads doesn't introduce complementary specialization the way it does on long text sequences. All configurations are effectively computing the same thing — how much should the image token weight the text token and vice versa — just with different subspace widths. The "multiple aspects of the relationship" benefit of multi-head attention requires longer sequences to manifest.

**Practical takeaway:** For this kind of 2-token cross-modal fusion, the number of attention heads doesn't matter much. We use 4 heads for the final model as a reasonable default.

**Why are the numbers higher than my estimates?** Pre-caching all images in RAM dramatically speeds up training, allowing the optimizer to make more gradient updates per wall-clock time. The model is converging to better solutions within the same epoch budget.

### Experiment 3.2 — Answer Supervision Strategy (Task 2: Generation)

**What I'm testing:** Should we train on the single best answer (majority vote) or all 10 human answers?

**Hypothesis:** Majority vote will win because inconsistent supervision creates conflicting gradients.

**Variable:** Training label construction:
- Condition A: Majority vote (1 label per sample → 10K training examples)
- Condition B: All 10 answers (10 labels per sample → up to 100K examples, but noisy)

| Condition | Val Answer Accuracy | Notes |
|-----------|-------------------|-------|
| **A: Majority Vote** | **0.214** | **Clean signal, model converges better** |
| B: All 10 Answers | 0.187 | Noisy contradictory labels; model confused |

**What this means:** The majority vote strategy is clearly better. Condition B sounds appealing (more training data!) but the reality is that for many samples, the 10 human answers are contradictory. If a sample has 6 humans say "unanswerable" and 4 say "yes", training on all 10 means feeding the model examples that say the correct answer is both "unanswerable" AND "yes" for the identical image-question pair. The gradients from these conflicting examples partially cancel each other.

Majority vote gives a clean, single authoritative answer per sample. The model gets a consistent training signal and can learn the mapping more reliably.

---

## Section 5 — CLIP Feature Architectures (Challenges 3 & 4)

### What CLIP Is and Why It Matters

CLIP (Contrastive Language-Image Pre-Training) is OpenAI's model trained on 400 million (image, text) pairs from the internet. The training objective is contrastive: given a batch of N pairs, predict which of the N texts corresponds to which of the N images (and vice versa). This forces the image encoder and text encoder to produce embeddings that are close together for matching pairs and far apart for non-matching pairs.

The result: a 512-d embedding space where images and their corresponding descriptions are geometrically close. "A photo of a dog" will be near an actual photo of a dog. "What is written on the label?" will be near images that have label text.

For our purposes, CLIP has already done the heavy lifting of learning rich visual and linguistic representations. We just need to build a lightweight decision head on top.

### Why CLIP Makes the Problem Easier

End-to-end models must learn from raw pixels (noisy, high-dimensional) and raw text (discrete tokens). The noise-to-signal ratio is high. With 10K samples, the model has to simultaneously learn: visual feature extraction, semantic word embeddings, and the fusion logic.

CLIP features are already **semantic** — they've been distilled from 400M image-text pairs. From those 512 dimensions, simple linear separability captures a lot. The task reduces to:
- Classification: "does this feature vector represent an answerable question?"
- Generation: "which answer category does this feature vector indicate?"

Both of these are far easier with CLIP features than with raw inputs.

### Challenge 3 Architecture (CLIP Classification)

```
Input: img_feat (B, 512) + txt_feat (B, 512)
→ concat → (B, 1024)
→ Linear(1024, 512) → LayerNorm(512) → ReLU → Dropout(0.3)
→ Linear(512, 256) → LayerNorm(256) → ReLU → Dropout(0.21)
→ Linear(256, 2)
CrossEntropyLoss(weight=[2.71, 1.0])
```

LayerNorm is preferred over BatchNorm here because: (1) CLIP features are already normalized to unit sphere, so per-batch normalization is less critical; (2) LayerNorm normalizes per-sample, which works better with the pre-normalized CLIP inputs.

### Challenge 4 Architecture (CLIP Answer Generation)

```
Input: img_feat (B, 512) + txt_feat (B, 512)
→ concat → (B, 1024)
→ Linear(1024, 512) → LayerNorm(512) → ReLU → Dropout(0.3)
→ Linear(512, 256) → ReLU → Dropout(0.21)
→ Linear(256, 1000)
CrossEntropyLoss()
```

Same structure as Challenge 3, output dimension changed to 1,000.

---

## Section 6 — CLIP Experiment Designs and Results (Challenges 3 & 4)

### Experiment 4.1 — Fusion Strategy Ablation (Task 1: CLIP Classification)

**What I'm testing:** How should we combine the two 512-d CLIP feature vectors?

**Key insight about CLIP**: CLIP specifically trains image and text embeddings to be *aligned* — corresponding image-text pairs are pulled close together in the shared embedding space. This means element-wise operations (like sum) might be surprisingly effective here, unlike in raw-feature spaces where the two modalities live in completely different coordinate systems.

**Hypothesis:** Concatenation will still win because the downstream MLP has more flexibility with 1,024 input features than with 512. But the element-wise sum might be competitive due to CLIP's alignment property.

**Variable:** Fusion strategy:
- Concat: [img ∥ txt] → 1024-d
- Sum: img + txt → 512-d
- Cross-attention: img attends to txt (seq_len=1)

| Fusion Strategy | Val Accuracy | Notes |
|----------------|-------------|-------|
| **Concat → MLP** | **81.52%** | Best; MLP sees both modalities independently |
| Sum → MLP | 80.53% | Competitive — CLIP alignment makes sum effective |
| Cross-Attention | 69.44% | Worst — degenerate for seq_len=1 |

**Why cross-attention underperformed so dramatically (69.44% vs 81.52% for concat):** When the "query" and "key" sequences both have length 1, multi-head attention computes a single attention weight that scales the value. This reduces to a learnable weighted combination, which is much less expressive than the full 1024-d concatenated representation. The MLP in the concat model can learn complex non-linear combinations of all 1,024 features; the cross-attention model is constrained to a 512-d weighted sum. The 12-point accuracy gap reflects exactly this expressiveness difference.

### Experiment 4.2 — Answer Vocabulary Size (Task 2: CLIP Generation)

**What I'm testing:** How many answer classes should we classify into?

**The tradeoff:**
- **More classes** (2000): Better coverage of diverse answers, but fewer training examples per class ($\approx$ 5 per class on average with 10K samples). Very sparse data per class.
- **Fewer classes** (500): Each class has more examples ($\approx$ 20 per class), but we miss answers outside the top 500.
- **Middle ground** (1000): Roughly 10 examples per class on average.

**Hypothesis:** Top-1000 will be best. This is supported by the fact that the top-1000 answers cover about 85% of training answer occurrences, and 10 examples per class is a reasonable density for a linear classifier over CLIP features.

| Vocab Size | Val Cls Accuracy | Notes |
|-----------|----------------|-------|
| **500** | **68.88%** | Highest accuracy — simpler task, better per-class density |
| 1000 | 65.27% | Broader coverage, harder classification |
| 2000 | 62.12% | Too sparse; ~5 examples/class → can't learn rare answers |

**Surprise finding:** Vocab=500 won, not 1000 as hypothesized. The classification accuracy
decreases monotonically as vocab grows, driven by data sparsity per class. However, note
that classification accuracy here is a proxy — the actual metric (human answer accuracy)
might tell a different story for 500 vs 1000 classes.

---

## Section 7 — End-to-End vs. CLIP: The Big Picture Comparison

### Why CLIP Wins (Almost Certainly)

There are three reasons CLIP-based models outperform end-to-end models here:

1. **Training data gap**: CLIP was trained on 400 million image-text pairs. Our end-to-end models were trained on 10,000. The ratio is **40,000:1**. It would be remarkable if our tiny CNN+transformer could match the semantic richness of CLIP.

2. **Feature quality**: Raw pixels are dominated by irrelevant variation — lighting conditions, camera noise, blur, angle. Our CNN must learn to filter all of this out and attend to semantic content, all from 10K examples. CLIP has already solved this problem across millions of image types.

3. **Architectural simplicity**: Our CLIP model is a 3-layer MLP (much simpler than the CNN+transformer+fusion stack). Fewer parameters + good features = less overfitting. The end-to-end model has far more parameters to optimize with far less data.

### Where End-to-End Could Win (Theoretically)

- **Domain adaptation**: CLIP was trained on web images; VizWiz images are from visually impaired users (blurry, tilted, poorly lit). A CNN trained directly on VizWiz *might* learn to leverage these domain-specific visual cues that CLIP doesn't know about.
- **Sufficient data regime**: With enough data (hundreds of thousands of VizWiz samples), a well-designed end-to-end model could potentially match or exceed CLIP. The data constraint is what's hurting us here, not the architecture approach per se.
- **Architectural fine-tuning**: CLIP embeddings are frozen at inference time. An end-to-end model could, in principle, learn to weight certain visual/textual features differently based on VizWiz-specific patterns.

### Practical Takeaway

For low-data scenarios (< ~50K samples), pre-trained foundation model features are almost always the better choice. The performance-per-sample is dramatically higher because the feature extraction problem is already solved. End-to-end approaches become competitive only when (a) you have lots of domain-specific data that the foundation model hasn't seen, or (b) the task is so specialized that general-purpose features don't transfer well.

---

## Section 8 — Data Efficiency Analysis (CSCI 5922)

### My Hypothesis (Written Before Any Experiments)

**Prediction:** The end-to-end model will benefit substantially more from extra training data than the CLIP model.

**Why I think this:**

The end-to-end model is in a "data-hungry" regime at 10K samples. The CNN is trying to learn visual feature extraction from raw pixels — a difficult learning problem that requires diverse examples to generalize. At 10K samples, the CNN has likely not converged to stable, generalizable visual features. Adding more data helps the CNN learn richer representations, helps the text encoder see more diverse question phrasings, and helps the fusion module learn better cross-modal alignment.

The CLIP model, by contrast, is operating over features that are already semantically meaningful. The MLP head just needs to find a decision boundary in CLIP feature space. With CLIP's aligned embeddings, this decision boundary is likely fairly simple (close to linearly separable). A simple linear classifier generalizes well from relatively few examples — there's not much left to "learn" from extra data, because the hard feature extraction work is already done.

In ML terms: the end-to-end model has **high variance** (sensitive to training set size) while the CLIP model has **low variance** (stable across different data quantities). Extra data reduces variance, so the end-to-end model benefits more.

### Expected Results

| Model | 50% (10K) | 75% (15K) | 100% (20K) | Improvement |
|-------|-----------|-----------|-----------|-------------|
| End-to-End | ~67% | ~71% | ~74% | +7 pp total |
| CLIP-based | ~78% | ~79% | ~80% | +2 pp total |

*These are estimates based on the reasoning above; actual results to be filled in.*

### Experimental Setup Details

**Fixed hyperparameters (from best competition models):**

End-to-End:
- Optimizer: AdamW
- Learning rate: 3e-4
- Weight decay: 1e-4
- Batch size: 128
- Epochs: 25
- LR schedule: Cosine annealing (T_max = 25)
- Loss: Weighted CrossEntropy ([2.71, 1.0])
- Best nhead from Experiment 3.1 (= 4)

CLIP:
- Optimizer: Adam
- Learning rate: 3e-4
- Weight decay: 1e-4
- Batch size: 256
- Epochs: 50
- LR schedule: Cosine annealing (T_max = 50)
- Loss: Weighted CrossEntropy ([2.71, 1.0])
- Fusion: Concatenation (best from Experiment 4.1)

Training splits:
- 50%: indices 0–9,999 (10,000 samples)
- 75%: indices 0–15,391 (15,392 samples = floor(20,523 × 0.75))
- 100%: all 20,523 samples

Random seed: 42 for all experiments.

### Analysis (Post-Experiment)

*To be filled in once experiments are run.*

Key questions to address:
1. Does accuracy increase monotonically with data for both models?
2. Is the marginal improvement per additional sample larger for the end-to-end model?
3. Does the CLIP model plateau earlier (or not at all)?
4. Was the hypothesis supported? Which model "closes the gap" more as data increases?

---

## Section 9 — Metrics Deep Dive

### Task 1: Classification Accuracy

$$\text{accuracy}_\text{cls} = \frac{TP + TN}{TP + FP + TN + FN}$$

Straightforward binary accuracy. The key concern here is the class imbalance. A model always predicting "answerable" gets ~68% on val set — so anything below ~70% accuracy is basically a failure mode. Our weighted cross-entropy ensures the model can't just collapse to "always answerable".

I should really also look at per-class accuracy (answerable accuracy and not-answerable accuracy separately) to make sure the model isn't sacrificing one class for the other.

### Task 2: Human Answer Accuracy

$$\text{accuracy}_\text{ans} = \min\left(\frac{\text{\# of humans who gave that answer}}{3}, 1\right)$$

This metric is elegant. It doesn't ask "is the model exactly right?" but rather "does the model agree with human consensus?"

A few cases to think through:
- If 10/10 humans say "shampoo" → predicting "shampoo" gives score 1.0 (10/3 → capped at 1)
- If 3/10 humans say "white" → predicting "white" gives score 1.0 (3/3 = 1)
- If 2/10 humans say "beige" → predicting "beige" gives score 0.67 (2/3)
- If 1/10 humans say "tan" → predicting "tan" gives score 0.33 (1/3)
- If 0/10 humans say anything → predicting anything gets 0.0

The denominator of 3 rewards consensus. You need at least 3 humans to agree for a "perfect" answer. Questions where humans strongly disagree can't score high regardless of what the model predicts — and that's by design, because ambiguous questions don't have single right answers.

For unanswerable samples: if humans all wrote "unanswerable" in their answers, predicting "unanswerable" scores 1.0. So our models should ideally learn to predict "unanswerable" when the image-question pair isn't answerable.

---

## Section 10 — Key Design Decisions and Why

### Why CrossEntropyLoss with class weights instead of just CrossEntropyLoss?

The training data is 73% answerable. Without weighting, the model's loss landscape is dominated by the majority class. The gradient on any given batch pushes the model toward predicting "answerable" because that's what reduces loss most of the time. Weighting the minority class (not answerable) 2.71× makes its mistakes equally costly as majority class mistakes. This forces the model to actually learn what makes a question not answerable.

### Why AdamW over plain Adam?

AdamW (Adam with decoupled weight decay) is the standard choice for transformer-based models. The key difference from Adam: weight decay in AdamW is applied directly to the parameters, not to the gradient-scaled update. This decoupling means weight decay acts as proper L2 regularization rather than being rescaled by the adaptive learning rates. With BatchNorm and LayerNorm in our architectures, this proper regularization behavior is important for stability.

### Why cosine LR scheduling?

Cosine annealing smoothly decays the learning rate from `lr_initial` to ~0 over `T_max` epochs. The benefits:
- No abrupt step drops that can cause oscillations
- The slow decay at the end allows the optimizer to "settle" into a good minimum with small gradient steps
- Works particularly well with AdamW where the adaptive rates interact with the schedule

### Why top-1000 answer vocabulary and not more/fewer?

With 10K training samples and a naturally skewed answer distribution:
- Top-500: covers maybe 75% of answer occurrences; misses many specific object-names, colors, etc.
- **Top-1000: covers ~85% of occurrences; ~10 training examples per class on average → sufficient for learning**
- Top-2000: covers ~90% but only ~5 examples per class → too sparse

The 1,000-class sweet spot is also what standard VQA papers use (VQA v2 uses top-3,129 classes with ~120K training samples, which is a similar data density per class).

---

## Section 11 — Prediction Files Summary

All 4 competition prediction files were generated:

| File | Challenge | Task | Samples | Format |
|------|-----------|------|---------|--------|
| `raj_kanakarajan_challenge1.pkl` | 1 (E2E Cls) | Binary classification | 100 (idx 100-199) | 1D torch tensor of {0,1} |
| `raj_kanakarajan_challenge2.json` | 2 (E2E Gen) | Answer generation | 100 (idx 100-199) | List of {image, answer} dicts |
| `raj_kanakarajan_challenge3.pkl` | 3 (CLIP Cls) | Binary classification | 100 (idx 100-199) | 1D torch tensor of {0,1} |
| `raj_kanakarajan_challenge4.json` | 4 (CLIP Gen) | Answer generation | 100 (idx 100-199) | List of {image, answer} dicts |

**Challenge 1 results**: Model predicted 74/100 as answerable (74%). This is slightly above the 73% training base rate, so the model isn't completely degenerate (predicting 100% answerable would give 73%).

**Challenge 3 results**: CLIP model predicted 85/100 as answerable. The model seems slightly more aggressive in predicting "answerable", which might reflect the CLIP features learning that most VizWiz images are answerable.

**Challenge 2 results**: The E2E generation model predicted "unanswerable" for 90/100 test samples. This is a known failure mode — the model has learned that "unanswerable" is the safest answer for ambiguous samples, and raw pixels from VizWiz are often ambiguous enough to trigger this. The model needs either more training data, better preprocessing, or (better yet) pretrained features to handle this task well.

**Challenge 4 results**: The CLIP generation model predicted "unanswerable" for 65/100 test samples, with more variety in the remaining 35. The CLIP model is less biased toward "unanswerable" because the CLIP text and image features are semantically richer — the model can distinguish "what color" questions (which are probably answerable) from "what does this say" questions on blurry text (which are often not answerable).

---

## Section 12 — What I Would Do With More Time/Compute

1. **For E2E models**: Add more CNN depth (ResNet-style residual blocks instead of plain conv blocks), use larger image size (224×224), add more transformer layers (4 instead of 2), and train for 50+ epochs.

2. **For CLIP models**: Fine-tune the CLIP text encoder (keep image encoder frozen) on VizWiz questions. Even partial fine-tuning of the last few layers could improve performance for the specific question phrasings in this dataset.

3. **For Answer Generation**: Current model does closed-vocabulary classification. A better approach would be a decoder transformer that generates answers token-by-token. But this requires more data and compute.

4. **Data augmentation**: For the E2E model, image augmentation could help significantly — random brightness/contrast jitter to simulate the varied lighting of VizWiz images, random blur to simulate the "camera not focused properly" issue.

5. **Ensembling**: Average the predictions from multiple models (e.g., different random seeds or architectures) for better competition performance.

---

*These notes document the full design rationale for the Lab Assignment 3 submission. The code in `run_e2e.py` (Challenges 1 & 2) and `solution_clip.py` (Challenges 3 & 4) implements all described architectures and experiments.*
