"""
Lab Assignment 3 — Challenges 1 & 2: Custom End-to-End Multi-Modal Architectures
CSCI 5922 — Neural Networks and Deep Learning, Spring 2026

Challenge 1: Answerability binary classification (custom CNN + Transformer)
Challenge 2: Answer generation (same encoder, classification over answer vocab)

Rules:
  - NO pretrained weights anywhere
  - Must use self-attention AND/OR cross-attention
  - Max 10,000 training samples
"""

import os
import json
import math
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE  = "/Users/raj/CUB/Sem2/NNDL/Lab_Assignment_3"
ANNOT = os.path.join(BASE, "Lab Assignment 3 Annotations")

TRAIN_JSON  = os.path.join(ANNOT, "train.json")
VAL_JSON    = os.path.join(ANNOT, "val.json")
TEST_JSON   = os.path.join(ANNOT, "test.json")
TRAIN_IMGS  = os.path.join(BASE, "train")
VAL_IMGS    = os.path.join(BASE, "val")
TEST_IMGS   = os.path.join(BASE, "test")

MAX_TRAIN_SAMPLES = 10000  # 50% cap
MAX_SEQ_LEN       = 20     # max question tokens
TEXT_VOCAB_SIZE   = 5002   # 5000 words + <PAD> + <UNK>
ANSWER_VOCAB_SIZE = 1000   # top-1000 most common answers
IMG_SIZE          = 224    # image resize
SEED              = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────

# Image transforms — NO pretrained normalization needed,
# but ImageNet stats are a good prior for general images
IMG_TRANSFORM_TRAIN = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

IMG_TRANSFORM_EVAL = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def tokenize(text):
    """Lowercase, remove punctuation, split on whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def build_text_vocab(annotations, max_vocab=5000):
    """Build word vocabulary from all questions in the dataset."""
    counter = Counter()
    for sample in annotations:
        tokens = tokenize(sample["question"])
        counter.update(tokens)
    # <PAD>=0, <UNK>=1, then top words
    words = ["<PAD>", "<UNK>"] + [w for w, _ in counter.most_common(max_vocab)]
    w2i = {w: i for i, w in enumerate(words)}
    return w2i


def encode_question(question, w2i, max_len=MAX_SEQ_LEN):
    """Tokenize and encode a question to a fixed-length integer tensor."""
    tokens = tokenize(question)[:max_len]
    ids    = [w2i.get(t, 1) for t in tokens]  # 1 = <UNK>
    # Pad to max_len
    ids += [0] * (max_len - len(ids))          # 0 = <PAD>
    return torch.tensor(ids, dtype=torch.long)


def build_answer_vocab(annotations, vocab_size=ANSWER_VOCAB_SIZE):
    """Build closed answer vocabulary from training annotations."""
    counter = Counter()
    for s in annotations:
        for a in s["answers"]:
            counter[a["answer"].lower().strip()] += 1
    most_common = [ans for ans, _ in counter.most_common(vocab_size)]
    ans2idx = {ans: i for i, ans in enumerate(most_common)}
    idx2ans = {i: ans for ans, i in ans2idx.items()}
    return ans2idx, idx2ans


def get_majority_answer(sample, ans2idx):
    """Majority-vote answer index; fallback to index 0."""
    counter = Counter(a["answer"].lower().strip() for a in sample["answers"])
    for ans, _ in counter.most_common():
        if ans in ans2idx:
            return ans2idx[ans]
    return 0


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class VizWizDataset(Dataset):
    """
    Loads images from disk and encodes questions as token ids.
    Labels are either binary (classification) or answer-vocab index (generation).
    """

    def __init__(self, annotations, img_dir, w2i, ans2idx,
                 transform=IMG_TRANSFORM_EVAL, task="classification",
                 max_samples=None):
        if max_samples:
            annotations = annotations[:max_samples]
        self.annotations = annotations
        self.img_dir     = img_dir
        self.w2i         = w2i
        self.ans2idx     = ans2idx
        self.transform   = transform
        self.task        = task

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations[idx]

        # ── Load and transform image ──
        img_path = os.path.join(self.img_dir, sample["image"])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # If image is corrupted, return a blank image
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=128)
        img_tensor = self.transform(img)  # (3, 224, 224)

        # ── Encode question ──
        q_tensor = encode_question(sample["question"], self.w2i)  # (20,)

        # ── Label ──
        if self.task == "classification":
            label = torch.tensor(sample["answerable"], dtype=torch.long)
        else:  # generation
            label = torch.tensor(get_majority_answer(sample, self.ans2idx),
                                 dtype=torch.long)

        return img_tensor, q_tensor, label


class VizWizTestDataset(Dataset):
    """Test split — no labels."""

    def __init__(self, annotations, img_dir, w2i, transform=IMG_TRANSFORM_EVAL):
        self.annotations = annotations
        self.img_dir     = img_dir
        self.w2i         = w2i
        self.transform   = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        img_path = os.path.join(self.img_dir, sample["image"])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=128)
        img_tensor = self.transform(img)
        q_tensor   = encode_question(sample["question"], self.w2i)
        return img_tensor, q_tensor, sample["image"]


# ─────────────────────────────────────────────────────────────
# Model Components
# ─────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """
    Custom image encoder: 4 conv blocks → global avg pool → 512-d embedding.
    No pretrained weights — trained from scratch.
    """

    def __init__(self, out_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 224 → 112

            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 112 → 56

            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 56 → 28

            # Block 4: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 28 → 14
        )
        # Global average pooling → flatten → project to out_dim
        self.gap     = nn.AdaptiveAvgPool2d((4, 4))
        self.project = nn.Sequential(
            nn.Flatten(),                       # 256 * 4 * 4 = 4096
            nn.Linear(256 * 4 * 4, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        feat = self.encoder(x)           # (B, 256, 14, 14)
        feat = self.gap(feat)            # (B, 256, 4, 4)
        return self.project(feat)        # (B, 512)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to token embeddings."""

    def __init__(self, d_model, max_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)             # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TextEncoder(nn.Module):
    """
    Text encoder: Embedding → Positional Encoding → Transformer Encoder → pool → 512-d
    Uses self-attention via TransformerEncoderLayer.
    """

    def __init__(self, vocab_size=TEXT_VOCAB_SIZE, d_model=256, nhead=4,
                 num_layers=2, dim_ff=512, out_dim=512, dropout=0.1):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.project     = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, tokens, padding_mask=None):
        """
        tokens: (B, seq_len) integer token ids
        Returns: (B, out_dim) text feature
        """
        x = self.embed(tokens)          # (B, seq_len, d_model)
        x = self.pos_enc(x)
        # Create padding mask: True where token == <PAD>
        if padding_mask is None:
            padding_mask = (tokens == 0)  # (B, seq_len)
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # (B, seq_len, d_model)
        # Mean pooling over non-padding tokens
        mask_expanded = (~padding_mask).unsqueeze(-1).float()        # (B, seq_len, 1)
        x = (x * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)  # (B, d_model)
        return self.project(x)          # (B, out_dim)


class ModalityFusion(nn.Module):
    """
    Cross-modal fusion via self-attention over [image_token, text_token].
    Two modality tokens attend to each other via self-attention.
    This is equivalent to cross-attention when sequence length is 2.
    """

    def __init__(self, d_model=512, nhead=4, dropout=0.1):
        super().__init__()
        # Self-attention layer that processes both modality tokens together
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True
        )
        self.norm1  = nn.LayerNorm(d_model)
        self.ffn    = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2  = nn.LayerNorm(d_model)
        self.project = nn.Linear(d_model * 2, d_model)

    def forward(self, img_feat, txt_feat):
        """
        img_feat: (B, 512)
        txt_feat: (B, 512)
        Returns:  (B, 512) fused representation
        """
        # Stack as sequence: (B, 2, 512)
        tokens = torch.stack([img_feat, txt_feat], dim=1)

        # Self-attention over the 2 tokens (image attends to text and vice versa)
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)  # residual + norm

        # Feed-forward
        ffn_out = self.ffn(tokens)
        tokens  = self.norm2(tokens + ffn_out)  # residual + norm

        # Concatenate both tokens and project
        # tokens: (B, 2, 512) → flatten → (B, 1024) → (B, 512)
        fused = tokens.reshape(tokens.size(0), -1)  # (B, 1024)
        return self.project(fused)                  # (B, 512)


# ─────────────────────────────────────────────────────────────
# Full Multi-Modal Models
# ─────────────────────────────────────────────────────────────

class E2EClassifier(nn.Module):
    """
    Challenge 1: End-to-end binary classifier.
    CNN image encoder → Transformer text encoder → cross-modal fusion → 2-class head
    """

    def __init__(self, text_vocab_size=TEXT_VOCAB_SIZE, d_model=512,
                 nhead=4, dropout=0.3):
        super().__init__()
        self.img_encoder = SimpleCNN(out_dim=d_model)
        self.txt_encoder = TextEncoder(vocab_size=text_vocab_size, d_model=256,
                                       nhead=4, num_layers=2, dim_ff=512,
                                       out_dim=d_model)
        self.fusion      = ModalityFusion(d_model=d_model, nhead=nhead,
                                          dropout=0.1)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, images, tokens):
        img_feat  = self.img_encoder(images)                     # (B, 512)
        txt_feat  = self.txt_encoder(tokens)                     # (B, 512)
        fused     = self.fusion(img_feat, txt_feat)              # (B, 512)
        return self.classifier(fused)                            # (B, 2)


class E2EGenerator(nn.Module):
    """
    Challenge 2: End-to-end closed-vocabulary answer generator.
    Same architecture as E2EClassifier but with vocab-size output head.
    """

    def __init__(self, text_vocab_size=TEXT_VOCAB_SIZE, d_model=512,
                 nhead=4, dropout=0.3, answer_vocab_size=ANSWER_VOCAB_SIZE):
        super().__init__()
        self.img_encoder = SimpleCNN(out_dim=d_model)
        self.txt_encoder = TextEncoder(vocab_size=text_vocab_size, d_model=256,
                                       nhead=4, num_layers=2, dim_ff=512,
                                       out_dim=d_model)
        self.fusion      = ModalityFusion(d_model=d_model, nhead=nhead,
                                          dropout=0.1)
        self.generator   = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, answer_vocab_size),
        )

    def forward(self, images, tokens):
        img_feat = self.img_encoder(images)
        txt_feat = self.txt_encoder(tokens)
        fused    = self.fusion(img_feat, txt_feat)
        return self.generator(fused)                             # (B, vocab_size)


# ─────────────────────────────────────────────────────────────
# Training + Evaluation
# ─────────────────────────────────────────────────────────────

def compute_accuracy(logits, labels):
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for images, tokens, labels in loader:
        images = images.to(device)
        tokens = tokens.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images, tokens)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc  += compute_accuracy(logits, labels)
        n          += 1

    return total_loss / n, total_acc / n


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for images, tokens, labels in loader:
            images = images.to(device)
            tokens = tokens.to(device)
            labels = labels.to(device)
            logits = model(images, tokens)
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            total_acc  += compute_accuracy(logits, labels)
            n          += 1
    return total_loss / n, total_acc / n


def compute_answer_accuracy(preds_text, val_data, offset=0):
    """Human answer accuracy metric (Eq. 2 from assignment)."""
    total = 0.0
    for i, pred in enumerate(preds_text):
        sample  = val_data[offset + i]
        humans  = [a["answer"].lower().strip() for a in sample["answers"]]
        count   = humans.count(pred.lower().strip())
        total  += min(count / 3.0, 1.0)
    return total / len(preds_text)


def train_model(model, train_loader, val_loader, optimizer, scheduler,
                criterion, device, num_epochs=25, task_name=""):
    best_val_acc = 0.0
    best_state   = None

    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"[{task_name}] Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
                  f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} | "
                  f"Best: {best_val_acc:.4f}")

    print(f"\n[{task_name}] Best Val Accuracy: {best_val_acc:.4f}")
    model.load_state_dict(best_state)
    return model, best_val_acc


# ─────────────────────────────────────────────────────────────
# Experiment 3.1 — Attention Heads Ablation (Challenge 1)
# ─────────────────────────────────────────────────────────────

def experiment_3_1_attention_heads(train_data, val_data, w2i):
    """
    Ablate number of attention heads in the cross-modal fusion layer.
    Tests: nhead ∈ {1, 2, 4, 8}
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3.1 — Attention Heads Ablation (Challenge 1)")
    print("="*60)

    results = {}
    cls_weights = torch.tensor([14991 / 5532, 1.0]).to(device)
    criterion   = nn.CrossEntropyLoss(weight=cls_weights)

    for nhead in [1, 2, 4, 8]:
        print(f"\n--- nhead = {nhead} ---")

        model = E2EClassifier(text_vocab_size=len(w2i), nhead=nhead).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

        train_ds = VizWizDataset(train_data, TRAIN_IMGS, w2i, ans2idx={},
                                  transform=IMG_TRANSFORM_TRAIN, task="classification",
                                  max_samples=MAX_TRAIN_SAMPLES)
        val_ds   = VizWizDataset(val_data, VAL_IMGS, w2i, ans2idx={},
                                  task="classification")

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False,
                                  num_workers=4, pin_memory=True)

        _, best_acc = train_model(model, train_loader, val_loader,
                                   optimizer, scheduler, criterion, device,
                                   num_epochs=20, task_name=f"Exp3.1-nhead{nhead}")
        results[nhead] = best_acc

    print("\n--- Experiment 3.1 Results ---")
    for nhead, acc in results.items():
        print(f"  nhead={nhead}: Val Acc = {acc:.4f}")
    return results


# ─────────────────────────────────────────────────────────────
# Experiment 3.2 — Answer Supervision Strategy (Challenge 2)
# ─────────────────────────────────────────────────────────────

def majority_vote_answer(sample, ans2idx):
    """Standard majority vote."""
    counter = Counter(a["answer"].lower().strip() for a in sample["answers"])
    for ans, _ in counter.most_common():
        if ans in ans2idx:
            return ans2idx[ans]
    return 0


def confidence_filtered_answer(sample, ans2idx):
    """Majority vote using only high-confidence answers."""
    confident = [a["answer"].lower().strip()
                 for a in sample["answers"]
                 if a["answer_confidence"] == "yes"]
    if not confident:
        return majority_vote_answer(sample, ans2idx)
    counter = Counter(confident)
    for ans, _ in counter.most_common():
        if ans in ans2idx:
            return ans2idx[ans]
    return 0


class VizWizDatasetAllAnswers(Dataset):
    """Variant: train on all 10 answers (data augmentation approach)."""

    def __init__(self, annotations, img_dir, w2i, ans2idx,
                 transform=IMG_TRANSFORM_TRAIN, max_samples=None):
        if max_samples:
            annotations = annotations[:max_samples]
        # Expand: each sample becomes 10 sub-samples
        self.items   = []
        self.img_dir = img_dir
        self.w2i     = w2i
        self.ans2idx = ans2idx
        self.transform = transform

        for sample in annotations:
            for a in sample["answers"]:
                ans_str = a["answer"].lower().strip()
                label   = ans2idx.get(ans_str, 0)
                self.items.append((sample["image"], sample["question"], label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_name, question, label = self.items[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=128)
        img_tensor = self.transform(img)
        q_tensor   = encode_question(question, self.w2i)
        return img_tensor, q_tensor, torch.tensor(label, dtype=torch.long)


def experiment_3_2_supervision_strategy(train_data, val_data, w2i, ans2idx, idx2ans):
    """
    Compare 3 supervision strategies for answer generation training:
      A: Majority vote labels
      B: All 10 answers (augmented)
      C: Confidence-filtered majority vote
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3.2 — Answer Supervision Strategy (Challenge 2)")
    print("="*60)

    results = {}
    criterion = nn.CrossEntropyLoss()

    # ── Condition A: Majority Vote ──
    print("\n--- Condition A: Majority Vote ---")
    model_a = E2EGenerator(text_vocab_size=len(w2i),
                           answer_vocab_size=len(ans2idx)).to(device)
    opt_a   = torch.optim.AdamW(model_a.parameters(), lr=3e-4, weight_decay=1e-4)
    sch_a   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=20)

    train_ds_a = VizWizDataset(train_data, TRAIN_IMGS, w2i, ans2idx,
                                transform=IMG_TRANSFORM_TRAIN, task="generation",
                                max_samples=MAX_TRAIN_SAMPLES)
    val_ds     = VizWizDataset(val_data, VAL_IMGS, w2i, ans2idx, task="generation")

    train_ldr_a = DataLoader(train_ds_a, batch_size=64, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_ldr     = DataLoader(val_ds, batch_size=64, shuffle=False,
                              num_workers=4, pin_memory=True)

    model_a, _ = train_model(model_a, train_ldr_a, val_ldr, opt_a, sch_a,
                               criterion, device, num_epochs=20,
                               task_name="Exp3.2-MajorityVote")

    # Compute human answer accuracy
    model_a.eval()
    preds_a = []
    with torch.no_grad():
        for imgs, toks, _ in val_ldr:
            logits = model_a(imgs.to(device), toks.to(device))
            preds_a.extend([idx2ans[i] for i in logits.argmax(-1).cpu().tolist()])
    ans_acc_a = compute_answer_accuracy(preds_a, val_data)
    results["A_majority_vote"] = ans_acc_a
    print(f"  Condition A Ans Acc: {ans_acc_a:.4f}")

    # ── Condition B: All Answers (augmented) ──
    print("\n--- Condition B: All Answers (Augmented) ---")
    model_b  = E2EGenerator(text_vocab_size=len(w2i),
                            answer_vocab_size=len(ans2idx)).to(device)
    opt_b    = torch.optim.AdamW(model_b.parameters(), lr=3e-4, weight_decay=1e-4)
    sch_b    = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=20)

    train_ds_b  = VizWizDatasetAllAnswers(train_data, TRAIN_IMGS, w2i, ans2idx,
                                           max_samples=MAX_TRAIN_SAMPLES)
    train_ldr_b = DataLoader(train_ds_b, batch_size=64, shuffle=True,
                              num_workers=4, pin_memory=True)

    model_b, _ = train_model(model_b, train_ldr_b, val_ldr, opt_b, sch_b,
                               criterion, device, num_epochs=20,
                               task_name="Exp3.2-AllAnswers")

    model_b.eval()
    preds_b = []
    with torch.no_grad():
        for imgs, toks, _ in val_ldr:
            logits = model_b(imgs.to(device), toks.to(device))
            preds_b.extend([idx2ans[i] for i in logits.argmax(-1).cpu().tolist()])
    ans_acc_b = compute_answer_accuracy(preds_b, val_data)
    results["B_all_answers"] = ans_acc_b
    print(f"  Condition B Ans Acc: {ans_acc_b:.4f}")

    print("\n--- Experiment 3.2 Results ---")
    for cond, acc in results.items():
        print(f"  {cond}: Ans Acc = {acc:.4f}")
    return results


# ─────────────────────────────────────────────────────────────
# Train Final Competition Models
# ─────────────────────────────────────────────────────────────

def train_challenge1_final(train_data, val_data, w2i, best_nhead=4):
    """Train final Challenge 1 model (E2E classification)."""
    print("\n" + "="*60)
    print("FINAL MODEL — Challenge 1 (E2E Classification)")
    print("="*60)

    model = E2EClassifier(text_vocab_size=len(w2i), nhead=best_nhead).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    cls_weights = torch.tensor([14991 / 5532, 1.0]).to(device)
    criterion   = nn.CrossEntropyLoss(weight=cls_weights)

    train_ds = VizWizDataset(train_data, TRAIN_IMGS, w2i, ans2idx={},
                              transform=IMG_TRANSFORM_TRAIN, task="classification",
                              max_samples=MAX_TRAIN_SAMPLES)
    val_ds   = VizWizDataset(val_data, VAL_IMGS, w2i, ans2idx={},
                              task="classification")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False,
                               num_workers=4, pin_memory=True)

    model, _ = train_model(model, train_loader, val_loader,
                            optimizer, scheduler, criterion, device,
                            num_epochs=25, task_name="Challenge1")
    return model


def train_challenge2_final(train_data, val_data, w2i, ans2idx):
    """Train final Challenge 2 model (E2E answer generation)."""
    print("\n" + "="*60)
    print("FINAL MODEL — Challenge 2 (E2E Answer Generation)")
    print("="*60)

    model = E2EGenerator(text_vocab_size=len(w2i),
                         answer_vocab_size=len(ans2idx)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    criterion = nn.CrossEntropyLoss()

    train_ds = VizWizDataset(train_data, TRAIN_IMGS, w2i, ans2idx,
                              transform=IMG_TRANSFORM_TRAIN, task="generation",
                              max_samples=MAX_TRAIN_SAMPLES)
    val_ds   = VizWizDataset(val_data, VAL_IMGS, w2i, ans2idx, task="generation")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False,
                               num_workers=4, pin_memory=True)

    model, _ = train_model(model, train_loader, val_loader,
                            optimizer, scheduler, criterion, device,
                            num_epochs=25, task_name="Challenge2")
    return model


# ─────────────────────────────────────────────────────────────
# Prediction Generation
# ─────────────────────────────────────────────────────────────

def generate_challenge1_predictions(model, test_data, w2i, output_path):
    """Generate .pkl file for Challenge 1 (binary classification, indices 100-199)."""
    model.eval()

    sub_data = test_data[100:200]
    test_ds  = VizWizTestDataset(sub_data, TEST_IMGS, w2i)
    loader   = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    all_preds = []
    with torch.no_grad():
        for imgs, toks, _ in loader:
            logits = model(imgs.to(device), toks.to(device))
            preds  = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)

    preds_tensor = torch.tensor(all_preds, dtype=torch.long)
    torch.save(preds_tensor, output_path)
    print(f"Challenge 1 predictions saved to: {output_path}")
    print(f"  Shape: {preds_tensor.shape} | Answerable: {preds_tensor.sum().item()} / 100")
    return preds_tensor


def generate_challenge2_predictions(model, test_data, w2i, idx2ans, output_path):
    """Generate .json file for Challenge 2 (answer generation, indices 100-199)."""
    model.eval()

    sub_data = test_data[100:200]
    test_ds  = VizWizTestDataset(sub_data, TEST_IMGS, w2i)
    loader   = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    all_answers = []
    all_images  = []
    with torch.no_grad():
        for imgs, toks, img_names in loader:
            logits     = model(imgs.to(device), toks.to(device))
            pred_idxs  = logits.argmax(dim=-1).cpu().tolist()
            all_answers.extend([idx2ans[i] for i in pred_idxs])
            all_images.extend(list(img_names))

    predictions = [{"image": img, "answer": ans}
                   for img, ans in zip(all_images, all_answers)]

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Challenge 2 predictions saved to: {output_path}")
    print("  Sample predictions:")
    for p in predictions[:5]:
        print(f"    {p['image']}: {p['answer']}")
    return predictions


# ─────────────────────────────────────────────────────────────
# Section 5 — Data Efficiency for E2E model (5922 only)
# ─────────────────────────────────────────────────────────────

def data_efficiency_e2e(train_data, val_data, w2i):
    """
    Train E2E classification model at 50%, 75%, 100% of training data.
    Hyperparameters are fixed from Challenge 1 final model.
    """
    print("\n" + "="*60)
    print("SECTION 5 — Data Efficiency (E2E Model)")
    print("="*60)

    total = len(train_data)  # 20,523
    fractions = {
        "50%  (10,000)": 10000,
        "75%  (15,392)": int(total * 0.75),
        "100% (20,523)": total,
    }

    results = {}
    cls_weights = torch.tensor([14991 / 5532, 1.0]).to(device)
    criterion   = nn.CrossEntropyLoss(weight=cls_weights)

    for label, n_samples in fractions.items():
        print(f"\n--- Training with {label} samples ---")
        model = E2EClassifier(text_vocab_size=len(w2i), nhead=4).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)

        train_ds = VizWizDataset(train_data, TRAIN_IMGS, w2i, ans2idx={},
                                  transform=IMG_TRANSFORM_TRAIN, task="classification",
                                  max_samples=n_samples)
        val_ds   = VizWizDataset(val_data, VAL_IMGS, w2i, ans2idx={},
                                  task="classification")

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                                   num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False,
                                   num_workers=4, pin_memory=True)

        _, best_acc = train_model(model, train_loader, val_loader, opt, sch,
                                   criterion, device, num_epochs=25,
                                   task_name=f"DataEff-E2E-{label}")
        results[label] = best_acc

    print("\n--- Data Efficiency Results (E2E) ---")
    for label, acc in results.items():
        print(f"  {label}: Val Acc = {acc:.4f}")
    return results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("Loading annotations...")
    with open(TRAIN_JSON) as f: train_data = json.load(f)
    with open(VAL_JSON)   as f: val_data   = json.load(f)
    with open(TEST_JSON)  as f: test_data  = json.load(f)

    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # Build vocabularies from training data
    print("Building text vocabulary...")
    w2i = build_text_vocab(train_data, max_vocab=5000)
    print(f"Text vocab size: {len(w2i)}")

    print("Building answer vocabulary...")
    ans2idx, idx2ans = build_answer_vocab(train_data, vocab_size=ANSWER_VOCAB_SIZE)
    print(f"Answer vocab size: {len(ans2idx)}")

    # ── Experiment 3.1: Attention head ablation (uncomment to run) ──
    # exp3_1_results = experiment_3_1_attention_heads(train_data, val_data, w2i)

    # ── Experiment 3.2: Supervision strategy (uncomment to run) ──
    # exp3_2_results = experiment_3_2_supervision_strategy(
    #     train_data, val_data, w2i, ans2idx, idx2ans)

    # ── Train final Challenge 1 model ──
    model1 = train_challenge1_final(train_data, val_data, w2i, best_nhead=4)

    # ── Train final Challenge 2 model ──
    model2 = train_challenge2_final(train_data, val_data, w2i, ans2idx)

    # ── Save predictions ──
    preds1 = generate_challenge1_predictions(
        model1, test_data, w2i,
        os.path.join(BASE, "raj_kanakarajan_challenge1.pkl")
    )

    preds2 = generate_challenge2_predictions(
        model2, test_data, w2i, idx2ans,
        os.path.join(BASE, "raj_kanakarajan_challenge2.json")
    )

    # ── Section 5: Data Efficiency (5922 only) ──
    # data_efficiency_e2e(train_data, val_data, w2i)

    print("\nDone! E2E models complete.")


if __name__ == "__main__":
    main()
