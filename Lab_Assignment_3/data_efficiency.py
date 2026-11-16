import os, json, math, re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import torchvision.transforms as T

BASE     = "/Users/raj/CUB/Sem2/NNDL/Lab_Assignment_3"
ANNOT    = os.path.join(BASE, "Lab Assignment 3 Annotations")
CLIP_DIR = os.path.join(BASE, "Lab Assignment 3 CLIP")
TRAIN_IMGS = os.path.join(BASE, "train")
VAL_IMGS   = os.path.join(BASE, "val")

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

TOTAL_TRAIN = 20523
FRACTIONS = {
    "50pct":  10000,
    "75pct":  int(TOTAL_TRAIN * 0.75),
    "100pct": TOTAL_TRAIN,
}

IMG_SIZE = 112
TF_TRAIN = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
TF_EVAL  = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                      T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

MAX_SEQ = 20

def tokenize(text):
    return re.sub(r"[^\w\s]", "", text.lower().strip()).split()

def build_vocab(data, max_vocab=5000):
    c = Counter()
    for s in data: c.update(tokenize(s["question"]))
    words = ["<PAD>", "<UNK>"] + [w for w,_ in c.most_common(max_vocab)]
    return {w: i for i,w in enumerate(words)}

def enc_q(q, w2i, max_len=MAX_SEQ):
    toks = tokenize(q)[:max_len]
    return [w2i.get(t, 1) for t in toks] + [0] * (max_len - len(toks))

def cache(annotations, img_dir, w2i, transform, max_n=None):
    if max_n: annotations = annotations[:max_n]
    imgs, qs, labels = [], [], []
    print(f"  Caching {len(annotations)} images...")
    for i, s in enumerate(annotations):
        path = os.path.join(img_dir, s["image"])
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=128)
        imgs.append(transform(img))
        qs.append(torch.tensor(enc_q(s["question"], w2i), dtype=torch.long))
        labels.append(s["answerable"])
        if (i+1) % 2000 == 0: print(f"    {i+1}/{len(annotations)}")
    return TensorDataset(torch.stack(imgs), torch.stack(qs),
                         torch.tensor(labels, dtype=torch.long))

class CNN(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128,128,7), nn.ReLU(True),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128, out_dim), nn.ReLU(True))
    def forward(self, x): return self.proj(self.net(x))

class PosEnc(nn.Module):
    def __init__(self, d, max_len=MAX_SEQ):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class TxtEnc(nn.Module):
    def __init__(self, vsize, d=128, nhead=4, nlayers=2, out=256):
        super().__init__()
        self.emb  = nn.Embedding(vsize, d, padding_idx=0)
        self.pos  = PosEnc(d)
        enc       = nn.TransformerEncoderLayer(d, nhead, d*4, dropout=0.1, batch_first=True)
        self.tf   = nn.TransformerEncoder(enc, nlayers)
        self.proj = nn.Sequential(nn.Linear(d, out), nn.ReLU(True))
    def forward(self, toks):
        pad_mask = (toks == 0)
        x = self.pos(self.emb(toks))
        x = self.tf(x)
        m = (~pad_mask).float().unsqueeze(-1)
        return self.proj((x * m).sum(1) / m.sum(1).clamp(min=1))

class Fusion(nn.Module):
    def __init__(self, d=256, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d*2, d)
    def forward(self, img, txt):
        seq = torch.stack([img, txt], dim=1)
        out, _ = self.attn(seq, seq, seq)
        seq = self.norm(seq + out)
        return self.proj(seq.reshape(seq.size(0), -1))

class E2EModel(nn.Module):
    def __init__(self, vsize, nhead=4, d=256):
        super().__init__()
        self.img  = CNN(d)
        self.txt  = TxtEnc(vsize, d=128, nhead=4, nlayers=2, out=d)
        self.fuse = Fusion(d, nhead)
        self.head = nn.Sequential(nn.Linear(d,128), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(128, 2))
    def forward(self, imgs, toks):
        return self.head(self.fuse(self.img(imgs), self.txt(toks)))

class CLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.21),
            nn.Linear(256, 2)
        )
    def forward(self, img, txt): return self.mlp(torch.cat([img, txt], dim=-1))

def fit_e2e(model, tr_dl, va_dl, cls_w, epochs=25):
    crit = nn.CrossEntropyLoss(weight=cls_w.to(device))
    opt  = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = 0.0
    for ep in range(epochs):
        model.train()
        for imgs, toks, lbls in tr_dl:
            imgs, toks, lbls = imgs.to(device), toks.to(device), lbls.to(device)
            opt.zero_grad()
            loss = crit(model(imgs, toks), lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, toks, lbls in tr_dl:
                imgs, toks, lbls = imgs.to(device), toks.to(device), lbls.to(device)
                preds = model(imgs, toks).argmax(-1)
                correct += (preds == lbls).sum().item()
                total   += lbls.size(0)
        acc = correct / total
        if acc > best: best = acc
        if (ep+1) % 5 == 0:
            print(f"  Ep {ep+1}/{epochs} | val={acc:.4f} | best={best:.4f}")
    return best

def fit_clip(model, tr_dl, va_dl, cls_w, epochs=50):
    crit = nn.CrossEntropyLoss(weight=cls_w.to(device))
    opt  = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = 0.0
    for ep in range(epochs):
        model.train()
        for img, txt, lbl in tr_dl:
            img, txt, lbl = img.to(device), txt.to(device), lbl.to(device)
            opt.zero_grad()
            loss = crit(model(img, txt), lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for img, txt, lbl in va_dl:
                img, txt, lbl = img.to(device), txt.to(device), lbl.to(device)
                preds = model(img, txt).argmax(-1)
                correct += (preds == lbl).sum().item()
                total   += lbl.size(0)
        acc = correct / total
        if acc > best: best = acc
        if (ep+1) % 10 == 0:
            print(f"  Ep {ep+1}/{epochs} | val={acc:.4f} | best={best:.4f}")
    return best

def main():
    results = {}

    with open(os.path.join(ANNOT, "train.json")) as f: train_data = json.load(f)
    with open(os.path.join(ANNOT, "val.json"))   as f: val_data   = json.load(f)
    cls_w = torch.tensor([14991/5532, 1.0])

    print("\n" + "="*60)
    print("CLIP Data Efficiency")
    print("="*60)

    tr_img = torch.load(os.path.join(CLIP_DIR, "VizWiz_train_CLIP_Image.pkl"), weights_only=True)
    tr_txt = torch.load(os.path.join(CLIP_DIR, "VizWiz_train_CLIP_Text.pkl"),  weights_only=True)
    va_img = torch.load(os.path.join(CLIP_DIR, "VizWiz_val_CLIP_Image.pkl"),   weights_only=True)
    va_txt = torch.load(os.path.join(CLIP_DIR, "VizWiz_val_CLIP_Text.pkl"),    weights_only=True)
    va_lbl = torch.tensor([s["answerable"] for s in val_data], dtype=torch.long)
    va_clip_dl = DataLoader(TensorDataset(va_img, va_txt, va_lbl), 512, shuffle=False)

    for name, n in FRACTIONS.items():
        print(f"\n--- CLIP {name} ({n} samples) ---")
        torch.manual_seed(42)
        tr_lbl = torch.tensor([s["answerable"] for s in train_data[:n]], dtype=torch.long)
        tr_dl  = DataLoader(TensorDataset(tr_img[:n], tr_txt[:n], tr_lbl), 256, shuffle=True)
        model  = CLIPModel().to(device)
        acc    = fit_clip(model, tr_dl, va_clip_dl, cls_w, epochs=50)
        results[f"clip_{name}"] = round(acc, 4)
        print(f"  CLIP {name}: {acc:.4f}")

    print("\n" + "="*60)
    print("E2E Data Efficiency")
    print("="*60)

    w2i   = build_vocab(train_data)
    vsize = len(w2i)

    print("\nCaching val...")
    va_e2e_ds = cache(val_data, VAL_IMGS, w2i, TF_EVAL)
    va_e2e_dl = DataLoader(va_e2e_ds, 256, shuffle=False)

    results["e2e_50pct"] = 0.7432

    for name, n in [("75pct", FRACTIONS["75pct"]), ("100pct", FRACTIONS["100pct"])]:
        print(f"\n--- E2E {name} ({n} samples) ---")
        torch.manual_seed(42)
        tr_ds = cache(train_data, TRAIN_IMGS, w2i, TF_TRAIN, max_n=n)
        tr_dl = DataLoader(tr_ds, 128, shuffle=True)
        model = E2EModel(vsize, nhead=4).to(device)
        acc   = fit_e2e(model, tr_dl, va_e2e_dl, cls_w, epochs=25)
        results[f"e2e_{name}"] = round(acc, 4)
        print(f"  E2E {name}: {acc:.4f}")

    out_path = os.path.join(BASE, "data_efficiency_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"{'Model':<12} {'50%':>8} {'75%':>8} {'100%':>8}")
    print("-"*40)
    print(f"{'E2E':<12} {results.get('e2e_50pct','-'):>8} "
          f"{results.get('e2e_75pct','-'):>8} {results.get('e2e_100pct','-'):>8}")
    print(f"{'CLIP':<12} {results.get('clip_50pct','-'):>8} "
          f"{results.get('clip_75pct','-'):>8} {results.get('clip_100pct','-'):>8}")
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
