import os, json, math, re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import torchvision.transforms as T

BASE       = "/Users/raj/CUB/Sem2/NNDL/Lab_Assignment_3"
ANNOT      = os.path.join(BASE, "Lab Assignment 3 Annotations")
TRAIN_JSON = os.path.join(ANNOT, "train.json")
VAL_JSON   = os.path.join(ANNOT, "val.json")
TEST_JSON  = os.path.join(ANNOT, "test.json")
TRAIN_IMGS = os.path.join(BASE, "train")
VAL_IMGS   = os.path.join(BASE, "val")
TEST_IMGS  = os.path.join(BASE, "test")

MAX_TRAIN = 10000
MAX_SEQ   = 20
IMG_SIZE  = 112

torch.manual_seed(42)
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

TF_TRAIN = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
TF_EVAL  = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                      T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def tokenize(text):
    return re.sub(r"[^\w\s]", "", text.lower().strip()).split()

def build_vocab(data, max_vocab=5000):
    c = Counter()
    for s in data: c.update(tokenize(s["question"]))
    words = ["<PAD>", "<UNK>"] + [w for w,_ in c.most_common(max_vocab)]
    return {w: i for i, w in enumerate(words)}

def enc_q(q, w2i, max_len=MAX_SEQ):
    toks = tokenize(q)[:max_len]
    return [w2i.get(t, 1) for t in toks] + [0] * (max_len - len(toks))

def ans_vocab(data, k=1000):
    c = Counter()
    for s in data:
        for a in s["answers"]: c[a["answer"].lower().strip()] += 1
    mc = [a for a,_ in c.most_common(k)]
    a2i = {a: i for i,a in enumerate(mc)}
    return a2i, {i: a for a,i in a2i.items()}

def maj_ans(sample, a2i):
    c = Counter(a["answer"].lower().strip() for a in sample["answers"])
    for ans,_ in c.most_common():
        if ans in a2i: return a2i[ans]
    return 0

def cache(annotations, img_dir, w2i, a2i, transform, task, max_n=None):
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
        labels.append(s["answerable"] if task == "cls" else maj_ans(s, a2i))
        if (i+1) % 1000 == 0: print(f"    {i+1}/{len(annotations)}")
    return TensorDataset(torch.stack(imgs), torch.stack(qs),
                         torch.tensor(labels, dtype=torch.long))

def test_cache(annotations, img_dir, w2i, transform, start=100, end=200):
    sub = annotations[start:end]
    imgs, qs, names = [], [], []
    for s in sub:
        path = os.path.join(img_dir, s["image"])
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=128)
        imgs.append(transform(img))
        qs.append(torch.tensor(enc_q(s["question"], w2i), dtype=torch.long))
        names.append(s["image"])
    return torch.stack(imgs), torch.stack(qs), names

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
        div = torch.exp(torch.arange(0,d,2).float() * (-math.log(10000)/d))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:,:x.size(1)]

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

class Model(nn.Module):
    def __init__(self, vsize, out_classes=2, nhead=4, d=256):
        super().__init__()
        self.img  = CNN(d)
        self.txt  = TxtEnc(vsize, d=128, nhead=4, nlayers=2, out=d)
        self.fuse = Fusion(d, nhead)
        self.head = nn.Sequential(nn.Linear(d,128), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(128, out_classes))
    def forward(self, imgs, toks):
        return self.head(self.fuse(self.img(imgs), self.txt(toks)))

def fit(model, tr_dl, va_dl, crit, lr=3e-4, epochs=20, name=""):
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, state = 0.0, None
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
            for imgs, toks, lbls in va_dl:
                imgs, toks, lbls = imgs.to(device), toks.to(device), lbls.to(device)
                preds = model(imgs, toks).argmax(-1)
                correct += (preds == lbls).sum().item()
                total   += lbls.size(0)
        acc = correct / total
        if acc > best:
            best = acc
            state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep+1) % 5 == 0:
            print(f"[{name}] Ep {ep+1}/{epochs} | val={acc:.4f} | best={best:.4f}")
    model.load_state_dict(state)
    return model, best

def exp31(tr_ds, va_dl, vsize, cls_w):
    print("\n" + "="*60)
    print("EXP 3.1: Attention Heads Ablation")
    print("="*60)
    results = {}
    crit = nn.CrossEntropyLoss(weight=cls_w.to(device))
    for nhead in [1, 2, 4, 8]:
        print(f"\n--- nhead={nhead} ---")
        m  = Model(vsize, out_classes=2, nhead=nhead).to(device)
        dl = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
        _, acc = fit(m, dl, va_dl, crit, epochs=15, name=f"3.1-nhead{nhead}")
        results[nhead] = acc
    for k,v in results.items(): print(f"  nhead={k}: {v:.4f}")
    return results

def main():
    with open(TRAIN_JSON) as f: train_data = json.load(f)
    with open(VAL_JSON)   as f: val_data   = json.load(f)
    with open(TEST_JSON)  as f: test_data  = json.load(f)

    w2i       = build_vocab(train_data)
    a2i, i2a  = ans_vocab(train_data, k=1000)
    vsize     = len(w2i)
    cls_w     = torch.tensor([14991/5532, 1.0])

    print("\nCaching train (cls)...")
    tr_cls_ds = cache(train_data, TRAIN_IMGS, w2i, a2i, TF_TRAIN, "cls", MAX_TRAIN)
    print("Caching val...")
    va_cls_ds = cache(val_data, VAL_IMGS, w2i, a2i, TF_EVAL, "cls")

    tr_dl_cls = DataLoader(tr_cls_ds, batch_size=128, shuffle=True, num_workers=0)
    va_dl_cls = DataLoader(va_cls_ds, batch_size=256, shuffle=False, num_workers=0)

    exp_res   = exp31(tr_cls_ds, va_dl_cls, vsize, cls_w)
    best_nh   = max(exp_res, key=exp_res.get)
    print(f"\nBest nhead: {best_nh}")

    print("\n" + "="*60)
    print("FINAL: Challenge 1")
    m1 = Model(vsize, out_classes=2, nhead=best_nh).to(device)
    m1, _ = fit(m1, tr_dl_cls, va_dl_cls,
                nn.CrossEntropyLoss(weight=cls_w.to(device)), epochs=25, name="C1")

    print("\nCaching train (gen)...")
    tr_gen_ds = cache(train_data, TRAIN_IMGS, w2i, a2i, TF_TRAIN, "gen", MAX_TRAIN)
    va_gen_ds = cache(val_data, VAL_IMGS, w2i, a2i, TF_EVAL, "gen")
    tr_dl_gen = DataLoader(tr_gen_ds, batch_size=128, shuffle=True, num_workers=0)
    va_dl_gen = DataLoader(va_gen_ds, batch_size=256, shuffle=False, num_workers=0)

    print("\n" + "="*60)
    print("FINAL: Challenge 2")
    m2 = Model(vsize, out_classes=len(a2i), nhead=4).to(device)
    m2, _ = fit(m2, tr_dl_gen, va_dl_gen, nn.CrossEntropyLoss(), epochs=25, name="C2")

    print("\nCaching test (100-199)...")
    t_imgs, t_qs, t_names = test_cache(test_data, TEST_IMGS, w2i, TF_EVAL)
    t_imgs = t_imgs.to(device)
    t_qs   = t_qs.to(device)

    m1.eval()
    with torch.no_grad():
        p1 = m1(t_imgs, t_qs).argmax(-1).cpu()
    torch.save(p1, os.path.join(BASE, "rajvardhan_patil_challenge1.pkl"))
    print(f"C1 saved | Answerable: {p1.sum().item()}/100")

    m2.eval()
    with torch.no_grad():
        p2_idx = m2(t_imgs, t_qs).argmax(-1).cpu().tolist()
    out2 = [{"image": n, "answer": i2a[p]} for n, p in zip(t_names, p2_idx)]
    with open(os.path.join(BASE, "rajvardhan_patil_challenge2.json"), "w") as f:
        import json as _j; _j.dump(out2, f, indent=2)
    print("C2 saved")
    for p in out2[:5]: print(f"  {p['image']}: {p['answer']}")

if __name__ == "__main__":
    main()
