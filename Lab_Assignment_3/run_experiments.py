import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from collections import Counter

BASE     = "/Users/raj/CUB/Sem2/NNDL/Lab_Assignment_3"
ANNOT    = os.path.join(BASE, "Lab Assignment 3 Annotations")
CLIP_DIR = os.path.join(BASE, "Lab Assignment 3 CLIP")

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

with open(os.path.join(ANNOT, "train.json")) as f: train_data = json.load(f)
with open(os.path.join(ANNOT, "val.json"))   as f: val_data   = json.load(f)

tr_img = torch.load(os.path.join(CLIP_DIR, "VizWiz_train_CLIP_Image.pkl"), weights_only=True)
tr_txt = torch.load(os.path.join(CLIP_DIR, "VizWiz_train_CLIP_Text.pkl"),  weights_only=True)
va_img = torch.load(os.path.join(CLIP_DIR, "VizWiz_val_CLIP_Image.pkl"),   weights_only=True)
va_txt = torch.load(os.path.join(CLIP_DIR, "VizWiz_val_CLIP_Text.pkl"),    weights_only=True)

MAX_TRAIN  = 10000
VOCAB_SIZE = 1000

def ans_vocab(data, k=VOCAB_SIZE):
    c = Counter()
    for s in data:
        for a in s["answers"]: c[a["answer"].lower().strip()] += 1
    mc  = [a for a,_ in c.most_common(k)]
    a2i = {a: i for i,a in enumerate(mc)}
    return a2i, {i: a for a,i in a2i.items()}

def maj_ans(sample, a2i):
    c = Counter(a["answer"].lower().strip() for a in sample["answers"])
    for ans,_ in c.most_common():
        if ans in a2i: return a2i[ans]
    return 0

a2i, i2a = ans_vocab(train_data)

class VQADataset(Dataset):
    def __init__(self, data, img_f, txt_f, a2i, task="cls", max_n=None):
        if max_n: data = data[:max_n]
        self.data  = data
        self.img_f = img_f[:len(data)]
        self.txt_f = txt_f[:len(data)]
        self.a2i   = a2i
        self.task  = task
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        s = self.data[idx]
        img = self.img_f[idx]
        txt = self.txt_f[idx]
        if self.task == "cls":
            return img, txt, torch.tensor(s["answerable"], dtype=torch.long)
        else:
            return img, txt, torch.tensor(maj_ans(s, self.a2i), dtype=torch.long)

class Classifier(nn.Module):
    def __init__(self, fdim=512, h=512, drop=0.3, fusion="concat"):
        super().__init__()
        self.fusion = fusion
        in_dim = fdim*2 if fusion == "concat" else fdim
        if fusion == "cross_attn":
            self.cross = nn.MultiheadAttention(fdim, 4, dropout=0.1, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(h, h//2), nn.LayerNorm(h//2), nn.ReLU(), nn.Dropout(drop*0.7),
            nn.Linear(h//2, 2)
        )
    def forward(self, img, txt):
        if self.fusion == "concat":
            x = torch.cat([img, txt], dim=-1)
        elif self.fusion == "sum":
            x = img + txt
        elif self.fusion == "cross_attn":
            out, _ = self.cross(img.unsqueeze(1), txt.unsqueeze(1), txt.unsqueeze(1))
            x = out.squeeze(1)
        return self.mlp(x)

class Generator(nn.Module):
    def __init__(self, vsize=VOCAB_SIZE, drop=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(drop*0.7),
            nn.Linear(256, vsize)
        )
    def forward(self, img, txt): return self.mlp(torch.cat([img, txt], dim=-1))

def fit(model, tr_dl, va_dl, crit, lr=3e-4, epochs=30):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
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
    print(f"  Best val acc: {best:.4f}")
    return best

va_cls_ds = VQADataset(val_data, va_img, va_txt, a2i, "cls")
va_gen_ds = VQADataset(val_data, va_img, va_txt, a2i, "gen")
va_cls_dl = DataLoader(va_cls_ds, 512, shuffle=False, num_workers=0)
va_gen_dl = DataLoader(va_gen_ds, 512, shuffle=False, num_workers=0)

cls_w    = torch.tensor([14991/5532, 1.0]).to(device)
cls_crit = nn.CrossEntropyLoss(weight=cls_w)
gen_crit = nn.CrossEntropyLoss()

results = {}

print("\n=== EXP 4.1: Fusion Strategy ===")
for fusion in ["concat", "sum", "cross_attn"]:
    print(f"  fusion={fusion}")
    torch.manual_seed(42)
    tr_ds = VQADataset(train_data, tr_img, tr_txt, a2i, "cls", MAX_TRAIN)
    tr_dl = DataLoader(tr_ds, 256, shuffle=True, num_workers=0)
    m = Classifier(fusion=fusion).to(device)
    acc = fit(m, tr_dl, va_cls_dl, cls_crit, epochs=30)
    results[f"exp4_1_{fusion}"] = round(acc, 4)

print("\n=== EXP 4.2: Vocab Size ===")
for vsize in [500, 1000, 2000]:
    print(f"  vocab={vsize}")
    a2i_v, _ = ans_vocab(train_data, k=vsize)
    tr_ds_v  = VQADataset(train_data, tr_img, tr_txt, a2i_v, "gen", MAX_TRAIN)
    va_ds_v  = VQADataset(val_data,   va_img, va_txt, a2i_v, "gen")
    tr_dl_v  = DataLoader(tr_ds_v, 256, shuffle=True, num_workers=0)
    va_dl_v  = DataLoader(va_ds_v, 512, shuffle=False, num_workers=0)
    torch.manual_seed(42)
    m = Generator(vsize=vsize).to(device)
    acc = fit(m, tr_dl_v, va_dl_v, gen_crit, epochs=30)
    results[f"exp4_2_vocab{vsize}"] = round(acc, 4)

print("\n=== Results ===")
for k,v in results.items(): print(f"  {k}: {v:.4f}")

with open(os.path.join(BASE, "experiment_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Saved to experiment_results.json")
