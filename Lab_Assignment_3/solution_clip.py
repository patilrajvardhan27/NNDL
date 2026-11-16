import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
from collections import Counter
import numpy as np

BASE     = "/Users/raj/CUB/Sem2/NNDL/Lab_Assignment_3"
ANNOT    = os.path.join(BASE, "Lab Assignment 3 Annotations")
CLIP_DIR = os.path.join(BASE, "Lab Assignment 3 CLIP")

TRAIN_JSON = os.path.join(ANNOT, "train.json")
VAL_JSON   = os.path.join(ANNOT, "val.json")
TEST_JSON  = os.path.join(ANNOT, "test.json")

TRAIN_IMG_F = os.path.join(CLIP_DIR, "VizWiz_train_CLIP_Image.pkl")
TRAIN_TXT_F = os.path.join(CLIP_DIR, "VizWiz_train_CLIP_Text.pkl")
VAL_IMG_F   = os.path.join(CLIP_DIR, "VizWiz_val_CLIP_Image.pkl")
VAL_TXT_F   = os.path.join(CLIP_DIR, "VizWiz_val_CLIP_Text.pkl")
TEST_IMG_F  = os.path.join(CLIP_DIR, "VizWiz_test_CLIP_Image.pkl")
TEST_TXT_F  = os.path.join(CLIP_DIR, "VizWiz_test_CLIP_Text.pkl")

MAX_TRAIN = 10000
VOCAB_SIZE = 1000

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def ans_vocab(train_data, k=VOCAB_SIZE):
    c = Counter()
    for s in train_data:
        for a in s["answers"]: c[a["answer"].lower().strip()] += 1
    mc = [a for a,_ in c.most_common(k)]
    a2i = {a: i for i,a in enumerate(mc)}
    return a2i, {i: a for a,i in a2i.items()}

def maj_ans(sample, a2i):
    c = Counter(a["answer"].lower().strip() for a in sample["answers"])
    for ans,_ in c.most_common():
        if ans in a2i: return a2i[ans]
    return 0


class VQADataset(Dataset):
    def __init__(self, ann, img_f, txt_f, a2i, max_n=None, task="cls"):
        if max_n: ann = ann[:max_n]
        self.ann   = ann
        self.img_f = img_f[:len(ann)]
        self.txt_f = txt_f[:len(ann)]
        self.a2i   = a2i
        self.task  = task

    def __len__(self): return len(self.ann)

    def __getitem__(self, idx):
        s = self.ann[idx]
        if self.task == "cls":
            lbl = torch.tensor(s["answerable"], dtype=torch.long)
        else:
            lbl = torch.tensor(maj_ans(s, self.a2i), dtype=torch.long)
        return self.img_f[idx], self.txt_f[idx], lbl


class TestDS(Dataset):
    def __init__(self, ann, img_f, txt_f):
        self.ann   = ann
        self.img_f = img_f
        self.txt_f = txt_f
    def __len__(self): return len(self.ann)
    def __getitem__(self, idx):
        return self.img_f[idx], self.txt_f[idx], self.ann[idx]["image"]


class Classifier(nn.Module):
    def __init__(self, fdim=512, h=512, drop=0.3, n_cls=2, fusion="concat"):
        super().__init__()
        self.fusion = fusion
        if fusion == "concat":
            in_dim = fdim * 2
        elif fusion == "sum":
            in_dim = fdim
        elif fusion == "cross_attention":
            in_dim = fdim
            self.cross_attn = nn.MultiheadAttention(fdim, 4, dropout=0.1, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(h, h//2), nn.LayerNorm(h//2), nn.ReLU(), nn.Dropout(drop*0.7),
            nn.Linear(h//2, n_cls),
        )

    def fuse(self, img, txt):
        if self.fusion == "concat":
            return torch.cat([img, txt], dim=-1)
        elif self.fusion == "sum":
            return img + txt
        elif self.fusion == "cross_attention":
            out, _ = self.cross_attn(img.unsqueeze(1), txt.unsqueeze(1), txt.unsqueeze(1))
            return out.squeeze(1)

    def forward(self, img, txt): return self.mlp(self.fuse(img, txt))


class Generator(nn.Module):
    def __init__(self, fdim=512, h=512, drop=0.3, vsize=VOCAB_SIZE):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(fdim*2, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(h, h//2), nn.ReLU(), nn.Dropout(drop*0.7),
            nn.Linear(h//2, vsize),
        )
    def forward(self, img, txt): return self.mlp(torch.cat([img, txt], dim=-1))


def cls_acc(logits, labels):
    return (logits.argmax(-1) == labels).float().mean().item()

def human_acc(preds_text, val_data, offset=0):
    total = 0.0
    for i, pred in enumerate(preds_text):
        s = val_data[offset + i]
        ans = [a["answer"].lower().strip() for a in s["answers"]]
        total += min(ans.count(pred.lower().strip()) / 3.0, 1.0)
    return total / len(preds_text)

def train_ep(model, loader, opt, crit):
    model.train()
    tot_loss = tot_acc = 0.0
    for img, txt, lbl in loader:
        img, txt, lbl = img.to(device), txt.to(device), lbl.to(device)
        opt.zero_grad()
        logits = model(img, txt)
        loss   = crit(logits, lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot_loss += loss.item()
        tot_acc  += cls_acc(logits, lbl)
    n = len(loader)
    return tot_loss/n, tot_acc/n

def eval_ep(model, loader, crit):
    model.eval()
    tot_loss = tot_acc = 0.0
    with torch.no_grad():
        for img, txt, lbl in loader:
            img, txt, lbl = img.to(device), txt.to(device), lbl.to(device)
            logits = model(img, txt)
            tot_loss += crit(logits, lbl).item()
            tot_acc  += cls_acc(logits, lbl)
    n = len(loader)
    return tot_loss/n, tot_acc/n

def fit(model, tr_dl, va_dl, opt, sch, crit, epochs=50, name=""):
    best, state = 0.0, None
    for ep in range(epochs):
        tr_loss, tr_acc = train_ep(model, tr_dl, opt, crit)
        va_loss, va_acc = eval_ep(model, va_dl, crit)
        if sch: sch.step()
        if va_acc > best:
            best  = va_acc
            state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep+1) % 10 == 0:
            print(f"[{name}] Ep {ep+1}/{epochs} | tr={tr_acc:.4f} va={va_acc:.4f} best={best:.4f}")
    model.load_state_dict(state)
    return model, best


def exp41(train_data, val_data, tr_img, tr_txt, va_img, va_txt):
    print("\n" + "="*60)
    print("EXP 4.1: Fusion Strategy Ablation")
    print("="*60)
    results = {}
    cls_w = torch.tensor([14991/5532, 1.0]).to(device)
    for fusion in ["concat", "sum", "cross_attention"]:
        print(f"\n--- {fusion} ---")
        model = Classifier(fusion=fusion).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
        crit  = nn.CrossEntropyLoss(weight=cls_w)
        tr_ds = VQADataset(train_data, tr_img, tr_txt, {}, MAX_TRAIN, "cls")
        va_ds = VQADataset(val_data, va_img, va_txt, {}, task="cls")
        tr_dl = DataLoader(tr_ds, 128, shuffle=True, num_workers=0)
        va_dl = DataLoader(va_ds, 256, shuffle=False, num_workers=0)
        _, acc = fit(model, tr_dl, va_dl, opt, sch, crit, 30, f"4.1-{fusion}")
        results[fusion] = acc
    for k,v in results.items(): print(f"  {k}: {v:.4f}")
    return results


def exp42(train_data, val_data, tr_img, tr_txt, va_img, va_txt):
    print("\n" + "="*60)
    print("EXP 4.2: Vocab Size Ablation")
    print("="*60)
    results = {}
    for vsize in [500, 1000, 2000]:
        print(f"\n--- vocab={vsize} ---")
        a2i, i2a = ans_vocab(train_data, k=vsize)
        model = Generator(vsize=vsize).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
        crit  = nn.CrossEntropyLoss()
        tr_ds = VQADataset(train_data, tr_img, tr_txt, a2i, MAX_TRAIN, "gen")
        va_ds = VQADataset(val_data, va_img, va_txt, a2i, task="gen")
        tr_dl = DataLoader(tr_ds, 128, shuffle=True, num_workers=0)
        va_dl = DataLoader(va_ds, 256, shuffle=False, num_workers=0)
        _, acc = fit(model, tr_dl, va_dl, opt, sch, crit, 30, f"4.2-v{vsize}")
        model.eval()
        preds = []
        with torch.no_grad():
            for img, txt, _ in va_dl:
                p = model(img.to(device), txt.to(device)).argmax(-1).cpu().tolist()
                preds.extend([i2a[x] for x in p])
        h_acc = human_acc(preds, val_data)
        results[vsize] = {"cls": acc, "human": h_acc}
        print(f"  vocab={vsize}: cls={acc:.4f} human={h_acc:.4f}")
    return results


def train_c3(train_data, val_data, tr_img, tr_txt, va_img, va_txt):
    print("\n" + "="*60)
    print("FINAL: Challenge 3")
    print("="*60)
    model = Classifier(fusion="concat").to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    cls_w = torch.tensor([14991/5532, 1.0]).to(device)
    crit  = nn.CrossEntropyLoss(weight=cls_w)
    tr_ds = VQADataset(train_data, tr_img, tr_txt, {}, MAX_TRAIN, "cls")
    va_ds = VQADataset(val_data, va_img, va_txt, {}, task="cls")
    tr_dl = DataLoader(tr_ds, 128, shuffle=True, num_workers=0)
    va_dl = DataLoader(va_ds, 256, shuffle=False, num_workers=0)
    model, _ = fit(model, tr_dl, va_dl, opt, sch, crit, 50, "C3")
    return model


def train_c4(train_data, val_data, tr_img, tr_txt, va_img, va_txt, a2i):
    print("\n" + "="*60)
    print("FINAL: Challenge 4")
    print("="*60)
    model = Generator(vsize=len(a2i)).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    crit  = nn.CrossEntropyLoss()
    tr_ds = VQADataset(train_data, tr_img, tr_txt, a2i, MAX_TRAIN, "gen")
    va_ds = VQADataset(val_data, va_img, va_txt, a2i, task="gen")
    tr_dl = DataLoader(tr_ds, 128, shuffle=True, num_workers=0)
    va_dl = DataLoader(va_ds, 256, shuffle=False, num_workers=0)
    model, _ = fit(model, tr_dl, va_dl, opt, sch, crit, 50, "C4")
    return model


def gen_c3(model, test_data, t_img, t_txt, out_path):
    model.eval()
    sub_img = t_img[100:200].to(device)
    sub_txt = t_txt[100:200].to(device)
    with torch.no_grad():
        preds = model(sub_img, sub_txt).argmax(-1).cpu()
    torch.save(preds, out_path)
    print(f"C3 saved: {out_path} | answerable={preds.sum().item()}/100")
    return preds


def gen_c4(model, test_data, t_img, t_txt, i2a, out_path):
    model.eval()
    sub_img = t_img[100:200].to(device)
    sub_txt = t_txt[100:200].to(device)
    with torch.no_grad():
        idx = model(sub_img, sub_txt).argmax(-1).cpu().tolist()
    out = [{"image": test_data[100+i]["image"], "answer": i2a[p]} for i, p in enumerate(idx)]
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"C4 saved: {out_path}")
    for p in out[:5]: print(f"  {p['image']}: {p['answer']}")
    return out


def main():
    print("Loading data...")
    with open(TRAIN_JSON) as f: train_data = json.load(f)
    with open(VAL_JSON)   as f: val_data   = json.load(f)
    with open(TEST_JSON)  as f: test_data  = json.load(f)

    print("Loading CLIP features...")
    tr_img = torch.load(TRAIN_IMG_F, weights_only=True)
    tr_txt = torch.load(TRAIN_TXT_F, weights_only=True)
    va_img = torch.load(VAL_IMG_F,   weights_only=True)
    va_txt = torch.load(VAL_TXT_F,   weights_only=True)
    t_img  = torch.load(TEST_IMG_F,  weights_only=True)
    t_txt  = torch.load(TEST_TXT_F,  weights_only=True)

    a2i, i2a = ans_vocab(train_data, VOCAB_SIZE)

    model3 = train_c3(train_data, val_data, tr_img, tr_txt, va_img, va_txt)
    model4 = train_c4(train_data, val_data, tr_img, tr_txt, va_img, va_txt, a2i)

    gen_c3(model3, test_data, t_img, t_txt,
           os.path.join(BASE, "rajvardhan_patil_challenge3.pkl"))
    gen_c4(model4, test_data, t_img, t_txt, i2a,
           os.path.join(BASE, "rajvardhan_patil_challenge4.json"))

    print("\nDone!")

if __name__ == "__main__":
    main()
