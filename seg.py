"""plan2ifc, segmentation version: "colour in the drawing", then turn the colours into IFC elements.

1. A U-Net predicts, for every pixel of the 512x512 building crop, whether it is WALL / DOOR / WINDOW / SLAB
   (4 independent yes/no maps, since walls sit on top of the slab).
2. A plain (non-learned) vectoriser turns those maps into elements: wall bands -> centre line + thickness,
   door/window blobs -> short segments, slab -> rectangle. All elements in this dataset are axis-aligned.
3. The same IFC writer, scoring and train/val/test split as main.py are reused, so results compare 1:1.

    python seg.py                    # train + evaluate, results in ./seg_out
    python seg.py --check_vectoriser # score the vectoriser on perfect (ground-truth) masks = upper bound
    python seg.py --pretrained       # ImageNet ResNet-34 encoder (needs torchvision + one-off weight download)
"""
import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import main as base
from main import CLASSES, IMG

MIN_RUN = 15  # px at 512: shortest wall the vectoriser keeps (5th percentile wall length is ~26 px)


# ----------------------------------------------------------------------------- targets
def masks_of(elems):
    return base.render(elems, IMG).astype(np.float32)  # (4, 512, 512), same rasteriser as the metric


# ----------------------------------------------------------------------------- vectoriser
def _runs(mask, horizontal):
    """Axis-aligned bands -> segments. Opening with a long thin kernel keeps only pixels that belong to a
    run of >= MIN_RUN px in that direction, so each component is one horizontal (or vertical) band."""
    k = np.ones((1, MIN_RUN) if horizontal else (MIN_RUN, 1), np.uint8)
    band = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(band, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        length = w if horizontal else h
        if length < MIN_RUN:
            continue
        t = area / length  # mean thickness, robust to bumps at junctions
        if horizontal:
            ys = np.where(lab[y:y + h, x:x + w] == i)[0] + y
            c = float(np.median(ys)) + 0.5
            p = [x, c, x + w, c]
        else:
            xs = np.where(lab[y:y + h, x:x + w] == i)[1] + x
            c = float(np.median(xs)) + 0.5
            p = [c, y, c, y + h]
        out.append((p, t))
    return out


def _blobs(mask, min_area=20):
    """Door / window blobs -> one segment along the long side."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < min_area:
            continue
        if w >= h:
            p, t = [x, y + h / 2, x + w, y + h / 2], h
        else:
            p, t = [x + w / 2, y, x + w / 2, y + h], w
        out.append((p, t))
    return out


def vectorise(prob, thr=0.5):
    """(4, H, W) probabilities -> element list in normalised [0, 1] crop coordinates."""
    R = prob.shape[-1]
    m = (prob > thr).astype(np.uint8)
    elems = []

    def add(cls, p, t):
        elems.append(dict(cls=cls, p=[v / R for v in p], t=float(t) / R, z0=None, z1=None))

    for p, t in _runs(m[0], True) + _runs(m[0], False):
        add("WALL", p, t)
    for ci, cls in ((1, "DOOR"), (2, "WINDOW")):
        for p, t in _blobs(m[ci]):
            add(cls, p, t)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m[3], 8)
    if n > 1:  # one slab per plan in this dataset: the biggest region
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, _ = st[i]
        if w * h > 0.01 * R * R:
            add("SLAB", [x, y, x + w, y + h], 0.0)
    return elems


# ----------------------------------------------------------------------------- model
def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Plain U-Net trained from scratch (~7.8M params)."""

    def __init__(self, ch=(32, 64, 128, 256, 512), n_out=len(CLASSES)):
        super().__init__()
        self.downs = nn.ModuleList()
        c = 1
        for o in ch:
            self.downs.append(cbr(c, o))
            c = o
        self.ups, self.dec = nn.ModuleList(), nn.ModuleList()
        for o in reversed(ch[:-1]):
            self.ups.append(nn.ConvTranspose2d(c, o, 2, 2))
            self.dec.append(cbr(2 * o, o))
            c = o
        self.head = nn.Conv2d(c, n_out, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.downs):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for up, dec, s in zip(self.ups, self.dec, reversed(skips[:-1])):
            x = dec(torch.cat([up(x), s], 1))
        return self.head(x)


class ResUNet(nn.Module):
    """U-Net whose encoder is an ImageNet-pretrained ResNet-34 (torchvision)."""

    def __init__(self, n_out=len(CLASSES)):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        self.l0 = nn.Sequential(r.conv1, r.bn1, r.relu)  # /2, 64
        self.l1 = nn.Sequential(r.maxpool, r.layer1)  # /4, 64
        self.l2, self.l3, self.l4 = r.layer2, r.layer3, r.layer4  # /8 128, /16 256, /32 512
        self.full = cbr(1, 32)  # full-resolution skip so thin walls stay sharp
        chans = [(512, 256, 256), (256, 128, 128), (128, 64, 64), (64, 64, 64), (64, 32, 32)]
        self.ups = nn.ModuleList(nn.ConvTranspose2d(i, o, 2, 2) for i, o, _ in chans)
        self.dec = nn.ModuleList(cbr(o + s, o) for _, o, s in chans)
        self.head = nn.Conv2d(32, n_out, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        rgb = ((1.0 - x).repeat(1, 3, 1, 1) - self.mean) / self.std  # back to black-on-white like photos
        s0 = self.l0(rgb)
        s1 = self.l1(s0)
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        y = self.l4(s3)
        for up, dec, s in zip(self.ups, self.dec, (s3, s2, s1, s0, self.full(x))):
            y = dec(torch.cat([up(y), s], 1))
        return self.head(y)


def seg_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.full((1, 4, 1, 1), 3.0,
                                                                                    device=logits.device))
    p = torch.sigmoid(logits)
    inter = (p * target).sum((2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((2, 3)) + target.sum((2, 3)) + 1)
    return bce + dice.mean()


# ----------------------------------------------------------------------------- train / eval
def to_batch(items, dev, aug):
    ims, tg = [], []
    for it in items:
        img, el = base.augment(it["img"], it["elems"], random.randrange(8)) if aug else (it["img"], it["elems"])
        ims.append(1.0 - img.astype(np.float32) / 255.0)
        tg.append(masks_of(el))
    return torch.from_numpy(np.stack(ims)[:, None]).to(dev), torch.from_numpy(np.stack(tg)).to(dev)


@torch.no_grad()
def predict(model, items, dev, bs=8):
    model.eval()
    rows = []
    for b in base.batches(items, bs):
        x, _ = to_batch(b, dev, False)
        prob = torch.sigmoid(model(x)).cpu().numpy()
        for it, pr in zip(b, prob):
            pred = vectorise(pr)
            rows.append(dict(item=it, pred=pred, prob=pr, **base.score(pred, it["elems"])))
    return rows


def train(model, tr, val, dev, epochs, bs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(tr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    best, best_state = -1e9, None
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for b in base.batches(tr, bs, shuffle=True):
            x, y = to_batch(b, dev, True)
            loss = seg_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot, n = tot + loss.item(), n + 1
        msg = f"[SEG] epoch {ep}/{epochs} loss {tot / n:.4f}"
        if val and (ep % 5 == 0 or ep == epochs):
            s = base.summarize(predict(model, val, dev))
            msg += f" | val reward {s['reward']:.3f} iou {s['iou']:.3f} f1 {s['f1']:.3f} n_pred {s['n_pred']:.1f}"
            if s["reward"] > best:
                best, best_state = s["reward"], {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(msg, flush=True)
    if best_state:
        model.load_state_dict(best_state)


def report(rows, out, px_per_m, defaults, write_files=True):
    import csv
    import ifcopenshell
    ok = 0
    if write_files:
        (out / "ifc").mkdir(parents=True, exist_ok=True)
        (out / "overlay").mkdir(exist_ok=True)
    for r in rows:
        it = r["item"]
        if not write_files:
            continue
        try:
            path = out / "ifc" / f"{it['name']}_pred.ifc"
            base.build_ifc(base.norm_to_world(r["pred"], it["crop"], px_per_m), path, defaults)
            ifcopenshell.open(str(path))
            ok += 1
        except Exception as ex:
            print(f"[ifc fail] {it['name']}: {ex}")
        vis = cv2.cvtColor(it["img"], cv2.COLOR_GRAY2BGR)
        g = base.render(it["elems"], IMG).max(0).astype(bool)
        p = base.render(r["pred"], IMG).max(0).astype(bool)
        vis[g & ~p], vis[p & ~g], vis[g & p] = (0, 200, 0), (0, 0, 220), (0, 220, 220)  # green=GT red=pred
        cv2.imwrite(str(out / "overlay" / f"{it['name']}.png"), vis)
    summ = base.summarize(rows)
    summ.update(n_test=len(rows), valid_ifc_rate=ok / max(1, len(rows)) if write_files else None)
    summ["per_class"] = {c: dict(iou=float(np.mean([r["per_class"][c]["iou"] for r in rows if c in r["per_class"]])),
                                 f1=float(np.mean([r["per_class"][c]["f1"] for r in rows if c in r["per_class"]])))
                         for c in CLASSES if any(c in r["per_class"] for r in rows)}
    if write_files:
        json.dump(summ, open(out / "summary.json", "w"), indent=2)
        with open(out / "per_sample.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "reward", "iou", "f1", "n_pred", "n_gt"])
            for r in rows:
                w.writerow([r["item"]["name"], f"{r['reward']:.4f}", f"{r['iou']:.4f}", f"{r['f1']:.4f}",
                            r["n_pred"], r["n_gt"]])
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="ifc_models")
    ap.add_argument("--out", default="seg_out")
    ap.add_argument("--cache", default="plan2ifc_out/cache.pkl", help="shared with main.py")
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_val", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--check_vectoriser", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {dev}")

    samples = base.scan_folder(args.dir, args.limit, args.cache)
    n_train = min(args.n_train, int(len(samples) * 0.8)) if args.limit else args.n_train
    train_all, test = samples[:n_train], samples[n_train:]
    n_val = min(args.n_val, len(train_all) // 10)
    tr, val = train_all[:-n_val], train_all[-n_val:]
    print(f"Split: train {len(tr)} | val {len(val)} | test {len(test)}  (identical to main.py)")
    px_per_m = float(np.median([s["px_per_m"] for s in train_all]))
    fb = dict(WALL=(0.0, 2.8), DOOR=(0.0, 2.1), WINDOW=(0.9, 2.1), SLAB=(-0.2, 0.0))
    defaults = {}
    for c in CLASSES:
        z = [(e["z0"], e["z1"]) for s in train_all for e in s["elems"] if e["cls"] == c]
        defaults[c] = (float(np.median([a for a, _ in z])), float(np.median([b for _, b in z]))) if z else fb[c]

    if args.check_vectoriser:  # perfect masks in -> how much does the vectoriser alone lose?
        rows = []
        for it in test:
            pred = vectorise(masks_of(it["elems"]))
            rows.append(dict(item=it, pred=pred, **base.score(pred, it["elems"])))
        print("VECTORISER UPPER BOUND (ground-truth masks)\n" + json.dumps(report(rows, out, px_per_m, defaults, False),
                                                                             indent=2))
        return

    model = (ResUNet() if args.pretrained else UNet()).to(dev)
    print(f"Model: {type(model).__name__}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    train(model, tr, val, dev, args.epochs, args.bs, args.lr)
    torch.save(model.state_dict(), out / "model.pt")
    summ = report(predict(model, test, dev), out, px_per_m, defaults)
    print("\nTEST RESULTS\n" + json.dumps(summ, indent=2))
    print(f"\nDone. Everything is in {out.resolve()}")


if __name__ == "__main__":
    main()
