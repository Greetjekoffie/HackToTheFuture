"""plan2ifc: learn floor-plan PNG -> IFC with supervised warm-start + RL fine-tuning.

Put this file next to the `ifc_models` folder and run:

    pip install torch numpy scipy opencv-python pillow ifcopenshell
    python plan2ifc.py

Quick trial on a few files first:   python plan2ifc.py --limit 40 --sft_epochs 3 --rl_epochs 2

Folder contents: resbim-X_plan.png + resbim-X_model.ifc pairs. Anything that does not match
that naming, has no partner, or fails to load/parse is skipped (and listed). The first 300
valid pairs (ordered by X) train (--n_train, default 400), the rest test. 30 of those are held out
for picking the best checkpoint. Results (metrics, generated IFCs, overlay images) go to ./plan2ifc_out.
Parsed pairs are cached in <out>/cache.pkl.

REGISTRATION: walls are detected in the PNG (thick filled bands, see wall_mask()). The network sees
a square crop around them, and the wall-pixel bbox is matched to the IFC wall footprint. Plans are
printed at a fixed scale (~181 or ~194 px/m), so pairs whose x and y scales disagree are skipped.
"""
import argparse
import copy
import csv
import json
import math
import os
import random
import re
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # let unsupported ops fall back to CPU on Mac

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment

# ----------------------------------------------------------------------------- constants
CLASSES = ["WALL", "DOOR", "WINDOW", "SLAB"]
IFC_TYPES = {"WALL": "IfcWall", "DOOR": "IfcDoor", "WINDOW": "IfcWindow", "SLAB": "IfcSlab"}
PAD, BOS, EOS = 0, 1, 2
CLS0 = 3
COORD0 = CLS0 + len(CLASSES)
FIELDS = 6  # class, x1, y1, x2, y2, thickness (slab: x0, y0, x1, y1, 0)
IMG = 512
NB = 256  # coordinate bins
MAX_ELEMS = 128
VOCAB = COORD0 + NB
SEQ_LEN = FIELDS * MAX_ELEMS + 2
D_MODEL, N_HEADS, ENC_LAYERS, DEC_LAYERS = 384, 6, 4, 6
T_MAX = 0.25


# ----------------------------------------------------------------------------- data loading
CROP_MARGIN = 0.12  # extra border around the detected walls (slabs / openings may stick out)


def wall_mask(a):
    """Walls are drawn as thick filled dark bands. Opening with a 13px square removes text, dimension
    lines, grid lines and door arcs; small leftovers (letters inside grid bubbles) are dropped by area."""
    dark = (a < 200).astype(np.uint8)
    wm = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((13, 13), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(wm, 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] > 3000
    return keep[lab]


def load_plan(path):
    """PNG -> (512x512 crop around the building, crop dict). The crop is found from the image alone
    (wall pixels), so the exact same thing happens at inference time without an IFC."""
    a = np.array(Image.open(path).convert("L"))
    if a.mean() < 127:
        a = 255 - a
    ys, xs = np.where(wall_mask(a))
    if len(xs) == 0:
        raise ValueError("no walls detected in image")
    wb = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    side = max(wb[2] - wb[0], wb[3] - wb[1]) * (1 + 2 * CROP_MARGIN)
    cx, cy = (wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2
    x0, y0, S = int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))
    canvas = np.full((S, S), 255, np.uint8)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + S, a.shape[1]), min(y0 + S, a.shape[0])
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0:sy1, sx0:sx1]
    img = cv2.resize(canvas, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return img, dict(x0=float(x0), y0=float(y0), side=float(S), wall_bbox=wb)


def parse_ifc(path):
    """IFC -> (elements in world metres, wall footprint extent). Wall/door/window = centre line + thickness."""
    import ifcopenshell
    import ifcopenshell.geom
    f = ifcopenshell.open(str(path))
    st = ifcopenshell.geom.settings()
    try:
        st.set("use-world-coords", True)
    except Exception:
        st.set(st.USE_WORLD_COORDS, True)
    elems, wallxy = [], []
    for name in CLASSES:
        for el in f.by_type(IFC_TYPES[name]):
            try:
                v = np.array(ifcopenshell.geom.create_shape(st, el).geometry.verts, float).reshape(-1, 3)
            except Exception:
                continue
            if len(v) < 3:
                continue
            xy = v[:, :2].astype(np.float32)
            z0, z1 = float(v[:, 2].min()), float(v[:, 2].max())
            if name == "SLAB":
                p = [float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())]
                t = 0.0
                if p[2] - p[0] < 1e-3 or p[3] - p[1] < 1e-3:
                    continue
            else:
                (cx, cy), (w, h), ang = cv2.minAreaRect(xy)
                if h > w:
                    w, h, ang = h, w, ang + 90.0
                if w < 1e-3:
                    continue
                th = np.deg2rad(ang)
                d = np.array([np.cos(th), np.sin(th)]) * w / 2
                c = np.array([cx, cy])
                p = [float(c[0] - d[0]), float(c[1] - d[1]), float(c[0] + d[0]), float(c[1] + d[1])]
                t = float(h)
                if name == "WALL":
                    wallxy.append(xy)
            elems.append(dict(cls=name, p=p, t=t, z0=z0, z1=z1))
    if not wallxy:
        raise ValueError("no walls found in IFC")
    a = np.concatenate(wallxy)
    return elems, (float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max()))


def fit_transform(extent, crop):
    """Register IFC to image: the wall-pixel bbox in the PNG == the wall footprint extent in the IFC.
    Plans are printed at a fixed scale, so x and y must give the same px/m; if not, the pair is rejected."""
    minx, miny, maxx, maxy = extent
    bx0, by0, bx1, by1 = crop["wall_bbox"]
    sx, sy = (bx1 - bx0) / max(maxx - minx, 1e-6), (by1 - by0) / max(maxy - miny, 1e-6)
    if abs(sx - sy) / max(sx, sy) > 0.03:
        raise ValueError(f"image/IFC registration mismatch (x {sx:.1f} vs y {sy:.1f} px/m)")
    s = (sx + sy) / 2
    # world -> pixel: px = bx0 + (X - minx) * s ; py = by0 + (maxy - Y) * s  (image y points down)
    return dict(s=s, bx0=bx0, by0=by0, minx=minx, maxy=maxy)


def world_to_norm(elems, tf, crop):
    S, x0, y0 = crop["side"], crop["x0"], crop["y0"]
    nx = lambda X: (tf["bx0"] + (X - tf["minx"]) * tf["s"] - x0) / S
    ny = lambda Y: (tf["by0"] + (tf["maxy"] - Y) * tf["s"] - y0) / S
    out = []
    for e in elems:
        p = e["p"]
        q = [nx(p[0]), ny(p[1]), nx(p[2]), ny(p[3])]
        if e["cls"] == "SLAB":
            q = [min(q[0], q[2]), min(q[1], q[3]), max(q[0], q[2]), max(q[1], q[3])]
        out.append(dict(cls=e["cls"], p=q, t=e["t"] * tf["s"] / S, z0=e["z0"], z1=e["z1"]))
    return out


def norm_to_world(elems, crop, px_per_m):
    """Inverse of world_to_norm, using only image information + the dataset print scale.
    World origin = centre of the detected walls."""
    S, x0, y0 = crop["side"], crop["x0"], crop["y0"]
    bx0, by0, bx1, by1 = crop["wall_bbox"]
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    wx = lambda n: (x0 + n * S - cx) / px_per_m
    wy = lambda n: (cy - (y0 + n * S)) / px_per_m
    out = []
    for e in elems:
        p = e["p"]
        q = [wx(p[0]), wy(p[1]), wx(p[2]), wy(p[3])]
        if e["cls"] == "SLAB":
            q = [min(q[0], q[2]), min(q[1], q[3]), max(q[0], q[2]), max(q[1], q[3])]
        out.append(dict(cls=e["cls"], p=q, t=e["t"] * S / px_per_m, z0=None, z1=None))
    return out


def augment(img, elems, k):
    """One of the 8 square symmetries (k in 0..7), applied identically to image and labels."""
    if k == 0:
        return img, elems
    flip, rot = k >= 4, k % 4

    def tp(x, y):
        if flip:
            x = 1.0 - x
        for _ in range(rot):  # np.rot90 (counter-clockwise on screen): (x, y) -> (y, 1 - x)
            x, y = y, 1.0 - x
        return x, y

    im = img[:, ::-1] if flip else img
    im = np.ascontiguousarray(np.rot90(im, rot))
    out = []
    for e in elems:
        p = e["p"]
        a, b = tp(p[0], p[1]), tp(p[2], p[3])
        q = [a[0], a[1], b[0], b[1]]
        if e["cls"] == "SLAB":
            q = [min(q[0], q[2]), min(q[1], q[3]), max(q[0], q[2]), max(q[1], q[3])]
        out.append(dict(e, p=q))
    return im, out


def _load_pair(n, png, ifc):
    img, crop = load_plan(png)
    elems, extent = parse_ifc(ifc)
    tf = fit_transform(extent, crop)
    return dict(name=f"resbim-{n}", img=img, crop=crop, elems=world_to_norm(elems, tf, crop), px_per_m=tf["s"])


def scan_folder(folder, limit=None, cache=None):
    """Return list of valid samples; print every skipped file with the reason."""
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder.resolve()}")
    pat = re.compile(r"^resbim-(\d+)_plan\.png$")
    cands, skipped = [], []
    for f in sorted(folder.iterdir()):
        if not f.is_file():
            continue
        m = pat.match(f.name)
        if not m:
            if not re.match(r"^resbim-\d+_model\.ifc$", f.name):
                skipped.append((f.name, "name does not match resbim-X_plan.png / resbim-X_model.ifc"))
            continue
        ifc = folder / f"resbim-{m.group(1)}_model.ifc"
        if not ifc.exists():
            skipped.append((f.name, "no matching _model.ifc"))
            continue
        cands.append((int(m.group(1)), f, ifc))
    for f in sorted(folder.glob("resbim-*_model.ifc")):
        if not (folder / f.name.replace("_model.ifc", "_plan.png")).exists():
            skipped.append((f.name, "no matching _plan.png"))
    cands.sort()
    if limit:
        cands = cands[:limit]
    memo = {}
    if cache and Path(cache).exists():
        import pickle
        memo = pickle.load(open(cache, "rb"))
    samples, changed = [], False
    for i, (n, png, ifc) in enumerate(cands):
        key = (png.name, png.stat().st_mtime, ifc.stat().st_mtime, CROP_MARGIN, IMG)
        try:
            if key not in memo:
                try:
                    memo[key] = _load_pair(n, png, ifc)
                except Exception as ex:
                    memo[key] = f"unreadable: {str(ex)[:80]}"
                changed = True
            if isinstance(memo[key], str):
                raise ValueError(memo[key])
            samples.append(memo[key])
        except Exception as ex:
            skipped.append((png.name, str(ex)))
        if (i + 1) % 50 == 0:
            print(f"  parsed {i + 1}/{len(cands)}", flush=True)
    if cache and changed:
        import pickle
        pickle.dump(memo, open(cache, "wb"))
    print(f"Valid pairs: {len(samples)} | skipped: {len(skipped)}")
    for name, why in skipped[:30]:
        print(f"  skipped {name}: {why}")
    if len(skipped) > 30:
        print(f"  ... and {len(skipped) - 30} more")
    return samples


# ----------------------------------------------------------------------------- tokens
def _q(v):
    return int(np.clip(round(v * (NB - 1)), 0, NB - 1))


def encode(elems):
    rows = []
    for e in elems:
        c = CLASSES.index(e["cls"])
        q = [_q(v) for v in e["p"]]
        if e["cls"] == "SLAB":
            q = [min(q[0], q[2]), min(q[1], q[3]), max(q[0], q[2]), max(q[1], q[3])]
            t = 0
        else:
            if (q[0], q[1]) > (q[2], q[3]):
                q = [q[2], q[3], q[0], q[1]]
            t = _q(math.sqrt(min(max(e["t"], 0.0), T_MAX) / T_MAX))
        rows.append((c, q, t))
    rows.sort(key=lambda r: (r[0], r[1][1], r[1][0], r[1][3], r[1][2]))
    toks = [BOS]
    for c, q, t in rows[:MAX_ELEMS]:
        toks += [CLS0 + c] + [COORD0 + v for v in q + [t]]
    toks.append(EOS)
    return toks + [PAD] * (SEQ_LEN - len(toks))


def decode(tokens):
    out, i = [], 1
    while i < len(tokens) and tokens[i] not in (EOS, PAD):
        c = tokens[i] - CLS0
        if not 0 <= c < len(CLASSES) or i + 5 >= len(tokens):
            break
        v = [t - COORD0 for t in tokens[i + 1:i + 6]]
        if any(not 0 <= x < NB for x in v):
            break
        slab = CLASSES[c] == "SLAB"
        out.append(dict(cls=CLASSES[c], p=[x / (NB - 1) for x in v[:4]],
                        t=0.0 if slab else (v[4] / (NB - 1)) ** 2 * T_MAX, z0=None, z1=None))
        i += FIELDS
    return out


def build_mask():
    m = np.zeros((SEQ_LEN, VOCAB), bool)
    m[0, PAD] = True
    for j in range(1, SEQ_LEN):
        slot = (j - 1) % FIELDS
        if j == SEQ_LEN - 1:
            m[j, EOS] = True
        elif slot == 0:
            m[j, CLS0:CLS0 + len(CLASSES)] = True
            m[j, EOS] = True
        else:
            m[j, COORD0:] = True
    return m


# ----------------------------------------------------------------------------- reward / metrics
_K = np.ones((3, 3), np.uint8)


def render(elems, R=128):
    m = np.zeros((len(CLASSES), R, R), np.uint8)
    for e in elems:
        c = CLASSES.index(e["cls"])
        p = np.array(e["p"], float) * R
        if e["cls"] == "SLAB":
            cv2.rectangle(m[c], (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, -1)
            continue
        a, b = p[:2], p[2:]
        d = b - a
        L = float(np.hypot(*d))
        if L < 1e-6:
            continue
        n = np.array([-d[1], d[0]]) / L * (e["t"] * R / 2)
        cv2.fillConvexPoly(m[c], np.round(np.array([a + n, b + n, b - n, a - n])).astype(np.int32), 1)
        cv2.line(m[c], (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))), 1, 1)
    return m


def _iou(a, b):
    a, b = cv2.dilate(a, _K), cv2.dilate(b, _K)
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def _f1(pred, gt, thr=0.02):
    if not pred or not gt:
        return 0.0
    ctr = lambda E: np.array([[(e["p"][0] + e["p"][2]) / 2, (e["p"][1] + e["p"][3]) / 2] for e in E])
    D = np.linalg.norm(ctr(pred)[:, None] - ctr(gt)[None], axis=-1)
    if pred[0]["cls"] != "SLAB":
        ang = lambda E: np.array([np.arctan2(e["p"][3] - e["p"][1], e["p"][2] - e["p"][0]) % np.pi for e in E])
        diff = np.abs(ang(pred)[:, None] - ang(gt)[None])
        D = np.where(np.minimum(diff, np.pi - diff) < 0.26, D, 1e3)
    r, c = linear_sum_assignment(D)
    tp = int(sum(D[i, j] < thr for i, j in zip(r, c)))
    if tp == 0:
        return 0.0
    pr, rc = tp / len(pred), tp / len(gt)
    return 2 * pr * rc / (pr + rc)


def score(pred, gt):
    """reward = 0.5*mean IoU + 0.5*mean matched-F1 - 0.002*excess elements."""
    mp, mg = render(pred), render(gt)
    ious, f1s, per = [], [], {}
    for i, name in enumerate(CLASSES):
        P = [e for e in pred if e["cls"] == name]
        G = [e for e in gt if e["cls"] == name]
        if not P and not G:
            continue
        iou, f1 = _iou(mp[i], mg[i]), _f1(P, G)
        ious.append(iou)
        f1s.append(f1)
        per[name] = dict(iou=iou, f1=f1)
    iou = float(np.mean(ious)) if ious else 0.0
    f1 = float(np.mean(f1s)) if f1s else 0.0
    return dict(reward=0.5 * iou + 0.5 * f1 - 0.002 * max(0, len(pred) - len(gt)),
                iou=iou, f1=f1, n_pred=len(pred), n_gt=len(gt), per_class=per)


# ----------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o1 = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.q2, self.kv2, self.o2 = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def _heads(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.h, D // self.h).transpose(1, 2)

    def forward(self, x, mem, cache=None):
        B, T, D = x.shape
        q, k, v = [self._heads(t) for t in self.qkv(self.ln1(x)).chunk(3, -1)]
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        a = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None)
        x = x + self.o1(a.transpose(1, 2).reshape(B, T, D))
        q2 = self._heads(self.q2(self.ln2(x)))
        if cache is not None and "mk" in cache:
            mk, mv = cache["mk"], cache["mv"]
        else:
            mk, mv = [self._heads(t) for t in self.kv2(mem).chunk(2, -1)]
            if cache is not None:
                cache["mk"], cache["mv"] = mk, mv
        a = F.scaled_dot_product_attention(q2, mk, mv)
        x = x + self.o2(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln3(x))


class Plan2Ifc(nn.Module):
    def __init__(self):
        super().__init__()
        d = D_MODEL
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
                                  nn.Conv2d(64, d, 4, 4))
        g = IMG // 16
        self.pos_enc = nn.Parameter(torch.randn(1, g * g, d) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d, N_HEADS, 4 * d, 0.0, activation="gelu", batch_first=True, norm_first=True), ENC_LAYERS)
        self.tok, self.pos_dec, self.slot = nn.Embedding(VOCAB, d), nn.Embedding(SEQ_LEN, d), nn.Embedding(FIELDS + 1, d)
        self.blocks = nn.ModuleList(Block(d, N_HEADS) for _ in range(DEC_LAYERS))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, VOCAB)
        self.register_buffer("mask", torch.from_numpy(build_mask()), persistent=False)
        self.register_buffer("slot_ids", torch.tensor([0] + [(j - 1) % FIELDS + 1 for j in range(1, SEQ_LEN)]),
                             persistent=False)

    def encode(self, img):
        return self.enc(self.stem(img).flatten(2).transpose(1, 2) + self.pos_enc)

    def embed(self, tokens, start):
        pos = torch.arange(start, start + tokens.size(1), device=tokens.device)
        return self.tok(tokens) + self.pos_dec(pos) + self.slot(self.slot_ids[pos])

    def forward(self, img, tokens):
        mem = self.encode(img)
        x = self.embed(tokens[:, :-1], 0)
        for b in self.blocks:
            x = b(x, mem)
        return self.head(self.ln(x)).float().masked_fill(~self.mask[1:tokens.size(1)], float("-inf"))

    def logprobs(self, img, tokens):
        tgt = tokens[:, 1:]
        lp = F.log_softmax(self.forward(img, tokens), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        valid = tgt != PAD
        return lp.masked_fill(~valid, 0.0), valid

    @torch.no_grad()
    def generate(self, img, greedy=False, temperature=1.0):
        B = img.size(0)
        mem = self.encode(img)
        caches = [{} for _ in self.blocks]
        tok = torch.full((B, 1), BOS, dtype=torch.long, device=img.device)
        out, done = [tok], torch.zeros(B, dtype=torch.bool, device=img.device)
        for i in range(SEQ_LEN - 1):
            x = self.embed(tok, i)
            for b, c in zip(self.blocks, caches):
                x = b(x, mem, c)
            logits = self.head(self.ln(x))[:, -1].float().masked_fill(~self.mask[i + 1], float("-inf"))
            nxt = logits.argmax(-1) if greedy else torch.multinomial(F.softmax(logits / temperature, -1), 1).squeeze(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            done = done | (nxt == EOS)
            out.append(nxt[:, None])
            tok = nxt[:, None]
            if done.all():
                break
        return torch.cat(out, 1)


# ----------------------------------------------------------------------------- helpers
def amp(dev):
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")


def imgs_of(items, dev):
    return torch.stack([(1.0 - torch.from_numpy(i["img"]).float() / 255.0).unsqueeze(0) for i in items]).to(dev)


def batches(items, bs, shuffle=False):
    idx = list(range(len(items)))
    if shuffle:
        random.shuffle(idx)
    for s in range(0, len(idx), bs):
        yield [items[i] for i in idx[s:s + bs]]


@torch.no_grad()
def predict(model, items, dev, bs=8):
    model.eval()
    rows = []
    for b in batches(items, bs):
        with amp(dev):
            toks = model.generate(imgs_of(b, dev), greedy=True).cpu().tolist()
        for it, t in zip(b, toks):
            pred = decode(t)
            rows.append(dict(item=it, pred=pred, **score(pred, it["elems"])))
    return rows


def summarize(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in ("reward", "iou", "f1", "n_pred", "n_gt")}


# ----------------------------------------------------------------------------- IFC writer
def build_ifc(elems, out_path, defaults):
    import ifcopenshell.api
    run = ifcopenshell.api.run

    def mat(x, y, z, th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], float)

    f = run("project.create_file")
    project = run("root.create_entity", f, ifc_class="IfcProject", name="plan2ifc")
    run("unit.assign_unit", f)
    ctx = run("context.add_context", f, context_type="Model")
    body = run("context.add_context", f, context_type="Model", context_identifier="Body",
               target_view="MODEL_VIEW", parent=ctx)
    site = run("root.create_entity", f, ifc_class="IfcSite", name="Site")
    bld = run("root.create_entity", f, ifc_class="IfcBuilding", name="Building")
    sto = run("root.create_entity", f, ifc_class="IfcBuildingStorey", name="Level 0")
    run("aggregate.assign_object", f, relating_object=project, products=[site])
    run("aggregate.assign_object", f, relating_object=site, products=[bld])
    run("aggregate.assign_object", f, relating_object=bld, products=[sto])
    for e in elems:
        cls = e["cls"]
        z0 = e["z0"] if e.get("z0") is not None else defaults[cls][0]
        z1 = e["z1"] if e.get("z1") is not None else defaults[cls][1]
        height = max(z1 - z0, 0.05)
        el = run("root.create_entity", f, ifc_class=IFC_TYPES[cls], name=cls.title())
        p = e["p"]
        if cls == "SLAB":
            x0, y0, x1, y1 = p
            rep = run("geometry.add_slab_representation", f, context=body, depth=height,
                      polyline=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            matrix = mat(0, 0, z0, 0.0)
        else:
            a, b = np.array(p[:2]), np.array(p[2:])
            length = float(np.hypot(*(b - a)))
            if length < 1e-3:
                continue
            t = max(e["t"], 0.02)
            th = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
            origin = a - np.array([-np.sin(th), np.cos(th)]) * t / 2
            rep = run("geometry.add_wall_representation", f, context=body, length=length, height=height, thickness=t)
            matrix = mat(origin[0], origin[1], z0, th)
        run("geometry.assign_representation", f, product=el, representation=rep)
        run("geometry.edit_object_placement", f, product=el, matrix=matrix)
        run("spatial.assign_container", f, relating_structure=sto, products=[el])
    f.write(str(out_path))


# ----------------------------------------------------------------------------- training stages
def run_sft(model, train, val, dev, epochs, bs, lr, aug=True):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * math.ceil(len(train) / bs)
    warm = max(1, min(200, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    best, best_state = -1.0, copy.deepcopy(model.state_dict())
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for b in batches(train, bs, shuffle=True):
            if aug:  # random flip/rotation so the decoder cannot just memorise one "average" plan
                b = [dict(zip(("img", "elems"), augment(i["img"], i["elems"], random.randrange(8)))) for i in b]
                toks = torch.tensor([encode(i["elems"]) for i in b], device=dev)
            else:
                toks = torch.tensor([i["toks"] for i in b], device=dev)
            with amp(dev):
                logits = model(imgs_of(b, dev), toks)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), toks[:, 1:].reshape(-1), ignore_index=PAD)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, n = tot + loss.item(), n + 1
        msg = f"[SFT] epoch {ep}/{epochs} loss {tot / n:.4f}"
        if val and (ep % 5 == 0 or ep == epochs):
            s = summarize(predict(model, val, dev))
            msg += f" | val reward {s['reward']:.3f} iou {s['iou']:.3f} f1 {s['f1']:.3f}"
            if s["reward"] > best:
                best, best_state = s["reward"], copy.deepcopy(model.state_dict())
        print(msg, flush=True)
    model.load_state_dict(best_state if val else model.state_dict())


def run_rl(model, train, val, dev, epochs, bs, group, lr, beta):
    """GRPO-style: sample `group` sequences per plan, reward = score vs ground truth,
    advantage = reward normalised within the group, KL penalty to the SFT policy."""
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    best, best_state = -1.0, copy.deepcopy(model.state_dict())
    if val:
        s = summarize(predict(model, val, dev))
        best = s["reward"]
        print(f"[RL] start | val reward {s['reward']:.3f} iou {s['iou']:.3f} f1 {s['f1']:.3f}", flush=True)
    for ep in range(1, epochs + 1):
        stats = []
        for b in batches(train, bs, shuffle=True):
            B = len(b)
            img = imgs_of(b, dev).repeat_interleave(group, 0)
            model.eval()
            with amp(dev):
                toks = model.generate(img, greedy=False)
            rewards = torch.tensor([score(decode(t), b[i // group]["elems"])["reward"]
                                    for i, t in enumerate(toks.cpu().tolist())], device=dev).view(B, group)
            adv = ((rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-4)).view(-1, 1)
            with torch.no_grad(), amp(dev):
                ref_lp, _ = ref.logprobs(img, toks)
            model.train()
            with amp(dev):
                lp, valid = model.logprobs(img, toks)
            d = ref_lp - lp
            kl = d.exp() - d - 1.0
            loss = (((-adv * lp + beta * kl) * valid).sum(1) / valid.sum(1).clamp(min=1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            stats.append(rewards.mean().item())
        msg = f"[RL] epoch {ep}/{epochs} sampled reward {np.mean(stats):.3f}"
        if val:
            s = summarize(predict(model, val, dev))
            msg += f" | val reward {s['reward']:.3f} iou {s['iou']:.3f} f1 {s['f1']:.3f}"
            if s["reward"] > best:
                best, best_state = s["reward"], copy.deepcopy(model.state_dict())
        print(msg, flush=True)
    if val:
        model.load_state_dict(best_state)


def evaluate(model, test, dev, out, px_per_m, defaults):
    rows = predict(model, test, dev)
    (out / "ifc").mkdir(parents=True, exist_ok=True)
    (out / "overlay").mkdir(exist_ok=True)
    import ifcopenshell
    ok = 0
    for r in rows:
        it = r["item"]
        try:
            path = out / "ifc" / f"{it['name']}_pred.ifc"
            build_ifc(norm_to_world(r["pred"], it["crop"], px_per_m), path, defaults)
            ifcopenshell.open(str(path))
            ok += 1
        except Exception as ex:
            print(f"[ifc fail] {it['name']}: {ex}")
        vis = cv2.cvtColor(it["img"], cv2.COLOR_GRAY2BGR)
        g = render(it["elems"], IMG).max(0).astype(bool)
        p = render(r["pred"], IMG).max(0).astype(bool)
        vis[g & ~p], vis[p & ~g], vis[g & p] = (0, 200, 0), (0, 0, 220), (0, 220, 220)  # green=GT red=pred
        cv2.imwrite(str(out / "overlay" / f"{it['name']}.png"), vis)
    summ = summarize(rows)
    summ.update(n_test=len(rows), valid_ifc_rate=ok / max(1, len(rows)))
    summ["per_class"] = {c: dict(iou=float(np.mean([r["per_class"][c]["iou"] for r in rows if c in r["per_class"]])),
                                 f1=float(np.mean([r["per_class"][c]["f1"] for r in rows if c in r["per_class"]])))
                         for c in CLASSES if any(c in r["per_class"] for r in rows)}
    json.dump(summ, open(out / "summary.json", "w"), indent=2)
    with open(out / "per_sample.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "reward", "iou", "f1", "n_pred", "n_gt"])
        for r in rows:
            w.writerow([r["item"]["name"], f"{r['reward']:.4f}", f"{r['iou']:.4f}", f"{r['f1']:.4f}", r["n_pred"], r["n_gt"]])
    print("\nTEST RESULTS\n" + json.dumps(summ, indent=2))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="ifc_models")
    ap.add_argument("--out", default="plan2ifc_out")
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_val", type=int, default=30)
    ap.add_argument("--sft_epochs", type=int, default=60)
    ap.add_argument("--rl_epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only use the first N pairs (quick trial)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():  # Apple Silicon GPU
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {dev}" + ("  (no GPU found: training will be very slow)" if dev.type == "cpu" else ""))

    samples = scan_folder(args.dir, args.limit, out / "cache.pkl")
    n_train = min(args.n_train, int(len(samples) * 0.8)) if args.limit else args.n_train
    if len(samples) <= n_train:
        raise SystemExit(f"Only {len(samples)} valid pairs; need more than {n_train} for a train/test split.")
    train_all, test = samples[:n_train], samples[n_train:]
    n_val = min(args.n_val, len(train_all) // 10)
    train, val = (train_all[:-n_val], train_all[-n_val:]) if n_val else (train_all, [])
    print(f"Split: train {len(train)} | val {len(val)} (from the first {n_train}) | test {len(test)}")

    trunc = sum(len(s["elems"]) > MAX_ELEMS for s in samples)
    if trunc:
        print(f"Note: {trunc} plans have > {MAX_ELEMS} elements; extras are cut (raise MAX_ELEMS at the top).")
    for s in samples:
        s["toks"] = encode(s["elems"])

    # dataset-level constants from the training split only
    scales = np.array([s["px_per_m"] for s in train_all])
    px_per_m = float(np.median(scales))
    print(f"Print scale: {px_per_m:.1f} px/m (range {scales.min():.1f}-{scales.max():.1f}); used to turn predictions into metres")
    fb = dict(WALL=(0.0, 2.8), DOOR=(0.0, 2.1), WINDOW=(0.9, 2.1), SLAB=(-0.2, 0.0))
    defaults = {}
    for c in CLASSES:
        z = [(e["z0"], e["z1"]) for s in train_all for e in s["elems"] if e["cls"] == c]
        defaults[c] = (float(np.median([a for a, _ in z])), float(np.median([b for _, b in z]))) if z else fb[c]

    model = Plan2Ifc().to(dev)
    run_sft(model, train, val, dev, args.sft_epochs, args.bs, 3e-4)
    run_rl(model, train, val, dev, args.rl_epochs, max(1, args.bs // 2), args.group, 1e-5, 0.02)
    torch.save(model.state_dict(), out / "model.pt")
    evaluate(model, test, dev, out, px_per_m, defaults)
    print(f"\nDone. Everything is in {out.resolve()}")


if __name__ == "__main__":
    main()