"""Floor-plan reader trained on CubiCasa5k: symbols, room types and walls in one network.

One U-Net (ImageNet ResNet-34 encoder, from seg.py) with three kinds of output map:
  * symbols  (door, window, toilet, sink, shower, bathtub, sauna stove): a small bright dot ("heat map")
    on the centre of every object. Counting = finding the peaks (CenterNet-style).
  * rooms    (kitchen, bedroom, bathroom, wc, balcony, sauna): a dot inside every room of that type. The
    network learns this from the room's label text (MH, KH, WC, ...) and its contents/shape.
  * wall     a yes/no map of wall pixels (door openings left open), used to build the 3D / IFC model.

Targets come from the SVG annotations. Definitions match how questions.jsonl was built (checked against
its answers: 99.8-100 % agreement per question template).

    python3 counter.py prepare      # parse SVGs -> data/cubicasa_prep (resized images, wall masks, points)
    python3 counter.py train        # -> counter_out/counter.pt (+ thresholds tuned on val)
    python3 counter.py eval         # count accuracy on the test split
    python3 counter.py show IMG     # draw detections -> counter_out/show.png
"""
import argparse
import json
import math
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "cubicasa5k"
PREP = ROOT / "data" / "cubicasa_prep"
OUT = ROOT / "counter_out"
OBJECTS = ["door", "window", "toilet", "sink", "shower", "bathtub", "sauna_stove"]
ROOMS = ["kitchen", "bedroom", "bathroom", "wc", "balcony", "sauna", "other"]
CLASSES = OBJECTS + ROOMS  # heat-map channels; one extra channel (last) = wall mask
N_OUT = len(CLASSES) + 1
LONG = 1024  # every plan is resized so its longest side is LONG px (train and inference)
CROP = 512
PEAK_K = {**{c: 7 for c in OBJECTS}, **{c: 31 for c in ROOMS}}  # NMS window (px) per class
HOLDOUT = {"colorful_10052"}  # benchmark plans: never trained on, even if CubiCasa lists them as train

SINKS = {"Sink", "DoubleSink", "RoundSink", "DoubleSinkRight", "SideSink", "CornerSink"}


# ----------------------------------------------------------------------------- SVG annotations
def _floats(s):
    return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", s or "")]


def _matrix(tr):
    """SVG transform string -> 3x3 matrix (matrix / translate / scale / rotate)."""
    M = np.eye(3)
    for kind, args in re.findall(r"(matrix|translate|scale|rotate)\s*\(([^)]*)\)", tr or ""):
        a = _floats(args)
        if kind == "matrix" and len(a) == 6:
            T = np.array([[a[0], a[2], a[4]], [a[1], a[3], a[5]], [0, 0, 1]])
        elif kind == "translate":
            T = np.array([[1, 0, a[0]], [0, 1, a[1] if len(a) > 1 else 0], [0, 0, 1]])
        elif kind == "scale":
            T = np.diag([a[0], a[1] if len(a) > 1 else a[0], 1])
        elif kind == "rotate":
            t = math.radians(a[0])
            R = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
            if len(a) == 3:
                R = np.array([[1, 0, a[1]], [0, 1, a[2]], [0, 0, 1]]) @ R @ np.array([[1, 0, -a[1]], [0, 1, -a[2]], [0, 0, 1]])
            T = R
        else:
            continue
        M = M @ T
    return M


def _tag(e):
    return e.tag.split("}")[-1]


def _first_poly(e, M):
    """First polygon / rect inside element e (the object's outline), transformed to plan coordinates."""
    def rec(x, M):
        M = M @ _matrix(x.get("transform"))
        t = _tag(x)
        if t in ("polygon", "polyline"):
            p = _floats(x.get("points"))
            if len(p) >= 6:
                return np.array(p).reshape(-1, 2), M
        if t == "rect":
            X, Y = float(x.get("x", 0)), float(x.get("y", 0))
            W, H = float(x.get("width", 0)), float(x.get("height", 0))
            return np.array([[X, Y], [X + W, Y], [X + W, Y + H], [X, Y + H]]), M
        for ch in x:
            r = rec(ch, M)
            if r is not None:
                return r
        return None
    r = rec(e, M)
    if r is None:
        return None
    P, M2 = r
    return (np.c_[P, np.ones(len(P))] @ M2.T)[:, :2]


def _name_label(e):
    for sub in e.iter():
        if "NameLabel" in (sub.get("class") or ""):
            txt = "".join(t.text or "" for t in sub.iter() if _tag(t) == "text").strip()
            return txt
    return ""


def object_class(cls_attr):
    c = (cls_attr or "").split()
    if not c:
        return None
    if c[0] == "Door":
        return None if len(c) > 1 and c[1] == "None" else "door"  # doorway without a leaf: not a door
    if c[0] == "Window":
        return "window"
    if c[0] == "FixedFurniture" and len(c) > 1:
        if c[1] == "Toilet":
            return "toilet"
        if c[1] in SINKS:
            return "sink"
        if c[1] == "Shower":
            return "shower"
        if c[1] == "Bathtub":
            return "bathtub"
        if "SaunaStove" in " ".join(c[1:]):
            return "sauna_stove"
    return None


def room_class(cls_attr, label):
    c = (cls_attr or "").split()[1:]
    lab = label.strip().upper()
    if lab.startswith("WC"):
        return "wc"
    if not c:
        return None
    if c[0] == "Kitchen":
        return "kitchen"
    if c[0] == "Bedroom":
        return "bedroom"
    if c[0] == "Bath":
        return "bathroom"
    if c[0] == "Outdoor" and len(c) > 1 and c[1] == "Balcony":
        return "balcony"
    if c[0] == "Sauna":
        return "sauna"
    # Keep all remaining annotated Space polygons as generic rooms. OCR labels
    # can specialize these at inference; dropping them made whole plans vanish
    # whenever CubiCasa used an unmapped type such as LivingRoom or Undefined.
    return "other"


def parse_svg(path):
    """-> dict(objects={cls: [box]}, rooms={cls: [polygon]}, walls=[polygon], doors=[polygon], n_floors)
    in SVG coordinates (= F1_scaled.png pixels)."""
    root = ET.parse(path).getroot()
    objs = {c: [] for c in OBJECTS}
    rooms = {c: [] for c in ROOMS}
    walls, door_polys, floors = [], [], 0

    def walk(e, Mp):
        nonlocal floors
        # _first_poly applies e's own transform itself, so it gets the parent matrix Mp
        # (applying it here as well would place rotated furniture twice-transformed, off the plan)
        M = Mp @ _matrix(e.get("transform"))
        cls = e.get("class") or ""
        head = cls.split()[0] if cls.split() else ""
        if head == "Floorplan":
            floors += 1
        if head == "Wall":
            P = _first_poly(e, Mp)
            if P is not None:
                walls.append(P)
        elif head == "Space":
            P = _first_poly(e, Mp)
            k = room_class(cls, _name_label(e))
            if P is not None and k:
                rooms[k].append(P)
        k = object_class(cls)
        if k is not None:
            P = _first_poly(e, Mp)
            if P is not None:
                x0, y0 = P.min(0)
                x1, y1 = P.max(0)
                if x1 - x0 > 1 and y1 - y0 > 1:
                    objs[k].append([float(x0), float(y0), float(x1), float(y1)])
                if k == "door" or cls.startswith("Door"):
                    door_polys.append(P)
            return  # nested parts of one object
        for ch in e:
            walk(ch, M)

    walk(root, np.eye(3))
    return dict(objects=objs, rooms=rooms, walls=walls, doors=door_polys, n_floors=max(1, floors))


def interior_point(poly, scale=1.0):
    """Point well inside a (possibly L-shaped) polygon: maximum of the distance transform."""
    P = np.asarray(poly, np.float64) * scale
    x0, y0 = P.min(0) - 2
    x1, y1 = P.max(0) + 2
    w, h = int(x1 - x0) + 1, int(y1 - y0) + 1
    if w < 3 or h < 3:
        c = P.mean(0)
        return float(c[0]), float(c[1])
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(P - [x0, y0]).astype(np.int32)], 1)
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(d.argmax()), d.shape)
    return float(x + x0), float(y + y0)


# ----------------------------------------------------------------------------- preprocessing
def to_gray_ink(rgb):
    """Any plan image -> float32 'ink' image in [0, 1] (1 = dark lines), same as seg.py's input."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    if g.mean() < 127:
        g = 255 - g
    return 1.0 - g.astype(np.float32) / 255.0


def resize_long(img, long=LONG):
    H, W = img.shape[:2]
    s = long / max(H, W)
    return cv2.resize(img, (max(1, round(W * s)), max(1, round(H * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC), s


def prepare():
    dirs = sorted({p.parent for p in RAW.rglob("model.svg")})
    if not dirs:
        sys.exit(f"No model.svg found under {RAW}. Unzip cubicasa5k.zip there first.")
    PREP.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name in ("train", "val", "test"):
        for f in RAW.rglob(f"{name}.txt"):
            for line in f.read_text().split():
                parts = line.strip("/").split("/")
                splits[f"{parts[-2]}_{parts[-1]}"] = name
    meta, bad = [], 0
    counts = {c: 0 for c in CLASSES}
    for i, d in enumerate(dirs):
        try:
            key = f"{d.parent.name}_{d.name}"
            rgb = cv2.imread(str(d / "F1_scaled.png"), cv2.IMREAD_COLOR)[..., ::-1]
            ann = parse_svg(d / "model.svg")  # SVG coordinates are F1_scaled.png pixels
            ink, s = resize_long(to_gray_ink(rgb))
            cv2.imwrite(str(PREP / f"{key}.png"), (ink * 255).astype(np.uint8))
            wall = np.zeros(ink.shape, np.uint8)
            for P in ann["walls"]:
                cv2.fillPoly(wall, [np.round(P * s).astype(np.int32)], 255)
            for P in ann["doors"]:  # door openings are gaps in the wall
                cv2.fillPoly(wall, [np.round(P * s).astype(np.int32)], 0)
            cv2.imwrite(str(PREP / f"{key}_wall.png"), wall)
            pts = {c: [] for c in CLASSES}
            size = {c: [] for c in CLASSES}
            for c, bs in ann["objects"].items():
                for b in bs:
                    pts[c].append([(b[0] + b[2]) / 2 * s, (b[1] + b[3]) / 2 * s])
                    size[c].append(max(b[2] - b[0], b[3] - b[1]) * s)
            for c, polys in ann["rooms"].items():
                for P in polys:
                    pts[c].append(list(interior_point(P, s)))
                    size[c].append(float(np.sqrt(cv2.contourArea(np.float32(P * s)))))
            for c in CLASSES:
                counts[c] += len(pts[c])
            split = "test" if key in HOLDOUT else splits.get(key, "train")
            meta.append(dict(key=key, split=split, h=ink.shape[0], w=ink.shape[1], scale=s, pts=pts, size=size,
                             n_floors=ann["n_floors"]))
        except Exception as ex:
            bad += 1
            print(f"  skip {d}: {ex}")
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(dirs)}", flush=True)
    json.dump(meta, open(PREP / "meta.json", "w"))
    by = {s: sum(m["split"] == s for m in meta) for s in ("train", "val", "test")}
    print(f"Prepared {len(meta)} plans ({bad} skipped) | split {by} | points {counts}")


# ----------------------------------------------------------------------------- targets / model
def sigma_of(c, size):
    if c in ROOMS:
        return 6.0
    return max(2.0, 0.18 * min(size, 60))


def heatmap(pts, sizes, h, w):
    hm = np.zeros((len(CLASSES), h, w), np.float32)
    for ci, c in enumerate(CLASSES):
        for (cx, cy), sz in zip(pts[c], sizes[c]):
            sig = sigma_of(c, sz)
            r = int(3 * sig) + 1
            xa, xb, ya, yb = max(0, int(cx) - r), min(w, int(cx) + r + 1), max(0, int(cy) - r), min(h, int(cy) + r + 1)
            if xa >= xb or ya >= yb:
                continue
            yy, xx = np.mgrid[ya:yb, xa:xb]
            g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig * sig))
            hm[ci, ya:yb, xa:xb] = np.maximum(hm[ci, ya:yb, xa:xb], g)
            iy, ix = int(round(cy)), int(round(cx))
            if 0 <= iy < h and 0 <= ix < w:
                hm[ci, iy, ix] = 1.0
    return hm


def focal_loss(logits, gt, alpha=2, beta=4):
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    pos_l = (torch.log(p) * (1 - p) ** alpha * pos).sum()
    neg_l = (torch.log(1 - p) * p ** alpha * (1 - gt) ** beta * (1 - pos)).sum()
    return -(pos_l + neg_l) / pos.sum().clamp(min=1)


def wall_loss(logits, gt):
    bce = F.binary_cross_entropy_with_logits(logits, gt)
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * gt).sum() + 1) / (p.sum() + gt.sum() + 1)
    return bce + dice


def build_model(pretrained=True):
    import seg
    if pretrained:
        m = seg.ResUNet(n_out=N_OUT)
    else:  # loading a checkpoint: no need to download ImageNet weights
        import torchvision
        orig = torchvision.models.resnet34
        torchvision.models.resnet34 = lambda weights=None: orig(weights=None)
        try:
            m = seg.ResUNet(n_out=N_OUT)
        finally:
            torchvision.models.resnet34 = orig
    with torch.no_grad():
        m.head.bias[:len(CLASSES)].fill_(-2.19)  # heat maps start at p ~ 0.1 (CenterNet init)
    return m


def load_checkpoint_weights(model, checkpoint):
    """Load a prior heat-map checkpoint while preserving channels if classes grew."""
    source = checkpoint["model"]
    target = model.state_dict()
    old_classes = checkpoint.get("classes") or [c for c in CLASSES if c != "other"]
    for name, value in source.items():
        if name in ("head.weight", "head.bias") and value.shape != target[name].shape:
            for cls in CLASSES:
                if cls in old_classes:
                    target[name][CLASSES.index(cls)].copy_(value[old_classes.index(cls)])
            # The wall channel is stored immediately after the old heat channels.
            target[name][len(CLASSES)].copy_(value[len(old_classes)])
        elif name in target and target[name].shape == value.shape:
            target[name] = value
    model.load_state_dict(target)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# ----------------------------------------------------------------------------- training data
def load_meta():
    return json.load(open(PREP / "meta.json"))


def sample_crop(m, aug=True):
    img = cv2.imread(str(PREP / f"{m['key']}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    wall = cv2.imread(str(PREP / f"{m['key']}_wall.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    pts = {c: [list(p) for p in m["pts"][c]] for c in CLASSES}
    sizes = {c: list(m["size"][c]) for c in CLASSES}
    if aug:
        s = math.exp(random.uniform(math.log(0.6), math.log(1.5)))  # scale jitter -> robust to plan resolution
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=interp)
        wall = cv2.resize(wall, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        pts = {c: [[v * s for v in p] for p in ps] for c, ps in pts.items()}
        sizes = {c: [v * s for v in vs] for c, vs in sizes.items()}
    H, W = img.shape
    ph, pw = max(0, CROP - H), max(0, CROP - W)
    if ph or pw:
        img, wall = np.pad(img, ((0, ph), (0, pw))), np.pad(wall, ((0, ph), (0, pw)))
        H, W = img.shape
    allp = [p for c in CLASSES for p in pts[c]]
    if allp and random.random() < 0.8:  # most crops centred near something to learn from
        cx, cy = random.choice(allp)
        x0 = int(np.clip(cx - random.uniform(0.15, 0.85) * CROP, 0, W - CROP))
        y0 = int(np.clip(cy - random.uniform(0.15, 0.85) * CROP, 0, H - CROP))
    else:
        x0, y0 = random.randint(0, W - CROP), random.randint(0, H - CROP)
    img, wall = img[y0:y0 + CROP, x0:x0 + CROP], wall[y0:y0 + CROP, x0:x0 + CROP]
    cp, cs = {}, {}
    for c in CLASSES:
        keep = [i for i, (x, y) in enumerate(pts[c]) if 0 <= x - x0 < CROP and 0 <= y - y0 < CROP]
        cp[c] = [[pts[c][i][0] - x0, pts[c][i][1] - y0] for i in keep]
        cs[c] = [sizes[c][i] for i in keep]
    tgt = np.concatenate([heatmap(cp, cs, CROP, CROP), wall[None]], 0)
    if aug:
        k = random.randrange(8)
        if k >= 4:
            img, tgt = img[:, ::-1], tgt[:, :, ::-1]
        img, tgt = np.rot90(img, k % 4), np.rot90(tgt, k % 4, axes=(1, 2))
        img = img * random.uniform(0.7, 1.0)  # line darkness varies between plan styles
    return np.ascontiguousarray(img), np.ascontiguousarray(tgt)


class CropSet(torch.utils.data.Dataset):
    def __init__(self, metas, n):
        self.metas, self.n = metas, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x, y = sample_crop(random.choice(self.metas))
        return torch.from_numpy(x)[None], torch.from_numpy(y)


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def forward_full(model, ink, dev):
    """ink image (already at LONG scale) -> (N_OUT, H, W) probabilities. Pads to a multiple of 32."""
    H, W = ink.shape
    ph, pw = (-H) % 32, (-W) % 32
    x = torch.from_numpy(np.pad(ink, ((0, ph), (0, pw))))[None, None].to(dev)
    return torch.sigmoid(model(x))[0, :, :H, :W]


def peak_values(p):
    """(N_OUT, H, W) -> per heat class: tensor of peak values and their (x, y)."""
    out = {}
    for ci, c in enumerate(CLASSES):
        k = PEAK_K[c]
        mx = F.max_pool2d(p[ci][None, None], k, 1, k // 2)[0, 0]
        ys, xs = torch.where((p[ci] == mx) & (p[ci] > 0.05))
        out[c] = (p[ci][ys, xs], xs, ys)
    return out


def _suppress_detections(dets, radius):
    """Greedy score-ordered suppression in source-image pixels.

    Heat-map maxima can be several pixels apart on a single large symbol. Use the
    annotated symbol size as the suppression scale instead of the tiny heat-map
    NMS kernel, while retaining distinct nearby objects.
    """
    kept = []
    for d in sorted(dets, key=lambda x: x["score"], reverse=True):
        x, y = d["center"]
        if all(math.hypot(x - k["center"][0], y - k["center"][1]) >= radius for k in kept):
            kept.append(d)
    return kept


def _instance_size(heat, x, y, fallback, calibration=None):
    """Estimate one symbol's annotated size from its predicted Gaussian heat spot.

    Training encodes object size as heat-map sigma (sigma = 0.18 * size).
    The half-maximum diameter therefore gives a per-instance size estimate;
    clamp it around the class median because probabilities are imperfect.
    """
    h, w = heat.shape
    radius = int(np.clip(round(fallback * 0.5), 8, 36))
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = heat[y0:y1, x0:x1]
    peak = float(heat[y, x])
    if peak <= 0:
        return float(fallback)
    mask = (patch >= peak * 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    lx, ly = x - x0, y - y0
    label = int(labels[ly, lx])
    if label <= 0 or label >= n:
        return float(fallback)
    diameter = max(int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT]))
    estimate = diameter / (2.355 * 0.18)
    if calibration:
        estimate = float(calibration["slope"]) * estimate + float(calibration["intercept"])
    return float(np.clip(estimate, max(8.0, 0.40 * fallback), max(12.0, 2.5 * fallback)))


_MODEL, _MTIME, _SIZE_CAL = None, None, {}


def load_trained():
    """Load counter_out/counter.pt; reloads automatically when training writes a newer checkpoint."""
    global _MODEL, _MTIME, _SIZE_CAL
    ck = OUT / "counter.pt"
    if not ck.exists():
        raise RuntimeError("The plan reader is not trained yet (run: python3 counter.py prepare && "
                           "python3 counter.py train).")
    mt = ck.stat().st_mtime
    if _MODEL is None or mt != _MTIME:
        st = torch.load(ck, map_location="cpu", weights_only=False)  # our own checkpoint
        dev = device()
        m = build_model(pretrained=False)
        load_checkpoint_weights(m, st)
        thresholds = {c: float(st.get("thr", {}).get(c, 0.1)) for c in CLASSES}
        # Older checkpoints have no learned `other` heat-map head. Its newly
        # initialized weights produce a dense cloud of false room centres at
        # the usual 0.1 threshold, so leave that class to geometric fallback
        # regions until it has actually been trained.
        if "other" not in (st.get("classes") or []):
            thresholds["other"] = 1.1
        _MODEL, _MTIME = (m.to(dev).eval(), thresholds, st.get("size", {}), dev), mt
        _SIZE_CAL = st.get("size_calibration", {})
    return _MODEL


def _plan_roi_from_walls(wall, source_shape):
    """Estimate the drawing's building bounds from its wall map, excluding sheet legends."""
    h, w = wall.shape
    k = max(7, int(round(min(h, w) * 0.032)))
    k += 1 - (k % 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = (wall >= 0.35).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = map(int, stats[idx])
    if area < 0.012 * h * w or bw < 0.18 * w or bh < 0.18 * h or bw * bh > 0.78 * h * w:
        return None
    # Expand the wall-derived core so balcony edges and fixtures just inside
    # the exterior wall remain in scope, while sheet legends stay outside.
    px, py = int(bw * 0.06), int(bh * 0.06)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(w, x + bw + px), min(h, y + bh + py)
    source_h, source_w = source_shape[:2]
    return (x0 * source_w / w, y0 * source_h / h,
            x1 * source_w / w, y1 * source_h / h)


def analyse(rgb):
    """Public API: rgb image -> dict(
         objects={cls: [dict(box, center, score)]}  (input px),
         rooms={cls: [dict(center, score)]},
         wall=float32 wall-probability map at LONG scale, scale=s (LONG px per input px))."""
    m, thr, size, dev = load_trained()
    ink, s = resize_long(to_gray_ink(rgb))
    p = forward_full(m, ink, dev)
    wall = p[-1].detach().cpu().numpy()
    plan_roi = _plan_roi_from_walls(wall, rgb.shape)
    if plan_roi is not None:
        x0, y0, x1, y1 = plan_roi
        map_h, map_w = wall.shape
        image_h, image_w = rgb.shape[:2]
        mx0, my0 = int(x0 * map_w / image_w), int(y0 * map_h / image_h)
        mx1, my1 = int(np.ceil(x1 * map_w / image_w)), int(np.ceil(y1 * map_h / image_h))
        in_plan = np.zeros_like(wall, dtype=bool)
        in_plan[max(0, my0):min(map_h, my1), max(0, mx0):min(map_w, mx1)] = True
        wall[~in_plan] = 0.0
    pk = peak_values(p)
    objects, rooms = {}, {}
    for c in CLASSES:
        v, xs, ys = pk[c]
        keep = v >= thr[c]
        if plan_roi is not None:
            x0, y0, x1, y1 = plan_roi
            keep &= (xs / s >= x0) & (xs / s <= x1) & (ys / s >= y0) & (ys / s <= y1)
        dets = []
        half_default = size.get(c, 20) / 2
        heat = p[CLASSES.index(c)].detach().cpu().numpy() if c in OBJECTS else None
        for x, y, sc in zip(xs[keep].cpu().tolist(), ys[keep].cpu().tolist(), v[keep].cpu().tolist()):
            d = dict(center=(x / s, y / s), score=float(sc))
            if c in OBJECTS:
                # The target heat-map width is derived from each SVG symbol's
                # own bounds, so estimate each detection separately.
                obj_size = _instance_size(heat, x, y, size.get(c, 20), _SIZE_CAL.get(c))
                half = obj_size / 2
                d["size"] = obj_size / s  # input-image pixels, matching center/box
                d["box"] = ((x - half) / s, (y - half) / s, (x + half) / s, (y + half) / s)
            dets.append(d)
        if c in OBJECTS:
            # size is measured at LONG resolution; detections are returned in input pixels.
            # A modest fraction removes multiple responses on one symbol without
            # merging separate fixtures placed next to one another.
            dets = _suppress_detections(dets, max(2.0, 0.35 * size.get(c, 20) / s))
        (objects if c in OBJECTS else rooms)[c] = dets
    # Door and window classifiers can both fire on one opening. Keep the stronger
    # response when their centres nearly coincide; same-class suppression above
    # handles the common two/three-window case.
    openings = sorted(((c, d) for c in ("door", "window") for d in objects[c]),
                      key=lambda item: item[1]["score"], reverse=True)
    clean = []
    opening_radius = max(2.0, 0.22 * max(size.get("door", 20), size.get("window", 20)) / s)
    for cls, d in openings:
        if all(cls == kcls or math.dist(d["center"], kd["center"]) >= opening_radius
               for kcls, kd in clean):
            clean.append((cls, d))
    objects["door"] = [d for cls, d in clean if cls == "door"]
    objects["window"] = [d for cls, d in clean if cls == "window"]
    # Prefer a fine-tuned YOLO symbol detector when present. The heat-map model
    # remains responsible for room labels and walls, and is the fallback until
    # the YOLO weights have been trained.
    import yolo_objects
    yolo = yolo_objects.detect(rgb, roi=plan_roi)
    object_detector = yolo_objects.detector_name() if yolo is not None else "heat-map detector"
    if yolo is not None:
        objects = yolo
    return dict(objects=objects, rooms=rooms, wall=wall, scale=s,
                object_detector=object_detector)


def detect(rgb):
    """Back-compatible: objects only."""
    return analyse(rgb)["objects"]


# ----------------------------------------------------------------------------- train / eval
def count_errors(model, metas, dev, thr_grid):
    errs = {c: {t: [] for t in thr_grid} for c in CLASSES}
    model.eval()
    for m in metas:
        ink = cv2.imread(str(PREP / f"{m['key']}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        pk = peak_values(forward_full(model, ink, dev))
        for c in CLASSES:
            gt = len(m["pts"][c])
            v = pk[c][0]
            for t in thr_grid:
                errs[c][t].append(abs(int((v >= t).sum()) - gt))
    return errs


def median_sizes(metas):
    return {c: float(np.median([v for m in metas for v in m["size"][c]] or [20])) for c in CLASSES}


@torch.no_grad()
def calibrate_door_threshold(model, metas, dev, door_size):
    """Choose a door threshold by spatial precision, prioritising few false doors.

    A prediction counts as a match only when it is close to a labelled door centre.
    Among thresholds with at least 90% precision, select the one with best recall;
    if none qualifies, use the best F0.5 threshold (precision weighted).
    """
    grid = [round(float(t), 3) for t in np.arange(0.10, 0.951, 0.025)]
    totals = {t: [0, 0, 0] for t in grid}  # TP, FP, FN
    radius = max(6.0, float(door_size) * 0.45)
    model.eval()
    for meta in metas:
        ink = cv2.imread(str(PREP / f"{meta['key']}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        values, xs, ys = peak_values(forward_full(model, ink, dev))["door"]
        scores = values.cpu().numpy()
        points = np.c_[xs.cpu().numpy(), ys.cpu().numpy()]
        gt = np.asarray(meta["pts"]["door"], dtype=np.float32).reshape(-1, 2)
        for threshold in grid:
            pred = points[scores >= threshold]
            pairs = []
            if len(pred) and len(gt):
                dist = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
                pairs = sorted((float(dist[i, j]), i, j) for i, j in zip(*np.where(dist <= radius)))
            used_pred, used_gt = set(), set()
            for _, i, j in pairs:
                if i not in used_pred and j not in used_gt:
                    used_pred.add(i)
                    used_gt.add(j)
            tp = len(used_pred)
            totals[threshold][0] += tp
            totals[threshold][1] += len(pred) - tp
            totals[threshold][2] += len(gt) - tp
    rows = {}
    for t, (tp, fp, fn) in totals.items():
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f05 = 1.25 * precision * recall / max(0.25 * precision + recall, 1e-9)
        rows[t] = dict(precision=precision, recall=recall, f05=f05, tp=tp, fp=fp, fn=fn)
    eligible = [t for t in grid if rows[t]["precision"] >= 0.90]
    chosen = max(eligible, key=lambda t: (rows[t]["recall"], t)) if eligible else max(grid, key=lambda t: (rows[t]["f05"], t))
    return chosen, rows[chosen]


def train(args):
    meta = load_meta()
    tr = [m for m in meta if m["split"] == "train"]
    va = [m for m in meta if m["split"] == "val"][:args.n_val]
    dev = device()
    random.seed(0), np.random.seed(0), torch.manual_seed(0)
    model = build_model().to(dev)
    if args.resume and (OUT / "counter.pt").exists():
        prior = torch.load(OUT / "counter.pt", map_location="cpu", weights_only=False)
        load_checkpoint_weights(model, prior)
        print(f"Fine-tuning existing checkpoint from epoch {prior.get('epoch', '?')}", flush=True)
    print(f"Device {dev} | train {len(tr)} | val {len(va)} | params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    steps_per_epoch = args.crops // args.bs
    dl = torch.utils.data.DataLoader(CropSet(tr, args.crops), batch_size=args.bs, num_workers=args.workers,
                                     persistent_workers=args.workers > 0, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total, pct_start=max(0.05, 3 / total))
    OUT.mkdir(exist_ok=True)
    grid = [round(t, 2) for t in np.arange(0.1, 0.71, 0.05)]
    nh = len(CLASSES)
    best = 1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, totw = time.time(), 0.0, 0.0
        for i, (x, y) in enumerate(dl):
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            lh = focal_loss(out[:, :nh], y[:, :nh])
            lw = wall_loss(out[:, nh], y[:, nh])
            loss = lh + lw
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot, totw = tot + lh.item(), totw + lw.item()
        errs = count_errors(model, va, dev, grid)
        thr = {c: float(min(grid, key=lambda t: np.mean(errs[c][t]))) for c in CLASSES}
        mae = {c: float(np.mean(errs[c][thr[c]])) for c in CLASSES}
        exact = {c: float(np.mean(np.array(errs[c][thr[c]]) == 0)) for c in CLASSES}
        score = float(np.mean(list(mae.values())))
        print(f"epoch {ep}/{args.epochs} heat {tot / (i + 1):.3f} wall {totw / (i + 1):.3f} | {time.time() - t0:.0f}s | "
              f"val exact-count: " + " ".join(f"{c} {exact[c]:.0%}" for c in CLASSES), flush=True)
        if score < best:
            best = score
            torch.save(dict(model=model.state_dict(), thr=thr, size=median_sizes(tr), val_mae=mae, val_exact=exact,
                            epoch=ep, classes=CLASSES), OUT / "counter.pt")
            print(f"  saved (mean count error {score:.2f})", flush=True)


def calibrate():
    meta = load_meta()
    va = [m for m in meta if m["split"] == "val"]
    model, thr, size, dev = load_trained()
    threshold, stats = calibrate_door_threshold(model, va, dev, size.get("door", 20))
    checkpoint = torch.load(OUT / "counter.pt", map_location="cpu", weights_only=False)
    checkpoint["thr"]["door"] = float(threshold)
    checkpoint["door_calibration"] = dict(validation_plans=len(va), threshold=float(threshold), **stats)
    torch.save(checkpoint, OUT / "counter.pt")
    print("Door validation calibration: " + json.dumps(checkpoint["door_calibration"], indent=2), flush=True)


def evaluate():
    meta = load_meta()
    te = [m for m in meta if m["split"] == "test"]
    model, thr, _, dev = load_trained()
    errs = count_errors(model, te, dev, sorted(set(thr.values())))
    res = {c: dict(exact=float(np.mean(np.array(errs[c][thr[c]]) == 0)),
                   within1=float(np.mean(np.array(errs[c][thr[c]]) <= 1)),
                   mae=float(np.mean(errs[c][thr[c]])),
                   mean_gt=float(np.mean([len(m["pts"][c]) for m in te]))) for c in CLASSES}
    print(f"TEST ({len(te)} plans)\n" + json.dumps(res, indent=2))
    json.dump(res, open(OUT / "test_metrics.json", "w"), indent=2)


def show(path):
    rgb = cv2.imread(path, cv2.IMREAD_COLOR)[..., ::-1].copy()
    a = analyse(rgb)
    vis = rgb.copy()
    for c, ds in a["objects"].items():
        for d in ds:
            x0, y0, x1, y1 = map(int, d["box"])
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 90, 31), 2)
            cv2.putText(vis, c, (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 90, 31), 1)
    for c, ds in a["rooms"].items():
        for d in ds:
            x, y = map(int, d["center"])
            cv2.putText(vis, c.upper(), (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (47, 91, 211), 2)
    OUT.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / "show.png"), vis[..., ::-1])
    print({c: len(v) for c, v in {**a["objects"], **a["rooms"]}.items()}, "->", OUT / "show.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "train", "eval", "show", "calibrate"])
    ap.add_argument("path", nargs="?")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--crops", type=int, default=4000, help="random crops per epoch")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_val", type=int, default=150)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="fine-tune from counter_out/counter.pt")
    a = ap.parse_args()
    {"prepare": prepare, "train": lambda: train(a), "eval": evaluate, "show": lambda: show(a.path),
     "calibrate": calibrate}[a.cmd]()


if __name__ == "__main__":
    main()
