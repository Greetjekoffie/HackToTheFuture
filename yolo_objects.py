"""YOLO object detection for floor-plan symbols.

Build a YOLO dataset from the CubiCasa SVG boxes already parsed by counter.py,
train with ``python3 yolo_objects.py train``, and use the saved detector through
counter.analyse. Tile sizing, overlap ownership, and small-plan upscaling follow
the approach in AistisV/yolo-segmentation-testing. Ultralytics is optional until
the detector is trained.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import counter

CLASSES = list(counter.OBJECTS)
DATASET = counter.OUT / "yolo_tile_dataset"
RUN = counter.OUT / "yolo_objects"
WEIGHTS = RUN / "weights" / "best.pt"
TOILET_RUN = counter.OUT / "yolo_toilet"
TOILET_WEIGHTS = TOILET_RUN / "weights" / "best.pt"
TOILET_DATASET = counter.OUT / "yolo_toilet_tiles_v2"
TILE_SIZE = 640
TILE_STRIDE = 480
IMAGE_SIZE = TILE_SIZE
PREDICT_BATCH = 16
# Small symbols disappear when a whole drawing is squeezed into one detector
# input. These cutoffs follow the reference pipeline; tile ownership handles
# duplicate predictions in overlaps, while 3D evidence retains each score.
CONFIDENCE = 0.05
TOILET_CONFIDENCE = 0.05

_MODEL = None
_MODEL_MTIME = None
_TOILET_MODEL = None
_TOILET_MODEL_MTIME = None


def _ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the YOLO dependency with `python3 -m pip install ultralytics`.") from exc
    return YOLO


def prepare():
    """Export full-resolution CubiCasa annotations as overlapping 640 px tiles."""
    meta_path = counter.PREP / "meta.json"
    if not meta_path.exists():
        raise RuntimeError("Prepared CubiCasa data is missing. Run `python3 counter.py prepare` first.")
    meta = json.loads(meta_path.read_text())
    svg_by_key = {f"{p.parent.parent.name}_{p.parent.name}": p for p in counter.RAW.rglob("model.svg")}
    for split in ("train", "val", "test"):
        (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    for row in meta:
        key, split = row["key"], row["split"]
        svg = svg_by_key.get(key)
        source = (svg.parent / "F1_scaled.png") if svg else None
        if source is None or not source.exists():
            continue
        rgb = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        ink = counter.to_gray_ink(rgb[..., ::-1])
        h, w = ink.shape
        scale = max(1, round(2000 / max(h, w))) if max(h, w) < 1200 else 1
        if scale > 1:
            ink = cv2.resize(ink, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        image = np.repeat(np.clip(ink * 255, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
        ann = counter.parse_svg(svg)
        h, w = image.shape[:2]
        for y in _tile_starts(h):
            for x in _tile_starts(w):
                labels = []
                for cls_id, cls in enumerate(CLASSES):
                    for bx0, by0, bx1, by1 in ann["objects"][cls]:
                        bx0, by0, bx1, by1 = bx0 * scale, by0 * scale, bx1 * scale, by1 * scale
                        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
                        if not (x <= cx < x + TILE_SIZE and y <= cy < y + TILE_SIZE):
                            continue
                        x0, y0 = max(0, bx0 - x), max(0, by0 - y)
                        x1, y1 = min(TILE_SIZE, bx1 - x), min(TILE_SIZE, by1 - y)
                        bw, bh = x1 - x0, y1 - y0
                        if bw < 2 or bh < 2:
                            continue
                        labels.append(f"{cls_id} {(cx - x) / TILE_SIZE:.6f} "
                                      f"{(cy - y) / TILE_SIZE:.6f} {bw / TILE_SIZE:.6f} {bh / TILE_SIZE:.6f}")
                if not labels and random.Random(f"{key}:{x}:{y}").random() > 0.15:
                    continue
                stem = f"{key}_{x}_{y}"
                dest = DATASET / "images" / split / f"{stem}.png"
                crop = image[y:min(y + TILE_SIZE, h), x:min(x + TILE_SIZE, w)]
                if crop.shape[:2] != (TILE_SIZE, TILE_SIZE):
                    crop = cv2.copyMakeBorder(crop, 0, TILE_SIZE - crop.shape[0], 0,
                                              TILE_SIZE - crop.shape[1], cv2.BORDER_CONSTANT, value=0)
                Image.fromarray(crop).save(dest)
                (DATASET / "labels" / split / f"{stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))

    yaml = [f"path: {DATASET.resolve()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml.extend(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    (DATASET / "dataset.yaml").write_text("\n".join(yaml) + "\n")
    print(f"YOLO tile dataset ready at {DATASET} ({len(meta)} annotated plans)")


def _balanced_tile_list(split, limit, seed):
    """Return a deterministic, class-balanced list of tile paths for quick training."""
    label_files = sorted((DATASET / "labels" / split).glob("*.txt"))
    by_class = {i: [] for i in range(len(CLASSES))}
    negatives = []
    for label in label_files:
        ids = {int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()}
        if ids:
            for class_id in ids:
                if class_id in by_class:
                    by_class[class_id].append(label.stem)
        else:
            negatives.append(label.stem)
    rng = random.Random(seed)
    for stems in by_class.values():
        rng.shuffle(stems)
    rng.shuffle(negatives)

    positive_limit = max(1, int(limit * 0.9))
    cursor = {class_id: 0 for class_id in by_class}
    selected, class_hits = set(), {class_id: 0 for class_id in by_class}
    while len(selected) < positive_limit:
        available = [c for c, stems in by_class.items() if cursor[c] < len(stems)]
        if not available:
            break
        cls = min(available, key=lambda c: class_hits[c] / max(1, len(by_class[c])))
        stems = by_class[cls]
        candidate = None
        while cursor[cls] < len(stems) and candidate is None:
            name = stems[cursor[cls]]
            cursor[cls] += 1
            if name not in selected:
                candidate = name
        if candidate is None:
            continue
        selected.add(candidate)
        label_path = DATASET / "labels" / split / f"{candidate}.txt"
        for line in label_path.read_text().splitlines():
            if line.strip():
                class_id = int(line.split()[0])
                if class_id in class_hits:
                    class_hits[class_id] += 1

    selected = list(selected)
    rng.shuffle(selected)
    negative_count = min(len(negatives), limit - len(selected))
    selected.extend(negatives[:negative_count])
    return [str((DATASET / "images" / split / f"{name}.png").resolve()) for name in selected]


def _balanced_dataset_config(max_train_tiles, max_val_tiles):
    train_list = DATASET / "train_balanced.txt"
    val_list = DATASET / "val_balanced.txt"
    train_list.write_text("\n".join(_balanced_tile_list("train", max_train_tiles, 640)) + "\n")
    val_list.write_text("\n".join(_balanced_tile_list("val", max_val_tiles, 641)) + "\n")
    config = DATASET / "balanced_dataset.yaml"
    yaml = [f"path: {DATASET.resolve()}", f"train: {train_list.resolve()}",
            f"val: {val_list.resolve()}", "names:"]
    yaml.extend(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    config.write_text("\n".join(yaml) + "\n")
    return config


def _training_device(requested):
    mps_available = bool(torch.backends.mps.is_available())
    if requested == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but is unavailable in this Python environment. "
                           "Check torch.backends.mps.is_available() before training.")
    if requested:
        return requested
    if mps_available:
        return "mps"
    if torch.cuda.is_available():
        return "0"
    raise RuntimeError("No supported GPU is available. This run would fall back to CPU; "
                       "pass --device cpu only if you intend to train on CPU.")


def train(epochs=8, batch=8, imgsz=IMAGE_SIZE, device="", weights=None,
          max_train_tiles=3000, max_val_tiles=1000, workers=2):
    """Fine-tune YOLO on a balanced tile subset; increase the limits for GPU runs."""
    config = DATASET / "dataset.yaml"
    if not config.exists():
        prepare()
    config = _balanced_dataset_config(max_train_tiles, max_val_tiles)
    YOLO = _ultralytics()
    initial_weights = weights or (str(WEIGHTS) if WEIGHTS.exists() else "yolov8n.pt")
    model = YOLO(initial_weights)
    model.train(data=str(config), epochs=epochs, imgsz=imgsz, batch=batch,
                project=str(counter.OUT), name="yolo_objects", exist_ok=True,
                device=_training_device(device), workers=workers, patience=max(3, min(8, epochs // 2)),
                close_mosaic=min(10, max(0, epochs - 1)),
                degrees=0, translate=0.04, scale=0.25, fliplr=0.5, flipud=0.5,
                hsv_h=0, hsv_s=0, hsv_v=0.15, plots=True)
    print(f"Best weights: {WEIGHTS}")


def prepare_toilet_data(max_positive_tiles=1000, negative_ratio=2):
    """Make a focused toilet-tile set, retaining hard background examples."""
    config = DATASET / "dataset.yaml"
    if not config.exists():
        prepare()
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (TOILET_DATASET / kind / split).mkdir(parents=True, exist_ok=True)
    by_split = {split: {"positive": {}, "negative": {}} for split in ("train", "val")}
    for split in ("train", "val"):
        for label in (DATASET / "labels" / split).glob("*.txt"):
            lines = label.read_text().splitlines()
            toilet = ["0 " + " ".join(line.split()[1:]) for line in lines
                      if line.split() and line.split()[0] == "2"]
            group = "positive" if toilet else "negative"
            by_split[split][group][label.stem] = toilet
    rng = random.Random(20261114)
    positive, negative = by_split["train"]["positive"], by_split["train"]["negative"]
    train_pos = list(positive.keys())
    train_neg = list(negative.keys())
    rng.shuffle(train_pos)
    rng.shuffle(train_neg)
    selected_train = train_pos[:max_positive_tiles]
    selected_train += train_neg[:min(len(train_neg), len(selected_train) * negative_ratio)]
    val_positive, val_negative = by_split["val"]["positive"], by_split["val"]["negative"]
    selected_val = list(val_positive.keys()) + list(val_negative.keys())
    for split, stems in (("train", selected_train), ("val", selected_val)):
        src_split = "train" if split == "train" else "val"
        split_pos, split_neg = ((positive, negative) if split == "train"
                                else (val_positive, val_negative))
        for stem in stems:
            src = DATASET / "images" / src_split / f"{stem}.png"
            if not src.exists():
                continue
            source_boxes = split_pos if stem in split_pos else split_neg
            boxes = source_boxes[stem]
            dst = TOILET_DATASET / "images" / split / f"{stem}.png"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
            (TOILET_DATASET / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(boxes) + ("\n" if boxes else ""))

    yaml = [f"path: {TOILET_DATASET.resolve()}", "train: images/train", "val: images/val", "names:", "  0: toilet"]
    (TOILET_DATASET / "dataset.yaml").write_text("\n".join(yaml) + "\n")
    print(f"Toilet-focused tiles: {len(selected_train)} train and {len(selected_val)} validation candidates")
    return TOILET_DATASET / "dataset.yaml"


def train_toilet(epochs=8, batch=8, imgsz=IMAGE_SIZE, device="", workers=2):
    config = prepare_toilet_data()
    YOLO = _ultralytics()
    starting = str(TOILET_WEIGHTS) if TOILET_WEIGHTS.exists() else (str(WEIGHTS) if WEIGHTS.exists() else "yolov8n.pt")
    model = YOLO(starting)
    model.train(data=str(config), epochs=epochs, imgsz=imgsz, batch=batch,
                project=str(counter.OUT), name="yolo_toilet", exist_ok=True,
                device=_training_device(device), workers=workers, patience=max(3, min(6, epochs // 2)),
                close_mosaic=min(5, max(0, epochs - 1)),
                degrees=0, translate=0.04, scale=0.25, fliplr=0.5, flipud=0.5,
                hsv_h=0, hsv_s=0, hsv_v=0.15, plots=True)
    print(f"Toilet specialist weights: {TOILET_WEIGHTS}")


def detector_name():
    return "YOLO tile detector + toilet specialist" if TOILET_WEIGHTS.exists() else "YOLO tile detector"


def _tile_starts(length, tile=TILE_SIZE, stride=TILE_STRIDE):
    if length <= tile:
        return [0]
    return sorted(set(list(range(0, length - tile, stride)) + [length - tile]))


def _predict_tiled(model, image, scale, confidence, detector, classes=None):
    """Run overlapping tiles and emit each instance from its nearest tile centre.

    image is the full-resolution, white-background ink raster. Returned boxes
    and centres use the uploaded image's pixel coordinates.
    """
    h, w = image.shape[:2]
    tiles = [(x, y) for y in _tile_starts(h) for x in _tile_starts(w)]
    tile_centres = [(x + TILE_SIZE / 2, y + TILE_SIZE / 2) for x, y in tiles]
    out = []
    for start in range(0, len(tiles), PREDICT_BATCH):
        batch_tiles = tiles[start:start + PREDICT_BATCH]
        crops = []
        for x, y in batch_tiles:
            crop = image[y:min(y + TILE_SIZE, h), x:min(x + TILE_SIZE, w)]
            if crop.shape[:2] != (TILE_SIZE, TILE_SIZE):
                crop = cv2.copyMakeBorder(crop, 0, TILE_SIZE - crop.shape[0], 0,
                                          TILE_SIZE - crop.shape[1], cv2.BORDER_CONSTANT, value=0)
            crops.append(np.repeat(crop[..., None], 3, axis=2))
        result = model.predict(crops, imgsz=TILE_SIZE,
                               conf=confidence, verbose=False)
        for offset, r in enumerate(result):
            tile_index = start + offset
            tx, ty = tiles[tile_index]
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            scores = r.boxes.conf.cpu().numpy()
            for (x0, y0, x1, y1), class_id, score in zip(boxes, class_ids, scores):
                if classes is not None and class_id not in classes:
                    continue
                cx, cy = tx + float((x0 + x1) / 2), ty + float((y0 + y1) / 2)
                owner = min(range(len(tile_centres)), key=lambda i:
                            (tile_centres[i][0] - cx) ** 2 + (tile_centres[i][1] - cy) ** 2)
                if owner != tile_index:
                    continue
                ox0, oy0 = max(0.0, tx + float(x0)), max(0.0, ty + float(y0))
                ox1, oy1 = min(float(w), tx + float(x1)), min(float(h), ty + float(y1))
                ox0, oy0, ox1, oy1 = (v / scale for v in (ox0, oy0, ox1, oy1))
                out.append((class_id, dict(box=(ox0, oy0, ox1, oy1),
                                           center=((ox0 + ox1) / 2, (oy0 + oy1) / 2),
                                           size=max(ox1 - ox0, oy1 - oy0), score=float(score),
                                           detector=detector)))
    return out


def _merge_toilet_candidates(primary, specialist):
    """Union toilet candidates from the shared model and focused model once each."""
    ranked = sorted(primary + specialist, key=lambda d: d[1]["score"], reverse=True)
    kept = []
    for class_id, candidate in ranked:
        x0, y0, x1, y1 = candidate["box"]
        cx, cy = candidate["center"]
        size = max(candidate["size"], 1.0)
        duplicate = False
        for _, other in kept:
            a0, b0, a1, b1 = other["box"]
            ox, oy = other["center"]
            intersection = max(0.0, min(x1, a1) - max(x0, a0)) * max(0.0, min(y1, b1) - max(y0, b0))
            union = max(1.0, (x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - intersection)
            close = np.hypot(cx - ox, cy - oy) < max(6.0, 0.35 * max(size, other["size"]))
            if intersection / union >= 0.15 or close:
                duplicate = True
                break
        if not duplicate:
            kept.append((class_id, candidate))
    return [d for _, d in kept]


def detect(rgb, roi=None):
    """Return source-image detections, or None if no trained YOLO checkpoint exists."""
    if not load_models():
        return None
    h, w = rgb.shape[:2]
    ink = counter.to_gray_ink(rgb)
    # Match the reference's small-plan upscaling so symbols occupy enough
    # pixels for detection. Never downscale large drawings before tiling.
    scale = max(1, round(2000 / max(h, w))) if max(h, w) < 1200 else 1
    if scale > 1:
        ink = cv2.resize(ink, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    image = np.clip(ink * 255, 0, 255).astype(np.uint8)
    out = {cls: [] for cls in CLASSES}
    def in_plan(detection):
        if roi is None:
            return True
        x0, y0, x1, y1 = roi
        cx, cy = detection["center"]
        return x0 <= cx <= x1 and y0 <= cy <= y1

    primary_detections = [d for d in _predict_tiled(_MODEL, image, scale, CONFIDENCE, "YOLO")
                          if in_plan(d[1])]
    for class_id, detection in primary_detections:
        out[CLASSES[class_id]].append(detection)
    if _TOILET_MODEL is not None:
        toilet_detections = _predict_tiled(_TOILET_MODEL, image, scale, TOILET_CONFIDENCE,
                                           "YOLO toilet specialist")
        toilet_detections = [d for d in toilet_detections if in_plan(d[1])]
        primary = [(CLASSES.index("toilet"), d) for d in out["toilet"]]
        out["toilet"] = _merge_toilet_candidates(primary, toilet_detections)
    return out


def load_models():
    """Load the current best checkpoints; return False when no base detector exists."""
    global _MODEL, _MODEL_MTIME, _TOILET_MODEL, _TOILET_MODEL_MTIME
    if not WEIGHTS.is_file():
        return False
    mtime = WEIGHTS.stat().st_mtime
    if _MODEL is None or mtime != _MODEL_MTIME:
        _MODEL = _ultralytics()(str(WEIGHTS))
        _MODEL_MTIME = mtime
    if TOILET_WEIGHTS.is_file():
        toilet_mtime = TOILET_WEIGHTS.stat().st_mtime
        if _TOILET_MODEL is None or toilet_mtime != _TOILET_MODEL_MTIME:
            _TOILET_MODEL = _ultralytics()(str(TOILET_WEIGHTS))
            _TOILET_MODEL_MTIME = toilet_mtime
    else:
        _TOILET_MODEL = None
        _TOILET_MODEL_MTIME = None
    return True


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="export CubiCasa symbol boxes to YOLO format")
    tr = sub.add_parser("train", help="fine-tune YOLOv8n on the exported plans")
    tr.add_argument("--epochs", type=int, default=8)
    tr.add_argument("--batch", type=int, default=8)
    tr.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    tr.add_argument("--device", default="", help="optional torch device, for example mps or 0")
    tr.add_argument("--workers", type=int, default=2)
    tr.add_argument("--max-train-tiles", type=int, default=3000)
    tr.add_argument("--max-val-tiles", type=int, default=1000)
    tr.add_argument("--weights", default=None, help="optional starting checkpoint; defaults to the current best.pt")
    tr_toilet = sub.add_parser("train-toilet", help="fine-tune a balanced YOLO specialist for toilet symbols")
    tr_toilet.add_argument("--epochs", type=int, default=8)
    tr_toilet.add_argument("--batch", type=int, default=8)
    tr_toilet.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    tr_toilet.add_argument("--device", default="", help="optional torch device, for example mps or 0")
    tr_toilet.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "train-toilet":
        train_toilet(args.epochs, args.batch, args.imgsz, args.device, args.workers)
    else:
        train(args.epochs, args.batch, args.imgsz, args.device, args.weights,
              args.max_train_tiles, args.max_val_tiles, args.workers)


if __name__ == "__main__":
    main()
