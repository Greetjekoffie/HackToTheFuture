"""YOLO object detection for floor-plan symbols.

Build a YOLO dataset from the CubiCasa SVG boxes already parsed by counter.py,
train with ``python3 yolo_objects.py train``, and use the saved detector through
counter.analyse.  Ultralytics is optional until the detector is trained.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

import counter

CLASSES = list(counter.OBJECTS)
DATASET = counter.OUT / "yolo_dataset"
RUN = counter.OUT / "yolo_objects"
WEIGHTS = RUN / "weights" / "best.pt"
TOILET_RUN = counter.OUT / "yolo_toilet"
TOILET_WEIGHTS = TOILET_RUN / "weights" / "best.pt"
IMAGE_SIZE = 1024
CONFIDENCE = 0.25
TOILET_CONFIDENCE = 0.08

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
    """Export annotated CubiCasa plans to the standard YOLO detection layout."""
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
        image = counter.PREP / f"{key}.png"
        svg = svg_by_key.get(key)
        if not image.exists() or svg is None:
            continue
        dest = DATASET / "images" / split / f"{key}.png"
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(image.resolve())
        ann = counter.parse_svg(svg)
        h, w = row["h"], row["w"]
        s = float(row["scale"])
        labels = []
        for cls_id, cls in enumerate(CLASSES):
            for x0, y0, x1, y1 in ann["objects"][cls]:
                x0, x1 = np.clip(np.array([x0, x1]) * s, 0, w)
                y0, y1 = np.clip(np.array([y0, y1]) * s, 0, h)
                bw, bh = x1 - x0, y1 - y0
                if bw < 2 or bh < 2:
                    continue
                xc, yc = (x0 + x1) / (2 * w), (y0 + y1) / (2 * h)
                labels.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw / w:.6f} {bh / h:.6f}")
        (DATASET / "labels" / split / f"{key}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))

    yaml = [f"path: {DATASET.resolve()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml.extend(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    (DATASET / "dataset.yaml").write_text("\n".join(yaml) + "\n")
    print(f"YOLO dataset ready at {DATASET} ({len(meta)} annotated plans)")


def train(epochs=60, batch=4, imgsz=IMAGE_SIZE, device="", weights=None):
    """Fine-tune a small pretrained YOLO model on the exported plan symbols."""
    config = DATASET / "dataset.yaml"
    if not config.exists():
        prepare()
    YOLO = _ultralytics()
    initial_weights = weights or (str(WEIGHTS) if WEIGHTS.exists() else "yolov8n.pt")
    model = YOLO(initial_weights)
    model.train(data=str(config), epochs=epochs, imgsz=imgsz, batch=batch,
                project=str(counter.OUT), name="yolo_objects", exist_ok=True,
                device=device or None, workers=0, patience=12, close_mosaic=10,
                degrees=0, translate=0.04, scale=0.25, fliplr=0.5, flipud=0.5,
                hsv_h=0, hsv_s=0, hsv_v=0.15, plots=True)
    print(f"Best weights: {WEIGHTS}")


def prepare_toilet_data(positive_plans=500, negative_plans=150):
    """Make a small, balanced single-class set so toilet symbols get enough updates."""
    config = DATASET / "dataset.yaml"
    if not config.exists():
        prepare()
    meta = json.loads((counter.PREP / "meta.json").read_text())
    focus = counter.OUT / "yolo_toilet_dataset"
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (focus / kind / split).mkdir(parents=True, exist_ok=True)

    rows = [r for r in meta if r["split"] == "train"]
    positive, negative = [], []
    for row in rows:
        label = DATASET / "labels" / "train" / f"{row['key']}.txt"
        lines = label.read_text().splitlines() if label.exists() else []
        toilet = [line for line in lines if line.split()[0] == "2"]
        (positive if toilet else negative).append((row, toilet))
    rng = random.Random(20261114)
    rng.shuffle(positive)
    rng.shuffle(negative)
    chosen = positive[:positive_plans] + negative[:negative_plans]
    for row, boxes in chosen:
        key = row["key"]
        src = DATASET / "images" / "train" / f"{key}.png"
        dst = focus / "images" / "train" / f"{key}.png"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        yolo_boxes = ["0 " + " ".join(line.split()[1:]) for line in boxes]
        (focus / "labels" / "train" / f"{key}.txt").write_text("\n".join(yolo_boxes) + ("\n" if yolo_boxes else ""))

    for row in (r for r in meta if r["split"] == "val"):
        key = row["key"]
        src = DATASET / "images" / "val" / f"{key}.png"
        if not src.exists():
            continue
        dst = focus / "images" / "val" / f"{key}.png"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        label = DATASET / "labels" / "val" / f"{key}.txt"
        lines = label.read_text().splitlines() if label.exists() else []
        toilets = ["0 " + " ".join(line.split()[1:]) for line in lines if line.split()[0] == "2"]
        (focus / "labels" / "val" / f"{key}.txt").write_text("\n".join(toilets) + ("\n" if toilets else ""))

    yaml = [f"path: {focus.resolve()}", "train: images/train", "val: images/val", "names:", "  0: toilet"]
    (focus / "dataset.yaml").write_text("\n".join(yaml) + "\n")
    print(f"Toilet-focused dataset: {len(positive[:positive_plans])} positive and "
          f"{len(negative[:negative_plans])} negative plans; validation uses all CubiCasa val plans")
    return focus / "dataset.yaml"


def train_toilet(epochs=12, batch=8, imgsz=IMAGE_SIZE, device=""):
    config = prepare_toilet_data()
    YOLO = _ultralytics()
    starting = str(TOILET_WEIGHTS) if TOILET_WEIGHTS.exists() else (str(WEIGHTS) if WEIGHTS.exists() else "yolov8n.pt")
    model = YOLO(starting)
    model.train(data=str(config), epochs=epochs, imgsz=imgsz, batch=batch,
                project=str(counter.OUT), name="yolo_toilet", exist_ok=True,
                device=device or None, workers=0, patience=6, close_mosaic=min(5, max(0, epochs - 1)),
                degrees=0, translate=0.04, scale=0.25, fliplr=0.5, flipud=0.5,
                hsv_h=0, hsv_s=0, hsv_v=0.15, plots=True)
    print(f"Toilet specialist weights: {TOILET_WEIGHTS}")


def detector_name():
    return "YOLO + toilet specialist" if TOILET_WEIGHTS.exists() else "YOLO"


def detect(rgb):
    """Return source-image detections, or None if no trained YOLO checkpoint exists."""
    global _MODEL, _MODEL_MTIME, _TOILET_MODEL, _TOILET_MODEL_MTIME
    if not WEIGHTS.exists():
        return None
    mtime = WEIGHTS.stat().st_mtime
    if _MODEL is None or mtime != _MODEL_MTIME:
        _MODEL = _ultralytics()(str(WEIGHTS))
        _MODEL_MTIME = mtime
    h, w = rgb.shape[:2]
    ink, scale = counter.resize_long(counter.to_gray_ink(rgb), counter.LONG)
    resized = np.repeat(np.clip(ink * 255, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    result = _MODEL.predict(resized, imgsz=IMAGE_SIZE, conf=CONFIDENCE, verbose=False)[0]
    out = {cls: [] for cls in CLASSES}
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        for (x0, y0, x1, y1), class_id, score in zip(boxes, classes, scores):
            cls = CLASSES[class_id]
            x0, y0, x1, y1 = (float(v / scale) for v in (x0, y0, x1, y1))
            out[cls].append(dict(box=(x0, y0, x1, y1), center=((x0 + x1) / 2, (y0 + y1) / 2),
                                 size=max(x1 - x0, y1 - y0), score=float(score), detector="YOLO"))
    if TOILET_WEIGHTS.exists():
        toilet_mtime = TOILET_WEIGHTS.stat().st_mtime
        if _TOILET_MODEL is None or toilet_mtime != _TOILET_MODEL_MTIME:
            _TOILET_MODEL = _ultralytics()(str(TOILET_WEIGHTS))
            _TOILET_MODEL_MTIME = toilet_mtime
        toilet_result = _TOILET_MODEL.predict(resized, imgsz=IMAGE_SIZE, conf=TOILET_CONFIDENCE,
                                              verbose=False)[0]
        specialist = []
        if toilet_result.boxes is not None:
            for box, score in zip(toilet_result.boxes.xyxy.cpu().numpy(), toilet_result.boxes.conf.cpu().numpy()):
                x0, y0, x1, y1 = (float(v / scale) for v in box)
                specialist.append(dict(box=(x0, y0, x1, y1), center=((x0 + x1) / 2, (y0 + y1) / 2),
                                       size=max(x1 - x0, y1 - y0), score=float(score),
                                       detector="YOLO toilet specialist"))
        out["toilet"] = specialist
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="export CubiCasa symbol boxes to YOLO format")
    tr = sub.add_parser("train", help="fine-tune YOLOv8n on the exported plans")
    tr.add_argument("--epochs", type=int, default=60)
    tr.add_argument("--batch", type=int, default=4)
    tr.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    tr.add_argument("--device", default="", help="optional torch device, for example mps or 0")
    tr.add_argument("--weights", default=None, help="optional starting checkpoint; defaults to the current best.pt")
    tr_toilet = sub.add_parser("train-toilet", help="fine-tune a balanced YOLO specialist for toilet symbols")
    tr_toilet.add_argument("--epochs", type=int, default=12)
    tr_toilet.add_argument("--batch", type=int, default=8)
    tr_toilet.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    tr_toilet.add_argument("--device", default="", help="optional torch device, for example mps or 0")
    args = ap.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "train-toilet":
        train_toilet(args.epochs, args.batch, args.imgsz, args.device)
    else:
        train(args.epochs, args.batch, args.imgsz, args.device, args.weights)


if __name__ == "__main__":
    main()
