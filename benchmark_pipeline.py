"""Held-out benchmark for plan-to-model detection, wall coverage and opening sizes.

Uses prepared CubiCasa test annotations. The optional door size calibration is
fit only on validation plans; test plans are used only for the reported metrics.

Run: python3 benchmark_pipeline.py --n 30
"""
import argparse
import json

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import counter


def sample_split(meta, split, n):
    rows = [m for m in meta if m["split"] == split]
    if n and len(rows) > n:
        ids = np.linspace(0, len(rows) - 1, n).round().astype(int)
        rows = [rows[i] for i in ids]
    return rows


def match(cls, preds, gt, radius):
    gt = np.asarray(gt, np.float32).reshape(-1, 2)
    pc = np.asarray([d["center"] for d in preds], np.float32).reshape(-1, 2)
    if len(pc) and len(gt):
        distance = np.linalg.norm(pc[:, None, :] - gt[None, :, :], axis=2)
        ri, ci = linear_sum_assignment(distance)
        pairs = [(int(r), int(c), float(distance[r, c])) for r, c in zip(ri, ci)
                 if distance[r, c] <= radius]
    else:
        pairs = []
    return pairs, pc


def collect(rows, train_sizes, report_progress=True):
    obj_stats = {c: dict(tp=0, fp=0, fn=0, center_errors=[], size_pairs=[]) for c in counter.OBJECTS}
    room_stats = {c: dict(tp=0, fp=0, fn=0, center_errors=[]) for c in counter.ROOMS}
    wall_tp = wall_fp = wall_fn = 0
    door_pairs_for_fit = []
    for index, row in enumerate(rows, 1):
        path = counter.PREP / f"{row['key']}.png"
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        a = counter.analyse(rgb)
        scale = a["scale"]
        for cls in counter.OBJECTS:
            preds = a["objects"][cls]
            centers = [(d["center"][0] * scale, d["center"][1] * scale) for d in preds]
            pred_copy = [dict(d, center=center) for d, center in zip(preds, centers)]
            radius = max(10.0, 0.45 * float(train_sizes.get(cls, 20)))
            pairs, _ = match(cls, pred_copy, row["pts"][cls], radius)
            st = obj_stats[cls]
            st["tp"] += len(pairs)
            st["fp"] += len(preds) - len(pairs)
            st["fn"] += len(row["pts"][cls]) - len(pairs)
            gt_sizes = row["size"][cls]
            for pi, gi, dist in pairs:
                st["center_errors"].append(dist)
                pred_size = float(preds[pi].get("size", train_sizes.get(cls, 20))) * scale
                pair = (pred_size, float(gt_sizes[gi]))
                st["size_pairs"].append(pair)
                if cls == "door":
                    door_pairs_for_fit.append(pair)
        for cls in counter.ROOMS:
            preds = a["rooms"][cls]
            pred_copy = [dict(d, center=(d["center"][0] * scale, d["center"][1] * scale)) for d in preds]
            pairs, _ = match(cls, pred_copy, row["pts"][cls], 25.0)
            st = room_stats[cls]
            st["tp"] += len(pairs)
            st["fp"] += len(preds) - len(pairs)
            st["fn"] += len(row["pts"][cls]) - len(pairs)
            st["center_errors"].extend(dist for _, _, dist in pairs)
        pred_wall = a["wall"] >= 0.5
        gt_wall = cv2.imread(str(counter.PREP / f"{row['key']}_wall.png"), cv2.IMREAD_GRAYSCALE) >= 128
        wall_tp += int(np.logical_and(pred_wall, gt_wall).sum())
        wall_fp += int(np.logical_and(pred_wall, ~gt_wall).sum())
        wall_fn += int(np.logical_and(~pred_wall, gt_wall).sum())
        if report_progress and (index % 5 == 0 or index == len(rows)):
            print(f"{row['split']}: {index}/{len(rows)} plans", flush=True)
    return dict(objects=obj_stats, rooms=room_stats, wall=(wall_tp, wall_fp, wall_fn), door_pairs=door_pairs_for_fit)


def cls_report(stats):
    out = {}
    for cls, st in stats.items():
        precision = st["tp"] / max(1, st["tp"] + st["fp"])
        recall = st["tp"] / max(1, st["tp"] + st["fn"])
        out[cls] = dict(tp=st["tp"], fp=st["fp"], fn=st["fn"],
                        precision=round(precision, 3), recall=round(recall, 3),
                        f1=round(2 * precision * recall / max(precision + recall, 1e-9), 3),
                        center_mae_px=round(float(np.mean(st["center_errors"])), 2) if st["center_errors"] else None)
    return out


def size_report(stats, medians, calibration=None):
    out = {}
    for cls in ("door", "window"):
        pairs = stats[cls]["size_pairs"]
        if not pairs:
            out[cls] = dict(matched=0)
            continue
        pred, truth = np.asarray(pairs, np.float64).T
        if calibration and cls in calibration:
            fit = calibration[cls]
            pred = np.clip(fit["slope"] * pred + fit["intercept"],
                           max(8.0, 0.4 * medians.get(cls, 20)), max(12.0, 2.5 * medians.get(cls, 20)))
        baseline = np.full_like(truth, medians.get(cls, 20))
        out[cls] = dict(matched=int(len(truth)),
                        mae_px=round(float(np.abs(pred - truth).mean()), 2),
                        mape_pct=round(float((np.abs(pred - truth) / np.maximum(truth, 1)).mean() * 100), 2),
                        bias_px=round(float((pred - truth).mean()), 2),
                        fixed_median_mae_px=round(float(np.abs(baseline - truth).mean()), 2))
    return out


def wall_report(counts):
    tp, fp, fn = counts
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return dict(iou=round(tp / max(1, tp + fp + fn), 4), precision=round(precision, 4), recall=round(recall, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="evenly sampled plans from validation and test")
    args = ap.parse_args()
    meta = counter.load_meta()
    _, _, train_sizes, _ = counter.load_trained()
    val_rows = sample_split(meta, "val", args.n)
    test_rows = sample_split(meta, "test", args.n)
    val = collect(val_rows, train_sizes)
    val_size_before = size_report(val["objects"], train_sizes)
    door_pairs = np.asarray(val["door_pairs"], np.float64)
    calibration = {}
    calibration_saved = False
    if len(door_pairs) >= 10:
        slope, intercept = np.polyfit(door_pairs[:, 0], door_pairs[:, 1], 1)
        slope = float(np.clip(slope, 0.25, 4.0))
        fit = dict(slope=slope, intercept=float(intercept))
        candidate = size_report(val["objects"], train_sizes, {"door": fit})
        if candidate["door"]["mae_px"] < val_size_before["door"]["mae_px"]:
            calibration["door"] = fit
            checkpoint_path = counter.OUT / "counter.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint.setdefault("size_calibration", {})["door"] = fit
            torch.save(checkpoint, checkpoint_path)
            counter._MODEL, counter._MTIME, counter._SIZE_CAL = None, None, {}
            counter.load_trained()
            calibration_saved = True
    test = collect(test_rows, train_sizes)
    result = dict(
        benchmark="image-to-model benchmark on prepared CubiCasa holdout",
        validation_plans=len(val_rows), test_plans=len(test_rows),
        units="pixel measurements on the 1024-pixel long-side plan",
        validation_door_size_before=val_size_before.get("door"),
        door_size_calibration=calibration.get("door"), calibration_saved=calibration_saved,
        test=dict(object_detection=cls_report(test["objects"]),
                  room_detection=cls_report(test["rooms"]),
                  wall_segmentation=wall_report(test["wall"]),
                  opening_size=size_report(test["objects"], train_sizes)),
    )
    counter.OUT.mkdir(exist_ok=True)
    (counter.OUT / "pipeline_benchmark.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
