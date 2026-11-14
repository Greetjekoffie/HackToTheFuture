"""Score the Q&A pipeline on a JSONL benchmark (one {"path", "question", "answer", "unit", "type"} per line).

    python3 qa_eval.py questions.jsonl              # English questions only (others are skipped)
    python3 qa_eval.py questions.jsonl --show_all   # print every question, not just the wrong ones

Counts are correct when exact; areas when within 2 % of the true value.
Paths like data/cubicasa5k/colorful-10052.png are mapped to the unzipped CubiCasa layout
(data/cubicasa5k/cubicasa5k/colorful/10052/F1_original.png) if the flat file does not exist.
"""
import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import qa


def resolve(path):
    p = Path(path)
    if p.exists():
        return p
    m = re.search(r"cubicasa5k/(colorful|high_quality_architectural|high_quality)-(\d+)\.png$", str(path))
    if m:
        for name in ("F1_original.png", "F1_scaled.png"):
            q = Path("data/cubicasa5k/cubicasa5k") / m.group(1) / m.group(2) / name
            if q.exists():
                return q
    return None


def correct(pred, gt, unit):
    if pred is None:
        return False
    if unit == "m2":
        return abs(pred - gt) <= 0.02 * abs(gt)
    return round(pred) == round(gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--show_all", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    plans, by_kind, skipped, missing = {}, defaultdict(lambda: [0, 0]), 0, set()
    for r in rows:
        if not qa.is_english(r["question"]):
            skipped += 1
            continue
        p = resolve(r["path"])
        if p is None:
            missing.add(r["path"])
            continue
        if p not in plans:
            t = time.time()
            plans[p] = qa.FloorPlan.load(p)
            print(f"loaded {p} ({time.time() - t:.1f}s, {len(plans[p].lines)} text lines, {plans[p].text_source})")
        try:
            res = plans[p].ask(r["question"])
        except Exception as ex:
            res = dict(answer=None, explanation=f"error: {ex}")
        kind = qa.parse_question(r["question"])
        key = kind["kind"] + (f":{kind['cls']}" if "cls" in kind else "")
        ok = correct(res["answer"], r["answer"], r.get("unit"))
        by_kind[key][0] += ok
        by_kind[key][1] += 1
        if a.show_all or not ok:
            print(f"  {'OK ' if ok else 'BAD'} gt={r['answer']} pred={res['answer']} [{key}] {r['question'][:80]}")
            if not ok:
                print(f"      {res.get('explanation', '')[:200]}")
    tot = [sum(v[0] for v in by_kind.values()), sum(v[1] for v in by_kind.values())]
    print(f"\nAccuracy {tot[0]}/{tot[1]} = {tot[0] / max(1, tot[1]):.0%}  (skipped {skipped} non-English)")
    for k, (c, n) in sorted(by_kind.items()):
        print(f"  {k:16s} {c}/{n}")
    if missing:
        print("Missing files: " + ", ".join(sorted(missing)))


if __name__ == "__main__":
    main()
