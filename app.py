"""Local web app: upload a floor plan (PNG / JPG / PDF), ask English questions about it, and see the answer
highlighted on the plan and in a 3D model built from it (downloadable as IFC).

    .venv-yolo312/bin/python app.py  # loads counter_out/yolo_objects/weights/best.pt
    # then open http://localhost:8000

All analysis runs on this machine (OCR, the trained plan reader, question rules, IFC export). Standard
library HTTP server only. The page loads the three.js viewer library from a CDN.

    POST /api/plan   body = file, header X-Filename   -> plan id + preview
    POST /api/ask    {"id", "question"}                -> answer, explanation, evidence (image boxes + 3D ids)
    GET  /api/scene?id=...                             -> 3D scene (JSON, metres)
    GET  /api/ifc?id=...                               -> the same model as an IFC4 file
"""
import argparse
import base64
import errno
import io
import json
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import qa

WEB = Path(__file__).parent / "web"
MAX_UPLOAD = 60 * 1024 * 1024
PREVIEW_LONG_SIDE = 1800
PLANS = OrderedDict()  # id -> FloorPlan (small LRU)
LOCK = threading.Lock()  # OCR / torch models are not thread safe
_RAW_PLAN_INDEX = None


def preview_data_url(rgb):
    img = Image.fromarray(rgb)
    s = PREVIEW_LONG_SIDE / max(img.size)
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _mask_data_url(mask, color, alpha=105):
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color, alpha)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def comparison_data(fp):
    """Build model overlays and ground truth when this is an exact dataset plan."""
    global _RAW_PLAN_INDEX
    import counter

    height, width = fp.rgb.shape[:2]
    analysis = fp.analysis()
    pred_wall = cv2.resize(analysis["wall"], (width, height), interpolation=cv2.INTER_LINEAR) >= 0.5
    predicted = {
        name: [dict(box=list(map(float, d["box"])), score=float(d.get("score", 0)))
               for d in detections if d.get("box")]
        for name, detections in analysis["objects"].items()
    }
    reference_wall = np.zeros((height, width), dtype=np.uint8)
    reference = {}
    reference_rooms = {}
    reference_available = False
    status = "No matching labeled source plan was found; reference labels aren't available for this upload."

    if counter.RAW.is_dir():
        if _RAW_PLAN_INDEX is None:
            _RAW_PLAN_INDEX = {
                f"{p.parent.parent.name}_{p.parent.name}": p
                for p in counter.RAW.rglob("model.svg")
            }
        source = _RAW_PLAN_INDEX.get(Path(fp.name).stem)
        if source is not None:
            bgr = cv2.imread(str(source.parent / "F1_scaled.png"), cv2.IMREAD_COLOR)
            if bgr is not None:
                truth_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if truth_rgb.shape == fp.rgb.shape and np.mean(np.abs(truth_rgb.astype(np.int16) - fp.rgb.astype(np.int16))) <= 4:
                    ann = counter.parse_svg(source)
                    for polygon in ann["walls"]:
                        cv2.fillPoly(reference_wall, [np.round(polygon).astype(np.int32)], 1)
                    for polygon in ann["doors"]:
                        cv2.fillPoly(reference_wall, [np.round(polygon).astype(np.int32)], 0)
                    reference = {
                        name: [list(map(float, box)) for box in boxes]
                        for name, boxes in ann["objects"].items()
                    }
                    reference_rooms = {
                        name: [[[float(x), float(y)] for x, y in polygon]
                               for polygon in polygons]
                        for name, polygons in ann["rooms"].items()
                    }
                    reference_available = True
                    status = "Reference labels matched from CubiCasa5K."

    return dict(
        reference_available=reference_available, status=status,
        predicted_wall=_mask_data_url(pred_wall, (255, 89, 44)),
        reference_wall=_mask_data_url(reference_wall.astype(bool), (0, 190, 150)),
        predicted_objects=predicted, reference_objects=reference,
        predicted_rooms=analysis["rooms"], reference_rooms=reference_rooms,
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {fmt % args}")

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        query = dict(p.split("=", 1) for p in self.path.split("?", 1)[1].split("&") if "=" in p) if "?" in self.path else {}
        if path == "/api/compare":
            fp = PLANS.get(query.get("id", ""))
            if fp is None:
                return self._json(404, dict(error="Plan not found. Upload it again."))
            try:
                with LOCK:
                    result = comparison_data(fp)
                return self._json(200, result)
            except Exception as ex:
                traceback.print_exc()
                return self._json(500, dict(error=f"{type(ex).__name__}: {ex}"))
        if path in ("/api/scene", "/api/ifc"):
            fp = PLANS.get(query.get("id", ""))
            if fp is None:
                return self._json(404, dict(error="Plan not found. Upload it again."))
            try:
                with LOCK:
                    scene = fp.scene()
                if path == "/api/scene":
                    return self._json(200, scene)
                import tempfile
                import model3d
                stem = Path(fp.name).stem or "plan"
                with tempfile.TemporaryDirectory() as td:
                    out = Path(td) / f"{stem}.ifc"
                    with LOCK:
                        model3d.write_ifc(scene, out, name=stem)
                    body = out.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/x-step")
                self.send_header("Content-Disposition", f'attachment; filename="{stem}.ifc"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as ex:
                traceback.print_exc()
                self._json(500, dict(error=f"{type(ex).__name__}: {ex}"))
            return
        if path in ("/", "/index.html"):
            body = (WEB / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json(404, dict(error="not found"))

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_UPLOAD:
            raise ValueError("file too large (max 60 MB)")
        return self.rfile.read(n)

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            if path == "/api/plan":
                data = self._body()
                name = self.headers.get("X-Filename", "upload")
                t = time.time()
                with LOCK:
                    fp = qa.FloorPlan.load(data=data, name=name)
                pid = uuid.uuid4().hex[:12]
                PLANS[pid] = fp
                while len(PLANS) > 20:
                    PLANS.popitem(last=False)
                H, W = fp.rgb.shape[:2]
                self._json(200, dict(id=pid, name=name, width=W, height=H, image=preview_data_url(fp.rgb),
                                     text_source=fp.text_source, n_text=len(fp.lines),
                                     seconds=round(time.time() - t, 1)))
            elif path == "/api/ask":
                req = json.loads(self._body() or b"{}")
                fp = PLANS.get(req.get("id"))
                if fp is None:
                    return self._json(404, dict(error="Plan not found. Upload it again."))
                q = (req.get("question") or "").strip()
                if not q:
                    return self._json(400, dict(error="Empty question."))
                if not qa.is_english(q):
                    return self._json(200, dict(answer=None, unit=None, type=None, evidence=[],
                                                intent=dict(kind="unsupported"),
                                                explanation="Only English questions are supported."))
                t = time.time()
                with LOCK:
                    res = fp.ask(q)
                res["intent"] = qa.parse_question(q)
                res["seconds"] = round(time.time() - t, 1)
                self._json(200, res)
            else:
                self._json(404, dict(error="not found"))
        except Exception as ex:
            traceback.print_exc()
            self._json(500, dict(error=f"{type(ex).__name__}: {ex}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    import yolo_objects
    try:
        if yolo_objects.load_models():
            print(f"Web detector: {yolo_objects.detector_name()} ({yolo_objects.WEIGHTS})")
            if yolo_objects.TOILET_WEIGHTS.is_file():
                print(f"Toilet specialist: {yolo_objects.TOILET_WEIGHTS}")
        else:
            print("Web detector: YOLO best.pt not found; using the heat-map fallback.")
    except Exception as ex:
        raise SystemExit(f"Could not load YOLO weights for the website: {type(ex).__name__}: {ex}\n"
                         "Run the app with its YOLO-enabled virtualenv.") from ex
    try:
        srv = ThreadingHTTPServer((a.host, a.port), Handler)
    except OSError as ex:
        if ex.errno == errno.EADDRINUSE:
            raise SystemExit(f"Port {a.port} is already in use. The app may already be running at "
                             f"http://localhost:{a.port}; otherwise start another copy with --port {a.port + 1}.")
        raise
    print(f"Floor plan Q&A running on http://localhost:{a.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
