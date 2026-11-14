"""Local web app: upload a floor plan (PNG / JPG / PDF), ask English questions about it, and see the answer
highlighted on the plan and in a 3D model built from it (downloadable as IFC).

    python3 app.py            # then open http://localhost:8000

All analysis runs on this machine (OCR, the trained plan reader, question rules, IFC export). Standard
library HTTP server only. The page loads the three.js viewer library from a CDN.

    POST /api/plan   body = file, header X-Filename   -> plan id + preview
    POST /api/ask    {"id", "question"}                -> answer, explanation, evidence (image boxes + 3D ids)
    GET  /api/scene?id=...                             -> 3D scene (JSON, metres)
    GET  /api/ifc?id=...                               -> the same model as an IFC4 file
"""
import argparse
import base64
import io
import json
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

import qa

WEB = Path(__file__).parent / "web"
MAX_UPLOAD = 60 * 1024 * 1024
PREVIEW_LONG_SIDE = 1800
PLANS = OrderedDict()  # id -> FloorPlan (small LRU)
LOCK = threading.Lock()  # OCR / torch models are not thread safe


def preview_data_url(rgb):
    img = Image.fromarray(rgb)
    s = PREVIEW_LONG_SIDE / max(img.size)
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


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
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"Floor plan Q&A running on http://localhost:{a.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
