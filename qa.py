"""Floor-plan question answering, fully local (no LLM).

    fp = FloorPlan.load("plan.pdf");  fp.ask("How many toilets (WC) are drawn on this floor plan?")

Pipeline
  1. Load: PNG/JPG -> image + OCR text lines (RapidOCR). PDF -> rendered page + the PDF's own text layer.
  2. Read the plan with the trained network (counter.py): symbols (doors, windows, toilets, sinks, showers,
     bathtubs, sauna stoves), room types (kitchen, bedroom, bathroom, WC, balcony, sauna) and walls.
  3. Build the 3D / IFC model from that (model3d.py).
  4. Parse the English question with keyword rules -> one intent -> answer + explanation + evidence.
     Every evidence item has an image box (2D overlay) and the id of the 3D element it refers to, so the
     viewer can highlight exactly what was counted.
"""
import difflib
import io
import re
import unicodedata
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

RENDER_LONG_SIDE = 2400  # px, longest side when rasterising PDFs

OUTDOOR_WORDS = ["balkon", "balcony", "balcon", "loggia", "terrasse", "terrace", "terras", "sitzplatz", "patio",
                 "garten", "garden", "jardin", "aussen", "outdoor", "veranda", "dachterrasse", "parvek"]
NON_ROOM_WORDS = ["schacht", "shaft", "gaine", "void", "vide", "leerraum", "luftraum", "installation",
                  "steigzone", "lift", "aufzug", "elevator", "ascenseur"]
KITCHEN_WORDS = ["küche", "kuche", "kitchen", "cuisine", "kochen", "keittiö", "kitchenette"]
LIVING_WORDS = ["wohnen", "living", "séjour", "sejour", "olohuone", "wohnzimmer"]

AREA_RE = re.compile(r"(\d{1,4}(?:[.,]\d{1,2})?)\s*(?:m²|m2|m\^2|qm|sqm)", re.I)
NUM_ONLY_RE = re.compile(r"^[\d\s.,:/x×+\-–()]+$")

NAMES = dict(door="door", window="window", toilet="toilet", sink="sink", shower="shower", bathtub="bathtub",
             sauna_stove="sauna stove", kitchen="kitchen", bedroom="bedroom", bathroom="bathroom", wc="WC room",
             balcony="balcony", sauna="sauna", living_room="living room", dining_room="dining room", hall="hall",
             corridor="corridor", office="office", storage="storage room", laundry="laundry room",
             utility="utility room", closet="closet", entry="entry", garage="garage", other="room")

ROOM_LABEL_PATTERNS = [
    ("wc", r"\b(wc|w c|toilet|toilette|lavatory|restroom|powder room)\b"),
    ("kitchen", r"\b(kitchen|kitchenette|cuisine|cocina|cucina|kuche|küche|køkken|kjøkken|keittiö|keittio|kh|kch|kochen)\b"),
    ("bedroom", r"\b(bed(room)?|bedrm|br|mh|sleeping room|schlafzimmer|kinderzimmer|schlafen|schlaf|makuuhuone|sovrum|soveværelse|chambre|dormitorio|camera|sz)\b"),
    ("bathroom", r"\b(bath(room)?|washroom|shower room|wet room|badezimmer|dusche|kylpyhuone|pesuhuone|badrum|baño|bagno|salle de bain|salle d eau|ph|bad|ba)\b"),
    ("living_room", r"\b(living(room)?|sitting room|lounge|family room|salon|salón|soggiorno|liv|lr|oh|ho|wz|wohnen|wohnzimmer|séjour|sejour|woonkamer|vardagsrum|olohuone|stue)\b"),
    ("dining_room", r"\b(dining(room)?|comedor|dr|esszimmer|salle à manger|ruokailuhuone|spisestue|matrum)\b"),
    ("balcony", r"\b(balcony|balcon|balkon|loggia|terrace|terrasse|terras|parveke|balkong|veranda|porch)\b"),
    ("sauna", r"\bsauna\b"),
    ("corridor", r"\b(corridor|hallway|passage|flur|gang|couloir|käytävä|kaytava|treppenhaus|porras|portaikko)\b"),
    ("hall", r"\b(hall|foyer|lobby|diele|vestibule|aula)\b"),
    ("entry", r"\b(entry|entrance|windfang|entrée|entree|entré|przedpokój|eteinen|et)\b"),
    ("office", r"\b(office|study|büro|buro|bureau|arbeitszimmer|library|bibliothek)\b"),
    ("laundry", r"\b(laundry|laund(e)?ry room|wash room|waschen|utility laundry|hwr)\b"),
    ("utility", r"\b(utility|mechanical|plant room|technical room|boiler room|technik|technicalroom)\b"),
    ("garage", r"\bgarage\b"),
    ("closet", r"\b(closet|walk[ -]?in|wardrobe|dressing room|garderobe|ankleide|vestiaire|komero|vaatehuone|pukuhuone|vh)\b"),
    ("storage", r"\b(storage|store room|storeroom|abstellraum|pantry|vorrat|cellar|basement|attic|varasto)\b"),
    ("other", r"\b(room|zimmer|habitación|habitacion|pièce|piece|space|raum|huone)\b"),
]


# ----------------------------------------------------------------------------- text helpers
def norm(s):
    s = unicodedata.normalize("NFKC", s).lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"[\"'«»“”„()\[\],;:]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s):
    return [t for t in norm(s).split(" ") if t]


def token_seq_match(label_toks, line_toks, fuzzy):
    n = len(label_toks)
    for i in range(len(line_toks) - n + 1):
        win = line_toks[i:i + n]
        if win == label_toks:
            return True
        if fuzzy and difflib.SequenceMatcher(None, " ".join(win), " ".join(label_toks)).ratio() >= 0.8:
            return True
    return False


def has_word(text, words):
    t = norm(text)
    return any(w in t for w in words)


@dataclass
class TextLine:
    text: str
    box: tuple  # x0, y0, x1, y1 in image pixels

    @property
    def center(self):
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)


# ----------------------------------------------------------------------------- the plan
@dataclass
class FloorPlan:
    name: str
    rgb: np.ndarray
    lines: list = field(default_factory=list)
    text_source: str = "ocr"
    _analysis: dict = None
    _scene: dict = None

    @classmethod
    def load(cls, path=None, data=None, name=None, ocr=True):
        """ocr=False delays OCR until a question needs the plan's text."""
        if data is None:
            data = open(path, "rb").read()
        name = name or (str(path) if path else "upload")
        if data[:5] == b"%PDF-" or name.lower().endswith(".pdf"):
            return cls._from_pdf(data, name)
        img = Image.open(io.BytesIO(data))
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
        fp = cls(name=name, rgb=np.array(img.convert("RGB")))
        fp.lines = ocr_lines(fp.rgb)
        return fp

    @classmethod
    def _from_pdf(cls, data, name):
        import pymupdf as fitz
        page = fitz.open(stream=data, filetype="pdf")[0]
        zoom = RENDER_LONG_SIDE / max(page.rect.width, page.rect.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].copy()
        lines = []
        for b in page.get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                txt = " ".join(s["text"] for s in ln["spans"]).strip()
                if txt:
                    x0, y0, x1, y1 = ln["bbox"]
                    lines.append(TextLine(txt, (x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom)))
        fp = cls(name=name, rgb=rgb, lines=lines, text_source="pdf")
        if sum(any(c.isalpha() for c in l.text) for l in lines) < 3:  # scanned PDF without a text layer
            fp.lines, fp.text_source = ocr_lines(rgb), "ocr"
        return fp

    def get_lines(self):
        if self.lines is None:
            self.lines = ocr_lines(self.rgb)
        return self.lines

    # --------------------------------------------------------------- model outputs (cached)
    def analysis(self):
        if self._analysis is None:
            import counter
            self._analysis = counter.analyse(self.rgb)
        return self._analysis

    def scene(self):
        if self._scene is None:
            import model3d
            self._scene = model3d.build(self.rgb, self.analysis())
            self._label_scene_rooms(self._scene)
        return self._scene

    def _label_scene_rooms(self, scene):
        """Attach readable room text to the nearest containing 3D room region."""
        room_elements = [e for e in scene["elements"] if e["kind"] == "room"]
        if not room_elements:
            return
        scale = self.analysis()["scale"] / scene["px_per_m"]
        non_room_ids = set()
        for line in self.room_labels():
            if not has_word(line.text, NON_ROOM_WORDS):
                continue
            point = (line.center[0] * scale, line.center[1] * scale)
            for room in room_elements:
                if room["id"].startswith("room-other-") and cv2.pointPolygonTest(
                        np.float32(room["poly"]), point, False) >= 0:
                    non_room_ids.add(room["id"])
        if non_room_ids:
            scene["elements"] = [e for e in scene["elements"] if e["id"] not in non_room_ids]
            room_elements = [e for e in room_elements if e["id"] not in non_room_ids]
        candidates = []
        for line in self.room_labels():
            label = AREA_RE.sub("", line.text).strip(" \t-–")
            if sum(ch.isalpha() for ch in label) < 2:
                continue
            kind = next((k for k, pat in ROOM_LABEL_PATTERNS if re.search(pat, norm(label))), None)
            if kind is None:
                continue
            cx, cy = line.center
            point = (cx * scale, cy * scale)
            nearby = []
            for room in room_elements:
                edge_dist = cv2.pointPolygonTest(np.float32(room["poly"]), point, True)
                if edge_dist >= -0.75:  # tolerate OCR labels sitting just outside a wall boundary
                    lx, ly = room.get("label_at", room["poly"][0])
                    dist = float(np.hypot(point[0] - lx, point[1] - ly))
                    nearby.append((max(0.0, -edge_dist), kind == "other", dist, room))
            if nearby:
                _, generic, dist, room = min(nearby, key=lambda item: (item[0], item[1], item[2]))
                candidates.append((generic, dist, room, label, kind))
        # Greedy one-to-one assignment prevents a single OCR line from naming
        # several rooms when their wall regions touch or share a component.
        assigned_rooms, assigned_labels = set(), set()
        for _, _, room, label, kind in sorted(candidates, key=lambda item: (item[0], item[1])):
            rid, key = id(room), norm(label)
            if rid in assigned_rooms or key in assigned_labels:
                continue
            room["name"] = label
            room["room"] = kind
            assigned_rooms.add(rid)
            assigned_labels.add(key)

    def room_at(self, x, y):
        """id of the 3D room (IfcSpace) whose floor contains image point (x, y), if any."""
        try:
            sc = self.scene()
        except Exception:
            return None
        s = self.analysis()["scale"]
        X, Z = x * s / sc["px_per_m"], y * s / sc["px_per_m"]
        for e in sc["elements"]:
            if e["kind"] == "room" and cv2.pointPolygonTest(np.float32(e["poly"]), (X, Z), False) >= 0:
                return e["id"]
        return None

    # --------------------------------------------------------------- answering
    def ask(self, question):
        it = parse_question(question)
        k = it["kind"]
        if k == "label":
            return self._label(it["label"])
        if k == "area":
            return self._area(it.get("label"), it.get("apartment"))
        if k == "objects":
            return self._objects(it["classes"], it.get("washbasin", False))
        if k == "room":
            return self._rooms(it["cls"])
        if k == "floors":
            return self._floors()
        if k == "apartment":
            return self._apartments()
        if k == "rooms":
            return self._all_rooms()
        return dict(answer=None, unit=None, type=None, evidence=[],
                    explanation="Sorry, I can't answer that yet. I understand counts of doors, windows, toilets, "
                                "sinks / washbasins, showers, bathtubs, saunas, kitchens, bedrooms, bathrooms, WC "
                                "rooms, balconies, floors and apartments, rooms with a given label, and areas "
                                "printed on the plan.")

    def _ev_obj(self, cls, i, d):
        return dict(box=d["box"], label=f"{NAMES[cls]} {d['score']:.2f}", id=f"{cls}-{i}")

    def _ev_room(self, cls, i, d):
        x, y = d["center"]
        r = 0.02 * max(self.rgb.shape[:2])
        return dict(box=(x - r, y - r, x + r, y + r), label=NAMES[cls], id=f"room-{cls}-{i}")

    def _objects(self, classes, washbasin=False):
        a = self.analysis()
        ev, parts, n = [], [], 0
        for c in classes:
            ds = list(enumerate(a["objects"].get(c, [])))
            if c == "sink" and washbasin:
                ds = [(i, d) for i, d in ds if not self._in_kitchen(d["center"])]
            n += len(ds)
            parts.append(f"{len(ds)} {NAMES[c]}{'s' if len(ds) != 1 else ''}")
            ev += [self._ev_obj(c, i, d) for i, d in ds]
        what = " and ".join(parts)
        extra = " (sinks in a detected kitchen are not counted as washbasins)" if washbasin else ""
        return dict(answer=n, unit="count", type="count", evidence=ev,
                    explanation=f"The plan reader found {what}{extra}. They are highlighted in the plan and in "
                                f"the 3D model.")

    def _in_kitchen(self, pt):
        rooms = self.analysis()["rooms"]
        cands = [(c, d["center"]) for c in ("kitchen", "bathroom", "wc", "sauna") for d in rooms.get(c, [])]
        if not cands:
            return False
        c, _ = min(cands, key=lambda cd: (cd[1][0] - pt[0]) ** 2 + (cd[1][1] - pt[1]) ** 2)
        return c == "kitchen"

    def _rooms(self, cls):
        scene = self.scene()
        s = self.analysis()["scale"] / scene["px_per_m"]
        ds = [e for e in scene["elements"] if e["kind"] == "room" and e.get("room") == cls]
        ev = []
        for i, room in enumerate(ds):
            x, y = room.get("label_at", room["poly"][0])
            x, y = x / s, y / s
            r = 0.02 * max(self.rgb.shape[:2])
            ev.append(dict(box=(x - r, y - r, x + r, y + r), label=room.get("name", NAMES[cls]), id=room["id"]))
        n = len(ds)
        expl = f"The plan reader found {n} {NAMES[cls]}{'s' if n != 1 else ''}"
        if cls == "sauna":  # a sauna is a sauna room or, when the room isn't marked, a sauna stove
            st = list(enumerate(self.analysis()["objects"].get("sauna_stove", [])))
            if len(st) > n:
                n = len(st)
                ev = [self._ev_obj("sauna_stove", i, d) for i, d in st]
                expl = f"Found {n} sauna stove{'s' if n != 1 else ''} (each marks a sauna)"
        return dict(answer=n, unit="count", type="count", evidence=ev,
                    explanation=expl + ", using enclosed room regions and nearby drawing labels.")

    def _floors(self):
        import model3d  # noqa: F401  (floors are separate building outlines in the wall map)
        a = self.analysis()
        m = (a["wall"] > 0.5).astype(np.uint8)
        m = cv2.dilate(m, np.ones((25, 25), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        areas = st[1:, cv2.CC_STAT_WIDTH] * st[1:, cv2.CC_STAT_HEIGHT]
        big = int((areas >= 0.25 * areas.max()).sum()) if len(areas) else 1
        sc = self.scene()
        ev = [dict(box=None, label="floor outline", id=e["id"]) for e in sc["elements"] if e["kind"] == "slab"]
        return dict(answer=max(1, big), unit="count", type="count", evidence=ev,
                    explanation=f"Counted {max(1, big)} separate building outline(s) in the wall map; each drawn "
                                f"floor is a separate outline.")

    def _label_hits(self, label):
        lt = tokens(label)
        return [l for l in self.get_lines() if token_seq_match(lt, tokens(l.text), self.text_source == "ocr")]

    def _ev_text(self, l):
        return dict(box=l.box, label=l.text, id=self.room_at(*l.center))

    def _label(self, label):
        hits = self._label_hits(label)
        src = "the PDF's text layer" if self.text_source == "pdf" else "text read by OCR"
        return dict(answer=len(hits), unit="count", type="count", evidence=[self._ev_text(l) for l in hits],
                    explanation=f"Found {len(hits)} label(s) matching '{label}' in {src}.")

    def room_labels(self):
        out = []
        for l in self.get_lines():
            t = AREA_RE.sub("", l.text).strip()
            if sum(c.isalpha() for c in t) >= 2 and not NUM_ONLY_RE.match(t) and len(t) <= 40:
                out.append(l)
        return out

    def _nearest_label(self, line, labels, max_dist):
        cx, cy = line.center
        best, bd = None, max_dist
        for l in labels:
            if l is line:
                continue
            d = np.hypot(l.center[0] - cx, (l.center[1] - cy) * 1.5)
            if d < bd:
                best, bd = l, d
        return best

    def _area(self, label=None, apartment=None):
        if apartment:
            return dict(answer=None, unit="m2", type="area", evidence=[],
                        explanation="Areas per apartment need apartment boundaries, which this version can't detect.")
        labels = self.room_labels()
        max_dist = 0.08 * max(self.rgb.shape[:2])
        used, skipped, total = [], [], 0.0
        lt = tokens(label) if label else None
        for l in self.get_lines():
            for m in AREA_RE.finditer(l.text):
                val = float(m.group(1).replace(",", "."))
                own = AREA_RE.sub("", l.text).strip()
                lab_text = own if sum(c.isalpha() for c in own) >= 2 else None
                if lab_text is None:
                    near = self._nearest_label(l, labels, max_dist)
                    lab_text = near.text if near else ""
                if lt is not None:
                    if token_seq_match(lt, tokens(lab_text), self.text_source == "ocr"):
                        used.append((lab_text, val, l))
                        total += val
                elif has_word(lab_text, OUTDOOR_WORDS + NON_ROOM_WORDS):
                    skipped.append((lab_text, val, l))
                else:
                    used.append((lab_text, val, l))
                    total += val
        if not used:
            return dict(answer=None, unit="m2", type="area", evidence=[],
                        explanation="No room areas (numbers followed by m²)" + (f" for '{label}'" if label else "")
                                    + " were found on this plan, so the area can't be read off it.")
        expl = f"Summed {len(used)} room area(s): " + ", ".join(f"{t or '?'} {v:g}" for t, v, _ in used) + "."
        if skipped:
            expl += " Excluded (outdoor / shafts / voids): " + ", ".join(f"{t} {v:g}" for t, v, _ in skipped) + "."
        return dict(answer=round(total, 2), unit="m2", type="area", explanation=expl,
                    evidence=[dict(self._ev_text(l), label=f"{v:g} m²") for _, v, l in used])

    def _apartments(self):
        kit = [e for e in self.scene()["elements"] if e["kind"] == "room" and e.get("room") == "kitchen"]
        txt = [l for l in self.room_labels() if has_word(l.text, KITCHEN_WORDS)]
        n = max(len(kit), len(txt), 1)
        ev = [dict(box=None, label=e.get("name", "Kitchen"), id=e["id"]) for e in kit] if len(kit) >= len(txt) \
            else [self._ev_text(l) for l in txt]
        return dict(answer=n, unit="count", type="count", evidence=ev,
                    explanation=f"Estimated from the number of kitchens ({n}); each apartment has one. "
                                f"Heuristic, not trained.")

    def _all_rooms(self):
        labs = [l for l in self.room_labels() if not has_word(l.text, OUTDOOR_WORDS + NON_ROOM_WORDS)]
        return dict(answer=len(labs), unit="count", type="count", evidence=[self._ev_text(l) for l in labs],
                    explanation=f"Counted {len(labs)} room name label(s): " + ", ".join(l.text for l in labs) + ".")


# ----------------------------------------------------------------------------- OCR
_OCR = None


def ocr_lines(rgb):
    global _OCR
    if _OCR is None:
        from rapidocr import EngineType, RapidOCR
        t = EngineType.TORCH
        _OCR = RapidOCR(params={"Det.engine_type": t, "Cls.engine_type": t, "Rec.engine_type": t,
                                "Global.log_level": "error"})
    H, W = rgb.shape[:2]
    s = min(3000 / max(H, W), 2.0)
    img = cv2.resize(rgb, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC) \
        if abs(s - 1) > 0.05 else rgb
    r = _OCR(img[..., ::-1])
    out = []
    if r.txts is None:
        return out
    for txt, box, sc in zip(r.txts, r.boxes, r.scores):
        if sc < 0.5 or not txt.strip():
            continue
        b = np.array(box) / s
        out.append(TextLine(txt.strip(), (float(b[:, 0].min()), float(b[:, 1].min()),
                                          float(b[:, 0].max()), float(b[:, 1].max()))))
    return out


# ----------------------------------------------------------------------------- question parsing (English)
ROOM_PATTERNS = [  # order matters: "bathrooms ... not counting separate WC rooms" is about bathrooms
    ("bathroom", r"\bbathrooms?\b|\bwashrooms?\b"),
    ("wc", r"\bwc rooms?\b|\btoilet rooms?\b|\bseparate wcs?\b"),
    ("bedroom", r"\bbedrooms?\b"),
    ("kitchen", r"\bkitchens?\b"),
    ("living_room", r"\bliving rooms?\b|\blounges?\b|\bfamily rooms?\b"),
    ("dining_room", r"\bdining rooms?\b|\bdining areas?\b"),
    ("balcony", r"\bbalcon(y|ies)\b"),
    ("sauna", r"\bsaunas?\b"),
    ("corridor", r"\bcorridors?\b|\bhallways?\b"),
    ("hall", r"\bhalls?\b|\bfoyers?\b|\blobbies\b"),
    ("entry", r"\bentries\b|\bentry rooms?\b|\bentrances?\b"),
    ("office", r"\boffices?\b|\bstudies\b|\blibraries\b"),
    ("laundry", r"\blaundries\b|\blaundry rooms?\b"),
    ("utility", r"\butility rooms?\b|\bmechanical rooms?\b|\bboiler rooms?\b"),
    ("storage", r"\bstorage rooms?\b|\bstorerooms?\b|\bpantries\b"),
    ("closet", r"\bclosets?\b|\bwardrobes?\b|\bdressing rooms?\b"),
    ("garage", r"\bgarages?\b"),
]


def parse_question(q):
    ql = q.lower()
    # greedy up to the last closing quote, so "labelled 'Cage d'escalier'" keeps its apostrophe
    lab = re.search(r"(?:labell?ed|called|named|marked)\s*(?:as\s*)?['\"‘«“„](.+)['\"’»”“]", q, re.I)
    if re.search(r"\barea\b|m²|\bm2\b|square met|\bsqm\b|floor space", ql):
        apt = re.search(r"\bapartment ([a-z])\b", ql)
        room = re.search(r"\broom\s+(?:[\d.]+\s*)?\(?['‘«“]([^'’»”]+)['’»”]", q, re.I)
        name = lab.group(1).strip() if lab else room.group(1).strip() if room else None
        return dict(kind="area", label=name, apartment=apt.group(1).upper() if apt else None)
    if lab:
        return dict(kind="label", label=lab.group(1).strip())
    if re.search(r"\b(floors|storeys|stories|levels)\b", ql) and "how many" in ql:
        return dict(kind="floors")
    if re.search(r"\bsinks?\b|\bwash ?-?basins?\b|\blavabos?\b", ql):  # before rooms: "kitchen and bathroom combined"
        only_wash = not re.search(r"\bsinks?\b|kitchen", ql)
        return dict(kind="objects", classes=["sink"], washbasin=only_wash)
    for cls, pat in ROOM_PATTERNS:
        if re.search(pat, ql):
            return dict(kind="room", cls=cls)
    if re.search(r"\bapartments?\b|\bresidential units?\b|\bdwellings?\b|\bflats\b|\bhousing units?\b", ql):
        return dict(kind="apartment")
    has_shower, has_tub = re.search(r"\bshowers?\b", ql), re.search(r"\bbath ?tubs?\b|\bbaths\b", ql)
    if has_shower or has_tub:
        return dict(kind="objects", classes=(["shower"] if has_shower else []) + (["bathtub"] if has_tub else []))
    if re.search(r"\btoilets?\b|\bwcs?\b|\bw\.c\.|\blavator(y|ies)\b|\btoilet bowls?\b", ql):
        return dict(kind="objects", classes=["toilet"])
    if re.search(r"\bsinks?\b|\bwash ?-?basins?\b|\blavabos?\b|\bbasins?\b", ql):
        only_wash = not re.search(r"\bsinks?\b|kitchen", ql)
        return dict(kind="objects", classes=["sink"], washbasin=only_wash)
    if re.search(r"\bdoors?\b|\bdoorways?\b", ql):
        return dict(kind="objects", classes=["door"])
    if re.search(r"\bwindows?\b", ql):
        return dict(kind="objects", classes=["window"])
    if re.search(r"\bhow many rooms\b|\bnumber of rooms\b", ql):
        return dict(kind="rooms")
    return dict(kind="unknown")


def is_english(q):
    return bool(re.search(r"\b(how many|what is|what's|which|total|are there|is there|how much)\b", q.lower()))
