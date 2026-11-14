"""Plan -> 3D building model (IFC file + a light JSON scene for the web viewer).

Uses the trained plan reader (counter.analyse):
  walls       wall-probability map -> straight wall segments (seg.py's vectoriser) -> IfcWall
  doors       detected door symbols -> IfcDoor, a leaf standing in the wall opening
  windows     detected window symbols, snapped onto the nearest wall -> IfcWindow
  fixtures    toilet / sink / shower / bathtub / sauna stove -> IfcSanitaryTerminal / IfcElectricAppliance,
              backed against the nearest wall
  rooms       detected room types -> IfcSpace: the free area around the room dot, bounded by walls
              (door openings closed), with its floor area in m²
  slab        building footprint -> IfcSlab

Every element gets an id such as "toilet-0" or "room-bedroom-1". qa.py puts these ids in its answer
evidence, so the viewer can highlight exactly the elements an answer was counted from.
Scale (pixels per metre) is estimated from the wall thickness (assumed 0.2 m), so sizes are approximate.
"""
import math
import uuid

import cv2
import numpy as np

import counter
import seg

WALL_H = 2.7
WALL_T_M = 0.2  # assumed typical wall thickness, used to estimate the drawing scale
FIXTURE = {  # width along wall, depth from wall, height  (metres)
    "toilet": (0.40, 0.70, 0.80),
    "sink": (0.60, 0.45, 0.85),
    "shower": (0.90, 0.90, 2.00),
    "bathtub": (1.70, 0.75, 0.55),
    "sauna_stove": (0.45, 0.45, 0.80),
}
IFC_OF = {"toilet": ("IfcSanitaryTerminal", "TOILETPAN"), "sink": ("IfcSanitaryTerminal", "WASHHANDBASIN"),
          "shower": ("IfcSanitaryTerminal", "SHOWER"), "bathtub": ("IfcSanitaryTerminal", "BATH"),
          "sauna_stove": ("IfcElectricAppliance", "FREESTANDINGELECTRICHEATER")}
ROOM_NAME = dict(kitchen="Kitchen", bedroom="Bedroom", bathroom="Bathroom", wc="WC", balcony="Balcony", sauna="Sauna")


def estimate_px_per_m(wall):
    """Median wall thickness (px) / assumed thickness (m)."""
    m = (wall > 0.5).astype(np.uint8)
    if m.sum() < 50:
        return 50.0
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    ridge = (d >= cv2.dilate(d, np.ones((3, 3), np.uint8))) & (d > 0.5)
    t = 2 * float(np.median(d[ridge]))
    return float(np.clip(t / WALL_T_M, 15, 400))


def build(plan_rgb, a):
    """a = counter.analyse(rgb). Returns (scene dict, element list); coordinates in metres.
    Image x -> world x, image y -> world -y (IFC), viewer uses (x, height, image-y)."""
    wall = a["wall"]
    s = a["scale"]  # LONG px per input px
    H, W = wall.shape
    ppm = estimate_px_per_m(wall)  # LONG px per metre
    m = (wall > 0.5).astype(np.uint8)
    to_m = lambda x, y: (x / ppm, y / ppm)  # LONG px -> metres (x, depth-into-image)
    els = []

    # ---- walls: horizontal / vertical runs (seg.py vectoriser) ----------------------------------------
    walls = []
    for horiz in (True, False):
        for p, t in seg._runs(m, horiz):
            x0, y0, x1, y1 = p
            walls.append(dict(a=to_m(x0, y0), b=to_m(x1, y1), t=max(t / ppm, 0.08)))
    # nearest wall pixel for every pixel (to orient fixtures and snap windows)
    _, lab = cv2.distanceTransformWithLabels(1 - m, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    wx = np.flatnonzero(m)  # wall pixels (= the zero pixels of 1 - m, each with its own label)
    lab_xy = np.zeros((lab.max() + 1, 2), np.float32)
    if len(wx):
        # label ids are assigned to the zero pixels of the input in raster order
        zero_idx = np.flatnonzero((1 - m).ravel() == 0)
        lab_flat = lab.ravel()
        lab_xy[lab_flat[zero_idx]] = np.c_[zero_idx % W, zero_idx // W]

    def nearest_wall(x, y):
        xi, yi = int(np.clip(round(x), 0, W - 1)), int(np.clip(round(y), 0, H - 1))
        if not len(wx):
            return None
        wxp, wyp = lab_xy[lab[yi, xi]]
        return float(wxp), float(wyp)

    def wall_dir(x, y):
        """Direction (unit vector) of the wall run through / nearest to point (x, y), in LONG px."""
        best, bd = None, 1e9
        for w in walls:
            ax, ay = w["a"][0] * ppm, w["a"][1] * ppm
            bx, by = w["b"][0] * ppm, w["b"][1] * ppm
            d = _pt_seg_dist(x, y, ax, ay, bx, by)
            if d < bd:
                best, bd = (bx - ax, by - ay, ax, ay, bx, by), d
        if best is None:
            return (1.0, 0.0), None, None
        L = math.hypot(best[0], best[1]) or 1
        ux, uy = best[0] / L, best[1] / L
        t = max(0.0, min(L, (x - best[2]) * ux + (y - best[3]) * uy))
        return (ux, uy), bd, (best[2] + t * ux, best[3] + t * uy)

    # ---- doors and windows ---------------------------------------------------------------------------
    openings = []
    for cls, z0, z1 in (("door", 0.0, 2.1), ("window", 0.9, 2.1)):
        width_px = max(counter_size(cls), 10)
        for i, d in enumerate(a["objects"].get(cls, [])):
            cx, cy = d["center"][0] * s, d["center"][1] * s
            (ux, uy), dist, projection = wall_dir(cx, cy)
            if dist is not None and dist < 3 * width_px and projection is not None:
                # Place each opening on the nearest wall run, not merely near its
                # nearest wall pixel. This keeps its orientation and position aligned.
                cx, cy = projection
            half = width_px / 2
            pa, pb = (cx - ux * half, cy - uy * half), (cx + ux * half, cy + uy * half)
            thick = 0.06 if cls == "door" else WALL_T_M + 0.04
            openings.append(dict(c=(cx, cy), u=(ux, uy), half=half, kind=cls))
            els.append(dict(id=f"{cls}-{i}", kind=cls, ifc="IfcDoor" if cls == "door" else "IfcWindow",
                            name=cls.title(), score=d["score"],
                            box=_seg_box(to_m(*pa), to_m(*pb), thick, z0, z1)))

    # Split wall solids around detected openings so the door/window elements sit
    # in a real gap instead of floating over a continuous wall.
    wall_i = 0
    for w in walls:
        ax, ay = w["a"][0] * ppm, w["a"][1] * ppm
        bx, by = w["b"][0] * ppm, w["b"][1] * ppm
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < 1:
            continue
        ux, uy = dx / length, dy / length
        cuts = []
        for op in openings:
            ox, oy = op["c"]
            along = (ox - ax) * ux + (oy - ay) * uy
            across = abs((ox - ax) * uy - (oy - ay) * ux)
            align = abs(ux * op["u"][0] + uy * op["u"][1])
            if across <= w["t"] * ppm / 2 + 5 and align >= 0.8 and -op["half"] < along < length + op["half"]:
                cuts.append((max(0.0, along - op["half"]), min(length, along + op["half"])))
        cuts.sort()
        merged = []
        for lo, hi in cuts:
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        cursor = 0.0
        for lo, hi in merged + [(length, length)]:
            if lo - cursor >= 2:
                start = (ax + cursor * ux, ay + cursor * uy)
                end = (ax + lo * ux, ay + lo * uy)
                els.append(dict(id=f"wall-{wall_i}", kind="wall", ifc="IfcWall", name="Wall",
                                box=_seg_box(to_m(*start), to_m(*end), w["t"] / ppm, 0.0, WALL_H)))
                wall_i += 1
            cursor = max(cursor, hi)

    # ---- fixtures ----------------------------------------------------------------------------------
    for cls, (fw, fd, fh) in FIXTURE.items():
        for i, d in enumerate(a["objects"].get(cls, [])):
            cx, cy = d["center"][0] * s, d["center"][1] * s
            nw = nearest_wall(cx, cy)
            if nw and math.hypot(nw[0] - cx, nw[1] - cy) > 1e-3:
                bx, by = nw[0] - cx, nw[1] - cy  # direction to the wall behind the fixture
                L = math.hypot(bx, by)
                bx, by = bx / L, by / L
            else:
                bx, by = 0.0, -1.0
            # three.js rotation.y = yaw maps local z to (sin yaw, cos yaw): local +z points at the wall behind
            yaw = math.atan2(bx, by)
            X, Z = to_m(cx, cy)
            ifc_cls, ptype = IFC_OF[cls]
            els.append(dict(id=f"{cls}-{i}", kind=cls, ifc=ifc_cls, ptype=ptype, name=cls.replace("_", " ").title(),
                            score=d["score"], box=dict(c=[X, fh / 2, Z], s=[fw, fh, fd], yaw=yaw)))

    # ---- rooms -------------------------------------------------------------------------------------
    door_px = max(counter_size("door"), 10)
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((int(door_px * 1.2), int(door_px * 1.2)), np.uint8))
    free = (closed == 0).astype(np.uint8)
    n, comp = cv2.connectedComponents(free, connectivity=4)
    border = set(np.unique(np.r_[comp[0], comp[-1], comp[:, 0], comp[:, -1]]).tolist())
    for cls, ds in a["rooms"].items():
        for i, d in enumerate(ds):
            cx, cy = d["center"][0] * s, d["center"][1] * s
            xi, yi = int(np.clip(cx, 0, W - 1)), int(np.clip(cy, 0, H - 1))
            k = comp[yi, xi]
            poly, area = None, None
            if k > 0 and k not in border:
                reg = (comp == k).astype(np.uint8)
                cs, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c = max(cs, key=cv2.contourArea)
                c = cv2.approxPolyDP(c, 2.0, True)[:, 0, :]
                poly = [list(to_m(float(x), float(y))) for x, y in c]
                area = float(reg.sum()) / ppm ** 2
            if poly is None:  # no closed region found: a small marker square
                X, Z = to_m(cx, cy)
                poly = [[X - .6, Z - .6], [X + .6, Z - .6], [X + .6, Z + .6], [X - .6, Z + .6]]
            els.append(dict(id=f"room-{cls}-{i}", kind="room", room=cls, ifc="IfcSpace", name=ROOM_NAME[cls],
                            score=d["score"], poly=poly, area=area, label_at=list(to_m(cx, cy))))

    # ---- slab: building footprint ------------------------------------------------------------------------
    cs, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in cs if cv2.contourArea(c) > 0.05 * max([cv2.contourArea(x) for x in cs] or [1])]
    for i, c in enumerate(big):
        c = cv2.approxPolyDP(c, 3.0, True)[:, 0, :]
        els.append(dict(id=f"slab-{i}", kind="slab", ifc="IfcSlab", name="Floor slab",
                        poly=[list(to_m(float(x), float(y))) for x, y in c]))

    scene = dict(units="m", px_per_m=ppm, size=[W / ppm, H / ppm], wall_height=WALL_H, elements=els)
    return scene


def counter_size(cls):
    try:
        return counter.load_trained()[2].get(cls, 20)
    except Exception:
        return 20


def _pt_seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0 if L2 == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def _seg_box(a, b, t, z0, z1):
    """Box around segment a-b (metres, image plane) with thickness t, from height z0 to z1."""
    (ax, az), (bx, bz) = a, b
    L = max(math.hypot(bx - ax, bz - az), 0.05)
    return dict(c=[(ax + bx) / 2, (z0 + z1) / 2, (az + bz) / 2], s=[L, z1 - z0, t],
                yaw=-math.atan2(bz - az, bx - ax))


# ----------------------------------------------------------------------------- IFC export
def write_ifc(scene, path, name="Floor plan"):
    import ifcopenshell
    import ifcopenshell.api
    import ifcopenshell.guid
    run = ifcopenshell.api.run
    f = run("project.create_file", version="IFC4")
    project = run("root.create_entity", f, ifc_class="IfcProject", name=name)
    run("unit.assign_unit", f)
    ctx = run("context.add_context", f, context_type="Model")
    body = run("context.add_context", f, context_type="Model", context_identifier="Body", target_view="MODEL_VIEW",
               parent=ctx)
    site = run("root.create_entity", f, ifc_class="IfcSite", name="Site")
    bld = run("root.create_entity", f, ifc_class="IfcBuilding", name="Building")
    sto = run("root.create_entity", f, ifc_class="IfcBuildingStorey", name="Level 0")
    run("aggregate.assign_object", f, relating_object=project, products=[site])
    run("aggregate.assign_object", f, relating_object=site, products=[bld])
    run("aggregate.assign_object", f, relating_object=bld, products=[sto])

    def place(el, x, y, z, yaw):
        c, s_ = math.cos(yaw), math.sin(yaw)
        M = np.array([[c, -s_, 0, x], [s_, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], float)
        run("geometry.edit_object_placement", f, product=el, matrix=M)

    for e in scene["elements"]:
        kw = dict(ifc_class=e["ifc"], name=e["name"])
        if e.get("ptype"):
            kw["predefined_type"] = e["ptype"]
        el = run("root.create_entity", f, **kw)
        el.Description = e["id"]  # links the IFC element to the viewer / the answer evidence
        if "box" in e:
            b = e["box"]
            L, Hh, T = b["s"]
            # viewer coords (x, up, z=image-y) -> IFC (x, -z, up). Viewer local x = (cos, -sin) in (x, z)
            # becomes (cos, sin) in IFC (x, y): the same angle.
            yaw = b["yaw"]
            cx, cy, cz = b["c"][0], -b["c"][2], b["c"][1] - Hh / 2
            rep = run("geometry.add_wall_representation", f, context=body, length=L, height=Hh, thickness=T)
            ox = cx - (L / 2) * math.cos(yaw) + (T / 2) * math.sin(yaw)
            oy = cy - (L / 2) * math.sin(yaw) - (T / 2) * math.cos(yaw)
            run("geometry.assign_representation", f, product=el, representation=rep)
            place(el, ox, oy, cz, yaw)
        else:
            depth = 0.2 if e["kind"] == "slab" else 0.02
            pts = [(float(x), -float(z)) for x, z in e["poly"]]
            if len(pts) < 3:
                continue
            rep = run("geometry.add_slab_representation", f, context=body, depth=depth, polyline=pts)
            run("geometry.assign_representation", f, product=el, representation=rep)
            place(el, 0, 0, -depth if e["kind"] == "slab" else 0.0, 0)
            if e["kind"] == "room":
                el.LongName = e["name"]
        if e["ifc"] == "IfcSpace":
            run("aggregate.assign_object", f, relating_object=sto, products=[el])
        else:
            run("spatial.assign_container", f, relating_structure=sto, products=[el])
    f.write(str(path))
    return path
