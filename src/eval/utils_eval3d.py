
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utils for VLM single-frame 3D detection evaluation (no-score friendly).
Supports two input styles:
  A) {"predict": "<text>", "label": "<text>"}
  B) {"pred": [[cls,x,y,z,w,l,h,yaw,(score)], ...], "gt": [...]}
"""
import json, re, math
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

# --------------------- Parsing ---------------------

TEXT_COUNT_RE = re.compile(r'There (?:is|are) (\d+) objects? in the current view', re.IGNORECASE)
TEXT_BOX_RE   = re.compile(
    r'\[([^,]+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*'
    r'([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]'
)

def parse_text_objects(text: str):
    """Return (declared_count, list_of_boxes as dict)."""
    if not isinstance(text, str):
        return 0, []
    m = TEXT_COUNT_RE.search(text or "")
    declared = int(m.group(1)) if m else 0
    boxes = []
    for m in TEXT_BOX_RE.findall(text or ""):
        cls = str(m[0]).strip().lower()
        x,y,z = float(m[1]), float(m[2]), float(m[3])
        w,l,h = float(m[4]), float(m[5]), float(m[6])
        yaw   = float(m[7])
        boxes.append({"cls": cls, "x":x, "y":y, "z":z, "w":w, "l":l, "h":h, "yaw":yaw, "score": 1.0})
    return declared, boxes

def _box_from_list(e):
    # Accept [cls,x,y,z,w,l,h,yaw] or [cls,x,y,z,w,l,h,yaw,score]
    if not (isinstance(e, (list, tuple)) and len(e) >= 8):
        raise ValueError(f"Invalid list box: {e}")
    cls = str(e[0]).strip().lower()
    x,y,z,w,l,h,yaw = map(float, e[1:8])
    score = float(e[8]) if len(e) >= 9 else 1.0
    return {"cls": cls, "x":x, "y":y, "z":z, "w":w, "l":l, "h":h, "yaw":yaw, "score":score}

def _box_from_dict(e):
    cls = str(e["cls"]).strip().lower()
    x,y,z = float(e["x"]), float(e["y"]), float(e["z"])
    w,l,h = float(e["w"]), float(e["l"]), float(e["h"])
    yaw   = float(e["yaw"])
    score = float(e.get("score", 1.0))
    return {"cls": cls, "x":x, "y":y, "z":z, "w":w, "l":l, "h":h, "yaw":yaw, "score":score}

def to_boxes(entry: Any) -> List[Dict]:
    out = []
    if isinstance(entry, (list, tuple)):
        for e in entry:
            if isinstance(e, (list, tuple)):
                out.append(_box_from_list(e))
            elif isinstance(e, dict):
                out.append(_box_from_dict(e))
            else:
                raise ValueError(f"Unsupported element: {e}")
    else:
        raise ValueError(f"Unsupported boxes container: {type(entry)}")
    return out

def load_dataset(path: str) -> List[Dict]:
    p = Path(path)
    data = []
    with open(p, "r", encoding="utf-8") as f:
        if p.suffix.lower() == ".jsonl":
            for line in f:
                line = line.strip()
                if not line: continue
                data.append(json.loads(line))
        else:
            data = json.load(f)
    # Normalize entries to {"gt":[dict...], "pred":[dict...], "declared_pred_count": int}
    normalized = []
    for item in data:
        if "predict" in item and "label" in item:
            dec_p, preds = parse_text_objects(item["predict"])
            dec_g, gts   = parse_text_objects(item["label"])
            normalized.append({"gt": gts, "pred": preds, "declared_pred_count": dec_p, "declared_gt_count": dec_g})
        else:
            gts   = to_boxes(item.get("gt", item.get("label", [])))
            preds = to_boxes(item.get("pred", []))
            normalized.append({"gt": gts, "pred": preds, "declared_pred_count": len(preds), "declared_gt_count": len(gts)})
    return normalized

# --------------------- Geometry ---------------------

def yaw_wrap(a: float) -> float:
    return (a + math.pi) % (2*math.pi) - math.pi

def bev_corners_xy(x: float, y: float, w: float, l: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([[ +l/2, +w/2],
                      [ +l/2, -w/2],
                      [ -l/2, -w/2],
                      [ -l/2, +w/2]], dtype=np.float64)
    R = np.array([[c, -s],[s, c]], dtype=np.float64)
    world = local @ R.T + np.array([x, y])
    return world  # 4x2

def poly_area(p: np.ndarray) -> float:
    if len(p) < 3: return 0.0
    x = p[:,0]; y = p[:,1]
    return 0.5 * float(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))

def clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    def inside(p, a, b):
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= -1e-12
    def intersect(p1, p2, a, b):
        x1,y1 = p1; x2,y2 = p2; x3,y3 = a; x4,y4 = b
        den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(den) < 1e-12:
            return p2
        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / den
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / den
        return np.array([px,py], dtype=np.float64)

    out = subject.copy()
    for i in range(len(clip)):
        a = clip[i]; b = clip[(i+1)%len(clip)]
        inp = out
        if len(inp) == 0: break
        out = []
        S = inp[-1]
        for E in inp:
            if inside(E, a, b):
                if not inside(S, a, b):
                    out.append(intersect(S, E, a, b))
                out.append(E)
            elif inside(S, a, b):
                out.append(intersect(S, E, a, b))
            S = E
        out = np.array(out, dtype=np.float64)
    return out

def bev_iou_xy(rect1: np.ndarray, rect2: np.ndarray) -> Tuple[float, float]:
    area1 = abs(poly_area(rect1))
    area2 = abs(poly_area(rect2))
    inter_poly = clip_polygon(rect1, rect2)
    inter_area = abs(poly_area(inter_poly)) if len(inter_poly) >= 3 else 0.0
    union = max(1e-9, area1 + area2 - inter_area)
    return float(inter_area / union), inter_area

def iou_3d(box_p: Dict, box_g: Dict) -> float:
    r1 = bev_corners_xy(box_p["x"], box_p["y"], box_p["w"], box_p["l"], box_p["yaw"])
    r2 = bev_corners_xy(box_g["x"], box_g["y"], box_g["w"], box_g["l"], box_g["yaw"])
    bev_iou, inter_bev = bev_iou_xy(r1, r2)
    z1_min, z1_max = box_p["z"] - box_p["h"]/2, box_p["z"] + box_p["h"]/2
    z2_min, z2_max = box_g["z"] - box_g["h"]/2, box_g["z"] + box_g["h"]/2
    z_overlap = max(0.0, min(z1_max, z2_max) - max(z1_min, z2_min))
    inter_vol = inter_bev * z_overlap
    vol1 = box_p["w"]*box_p["l"]*box_p["h"]
    vol2 = box_g["w"]*box_g["l"]*box_g["h"]
    union = max(1e-9, vol1 + vol2 - inter_vol)
    return float(inter_vol / union)

def center_distance(p: Dict, g: Dict) -> float:
    return math.hypot(p["x"] - g["x"], p["y"] - g["y"])

def ase_from_sizes(p: Dict, g: Dict) -> float:
    w1,l1,h1 = p["w"], p["l"], p["h"]
    w2,l2,h2 = g["w"], g["l"], g["h"]
    inter = max(0.0, min(w1,w2) * min(l1,l2) * min(h1,h2))
    vol1, vol2 = w1*l1*h1, w2*l2*h2
    iou = inter / max(1e-9, vol1 + vol2 - inter)
    return float(1.0 - iou)

# --------------------- Matching (Hungarian) ---------------------

BIG = 1e6

def match_per_frame(preds: List[Dict], gts: List[Dict], mode: str, delta: float, iou_th: float, class_aware: bool=True):
    """
    mode: 'center' or 'iou'
    Return: (matches[(pi,gi)], unmatched_pred_idx, unmatched_gt_idx)
    """
    if len(preds) == 0 and len(gts) == 0:
        return [], [], []
    if len(preds) == 0:
        return [], [], list(range(len(gts)))
    if len(gts) == 0:
        return [], list(range(len(preds))), []

    P,G = len(preds), len(gts)
    C = np.zeros((P,G), dtype=np.float64)
    for i,p in enumerate(preds):
        for j,g in enumerate(gts):
            if class_aware and p["cls"] != g["cls"]:
                C[i,j] = BIG
                continue
            if mode == "center":
                C[i,j] = center_distance(p,g)  # smaller is better
            else:
                C[i,j] = 1.0 - iou_3d(p,g)     # smaller is better

    row_ind, col_ind = linear_sum_assignment(C)
    used_p, used_g = set(), set()
    matches = []
    for i,j in zip(row_ind, col_ind):
        cost = C[i,j]
        if cost >= BIG/2:  # class mismatch
            continue
        if mode == "center":
            if cost <= delta + 1e-9:
                matches.append((i,j))
                used_p.add(i); used_g.add(j)
        else:
            iou = 1.0 - cost
            if iou >= iou_th - 1e-9:
                matches.append((i,j))
                used_p.add(i); used_g.add(j)
    unmatched_p = [i for i in range(P) if i not in used_p]
    unmatched_g = [j for j in range(G) if j not in used_g]
    return matches, unmatched_p, unmatched_g

# --------------------- Metrics ---------------------

def prf(tp:int, fp:int, fn:int):
    prec = tp / max(1, tp+fp)
    rec  = tp / max(1, tp+fn)
    f1   = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    return prec, rec, f1

def compute_ap_from_sorted(flags: np.ndarray, scores: np.ndarray, gt_count: int) -> float:
    """
    101-point interpolated AP. 'flags' is 1 for TP, 0 for FP.
    Assumes scores are sorted descending already.
    """
    if gt_count == 0:
        return float("nan")
    tp = flags.astype(np.float64)
    fp = 1.0 - tp
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    prec = cum_tp / np.maximum(1, cum_tp + cum_fp)
    rec  = cum_tp / float(gt_count)
    levels = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for r in levels:
        mask = rec >= r
        p = np.max(prec[mask]) if np.any(mask) else 0.0
        ap += p
    return float(ap / 101.0)

def has_scores(dataset: List[Dict]) -> bool:
    for it in dataset:
        for p in it["pred"]:
            if "score" in p and p["score"] != 1.0:
                return True
    return False

def collect_classes(dataset: List[Dict]) -> List[str]:
    s = set()
    for it in dataset:
        for g in it["gt"]:
            s.add(g["cls"])
        for p in it["pred"]:
            s.add(p["cls"])
    return sorted(list(s))
