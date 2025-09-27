#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, math, numpy as np
from collections import defaultdict
from utils_eval3d import load_dataset, match_per_frame, prf, center_distance, ase_from_sizes, yaw_wrap, collect_classes

def eval_f1_and_geo(dataset, delta: float = 5.0):
    per_cls_counts = defaultdict(lambda: {"tp":0,"fp":0,"fn":0})
    per_cls_geo    = defaultdict(lambda: {"ATE":[], "ASE":[], "AOE":[]})
    for item in dataset:
        # bucket by class
        g_by, p_by = defaultdict(list), defaultdict(list)
        for g in item["gt"]:   g_by[g["cls"]].append(g)
        for p in item["pred"]: p_by[p["cls"]].append(p)
        for c in set(list(g_by.keys()) + list(p_by.keys())):
            gtc = g_by.get(c, []); prc = p_by.get(c, [])
            matches, u_p, u_g = match_per_frame(prc, gtc, mode="center", delta=delta, iou_th=0.2, class_aware=True)
            per_cls_counts[c]["tp"] += len(matches)
            per_cls_counts[c]["fp"] += len(u_p)
            per_cls_counts[c]["fn"] += len(u_g)
            for i,j in matches:
                p, g = prc[i], gtc[j]
                per_cls_geo[c]["ATE"].append(center_distance(p,g))
                per_cls_geo[c]["ASE"].append(ase_from_sizes(p,g))
                per_cls_geo[c]["AOE"].append(abs(yaw_wrap(p["yaw"]-g["yaw"])))

    # micro/macro
    micro_tp = micro_fp = micro_fn = 0
    per_class_prf = {}
    precs, recs, f1s = [], [], []
    for c, d in per_cls_counts.items():
        P,R,F = prf(d["tp"], d["fp"], d["fn"])
        per_class_prf[c] = {"Precision":P, "Recall":R, "F1":F, "TP":d["tp"], "FP":d["fp"], "FN":d["fn"]}
        micro_tp += d["tp"]; micro_fp += d["fp"]; micro_fn += d["fn"]
        precs.append(P); recs.append(R); f1s.append(F)
    microP, microR, microF1 = prf(micro_tp, micro_fp, micro_fn)
    macroP = float(np.mean(precs)) if len(precs)>0 else 0.0
    macroR = float(np.mean(recs))  if len(recs)>0 else 0.0
    macroF = float(np.mean(f1s))   if len(f1s)>0 else 0.0

    # geo overall & per-class
    geo_overall = {"mATE": np.nan, "mASE": np.nan, "mAOE": np.nan}
    allA, allS, allO = [], [], []
    geo_per_class = {}
    for c, dd in per_cls_geo.items():
        a = float(np.mean(dd["ATE"])) if len(dd["ATE"])>0 else float('nan')
        s = float(np.mean(dd["ASE"])) if len(dd["ASE"])>0 else float('nan')
        o = float(np.mean(dd["AOE"])) if len(dd["AOE"])>0 else float('nan')
        geo_per_class[c] = {"ATE":a, "ASE":s, "AOE":o}
        if not math.isnan(a): allA.append(a)
        if not math.isnan(s): allS.append(s)
        if not math.isnan(o): allO.append(o)
    if len(allA)>0: geo_overall["mATE"] = float(np.mean(allA))
    if len(allS)>0: geo_overall["mASE"] = float(np.mean(allS))
    if len(allO)>0: geo_overall["mAOE"] = float(np.mean(allO))

    return {"micro":{"P":microP,"R":microR,"F1":microF1},
            "macro":{"P":macroP,"R":macroR,"F1":macroF},
            "per_class": per_class_prf,
            "geo_overall": geo_overall,
            "geo_per_class": geo_per_class}

def _ap_101(flags_sorted: np.ndarray, gt_count: int) -> float:
    tp = flags_sorted.astype(np.float64)
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

def _nanmean_list(vals):
    arr = np.array(vals, dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size > 0 else float('nan')

def eval_multi_deltas(dataset, deltas):
    """
    Run eval_f1_and_geo for multiple deltas and return:
      - per-delta results
      - averaged metrics across deltas (simple arithmetic mean over deltas)
    """
    per_delta = {}
    for d in deltas:
        per_delta[d] = eval_f1_and_geo(dataset, delta=d)

    # Average across deltas
    # 1) micro/macro
    avg_micro = {
        "P": _nanmean_list([per_delta[d]["micro"]["P"] for d in deltas]),
        "R": _nanmean_list([per_delta[d]["micro"]["R"] for d in deltas]),
        "F1": _nanmean_list([per_delta[d]["micro"]["F1"] for d in deltas]),
    }
    avg_macro = {
        "P": _nanmean_list([per_delta[d]["macro"]["P"] for d in deltas]),
        "R": _nanmean_list([per_delta[d]["macro"]["R"] for d in deltas]),
        "F1": _nanmean_list([per_delta[d]["macro"]["F1"] for d in deltas]),
    }
    # 2) geometry overall
    avg_geo_overall = {
        "mATE": _nanmean_list([per_delta[d]["geo_overall"]["mATE"] for d in deltas]),
        "mASE": _nanmean_list([per_delta[d]["geo_overall"]["mASE"] for d in deltas]),
        "mAOE": _nanmean_list([per_delta[d]["geo_overall"]["mAOE"] for d in deltas]),
    }
    # 3) per-class averages
    classes = set()
    for d in deltas:
        classes.update(per_delta[d]["per_class"].keys())
        classes.update(per_delta[d]["geo_per_class"].keys())
    avg_per_class = {}
    avg_geo_per_class = {}
    for c in sorted(classes):
        # PRF averages
        Ps = []; Rs = []; Fs = []
        for d in deltas:
            if c in per_delta[d]["per_class"]:
                Ps.append(per_delta[d]["per_class"][c]["Precision"])
                Rs.append(per_delta[d]["per_class"][c]["Recall"])
                Fs.append(per_delta[d]["per_class"][c]["F1"])
        avg_per_class[c] = {
            "Precision": _nanmean_list(Ps),
            "Recall":    _nanmean_list(Rs),
            "F1":        _nanmean_list(Fs),
            # Optionally provide a reference GT count (from first delta where present)
            "GT": next(((per_delta[d]["per_class"][c]["TP"] + per_delta[d]["per_class"][c]["FN"])
                        for d in deltas if c in per_delta[d]["per_class"]), 0)
        }
        # Geometry averages
        As = []; Ss = []; Os = []
        for d in deltas:
            if c in per_delta[d]["geo_per_class"]:
                As.append(per_delta[d]["geo_per_class"][c]["ATE"])
                Ss.append(per_delta[d]["geo_per_class"][c]["ASE"])
                Os.append(per_delta[d]["geo_per_class"][c]["AOE"])
        avg_geo_per_class[c] = {
            "ATE": _nanmean_list(As),
            "ASE": _nanmean_list(Ss),
            "AOE": _nanmean_list(Os)
        }

    averaged = {
        "micro": avg_micro,
        "macro": avg_macro,
        "geo_overall": avg_geo_overall,
        "per_class": avg_per_class,
        "geo_per_class": avg_geo_per_class,
    }
    return per_delta, averaged

def _parse_deltas(arg_val: str, default=5.0):
    if not arg_val:
        return [default]
    try:
        parts = [p.strip() for p in arg_val.split(",")]
        vals = [float(p) for p in parts if p]
        return vals if len(vals) > 0 else [default]
    except Exception:
        return [default]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="JSON or JSONL with entries: {'predict'/'label' text} or {'pred','gt'} structured")
    ap.add_argument("--delta", type=float, default=None, help="single center-distance threshold (m) for F1/geo")
    ap.add_argument("--deltas", type=str, default=None, help="comma-separated thresholds, e.g., '2,3,5'")
    args = ap.parse_args()

    dataset = load_dataset(args.file)

    # Backward compatible: if --delta is provided and --deltas not, run single
    if args.deltas is None and args.delta is not None:
        out = eval_f1_and_geo(dataset, delta=args.delta)
        print("== Center-distance matching @ δ = {:.1f} m ==".format(args.delta))
        print("micro: P={:.3f} R={:.3f} F1={:.3f}".format(out["micro"]["P"], out["micro"]["R"], out["micro"]["F1"]))
        print("macro: P={:.3f} R={:.3f} F1={:.3f}".format(out["macro"]["P"], out["macro"]["R"], out["macro"]["F1"]))
        print("mATE(m)={:.3f} | mASE={:.3f} | mAOE(rad)={:.3f}".format(
            out["geo_overall"]["mATE"], out["geo_overall"]["mASE"], out["geo_overall"]["mAOE"]))
        print("\nPer-class PRF:")
        for c, v in sorted(out["per_class"].items()):
            print("  {:>14s}: P={:.3f} R={:.3f} F1={:.3f} (TP={} FP={} FN={})".format(
                c, v["Precision"], v["Recall"], v["F1"], v["TP"], v["FP"], v["FN"]))
        print("\nPer-class geometry:")
        for c, v in sorted(out["geo_per_class"].items()):
            print("  {:>14s}: ATE={:.3f} ASE={:.3f} AOE={:.3f}".format(c, v["ATE"], v["ASE"], v["AOE"]))
        return

    # Multi-deltas path (default if neither provided: use [5.0])
    deltas = _parse_deltas(args.deltas, default=(args.delta if args.delta is not None else 5.0))
    per_delta, averaged = eval_multi_deltas(dataset, deltas)

    # Print per-delta
    for d in deltas:
        out = per_delta[d]
        print("== Center-distance matching @ δ = {:.1f} m ==".format(d))
        print("micro: P={:.3f} R={:.3f} F1={:.3f}".format(out["micro"]["P"], out["micro"]["R"], out["micro"]["F1"]))
        print("macro: P={:.3f} R={:.3f} F1={:.3f}".format(out["macro"]["P"], out["macro"]["R"], out["macro"]["F1"]))
        print("mATE(m)={:.3f} | mASE={:.3f} | mAOE(rad)={:.3f}".format(
            out["geo_overall"]["mATE"], out["geo_overall"]["mASE"], out["geo_overall"]["mAOE"]))
        print("")

    # Print averaged
    print("== AVERAGED over deltas: {} ==".format(", ".join("{:.1f}".format(x) for x in deltas)))
    print("micro(avg): P={:.3f} R={:.3f} F1={:.3f}".format(
        averaged["micro"]["P"], averaged["micro"]["R"], averaged["micro"]["F1"]))
    print("macro(avg): P={:.3f} R={:.3f} F1={:.3f}".format(
        averaged["macro"]["P"], averaged["macro"]["R"], averaged["macro"]["F1"]))
    print("mATE(avg)(m)={:.3f} | mASE(avg)={:.3f} | mAOE(avg)(rad)={:.3f}".format(
        averaged["geo_overall"]["mATE"], averaged["geo_overall"]["mASE"], averaged["geo_overall"]["mAOE"]))

    # Optional: per-class averaged summaries
    print("\nPer-class (avg over deltas) — Detection:")
    for c, v in sorted(averaged["per_class"].items()):
        print("  {:>14s}: P={:.3f} R={:.3f} F1={:.3f} | GT={}".format(
            c, v["Precision"], v["Recall"], v["F1"], v["GT"]))
    print("\nPer-class (avg over deltas) — Geometry:")
    for c, v in sorted(averaged["geo_per_class"].items()):
        print("  {:>14s}: ATE={:.3f} ASE={:.3f} AOE={:.3f}".format(
            c, v["ATE"], v["ASE"], v["AOE"]))

if __name__ == "__main__":
    main()