
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="JSON or JSONL with entries: {'predict'/'label' text} or {'pred','gt'} structured")
    ap.add_argument("--delta", type=float, default=5.0, help="center-distance threshold (m) for F1/geo")
    args = ap.parse_args()

    dataset = load_dataset(args.file)
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

if __name__ == "__main__":
    main()
