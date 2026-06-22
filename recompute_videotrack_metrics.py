"""Offline recompute of stage1 metrics from saved predictions.json using the
patched build_gt_tracks_videotrack (subsampled to vis_sfps=10)."""
import json, os, sys, argparse
sys.path.insert(0, "/ov2/feilong/simple_repo")
import datasets
from eval_ov2_tracking import (
    build_gt_tracks_videotrack, build_gt_masks_videotrack,
    evaluate_video_object_tracking, extract_tracks,
)

VIDEOTRACK_HF = "/ov2/feilong/reproduce/molmo_data/video_datasets/video_track/Molmo2-VideoTrackEval"
TASK_FILTER = {
    "animal": "APTv2", "dance": "dancetrack", "misc": "sav",
    "person": "personpath22", "sports": "sportsmot",
}

def aggregate(metrics_list):
    keys = ["precision","recall","f1","HOTA","DetA","AssA","DetPr","DetRe","LocA"]
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m]
        out[k] = sum(vals)/len(vals) if vals else 0.0
    return out

def run(root, sampling_fps=10, write=False):
    ds_full = datasets.load_from_disk(VIDEOTRACK_HF)
    cache = {}
    for task, filt in TASK_FILTER.items():
        pj = os.path.join(root, task, "predictions.json")
        if not os.path.isfile(pj):
            print("[skip] %s: no predictions.json" % task); continue
        preds = json.load(open(pj))
        if filt not in cache:
            cache[filt] = ds_full.filter(lambda ex, _f=filt: ex["video_dataset"] == _f)
        ds = cache[filt]
        by_id = {ex["id"]: ex for ex in ds}
        per_sample = []
        for p in preds:
            ex = by_id.get(p["id"])
            if ex is None:
                continue
            vfps = float(ex["fps"])
            w, h = int(ex["w"]), int(ex["h"])
            pred_tracks = extract_tracks(p["prediction"], w, h, vfps)
            gt_tracks = build_gt_tracks_videotrack(ex, sampling_fps=sampling_fps)
            gt_masks = build_gt_masks_videotrack(ex)
            m = evaluate_video_object_tracking(pred_tracks, gt_tracks, gt_masks, h, w)
            per_sample.append(m)
            if write:
                p["metrics"] = m
        agg = aggregate(per_sample)
        n = len(per_sample)
        print("  %-8s n=%4d  P=%.4f  R=%.4f  F1=%.4f  HOTA=%.4f  DetA=%.4f  AssA=%.4f" % (
            task, n, agg["precision"], agg["recall"], agg["f1"],
            agg["HOTA"], agg["DetA"], agg["AssA"]))
        if write:
            json.dump(preds, open(pj, "w"))
            mp = os.path.join(root, task, "metrics.json")
            json.dump({"task": task, "metrics": agg, "n": n, "total": n}, open(mp, "w"), indent=2)
            print("    -> wrote %s and %s" % (pj, mp))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/ov2/feilong/simple_repo/eval_output_ov2/videotrack")
    ap.add_argument("--sampling-fps", type=int, default=10)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    print("=== recompute stage1 metrics @ sampling_fps=%d on %s (write=%s) ===" % (
        a.sampling_fps, a.root, a.write))
    run(a.root, a.sampling_fps, write=a.write)
