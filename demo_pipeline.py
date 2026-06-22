#!/usr/bin/env python3
"""
Demo selection + rendering for OV2 mustshort.

Two phases controlled by --phase:
  select   single-process; ranks samples, classifies single/multi, writes selection.json
  render   torchrun-distributed; runs SAM2 + writes demo MP4s

Output dir layout:
  <out>/selection.json
  <out>/<task>/<single|multi>/rank<NN>_<video>_q<qid>_jf<JF>/
      original.mp4    (source frames as video)
      points.mp4      (frames + point markers)
      mask.mp4        (frames + mask alpha-blend)
      info.json       (expression, J, F, J&F, HOTA, num objects)
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/ov2/feilong/simple_repo")
sys.path.insert(0, str(REPO))


def _load_eval():
    spec = importlib.util.spec_from_file_location(
        "ev", str(REPO / "eval_sam2_tracking.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# Mustshort prediction roots
MUSTSHORT_ROOTS = {
    "ref-davis17": REPO / "eval_output_ov2/s33_8b_finalround_mustshort_davmevis_legacy4_fps1_fnf128_tpl0/ref-davis17/ref-davis17",
    "mevis":       REPO / "eval_output_ov2/s33_8b_finalround_mustshort_davmevis_legacy4_fps1_fnf128_tpl0/mevis/mevis",
    "ref-yt-vos":  REPO / "eval_output_ov2/s37_8b_finalround_mustshort_ytvos_legacy4_fps1_fnf128_tpl0/ref-yt-vos/ref-yt-vos",
    "reasonvos":   REPO / "eval_output_ov2/s36_8b_finalround_mustshort_ytreasonvos_legacy4_fps1_fnf128_tpl0/reasonvos/reasonvos",
}

# Where GT masks live (relative to repo data dir or absolute)
GT_MASK_ROOTS = {
    "ref-davis17": REPO / "data/Ref-DAVIS17/valid/MasksRLE",
    "mevis":       REPO / "data/MeViS/valid_u/MasksRLE",
    "ref-yt-vos":  REPO / "data/Ref-YT-VOS/valid/MasksRLE",
    "reasonvos":   REPO / "data/ReasonVOS/MasksRLE",
}


def gt_num_objects(task, video, qid):
    p = GT_MASK_ROOTS[task] / video / f"{qid}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return len(d) if isinstance(d, dict) else None


# ===== PHASE 1: SELECT =====

def phase_select(out_dir, jf_threshold=0.8, n_per_class=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection = {}
    summary_lines = []
    for task, root in MUSTSHORT_ROOTS.items():
        sam2 = json.load(open(root / "sam2_results/sam2_predictions.json"))
        stage1 = json.load(open(root / "predictions.json"))
        s1_idx = {(it["video"], str(it["qid"])): it for it in stage1}

        items = []
        for r in sam2:
            v, q = r["video"], str(r["qid"])
            if r["J&F"] <= jf_threshold:
                continue
            n_obj = gt_num_objects(task, v, q)
            if n_obj is None:
                continue
            s1 = s1_idx.get((v, q), {})
            items.append({
                "task": task,
                "idx": r.get("idx"),
                "video": v,
                "qid": q,
                "expression": r.get("expression", s1.get("expression", "")),
                "J": r["J"], "F": r["F"], "JF": r["J&F"], "HOTA": r["HOTA"],
                "num_frames": r.get("num_frames"),
                "n_obj": n_obj,
                "is_multi": n_obj > 1,
                "prediction": s1.get("prediction", ""),
            })
        single_pool = sorted([i for i in items if not i["is_multi"]], key=lambda x: -x["JF"])
        multi_pool  = sorted([i for i in items if i["is_multi"]],  key=lambda x: -x["JF"])
        if n_per_class is not None:
            single = single_pool[:n_per_class]
            multi  = multi_pool[:n_per_class]
        else:
            single = single_pool
            multi  = multi_pool
        selection[task] = {"single": single, "multi": multi}
        sjf = f"{single[0]['JF']:.3f}" if single else "-"
        mjf = f"{multi[0]['JF']:.3f}"  if multi  else "-"
        summary_lines.append(
            f"[{task}] total={len(items)} single_pool={sum(1 for i in items if not i['is_multi'])} "
            f"multi_pool={sum(1 for i in items if i['is_multi'])} "
            f"selected single={len(single)} multi={len(multi)} "
            f"top_JF single={sjf} multi={mjf}"
        )

    with open(out_dir / "selection.json", "w") as f:
        json.dump(selection, f, indent=2)
    with open(out_dir / "selection_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("\n".join(summary_lines))
    print(f"\nSaved: {out_dir / 'selection.json'}")


# ===== PHASE 2: RENDER =====

# Color palette for tracks
COLORS = [
    (66, 133, 244),   # blue
    (234, 67, 53),    # red
    (251, 188, 5),    # yellow
    (52, 168, 83),    # green
    (153, 102, 255),  # purple
    (255, 109, 1),    # orange
    (0, 200, 200),    # cyan
    (255, 0, 200),    # magenta
]


def encode_mp4(frames, out_path, fps=10):
    """Encode list of HxWx3 BGR uint8 frames to mp4 via cv2."""
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not vw.isOpened():
        return False
    for f in frames:
        vw.write(f)
    vw.release()
    return True


def load_jpeg_sequence(frames_dir):
    files = sorted(Path(frames_dir).iterdir(), key=lambda p: p.name)
    files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    return [cv2.imread(str(f)) for f in files]


def load_mp4(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out


def parse_points_for_render(prediction_text, n_frames, w, h, video_fps):
    """Returns dict: frame_idx -> list of (obj_id, x_px, y_px).

    Coords in pred are in 1000x1000 grid. Time is in seconds (since OV2 is
    sampled at 1 fps). Frame index = round(t * video_fps).
    """
    import re
    out = {}
    if not prediction_text:
        return out
    m = re.search(r'<tracks[^>]*coords="([^"]*)"', prediction_text)
    if not m:
        return out
    body = m.group(1)
    for entry in body.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split()
        if len(parts) < 4:
            continue
        try:
            t = float(parts[0])
            oid = int(parts[1])
            x_g = float(parts[2])
            y_g = float(parts[3])
        except ValueError:
            continue
        f = int(round(t * float(video_fps)))
        if f < 0 or f >= n_frames:
            continue
        x = int(round(x_g / 1000.0 * w))
        y = int(round(y_g / 1000.0 * h))
        out.setdefault(f, []).append((oid, x, y))
    return out


def overlay_mask(frame, mask_uint8, color, alpha=0.55):
    """Alpha-blend mask region with color."""
    if mask_uint8 is None:
        return frame
    if mask_uint8.shape[:2] != frame.shape[:2]:
        mask_uint8 = cv2.resize(mask_uint8, (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
    m = mask_uint8.astype(bool)
    overlay = frame.copy()
    overlay[m] = (1 - alpha) * frame[m] + alpha * np.array(color, dtype=np.float32)
    return overlay


def draw_points(frame, pts):
    out = frame.copy()
    for oid, x, y in pts:
        c = COLORS[oid % len(COLORS)]
        cv2.circle(out, (x, y), 9, (255, 255, 255), -1)
        cv2.circle(out, (x, y), 6, c, -1)
    return out


def draw_caption(frame, text, color=(255, 255, 255)):
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text[:80], (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def safe_dir_name(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:60]


def render_one_sample(item, ev, predictor, data_dir, out_root, fps_lookup):
    """Render demos for one selected sample. Returns dict of result info."""
    task = item["task"]
    task_cfg = ev.TASK_CONFIGS[task]
    video_id = item["video"]
    qid = item["qid"]

    meta = fps_lookup.get((task, video_id, qid))
    if meta is None:
        return {"err": f"no HF meta for {task}/{video_id}/q{qid}"}
    video_fps = float(meta["fps"])

    # Get frame source — must match what eval/SAM2 sees:
    #  - jpeg datasets: JPEG dir at native fps (already 6 fps in HF)
    #  - mp4 datasets (reasonvos): re-extract at fps=6 to match eval pipeline
    import tempfile, shutil
    cleanup_dir = None
    if task_cfg["source"] == "jpeg":
        frame_dir = Path(task_cfg["frames_dir"]) / video_id
        if not frame_dir.exists():
            return {"err": f"frames not found: {frame_dir}"}
        sam2_frame_dir = str(frame_dir)
        frames = load_jpeg_sequence(frame_dir)
    elif task_cfg["source"] == "mp4":
        video_path = Path("/ov2/feilong/simple_repo/data/ReasonVOS/videos") / f"{video_id}.mp4"
        if not video_path.exists():
            return {"err": f"video not found: {video_path}"}
        cleanup_dir = tempfile.mkdtemp(prefix=f"sam2_{task}_{video_id}_")
        ev.extract_frames_from_video(str(video_path), cleanup_dir, fps=video_fps)
        sam2_frame_dir = cleanup_dir
        frames = load_jpeg_sequence(cleanup_dir)
    else:
        return {"err": f"unsupported source {task_cfg['source']}"}

    if not frames:
        return {"err": "empty frames"}

    n_frames = len(frames)
    h, w = frames[0].shape[:2]

    # Parse OV2 points with correct fps mapping
    pts_by_frame = parse_points_for_render(item["prediction"], n_frames, w, h, video_fps)

    try:
        inference_state = predictor.init_state(
            video_path=sam2_frame_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        predictor.reset_state(inference_state)
        prompt_map = ev.build_sam2_prompt_map(
            parsed_points=ev.parse_prediction_points(item["prediction"]),
            video_width=w, video_height=h, video_fps=video_fps, num_frames=n_frames,
        )
        seg = ev.run_sam2_segmentation(predictor, prompt_map, inference_state)
        masks = seg["mask_prediction"]
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    # Render 3 streams
    sub = "multi" if item["is_multi"] else "single"
    rank_idx = item["_rank"]
    out_dir = (Path(out_root) / task / sub /
               f"rank{rank_idx:02d}_{safe_dir_name(video_id)}_q{qid}_jf{int(round(item['JF']*1000)):03d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cap_text = f"[{task}|{sub}|JF={item['JF']:.3f}] {item['expression']}"

    orig_frames, point_frames, mask_frames = [], [], []
    for i, fr in enumerate(frames):
        of = draw_caption(fr, cap_text)
        orig_frames.append(of)

        pts = pts_by_frame.get(i, [])
        pf = draw_points(fr, pts)
        pf = draw_caption(pf, cap_text)
        point_frames.append(pf)

        m = masks[i] if i < len(masks) else None
        mf = overlay_mask(fr, m, (66, 133, 244), alpha=0.5)
        # Also draw current points on mask view
        mf = draw_points(mf, pts)
        mf = draw_caption(mf, cap_text)
        mask_frames.append(mf)

    encode_mp4(orig_frames,  out_dir / "original.mp4", fps=video_fps)
    encode_mp4(point_frames, out_dir / "points.mp4",   fps=video_fps)
    encode_mp4(mask_frames,  out_dir / "mask.mp4",     fps=video_fps)

    info = {
        "task": task, "video": video_id, "qid": qid,
        "expression": item["expression"],
        "J": item["J"], "F": item["F"], "JF": item["JF"], "HOTA": item["HOTA"],
        "n_objects_gt": item["n_obj"],
        "is_multi": item["is_multi"],
        "rank_in_class": rank_idx,
        "num_frames": n_frames,
        "video_fps": video_fps,
        "prediction": item["prediction"],
    }
    with open(out_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    return {"ok": True, "out_dir": str(out_dir)}


def _build_fps_lookup(sel):
    """Load HF metadata for tasks present in selection, return (task,video,qid)->meta."""
    import datasets
    out = {}
    task_to_sub = {
        "ref-davis17": "tracking/ref-davis17/track/valid",
        "mevis":       "tracking/mevis",
        "ref-yt-vos":  "tracking/ref-yt-vos",
        "reasonvos":   "tracking/reasonvos",
    }
    needed_tasks = set(sel.keys())
    for task in needed_tasks:
        path = REPO / "data" / task_to_sub[task]
        ds = datasets.load_from_disk(str(path))
        for i in range(len(ds)):
            ex = ds[i]
            key = (task, ex["video"], str(ex["qid"]))
            out[key] = {"fps": float(ex["fps"]),
                        "width": int(ex["width"]), "height": int(ex["height"]),
                        "n_frames": int(ex["n_frames"])}
    return out


def phase_render(out_dir):
    out_dir = Path(out_dir)
    sel = json.load(open(out_dir / "selection.json"))

    ev = _load_eval()
    rank, world_size, device = ev.setup_distributed()

    # Flatten selection with rank index per (task, sub)
    all_items = []
    for task, by_sub in sel.items():
        for sub in ("single", "multi"):
            for i, it in enumerate(by_sub[sub], start=1):
                it["_rank"] = i
                all_items.append(it)
    # Shard
    my_items = [x for i, x in enumerate(all_items) if i % world_size == rank]

    print(f"[rank {rank}/{world_size}] processing {len(my_items)}/{len(all_items)} items")

    fps_lookup = _build_fps_lookup(sel)
    print(f"[rank {rank}] fps_lookup size: {len(fps_lookup)}")

    # Build SAM2 once per rank
    predictor = ev.build_sam2_video_predictor(
        ev.DEFAULT_SAM2_CODE_DIR, ev.DEFAULT_SAM2_CONFIG_PATH, ev.DEFAULT_SAM2_CHECKPOINT_PATH,
        device=device,
    )

    data_dir = REPO / "data"
    log_path = out_dir / f"render_rank{rank}.log"
    with open(log_path, "w") as logf:
        for k, item in enumerate(my_items):
            try:
                res = render_one_sample(item, ev, predictor, data_dir, out_dir, fps_lookup)
                msg = f"[rank {rank}] {k+1}/{len(my_items)} {item['task']}/{item['video']}/q{item['qid']} -> {res}"
            except Exception as e:
                import traceback
                msg = f"[rank {rank}] ERR {item['task']}/{item['video']}/q{item['qid']}: {e}\n{traceback.format_exc()}"
            print(msg, flush=True)
            logf.write(msg + "\n"); logf.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True, choices=["select", "render"])
    p.add_argument("--out-dir", default="/ov2/feilong/simple_repo/demos_mustshort")
    p.add_argument("--n-per-class", type=int, default=None,
                   help="If set, cap selection to top-N per class. Else take all above threshold.")
    p.add_argument("--jf-threshold", type=float, default=0.0,
                   help="Minimum J&F to include in selection (e.g. 0.8). Default 0 = include all.")
    args = p.parse_args()

    if args.phase == "select":
        phase_select(args.out_dir, jf_threshold=args.jf_threshold, n_per_class=args.n_per_class)
    else:
        phase_render(args.out_dir)


if __name__ == "__main__":
    main()
