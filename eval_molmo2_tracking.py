"""
Unified Multi-GPU Molmo2 tracking evaluation for 4 datasets:
  - ref-davis17  (Ref-DAVIS 2017, 244 queries, valid split)
  - mevis        (MeViS, 793 queries, valid_u split)
  - ref-yt-vos   (Ref-YouTube-VOS, 834 queries, valid split)
  - reasonvos    (ReasonVOS, 458 queries, test split)

Usage:
  # Multi-GPU evaluation
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 --master-port 29520 \\
      eval_tracking.py --model-path ./Molmo2-4B --task ref-davis17

  # Single-GPU
  python eval_tracking.py --model-path ./Molmo2-4B --task ref-yt-vos --gpu 0
"""

import argparse
import ast
import json
import logging
import os
import re
import sys
import time
from datetime import timedelta
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import datasets
import numpy as np
import torch
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# Per-task tracking parameters. `sampling_fps` is the requested model output
# cadence (FPS); for videotrack tasks it ALSO controls the GT subsampling step,
# so the GT frame grid matches the model output grid (avoids systematic recall
# loss). For davis-style tasks GT comes pre-aligned via frame_trajectories,
# so sampling_fps only affects the prompt template and is informational.
TASK_CONFIGS = {
    "ref-davis17": {
        "hf_subpath": "tracking/ref-davis17/track/valid",
        "videos_subpath": "Ref-DAVIS17/valid/videos",
        "masks_subpath": "Ref-DAVIS17/valid/MasksRLE",
        "display_name": "Ref-DAVIS17",
        "sampling_fps": 1,
    },
    "mevis": {
        "hf_subpath": "tracking/mevis",
        "videos_subpath": "MeViS/valid_u/videos",
        "masks_subpath": "MeViS/valid_u/MasksRLE",
        "display_name": "MeViS",
        "sampling_fps": 1,
    },
    "ref-yt-vos": {
        "hf_subpath": "tracking/ref-yt-vos",
        "videos_subpath": "Ref-YT-VOS/valid/videos",
        "masks_subpath": "Ref-YT-VOS/valid/MasksRLE",
        "display_name": "Ref-YouTube-VOS",
        "sampling_fps": 1,
    },
    "reasonvos": {
        "hf_subpath": "tracking/reasonvos",
        "videos_subpath": "ReasonVOS/videos",
        "masks_subpath": "ReasonVOS/MasksRLE",
        "display_name": "ReasonVOS",
        "sampling_fps": 1,
    },
}


# ---------------------------------------------------------------------------
# Molmo2-VideoTrackEval (5 sub-datasets) support — schema="videotrack"
# Reference: /ov2/feilong/reproduce/molmo2/olmo/data/molmo2_video_track_datasets.py
# ---------------------------------------------------------------------------
VIDEOTRACK_HF_PATH = "/ov2/feilong/reproduce/molmo_data/video_datasets/video_track/Molmo2-VideoTrackEval"
VIDEOTRACK_VIDEOS_ROOT = "/ov2/feilong/reproduce/molmo_data/video_datasets/video_track"
VIDEOTRACK_SOURCE_TO_DIR = {
    "APTv2": "APTv2/videos",
    "dancetrack": "DanceTrack/videos/val",
    "sav": "sav/sav_test/videos_fps6",
    "personpath22": "personpath22/videos/test",
    "sportsmot": "SportsMOT/videos/val",
}

# videotrack: model is asked at sampling_fps=1 (matches official training).
# build_gt_tracks_videotrack subsamples GT to this cadence; mismatched values
# would cause systematic recall loss, so keep consistent.
TASK_CONFIGS.update({
    "animal":  {"schema": "videotrack", "filter_dataset": "APTv2",        "display_name": "Molmo2-Animal",  "sampling_fps": 1},
    "dance":   {"schema": "videotrack", "filter_dataset": "dancetrack",   "display_name": "Molmo2-Dance",   "sampling_fps": 1},
    "misc":    {"schema": "videotrack", "filter_dataset": "sav",          "display_name": "Molmo2-Misc",    "sampling_fps": 1},
    "person":  {"schema": "videotrack", "filter_dataset": "personpath22", "display_name": "Molmo2-Person",  "sampling_fps": 1},
    "sports":  {"schema": "videotrack", "filter_dataset": "sportsmot",    "display_name": "Molmo2-Sports",  "sampling_fps": 1},
})


def resolve_videotrack_video_path(ex):
    sub = VIDEOTRACK_SOURCE_TO_DIR.get(ex["video_source"]) or VIDEOTRACK_SOURCE_TO_DIR.get(ex["video_dataset"])
    if sub is None:
        raise KeyError(f"Unknown video_source/video_dataset: {ex.get('video_source')}/{ex.get('video_dataset')}")
    return os.path.join(VIDEOTRACK_VIDEOS_ROOT, sub, f"{ex['clip']}.mp4")


def build_gt_tracks_videotrack(ex, sampling_fps=1):
    """Build gt_tracks from Molmo2-VideoTrackEval `points` field.
    points[i].points has length n_frames (one [x,y] per clip frame at video_fps).
    GT frames are subsampled to model output cadence (sampling_fps), so step =
    max(1, round(video_fps / sampling_fps)). Keeps frames 0, step, 2*step, ..."""
    mask_ids = list(ex["mask_id"])
    oid_to_idx = {str(mid): idx for idx, mid in enumerate(mask_ids)}
    fps = float(ex["fps"]) if ex["fps"] else 1.0
    obj_points = {str(p["object_id"]): p["points"] for p in (ex.get("points") or [])}
    seg_ranges = {str(s["object_id"]): list(s.get("segments", []) or [])
                  for s in (ex.get("segments") or [])}
    n = int(ex["n_frames"])
    step = max(1, int(round(fps / max(float(sampling_fps), 1e-6))))
    tracks = []
    for f in range(0, n, step):
        pts = {}
        for oid, plist in obj_points.items():
            if oid not in oid_to_idx:
                continue
            if f >= len(plist):
                continue
            xy = plist[f]
            if xy is None or len(xy) < 2:
                continue
            x, y = float(xy[0]), float(xy[1])
            vis = True
            if seg_ranges.get(oid):
                vis = any(a <= f <= b for a, b in seg_ranges[oid])
            if not vis or (x <= 0 and y <= 0):
                continue
            pts[oid_to_idx[oid]] = {"point": [x, y], "occluded": False}
        tracks.append({"frame": f, "time": f / fps, "points": pts})
    return tracks


def build_gt_masks_videotrack(ex):
    """Return {str(idx): [rle_or_None, ...]} aligned with build_gt_tracks_videotrack."""
    out = {}
    for idx, m in enumerate(ex.get("masks") or []):
        frames = m.get("masks", []) or []
        processed = []
        for fm in frames:
            if not fm or not fm.get("counts"):
                processed.append(None)
            else:
                processed.append({"counts": fm["counts"], "size": list(fm["size"])})
        out[str(idx)] = processed
    return out


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=120))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return rank, world_size, device
    else:
        return 0, 1, None


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def load_model_and_processor(model_path, device):
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModel
    except ImportError:
        from transformers import AutoModelForCausalLM as AutoModel
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype="auto",
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def run_inference(model, processor, device, video_path, prompt,
                  sampling_fps=1, max_new_tokens=2048, temperature=0.0):
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": f"file://{video_path}"},
            {"type": "text", "text": prompt},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    import decord as _decord
    _vr = _decord.VideoReader(video_path, ctx=_decord.cpu(0))
    vid_h, vid_w = _vr[0].shape[:2]
    del _vr
    inputs = processor(
        text=text, videos=[video_path],
        frame_sample_mode="fps", sampling_fps=sampling_fps, max_fps=10, num_frames=512,
        return_tensors="pt",
    ).to(device)
    gen_args = dict(**inputs, max_new_tokens=max_new_tokens)
    if temperature > 0:
        gen_args.update(do_sample=True, temperature=temperature, top_p=0.9)
    with torch.inference_mode():
        outputs = model.generate(**gen_args)
    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0].strip()
    return answer, vid_w, vid_h


_COORD_REGEX = re.compile(r'<(?:points|tracks).*? coords="([0-9\t:;, .]+)"/?>')
_FRAME_REGEX = re.compile(r'(?:^|\t|:|,|;)([0-9.]+) ([0-9. ]+)')
_POINTS_REGEX = re.compile(r'([0-9]+) ([0-9]{1,4}) ([0-9]{1,4})')


def parse_xml_tracks(text, width, height, video_fps):
    grouped = defaultdict(list)
    for coord_match in _COORD_REGEX.finditer(text):
        for frame_match in _FRAME_REGEX.finditer(coord_match.group(1)):
            timestamp = float(frame_match.group(1))
            point_str = frame_match.group(2)
            for pt_match in _POINTS_REGEX.finditer(point_str):
                ix = pt_match.group(1)
                x = float(pt_match.group(2)) / 1000.0 * width
                y = float(pt_match.group(3)) / 1000.0 * height
                if 0 <= x <= width and 0 <= y <= height:
                    grouped[timestamp].append((ix, x, y))
    out = []
    for ts in sorted(grouped):
        frame = round(ts * video_fps)
        points = {}
        for ix, x, y in grouped[ts]:
            if str(ix) not in points:
                points[str(ix)] = dict(point=[x, y])
        out.append(dict(time=ts, frame=frame, points=points))
    return out


def parse_time_dict_tracks(text, width, height, video_fps):
    timestamp_pattern = r"time\s+(\d+\.?\d*)\s*\n\s*(\{[^}]+\})"
    result = []
    for match in re.finditer(timestamp_pattern, text, re.MULTILINE):
        seconds = float(match.group(1).strip())
        try:
            obj_points = ast.literal_eval(match.group(2).strip())
        except (ValueError, SyntaxError):
            continue
        frame = round(seconds * video_fps)
        points = {}
        for oid, coords in obj_points.items():
            if len(coords) < 2:
                continue
            x, y = coords[0], coords[1]
            if max(x, y) > 100:
                continue
            px = float(x) / 100.0 * width
            py = float(y) / 100.0 * height
            occ = False
            if len(coords) == 3:
                occ = str(coords[2]).strip().lower() in ['yes', 'true', '1']
            points[str(int(oid))] = dict(point=[px, py], occluded=occ)
        if points:
            result.append(dict(time=seconds, frame=frame, points=points))
    return result


def extract_tracks(text, width, height, video_fps):
    tracks = parse_xml_tracks(text, width, height, video_fps)
    if tracks:
        return tracks
    return parse_time_dict_tracks(text, width, height, video_fps)


def ann_to_mask(mask_ann):
    from pycocotools import mask as mask_utils
    if isinstance(mask_ann, np.ndarray):
        return mask_ann
    if isinstance(mask_ann, list):
        h, w = mask_ann[0]['size']
        rle = mask_utils.merge(mask_utils.frPyObjects(mask_ann, h, w))
    elif isinstance(mask_ann['counts'], list):
        rle = mask_utils.frPyObjects(mask_ann, mask_ann['size'][0], mask_ann['size'][1])
    else:
        rle = mask_ann
    return mask_utils.decode(rle)


def load_masks_at_frame(gt_masks, frame_idx, height, width):
    empty = np.zeros((height, width), dtype=bool)
    masks = []
    for mask_id, mask_list in gt_masks.items():
        first = next((m for m in mask_list if m is not None), None)
        if first is None:
            masks.append(empty)
            continue
        binary = empty
        if isinstance(first, dict) and 'frame' in first:
            for m in mask_list:
                if m['mask'] is not None and m['frame'] <= frame_idx:
                    if m['frame'] == frame_idx:
                        binary = ann_to_mask(m['mask']).astype(bool)
                        break
        else:
            if frame_idx < len(mask_list) and mask_list[frame_idx] is not None:
                binary = ann_to_mask(mask_list[frame_idx]).astype(bool)
        masks.append(binary)
    return masks


def is_point_in_mask(point, mask):
    h, w = mask.shape
    x, y = point
    xi, yi = int(round(x)), int(round(y))
    return 0 <= xi < w and 0 <= yi < h and bool(mask[yi, xi])


def evaluate_frame(pred_points, gt_points, masks):
    ng = len(gt_points)
    np_ = len(pred_points)
    if ng == 0:
        score = float(np_ == 0)
        return score, score, score
    if np_ == 0:
        return 0.0, 0.0, 0.0
    dist_mat = cdist(np.array(pred_points), np.array(gt_points))
    ri, ci = linear_sum_assignment(dist_mat)
    correct = sum(1 for r, c in zip(ri, ci) if c < len(masks) and is_point_in_mask(pred_points[r], masks[c]))
    p = correct / np_
    r = correct / len(masks)
    f = 2 * p * r / (p + r + 1e-10) if (p + r) > 0 else 0.0
    return p, r, f


def evaluate_video_tracks_with_masks(pred_tracks, gt_tracks, gt_masks, height, width):
    pred_by = {e['frame']: e for e in (pred_tracks or [])}
    gt_by = {e['frame']: e for e in (gt_tracks or [])}
    all_frames = sorted(set(pred_by) | set(gt_by))
    if not all_frames:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
                'num_frames': 0, 'frames_with_pred': 0, 'frames_with_gt': 0}
    metrics = []
    for fidx in all_frames:
        pf = pred_by.get(fidx)
        gf = gt_by.get(fidx)
        if pf is None and gf is None:
            continue
        pp = [tuple(v['point']) for v in pf['points'].values()] if pf else []
        gp = [tuple(v['point']) for v in gf['points'].values()] if gf else []
        masks = load_masks_at_frame(gt_masks, fidx, height, width)
        p, r, f = evaluate_frame(pp, gp, masks)
        metrics.append({'precision': p, 'recall': r, 'f1': f,
                        'has_pred': pf is not None, 'has_gt': gf is not None})
    if not metrics:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
                'num_frames': 0, 'frames_with_pred': 0, 'frames_with_gt': 0}
    return {
        'precision': float(np.mean([m['precision'] for m in metrics])),
        'recall': float(np.mean([m['recall'] for m in metrics])),
        'f1': float(np.mean([m['f1'] for m in metrics])),
        'num_frames': len(metrics),
        'frames_with_pred': sum(1 for m in metrics if m['has_pred']),
        'frames_with_gt': sum(1 for m in metrics if m['has_gt']),
    }


class HOTAMetric:
    def __init__(self):
        self.alpha_thresholds = np.arange(0.05, 1.0, 0.05)

    def prepare_data(self, pred_tracks, gt_tracks, gt_masks, height, width):
        pred_tracks = pred_tracks or []
        gt_tracks = gt_tracks or []
        pred_by = {e['frame']: e for e in pred_tracks}
        gt_by = {e['frame']: e for e in gt_tracks}
        all_gt_ids, all_pred_ids = set(), set()
        for e in gt_tracks:
            all_gt_ids.update(e['points'].keys())
        for e in pred_tracks:
            all_pred_ids.update(e['points'].keys())
        gt_map = {str(oid): i for i, oid in enumerate(sorted(all_gt_ids))}
        pred_map = {str(oid): i for i, oid in enumerate(sorted(all_pred_ids))}
        all_frames = sorted(set(pred_by) | set(gt_by))
        data = {
            'num_gt_ids': len(all_gt_ids), 'num_tracker_ids': len(all_pred_ids),
            'num_timesteps': len(all_frames),
            'gt_ids': [], 'tracker_ids': [], 'similarity_scores': [],
            'num_gt_dets': 0, 'num_tracker_dets': 0,
        }
        for fidx in all_frames:
            pf = pred_by.get(fidx)
            gf = gt_by.get(fidx)
            gt_ids_l, gt_pts = [], []
            if gf:
                for oid, pd in sorted(gf['points'].items()):
                    gt_ids_l.append(gt_map[str(oid)])
                    gt_pts.append(pd['point'])
            pred_ids_l, pred_pts = [], []
            if pf:
                for oid, pd in sorted(pf['points'].items()):
                    pred_ids_l.append(pred_map[str(oid)])
                    pred_pts.append(pd['point'])
            gt_arr = np.array(gt_ids_l, dtype=int)
            pred_arr = np.array(pred_ids_l, dtype=int)
            data['gt_ids'].append(gt_arr)
            data['tracker_ids'].append(pred_arr)
            data['num_gt_dets'] += len(gt_arr)
            data['num_tracker_dets'] += len(pred_arr)
            sim = np.zeros((len(gt_pts), len(pred_pts)))
            if gt_pts and pred_pts:
                masks = load_masks_at_frame(gt_masks, fidx, height, width)
                gf_items = sorted(gf['points'].items())
                for i, (oid, _pd) in enumerate(gf_items):
                    try:
                        mid = int(oid)
                    except (TypeError, ValueError):
                        continue
                    if mid < 0 or mid >= len(masks):
                        continue
                    mask = masks[mid]
                    for j, pp in enumerate(pred_pts):
                        if is_point_in_mask(pp, mask):
                            sim[i, j] = 1.0
            data['similarity_scores'].append(sim)
        return data

    def compute(self, data):
        na = len(self.alpha_thresholds)
        res = {k: np.zeros(na) for k in [
            'HOTA_TP', 'HOTA_FN', 'HOTA_FP', 'HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'LocA'
        ]}
        if data['num_tracker_dets'] == 0 and data['num_gt_dets'] == 0:
            return {k: np.ones(na) for k in res}
        if data['num_tracker_dets'] == 0:
            res['HOTA_FN'] = np.full(na, data['num_gt_dets'])
            res['LocA'] = np.ones(na)
            return res
        if data['num_gt_dets'] == 0:
            res['HOTA_FP'] = np.full(na, data['num_tracker_dets'])
            res['LocA'] = np.ones(na)
            return res
        pot_matches = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        gt_count = np.zeros((data['num_gt_ids'], 1))
        pred_count = np.zeros((1, data['num_tracker_ids']))
        for t, (gids, pids) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            sim = data['similarity_scores'][t]
            denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            iou = np.zeros_like(sim)
            mask = denom > np.finfo(float).eps
            iou[mask] = sim[mask] / denom[mask]
            pot_matches[gids[:, None], pids[None, :]] += iou
            gt_count[gids] += 1
            pred_count[0, pids] += 1
        global_score = pot_matches / (gt_count + pred_count - pot_matches + np.finfo(float).eps)
        matches_counts = [np.zeros_like(pot_matches) for _ in self.alpha_thresholds]
        for t, (gids, pids) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            if len(gids) == 0:
                for a in range(na):
                    res['HOTA_FP'][a] += len(pids)
                continue
            if len(pids) == 0:
                for a in range(na):
                    res['HOTA_FN'][a] += len(gids)
                continue
            sim = data['similarity_scores'][t]
            score_mat = global_score[gids[:, None], pids[None, :]] * sim
            mr, mc = linear_sum_assignment(-score_mat)
            for a, alpha in enumerate(self.alpha_thresholds):
                matched = sim[mr, mc] >= alpha - np.finfo(float).eps
                amr, amc = mr[matched], mc[matched]
                nm = len(amr)
                res['HOTA_TP'][a] += nm
                res['HOTA_FN'][a] += len(gids) - nm
                res['HOTA_FP'][a] += len(pids) - nm
                if nm > 0:
                    res['LocA'][a] += sim[amr, amc].sum()
                    matches_counts[a][gids[amr], pids[amc]] += 1
        for a in range(na):
            mc = matches_counts[a]
            assA = mc / np.maximum(1, gt_count + pred_count - mc)
            res['AssA'][a] = np.sum(mc * assA) / np.maximum(1, res['HOTA_TP'][a])
        res['LocA'] = np.maximum(1e-10, res['LocA']) / np.maximum(1e-10, res['HOTA_TP'])
        res['DetRe'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'])
        res['DetPr'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FP'])
        res['DetA'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'] + res['HOTA_FP'])
        res['HOTA'] = np.sqrt(res['DetA'] * res['AssA'])
        return res


def evaluate_video_object_tracking(pred_tracks, gt_tracks, gt_masks, height, width):
    spatial = evaluate_video_tracks_with_masks(pred_tracks, gt_tracks, gt_masks, height, width)
    hota = HOTAMetric()
    hota_data = hota.prepare_data(pred_tracks, gt_tracks, gt_masks, height, width)
    hota_res = hota.compute(hota_data)
    alpha_05_idx = 9
    return {
        'precision': spatial['precision'],
        'recall': spatial['recall'],
        'f1': spatial['f1'],
        'HOTA': float(hota_res['HOTA'][alpha_05_idx]),
        'DetA': float(hota_res['DetA'][alpha_05_idx]),
        'AssA': float(hota_res['AssA'][alpha_05_idx]),
        'DetPr': float(hota_res['DetPr'][alpha_05_idx]),
        'DetRe': float(hota_res['DetRe'][alpha_05_idx]),
        'LocA': float(hota_res['LocA'][alpha_05_idx]),
        'num_frames': spatial['num_frames'],
        'frames_with_pred': spatial['frames_with_pred'],
        'frames_with_gt': spatial['frames_with_gt'],
    }


def build_gt_tracks(example):
    mask_ids = example["mask_id"]
    oid_to_idx = {mid: idx for idx, mid in enumerate(mask_ids)}
    tracks = []
    for fd in example["frame_trajectories"]:
        pts = {}
        for p in fd["points"]:
            ok = str(p["id"])
            if ok in oid_to_idx:
                pts[oid_to_idx[ok]] = {
                    "point": list(p["point"]),
                    "occluded": p.get("occluded", False),
                }
        tracks.append({"frame": fd["frame"], "time": fd["time"], "points": pts})
    tracks.sort(key=lambda x: x["frame"])
    return tracks


EVAL_PROMPTS = [
    "Track {label} in {fps} FPS.",
    "Track {label} at {fps} FPS",
    "Track the {label} in {fps} FPS",
    "Track all instances of '{label}' in this video, sampling at {fps} frames per second. Show the position coordinates at each timestamp.",
    "Follow the {label} throughout this video. Sample positions at {fps} FPS and mark each with coordinates.",
    "Identify and track each {label} in this video clip. Sample at {fps} frames per second and provide point coordinates.",
    "Monitor the movement of all {label} in this video. Sample at {fps} FPS and output their positions as coordinates.",
    "For each {label} in the video, track its position throughout the clip sampling at {fps} FPS. Return coordinates at each sampled frame.",
    "Track all instances of '{label}' in this video, sampling at {fps} frames per second. Show the position coordinates at each timestamp, given as <track coords='t id x y'>label</tracks>.",
    "For each {label} in the video, track its position throughout the clip sampling at {fps} FPS. Return coordinates at each sampled frame, given as <track coords='t id x y'>label</tracks>.",
]


def parse_args():
    p = argparse.ArgumentParser(description="Unified tracking evaluation for Molmo2")
    p.add_argument("--model-path", required=True, help="Path to Molmo2 model")
    p.add_argument("--task", required=True, choices=list(TASK_CONFIGS.keys()),
                   help="Dataset to evaluate on")
    p.add_argument("--data-dir", default=None, help="Root data directory")
    p.add_argument("--gpu", type=int, default=0, help="GPU id for single-GPU mode")
    p.add_argument("--max-examples", type=int, default=-1)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--sampling-fps", type=int, default=None,
                   help="Override per-task sampling_fps. If unset, uses TASK_CONFIGS[task]['sampling_fps'].")
    p.add_argument("--template-id", type=int, default=-1,
                   help="Prompt template index (0-9). -1=random per example")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--id-allowlist", default=None, help="Path to a text file with one example id per line; only those will be evaluated.")
    return p.parse_args()


def main():
    args = parse_args()
    task_cfg = TASK_CONFIGS[args.task]
    rank, world_size, device = setup_distributed()
    if device is None:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)

    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir / "data"
    videos_dir = (data_dir / task_cfg["videos_subpath"]) if task_cfg.get("videos_subpath") else None
    masks_dir = (data_dir / task_cfg["masks_subpath"]) if task_cfg.get("masks_subpath") else None
    hf_data = (data_dir / task_cfg["hf_subpath"]) if task_cfg.get("hf_subpath") else None
    output_dir = (Path(args.output_dir) / args.task) if args.output_dir else script_dir / "eval_output" / args.task
    log_dir = script_dir / "logs"
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{args.task}.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        log.addHandler(fh)

    is_vt = task_cfg.get("schema") == "videotrack"
    if is_main_process():
        log.info(f"Task: {task_cfg['display_name']}")
        if is_vt:
            log.info(f"Loading Molmo2-VideoTrackEval from {VIDEOTRACK_HF_PATH}, filter={task_cfg['filter_dataset']}")
        else:
            log.info(f"Loading dataset from {hf_data}")
    if is_vt:
        ds = datasets.load_from_disk(VIDEOTRACK_HF_PATH)
        ds = ds.filter(lambda ex, _f=task_cfg["filter_dataset"]: ex["video_dataset"] == _f)
    else:
        ds = datasets.load_from_disk(str(hf_data))
    if args.id_allowlist:
        _allow = set(l.strip() for l in open(args.id_allowlist) if l.strip())
        ds = ds.filter(lambda ex, _a=_allow: ex.get("id") in _a)
        if is_main_process():
            log.info(f"id-allowlist filter: kept {len(ds)} / {len(_allow)} requested")
    if args.smoke_test:
        args.max_examples = 5
    if args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))
    total = len(ds)

    indices = list(range(rank, total, world_size))
    if is_main_process():
        log.info(f"Total examples: {total}, world_size: {world_size}")
        log.info(f"Loading model from {args.model_path}")
    model, processor = load_model_and_processor(args.model_path, device)
    if is_main_process():
        log.info("Model loaded on all ranks")
    barrier()

    local_results = []
    start_time = time.time()

    for i, idx in enumerate(indices):
        ex = ds[idx]
        sfps = args.sampling_fps if args.sampling_fps is not None else int(task_cfg.get("sampling_fps", 1))
        if is_vt:
            vname = ex["clip"]
            expr = ex["exp"]
            vfps = float(ex["fps"])
            qid = str(ex["id"])
            w, h = int(ex["w"]), int(ex["h"])
            try:
                vpath = resolve_videotrack_video_path(ex)
            except KeyError as _e:
                log.warning(f"[rank {rank}] {_e}")
                continue
            mask_path = None
        else:
            vname = ex["video"]
            expr = ex["expression"]
            vfps = ex["fps"]
            qid = ex["qid"]
            w, h = ex["width"], ex["height"]
            vpath = str(videos_dir / f"{vname}.mp4")
            mask_path = masks_dir / vname / f"{qid}.json"
        if not os.path.exists(vpath):
            log.warning(f"[rank {rank}] Missing video: {vpath}")
            continue
        if (not is_vt) and (not mask_path.exists()):
            log.warning(f"[rank {rank}] Missing masks: {mask_path}")
            continue

        if args.template_id >= 0:
            prompt_tmpl = EVAL_PROMPTS[args.template_id % len(EVAL_PROMPTS)]
        else:
            import random as _rnd
            _rng = _rnd.Random(idx)
            prompt_tmpl = _rng.choice(EVAL_PROMPTS)
        prompt = prompt_tmpl.format(label=expr, fps=sfps)
        try:
            ans, vw, vh = run_inference(
                model, processor, device, vpath, prompt,
                sampling_fps=sfps, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            vw = vw or w
            vh = vh or h
            pred_tracks = extract_tracks(ans, vw, vh, vfps)
            if is_vt:
                gt_tracks = build_gt_tracks_videotrack(ex, sampling_fps=sfps)
                gt_masks = build_gt_masks_videotrack(ex)
            else:
                gt_tracks = build_gt_tracks(ex)
                gt_masks = json.load(open(mask_path))
            try:
                metrics = evaluate_video_object_tracking(pred_tracks, gt_tracks, gt_masks, h, w)
            except Exception:
                log.exception(f"[rank {rank}] Metric error (prediction kept): {vname}/{qid}")
                metrics = None
            local_results.append({
                "idx": idx, "id": ex.get("id"), "video": vname, "qid": qid,
                "expression": expr, "prediction": ans,
                "metrics": ({k: v for k, v in metrics.items()
                              if not isinstance(v, (list, np.ndarray))}
                             if metrics is not None else None),
            })
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(indices) - i - 1)
            if metrics is not None:
                log.info(
                    f"[rank {rank}] [{i+1}/{len(indices)}] {vname}/{qid}  "
                    f"F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                    f"HOTA={metrics['HOTA']:.3f} ETA={eta:.0f}s"
                )
            else:
                log.info(
                    f"[rank {rank}] [{i+1}/{len(indices)}] {vname}/{qid}  metrics=NA ETA={eta:.0f}s"
                )
        except Exception:
            log.exception(f"[rank {rank}] Error: {vname}/{qid}")
            continue

    barrier()

    if world_size > 1:
        rank_file = output_dir / f"results_rank{rank}.json"
        with open(rank_file, "w") as f:
            json.dump(local_results, f)
        barrier()
        if is_main_process():
            all_results = []
            for r in range(world_size):
                rf = output_dir / f"results_rank{r}.json"
                with open(rf) as f:
                    all_results.extend(json.load(f))
                rf.unlink()
            all_results.sort(key=lambda x: x["idx"])
            _save_final_results(all_results, output_dir, total, task_cfg['display_name'])
    else:
        _save_final_results(local_results, output_dir, total, task_cfg['display_name'])


def _save_final_results(results, output_dir, total, display_name):
    if not results:
        log.warning("No results!")
        return
    metric_keys = ['precision', 'recall', 'f1', 'HOTA', 'DetA', 'AssA']
    avg = {}
    for k in metric_keys:
        vals = [r["metrics"][k] for r in results if k in r["metrics"]]
        avg[k] = float(np.mean(vals)) if vals else 0.0
    log.info("=" * 70)
    log.info(f"{display_name} Results ({len(results)}/{total} examples):")
    log.info("  " + "  ".join(f"{k}={v:.4f}" for k, v in avg.items()))
    log.info("=" * 70)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"task": display_name, "metrics": avg, "n": len(results), "total": total}, f, indent=2)
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
