"""
Unified Multi-GPU OV2-4B tracking evaluation for 4 datasets:
  - ref-davis17  (Ref-DAVIS 2017, 244 queries, valid split)
  - mevis        (MeViS, 793 queries, valid_u split)
  - ref-yt-vos   (Ref-YouTube-VOS, 834 queries, valid split)
  - reasonvos    (ReasonVOS, 458 queries, test split)

Usage:
  # Multi-GPU evaluation (8 GPUs)
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 --master-port 29521 \
      eval_ov2_tracking.py --model-path /ov2/feilong/LLaVA-OneVision-2.0/examples/llava_onevision2/convert/tracking_molmo2_200k_iter_0001000_hf --task ref-davis17

  # Single-GPU
  python eval_ov2_tracking.py --model-path /ov2/feilong/LLaVA-OneVision-2.0/examples/llava_onevision2/convert/tracking_molmo2_200k_iter_0001000_hf --task ref-davis17 --gpu 0
"""

import argparse
import ast
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import datasets
import decord
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


TASK_CONFIGS = {
    "ref-davis17": {
        "hf_subpath": "tracking/ref-davis17/track/valid",
        "videos_subpath": "Ref-DAVIS17/valid/videos",
        "masks_subpath": "Ref-DAVIS17/valid/MasksRLE",
        "display_name": "Ref-DAVIS17",
    },
    "mevis": {
        "hf_subpath": "tracking/mevis",
        "videos_subpath": "MeViS/valid_u/videos",
        "masks_subpath": "MeViS/valid_u/MasksRLE",
        "display_name": "MeViS",
    },
    "ref-yt-vos": {
        "hf_subpath": "tracking/ref-yt-vos",
        "videos_subpath": "Ref-YT-VOS/valid/videos",
        "masks_subpath": "Ref-YT-VOS/valid/MasksRLE",
        "display_name": "Ref-YouTube-VOS",
    },
    "reasonvos": {
        "hf_subpath": "tracking/reasonvos",
        "videos_subpath": "ReasonVOS/videos",
        "masks_subpath": "ReasonVOS/MasksRLE",
        "display_name": "ReasonVOS",
    },
}


# ---------------------------------------------------------------------------
# Molmo2-VideoTrackEval (single HF dataset, filtered per sub-task)
# ---------------------------------------------------------------------------
VIDEOTRACK_HF_PATH = "/ov2/feilong/reproduce/molmo_data/video_datasets/video_track/Molmo2-VideoTrackEval"
VIDEOTRACK_VIDEOS_ROOT = "/ov2/feilong/reproduce/molmo_data/video_datasets/video_track"
VIDEOTRACK_SOURCE_TO_DIR = {
    "APTv2/videos": "APTv2/videos",
    "dancetrack/videos_val_split_5s/": "DanceTrack/videos/val",
    "dancetrack/videos_val_split_10s/": "DanceTrack/videos/val",
    "sam-v/videos_test_fps6/": "sav/sav_test/videos_fps6",
    "personpath22/videos_test_split_5s/": "personpath22/videos/test",
    "personpath22/videos_test_split_10s/": "personpath22/videos/test",
    "SportsMOT/videos_val_split_5s/": "SportsMOT/videos/val",
    "SportsMOT/videos_val_split_10s/": "SportsMOT/videos/val",
}

TASK_CONFIGS.update({
    "animal":  {"schema": "videotrack", "filter_dataset": "APTv2",        "display_name": "Molmo2-Animal"},
    "dance":   {"schema": "videotrack", "filter_dataset": "dancetrack",   "display_name": "Molmo2-Dance"},
    "misc":    {"schema": "videotrack", "filter_dataset": "sav",          "display_name": "Molmo2-Misc"},
    "person":  {"schema": "videotrack", "filter_dataset": "personpath22", "display_name": "Molmo2-Person"},
    "sports":  {"schema": "videotrack", "filter_dataset": "sportsmot",    "display_name": "Molmo2-Sports"},
})


def resolve_videotrack_video_path(ex):
    sub = VIDEOTRACK_SOURCE_TO_DIR.get(ex["video_source"])
    if sub is None:
        raise KeyError(f"Unknown video_source: {ex['video_source']}")
    return os.path.join(VIDEOTRACK_VIDEOS_ROOT, sub, f"{ex['clip']}.mp4")


def build_gt_tracks_videotrack(ex, sampling_fps=10):
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



# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# OV2 model loading and inference (chat mode)
# ---------------------------------------------------------------------------

def smart_resize(height, width, patch_size=14, min_pixels=None, max_pixels=None, max_resolution=None):
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid size: height={height}, width={width}")
    scale = 1.0
    pixels = height * width
    if min_pixels and pixels < min_pixels:
        scale = math.sqrt(min_pixels / pixels)
    if max_pixels and pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
    align_size = patch_size * 2
    resized_h = max(align_size, int(round(height * scale / align_size) * align_size))
    resized_w = max(align_size, int(round(width * scale / align_size) * align_size))
    # Clamp each dimension to max_resolution if specified
    if max_resolution and (resized_h > max_resolution or resized_w > max_resolution):
        clamp_scale = min(max_resolution / resized_h, max_resolution / resized_w)
        resized_h = max(align_size, int(round(resized_h * clamp_scale / align_size) * align_size))
        resized_w = max(align_size, int(round(resized_w * clamp_scale / align_size) * align_size))
    return resized_h, resized_w


def _format_dense_seconds(seconds: float) -> str:
    return f"{float(seconds):.2f}"


def _inject_timestamps_to_chat_text(text, visual_timestamps):
    """Inject <X.XX seconds> before each <|vision_start|> token."""
    token_iter = iter(visual_timestamps)

    def _replacer(match):
        try:
            ts = next(token_iter)
        except StopIteration:
            return match.group(0)
        if ts:
            return f"{ts}{match.group(0)}"
        return match.group(0)

    return re.sub(r"<\|vision_start\|>", _replacer, text)


def _recommended_tracking_max_new_tokens(num_frames, requested_max_new_tokens):
    """Auto-scale max_new_tokens based on frame count."""
    requested = max(int(requested_max_new_tokens), 1)
    dense_budget = 512 + max(int(num_frames), 1) * 128
    return max(requested, min(dense_budget, 32768))


def build_patch_positions(num_frames, total_frames, h, w, frame_indices=None, device=None):
    if torch.is_tensor(h):
        if device is None:
            device = h.device
        h = h.item()
    if torch.is_tensor(w):
        if device is None:
            device = w.device
        w = w.item()
    if frame_indices is None:
        frame_indices = torch.linspace(0, total_frames - 1, num_frames, device=device).long()
    else:
        if not torch.is_tensor(frame_indices):
            frame_indices = torch.as_tensor(frame_indices, device=device)
        elif device is not None:
            frame_indices = frame_indices.to(device=device)
        num_frames = len(frame_indices)
    frame_indices = frame_indices.to(dtype=torch.long)
    device = frame_indices.device
    t_ids = frame_indices.repeat_interleave(h * w)
    h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(num_frames)
    w_ids = torch.arange(w, device=device).repeat(h).repeat(num_frames)
    return torch.stack([t_ids, h_ids, w_ids], dim=-1)


def extract_video_frames_pil(video_path, max_frames=64, patch_size=14,
                             min_pixels=None, max_pixels=None, max_resolution=None,
                             fixed_num_frames=None, target_fps=None,
                             source_fps=None):
    """Extract frames from video as PIL Images, with optional resize.
    
    When source_fps and target_fps are both given, uses stride-based sampling
    (step = source_fps / target_fps) matching the reference pipeline.
    """
    vr = decord.VideoReader(video_path)
    frame_count = len(vr)
    fps = vr.get_avg_fps()
    if not fps or fps <= 0:
        fps = 30.0

    # Use provided source_fps if given, else use video's native fps
    effective_source_fps = source_fps if (source_fps and source_fps > 0) else fps

    # Stride-based sampling (matching reference pipeline)
    if target_fps is not None and target_fps > 0:
        step = max(float(effective_source_fps) / float(target_fps), 1.0)
        selected = []
        position = 0.0
        while len(selected) < max_frames:
            frame_idx = int(round(position))
            if frame_idx >= frame_count:
                break
            if not selected or frame_idx != selected[-1]:
                selected.append(frame_idx)
            position += step
        if not selected:
            selected = [0]
    elif fixed_num_frames is not None:
        target_count = fixed_num_frames
        if frame_count <= target_count:
            selected = list(range(frame_count))
        else:
            selected = np.linspace(0, frame_count - 1, target_count, dtype=int).tolist()
    else:
        duration = frame_count / fps
        if duration < 10:
            target_count = 8
        elif duration < 30:
            target_count = 16
        else:
            target_count = max_frames
        if frame_count <= target_count:
            selected = list(range(frame_count))
        else:
            selected = np.linspace(0, frame_count - 1, target_count, dtype=int).tolist()

    frames_pil = []
    for idx in selected:
        frame = vr[idx].asnumpy()
        if min_pixels or max_pixels:
            rh, rw = smart_resize(frame.shape[0], frame.shape[1], patch_size, min_pixels, max_pixels, max_resolution)
            if (rh, rw) != (frame.shape[0], frame.shape[1]):
                interp = cv2.INTER_AREA if rh < frame.shape[0] else cv2.INTER_LINEAR
                frame = cv2.resize(frame, (rw, rh), interpolation=interp)
        frames_pil.append(Image.fromarray(frame))

    return frames_pil, selected, fps


def load_model_and_processor(model_path, device, max_pixels=313600, min_pixels=56*56):
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype="auto", attn_implementation="flash_attention_2",
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(
        model_path, max_pixels=max_pixels, min_pixels=min_pixels, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, processor, tokenizer


def run_inference(model, processor, tokenizer, device, video_path, prompt,
                  max_pixels=313600, min_pixels=56*56, max_resolution=None,
                  fixed_num_frames=64, max_new_tokens=2048, temperature=0.0,
                  target_fps=None, source_fps=None):
    """Run OV2 inference using chat mode with patch_positions."""
    from qwen_vl_utils import process_vision_info

    # Extract frames (prefer fps-based sampling; fixed_num_frames is the cap)
    frames_pil, frame_indices, vid_fps = extract_video_frames_pil(
        video_path, max_frames=fixed_num_frames,
        patch_size=14, min_pixels=min_pixels, max_pixels=max_pixels,
        max_resolution=max_resolution,
        fixed_num_frames=None if target_fps else fixed_num_frames,
        target_fps=target_fps,
        source_fps=source_fps,
    )

    # Get original video dimensions
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    vid_h, vid_w = vr[0].shape[:2]
    del vr

    num_frames = len(frames_pil)

    # Build OpenAI-style messages with individual images (chat mode)
    system_prompt = "You are a helpful assistant."
    image_content = [{"type": "image", "image": img} for img in frames_pil]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": image_content + [{"type": "text", "text": prompt}]},
    ]

    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Inject timestamps before each <|vision_start|> token
    # Use source_fps (from dataset) for timestamp computation, matching reference pipeline
    ts_fps = source_fps if (source_fps and source_fps > 0) else vid_fps
    visual_timestamps = [
        f"<{_format_dense_seconds(float(sel_idx) / ts_fps)} seconds>"
        for sel_idx in frame_indices
    ]
    text = _inject_timestamps_to_chat_text(text, visual_timestamps)

    # Process vision info using qwen_vl_utils
    image_inputs, video_inputs = process_vision_info(messages)

    # Process inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Build patch_positions (do NOT collapse image_grid_thw — keep per-image entries)
    image_grid_thw = inputs.get("image_grid_thw", None)
    patch_positions = None
    if image_grid_thw is not None:
        target_device = inputs.input_ids.device
        # Sequential frame indices [0,1,...,N-1] to match training (ord_t variant)
        seq_frame_indices = torch.arange(num_frames, device=target_device, dtype=torch.long)
        patch_positions = build_patch_positions(
            num_frames=num_frames,
            total_frames=num_frames,
            h=image_grid_thw[0][1],
            w=image_grid_thw[0][2],
            frame_indices=seq_frame_indices,
            device=target_device,
        )

    # Filter unsupported keys
    unsupported_keys = ["second_per_grid_ts", "mm_token_type_ids"]
    filtered_inputs = {k: v for k, v in inputs.items() if k not in unsupported_keys}

    # Generate
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    gen_args = {
        **filtered_inputs,
        "max_new_tokens": _recommended_tracking_max_new_tokens(num_frames, max_new_tokens),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": pad_token_id,
        "num_beams": 1,
        "use_cache": True,
        "patch_positions": patch_positions,
    }
    if temperature > 0:
        gen_args.update(do_sample=True, temperature=temperature, top_p=0.9)

    with torch.inference_mode():
        outputs = model.generate(**gen_args)

    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0].strip()
    return answer, vid_w, vid_h


# ---------------------------------------------------------------------------
# Track parsing (identical to eval_tracking.py)
# ---------------------------------------------------------------------------

_COORD_REGEX = re.compile(r'<(?:points|tracks).*? coords="([0-9\t:;, .]+)"?/?>?')
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


# ---------------------------------------------------------------------------
# Mask and evaluation helpers (identical to eval_tracking.py)
# ---------------------------------------------------------------------------

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


def load_masks_at_frame(gt_masks, frame_idx, height, width, return_dict=False):
    empty = np.zeros((height, width), dtype=bool)
    masks = []
    masks_by_id = {}
    for mask_id, mask_list in gt_masks.items():
        first = next((m for m in mask_list if m is not None), None)
        if first is None:
            masks.append(empty)
            masks_by_id[str(mask_id)] = empty
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
        masks_by_id[str(mask_id)] = binary
    if return_dict:
        return masks_by_id
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
    # Only evaluate on GT frames (model may predict at denser timestamps)
    all_frames = sorted(gt_by.keys())
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
        # Only evaluate on GT frames (model may predict at denser timestamps)
        all_frames = sorted(gt_by.keys())
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
                masks_by_id = load_masks_at_frame(gt_masks, fidx, height, width, return_dict=True)
                empty_mask = np.zeros((height, width), dtype=bool)
                gt_oids_ordered = [oid for oid, _ in sorted(gf['points'].items())]
                for i, oid in enumerate(gt_oids_ordered):
                    mask = masks_by_id.get(str(oid), empty_mask)
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
    "Track the {label}",                       # 0 original default (no period)
    "Track the {label}.",                      # 1 with period
    "Track {label}.",                          # 2 no article
    "Please track {label}.",                   # 3 polite
    "Locate and trace {label}.",               # 4 alt verb
    "Where is {label}?",                       # 5 question
    "Can you track {label} in the video?",     # 6 full question
]


def parse_args():
    p = argparse.ArgumentParser(description="OV2-4B tracking evaluation")
    p.add_argument("--model-path", required=True, help="Path to OV2 model")
    p.add_argument("--task", required=True, choices=list(TASK_CONFIGS.keys()),
                   help="Dataset to evaluate on")
    p.add_argument("--data-dir", default=None, help="Root data directory")
    p.add_argument("--gpu", type=int, default=0, help="GPU id for single-GPU mode")
    p.add_argument("--max-examples", type=int, default=-1)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--sampling-fps", type=float, default=1,
                   help="Sample fps for stride-based frame selection (ref default=1, step=source_fps/sampling_fps)")
    p.add_argument("--fixed-num-frames", type=int, default=128,
                   help="Max number of frames (cap). Actual count decided by sampling-fps.")
    p.add_argument("--max-pixels", type=int, default=313600,
                   help="Max pixels per frame for resize")
    p.add_argument("--min-pixels", type=int, default=3136,
                   help="Min pixels per frame for resize (default=256*28*28)")
    p.add_argument("--max-resolution", type=int, default=None,
                   help="Max height/width per frame (e.g. 560). Overrides max-pixels if set.")
    p.add_argument("--template-id", type=int, default=0,
                   help="Prompt template index (0-9). -1=random per example")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--prompt-suffix", default="", help="Append this string after the prompt template")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.max_resolution is not None:
        args.max_pixels = args.max_resolution * args.max_resolution
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
    output_dir = (Path(args.output_dir) / args.task) if args.output_dir else script_dir / "eval_output_ov2" / args.task
    log_dir = script_dir / "logs"
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"ov2_{args.task}.log")
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
    if args.smoke_test:
        args.max_examples = 5
    if args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))
    total = len(ds)

    indices = list(range(rank, total, world_size))
    if is_main_process():
        log.info(f"Total examples: {total}, world_size: {world_size}")
        log.info(f"Loading model from {args.model_path}")
    model, processor, tokenizer = load_model_and_processor(
        args.model_path, device,
        max_pixels=args.max_pixels, min_pixels=args.min_pixels,
    )
    if is_main_process():
        log.info("Model loaded on all ranks")
    barrier()

    local_results = []
    start_time = time.time()

    for i, idx in enumerate(indices):
        ex = ds[idx]
        sfps = args.sampling_fps
        # videotrack: visual sampling at 10 FPS to match Molmo2 reference pipeline
        vis_sfps = 10 if is_vt else sfps
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
        if args.prompt_suffix:
            prompt = prompt + args.prompt_suffix
        try:
            ans, vw, vh = run_inference(
                model, processor, tokenizer, device, vpath, prompt,
                max_pixels=args.max_pixels, min_pixels=args.min_pixels, max_resolution=args.max_resolution,
                fixed_num_frames=args.fixed_num_frames,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                target_fps=vis_sfps,
                source_fps=vfps,
            )
            vw = vw or w
            vh = vh or h
            if i < 2:
                log.info(f"[rank {rank}] Model output (first 500 chars): {ans[:500]}")
                log.info(f"[rank {rank}] pred_tracks count: {len(extract_tracks(ans, vw, vh, vfps))}")
            pred_tracks = extract_tracks(ans, vw, vh, vfps)
            if is_vt:
                gt_tracks = build_gt_tracks_videotrack(ex, sampling_fps=vis_sfps)
                gt_masks = build_gt_masks_videotrack(ex)
            else:
                gt_tracks = build_gt_tracks(ex)
                _raw_masks = json.load(open(mask_path))
                # Remap mask keys so gt_masks[str(i)] aligns with gt_tracks integer
                # oid=i (see build_gt_tracks). Prefer direct match on mask_id[i],
                # else fall back to the i-th key in the raw mask dict.
                _raw_keys = list(_raw_masks.keys())
                _mids = ex['mask_id']
                gt_masks = {}
                for i, mid in enumerate(_mids):
                    if mid in _raw_masks:
                        gt_masks[str(i)] = _raw_masks[mid]
                    elif i < len(_raw_keys):
                        gt_masks[str(i)] = _raw_masks[_raw_keys[i]]
            metrics = evaluate_video_object_tracking(pred_tracks, gt_tracks, gt_masks, h, w)
            local_results.append({
                "idx": idx, "id": ex["id"], "video": vname, "qid": qid,
                "expression": expr, "prediction": ans,
                "metrics": {k: v for k, v in metrics.items()
                            if not isinstance(v, (list, np.ndarray))},
            })
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(indices) - i - 1)
            log.info(
                f"[rank {rank}] [{i+1}/{len(indices)}] {vname}/{qid}  "
                f"F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                f"HOTA={metrics['HOTA']:.3f} ETA={eta:.0f}s"
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
