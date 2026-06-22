"""
SAM2-based segmentation evaluation for tracking predictions.

Reads predictions.json (point-tracking output from eval_tracking.py),
feeds the predicted points into SAM2 to produce segmentation masks,
then computes J&F metrics against ground-truth masks.

Supports 4 datasets:
  - ref-davis17   (Ref-DAVIS 2017, 244 queries)
  - mevis         (MeViS valid_u, 793 queries)
  - ref-yt-vos    (Ref-YouTube-VOS, 834 queries)
  - reasonvos     (ReasonVOS, 458 queries)

Directory layout expected:
  simple_repo/
  ├── data/
  │   ├── Ref-DAVIS17/valid/MasksRLE/{video}/{qid}.json   (GT masks in RLE)
  │   ├── MeViS/valid_u/MasksRLE/{video}/{qid}.json
  │   ├── Ref-YT-VOS/valid/MasksRLE/{video}/{qid}.json
  │   ├── ReasonVOS/MasksRLE/{video}/{qid}.json
  │   ├── ReasonVOS/videos/{video}.mp4                    (source videos)
  │   └── tracking/{task}/                                (HF datasets with metadata)
  ├── eval_output/
  │   └── {task}/predictions.json                         (input point predictions)
  └── eval_sam2_tracking.py                               (this script)

Output:
  eval_output/{task}/sam2_results/
  ├── sam2_metrics.json          (J, F, J&F, HOTA scores)
  └── sam2_predictions.json      (per-query detailed results)

Usage (8-GPU):
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 --master-port 29600 \
      eval_sam2_tracking.py --task ref-davis17

  # All 4 datasets sequentially:
  for task in ref-davis17 mevis ref-yt-vos reasonvos; do
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 --master-port 29600 \
          eval_sam2_tracking.py --task $task
  done
"""

import argparse
import copy
import importlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
import yaml
from pycocotools import mask as mask_utils
from skimage.morphology import disk

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
EVAL_OUTPUT_DIR = SCRIPT_DIR / "eval_output"

DEFAULT_SAM2_CODE_DIR = "/ov2/zwk/lmms-eval-ov2/extension/sam2_1"
DEFAULT_SAM2_CONFIG_PATH = "/ov2/zwk/lmms-eval-ov2/extension/sam2.1_hiera_large.yaml"
DEFAULT_SAM2_CHECKPOINT_PATH = "/ov2/zwk/lmms-eval-ov2/extension/sam2.1_hiera_large.pt"

# Regex patterns for parsing prediction text
COORD_REGEX = re.compile(r'<(?:points|tracks).*? coords="([0-9\t:;, .]+)"/?>')
FRAME_REGEX = re.compile(r"(?:^|\t|:|,|;)([0-9.]+) ([0-9. ]+)")
POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{1,4}) ([0-9]{1,4})")

INT_PATTERN = re.compile(r"^[+-]?\d+$")
FLOAT_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

ALPHA_THRESHOLDS = np.arange(0.05, 1.0, 0.05)

# Dataset configurations
TASK_CONFIGS = {
    "ref-davis17": {
        "hf_subpath": "tracking/ref-davis17/track/valid",
        "frames_dir": "/video_vit/tracking/vosdata/Ref-DAVIS17/valid/JPEGImages",
        "masks_subpath": "Ref-DAVIS17/valid/MasksRLE",
        "display_name": "Ref-DAVIS17",
        "source": "jpeg",
    },
    "mevis": {
        "hf_subpath": "tracking/mevis",
        "frames_dir": "/video_vit/tracking/vosdata/MeViS/valid_u/JPEGImages",
        "masks_subpath": "MeViS/valid_u/MasksRLE",
        "display_name": "MeViS",
        "source": "jpeg",
    },
    "ref-yt-vos": {
        "hf_subpath": "tracking/ref-yt-vos",
        "frames_dir": "/video_vit/tracking/vosdata/Ref-YTB-VOS/valid/JPEGImages",
        "masks_subpath": "Ref-YT-VOS/valid/MasksRLE",
        "display_name": "Ref-YouTube-VOS",
        "source": "jpeg",
    },
    "reasonvos": {
        "hf_subpath": "tracking/reasonvos",
        "frames_dir": None,
        "videos_dir": "ReasonVOS/videos",
        "masks_subpath": "ReasonVOS/MasksRLE",
        "display_name": "ReasonVOS",
        "source": "mp4",
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
    "animal":  {"schema": "videotrack", "filter_dataset": "APTv2",        "display_name": "Molmo2-Animal",  "source": "videotrack"},
    "dance":   {"schema": "videotrack", "filter_dataset": "dancetrack",   "display_name": "Molmo2-Dance",   "source": "videotrack"},
    "misc":    {"schema": "videotrack", "filter_dataset": "sav",          "display_name": "Molmo2-Misc",    "source": "videotrack"},
    "person":  {"schema": "videotrack", "filter_dataset": "personpath22", "display_name": "Molmo2-Person",  "source": "videotrack"},
    "sports":  {"schema": "videotrack", "filter_dataset": "sportsmot",    "display_name": "Molmo2-Sports",  "source": "videotrack"},
})


def _resolve_videotrack_video_path(ex):
    sub = VIDEOTRACK_SOURCE_TO_DIR.get(ex["video_source"])
    if sub is None:
        raise KeyError(f"Unknown video_source: {ex['video_source']}")
    return os.path.join(VIDEOTRACK_VIDEOS_ROOT, sub, f"{ex['clip']}.mp4")


def load_gt_masks_videotrack(masks_list, num_frames, height, width):
    """Decode inline RLE masks from Molmo2-VideoTrackEval record to (T,H,W) uint8."""
    gt = np.zeros((num_frames, height, width), dtype=np.uint8)
    for obj in (masks_list or []):
        frames = obj.get("masks", []) or []
        for fi, fm in enumerate(frames):
            if fi >= num_frames:
                break
            if not fm or not fm.get("counts"):
                continue
            counts = fm["counts"]
            if isinstance(counts, str):
                counts = counts.encode("ascii")
            rle = {"counts": counts, "size": list(fm["size"])}
            decoded = mask_utils.decode(rle)
            if decoded.shape != (height, width):
                decoded = cv2.resize(decoded, (width, height), interpolation=cv2.INTER_NEAREST)
            gt[fi] = np.maximum(gt[fi], decoded)
    return gt



# ===================================================================
# 1. Distributed helpers
# ===================================================================

def setup_distributed():
    """Initialize distributed process group (torchrun) or single-GPU fallback."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return rank, world_size, device
    return 0, 1, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


# ===================================================================
# 2. Parse prediction text -> points
# ===================================================================

def parse_prediction_points(text):
    """
    Parse prediction text like:
      <tracks coords="0.0 1 398 729;1.0 1 398 690;...">...</tracks>

    Returns list of (timestamp, point_id, x_norm, y_norm) where x,y in [0,1000].
    """
    all_points = []
    for coord_match in COORD_REGEX.finditer(text):
        for frame_match in FRAME_REGEX.finditer(coord_match.group(1)):
            timestamp = float(frame_match.group(1))
            point_str = frame_match.group(2)
            for pt_match in POINTS_REGEX.finditer(point_str):
                point_id = int(pt_match.group(1))
                x = float(pt_match.group(2))
                y = float(pt_match.group(3))
                all_points.append((timestamp, point_id, x, y))
    return all_points


def build_sam2_prompt_map(parsed_points, video_width, video_height, video_fps, num_frames):
    """
    Convert parsed points to SAM2 prompt map.
    Returns: {(frame_idx, obj_id_str): [(x_abs, y_abs), ...]}
    """
    prompt_map = defaultdict(list)

    for timestamp, point_id, x_norm, y_norm in parsed_points:
        frame_idx = int(round(timestamp * video_fps))
        frame_idx = max(0, min(frame_idx, num_frames - 1))

        # Convert from 1000-scale normalized to absolute pixel coords
        x_abs = x_norm / 1000.0 * video_width
        y_abs = y_norm / 1000.0 * video_height
        x_abs = max(0.0, min(x_abs, float(video_width - 1)))
        y_abs = max(0.0, min(y_abs, float(video_height - 1)))

        obj_id = str(point_id)
        prompt_map[(frame_idx, obj_id)].append((x_abs, y_abs))

    return dict(prompt_map)


# ===================================================================
# 3. SAM2 predictor building
# ===================================================================

def _setup_sam2_import_path(sam2_code_dir):
    sam2_code_dir = Path(sam2_code_dir).resolve()
    if not sam2_code_dir.exists():
        raise FileNotFoundError(f"SAM2 code directory not found: {sam2_code_dir}")
    parent = str(sam2_code_dir.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def _rewrite_target_path(target):
    if target.startswith("damsam."):
        return target.replace("damsam.", "sam2_1.", 1)
    return target


def _rewrite_targets_in_config(cfg):
    if isinstance(cfg, dict):
        return {
            k: (_rewrite_target_path(v) if k == "_target_" and isinstance(v, str)
                else _rewrite_targets_in_config(v))
            for k, v in cfg.items()
        }
    if isinstance(cfg, list):
        return [_rewrite_targets_in_config(x) for x in cfg]
    return cfg


def _coerce_config_scalars(cfg):
    if isinstance(cfg, dict):
        return {k: _coerce_config_scalars(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_coerce_config_scalars(v) for v in cfg]
    if isinstance(cfg, str):
        value = cfg.strip()
        if INT_PATTERN.fullmatch(value):
            try:
                return int(value)
            except ValueError:
                return cfg
        if FLOAT_PATTERN.fullmatch(value):
            try:
                return float(value)
            except ValueError:
                return cfg
    return cfg


def _resolve_target(target):
    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _instantiate_from_config(cfg):
    if isinstance(cfg, dict):
        if "_target_" in cfg:
            cls = _resolve_target(str(cfg["_target_"]))
            kwargs = {k: _instantiate_from_config(v) for k, v in cfg.items() if k != "_target_"}
            return cls(**kwargs)
        return {k: _instantiate_from_config(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_instantiate_from_config(v) for v in cfg]
    return cfg


def build_sam2_video_predictor(sam2_code_dir, config_path, checkpoint_path, device):
    """Build a SAM2 video predictor from config YAML + checkpoint."""
    _setup_sam2_import_path(sam2_code_dir)

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"SAM2 config not found: {config_file}")
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_file}")

    with config_file.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    if not isinstance(raw_cfg, dict) or "model" not in raw_cfg:
        raise ValueError(f"Invalid SAM2 config: {config_file}")

    model_cfg = _coerce_config_scalars(_rewrite_targets_in_config(copy.deepcopy(raw_cfg["model"])))
    model_cfg["_target_"] = "sam2_1.sam2_video_predictor.SAM2VideoPredictor"
    model_cfg.setdefault("binarize_mask_from_pts_for_mem_enc", True)
    model_cfg.setdefault("fill_hole_area", 8)
    extra = model_cfg.get("sam_mask_decoder_extra_args") or {}
    extra.setdefault("dynamic_multimask_via_stability", True)
    extra.setdefault("dynamic_multimask_stability_delta", 0.05)
    extra.setdefault("dynamic_multimask_stability_thresh", 0.98)
    model_cfg["sam_mask_decoder_extra_args"] = extra

    predictor = _instantiate_from_config(model_cfg)
    checkpoint = torch.load(str(checkpoint_file), map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    predictor.load_state_dict(state_dict, strict=True)
    predictor = predictor.to(device).eval()
    return predictor


# ===================================================================
# 4. SAM2 segmentation
# ===================================================================

def mask_logits_to_uint8(mask_logits):
    """Convert SAM2 mask logits to binary uint8 mask (merged across objects)."""
    logits = mask_logits.detach()
    if logits.ndim == 4:
        logits = logits.squeeze(1)
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    binary = logits > 0
    merged = torch.any(binary, dim=0).to(torch.uint8)
    return merged.cpu()


def run_sam2_segmentation(predictor, prompt_map, inference_state):
    """
    Run SAM2 video segmentation given an inference state and prompt map.

    Args:
        predictor: SAM2 video predictor
        prompt_map: {(frame_idx, obj_id): [(x, y), ...]} absolute pixel coords
        inference_state: pre-initialized SAM2 state (from predictor.init_state)
            Caller should call predictor.reset_state(inference_state) before this
            if reusing state for a new expression on the same video.
    """
    num_frames = int(inference_state["num_frames"])
    video_h = int(inference_state["video_height"])
    video_w = int(inference_state["video_width"])

    frame_to_mask = {}
    sorted_prompts = sorted(prompt_map.items(), key=lambda item: (item[0][0], item[0][1]))

    if sorted_prompts:
        # Group prompts by frame index
        prompts_by_frame = defaultdict(list)
        for (frame_idx, obj_id), points in sorted_prompts:
            prompts_by_frame[frame_idx].append((obj_id, points))
        prompt_frames = sorted(prompts_by_frame)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available() and "cuda" in str(inference_state["device"])
            else nullcontext()
        )

        with torch.inference_mode(), autocast_ctx:
            # Forward propagation
            for idx, frame_idx in enumerate(prompt_frames):
                for obj_id, points in prompts_by_frame[frame_idx]:
                    points_tensor = torch.tensor(points, dtype=torch.float32)
                    labels_tensor = torch.ones((len(points),), dtype=torch.int32)
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=points_tensor,
                        labels=labels_tensor,
                        clear_old_points=True,
                        normalize_coords=True,
                    )

                next_prompt_frame = (
                    prompt_frames[idx + 1] if idx + 1 < len(prompt_frames) else num_frames
                )
                max_track = max(next_prompt_frame - frame_idx - 1, 0)
                for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=frame_idx,
                    max_frame_num_to_track=max_track,
                    reverse=False,
                ):
                    frame_to_mask[int(out_frame_idx)] = mask_logits_to_uint8(out_mask_logits)

            # Backward propagation from first prompt frame to frame 0
            first_prompt_frame = prompt_frames[0]
            if first_prompt_frame > 0:
                for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=first_prompt_frame,
                    max_frame_num_to_track=first_prompt_frame,
                    reverse=True,
                ):
                    frame_to_mask[int(out_frame_idx)] = mask_logits_to_uint8(out_mask_logits)

    # Assemble masks for all frames
    empty_mask = torch.zeros((video_h, video_w), dtype=torch.uint8)
    mask_prediction = []
    for frame_idx in range(num_frames):
        mask = frame_to_mask.get(frame_idx, empty_mask)
        mask_prediction.append(mask.numpy().astype(np.uint8))

    return {
        "mask_prediction": mask_prediction,
        "num_video_frames": num_frames,
        "video_height": video_h,
        "video_width": video_w,
        "num_prompt_frames": len({fi for fi, _ in prompt_map}),
        "num_prompt_points": sum(len(pts) for pts in prompt_map.values()),
    }


# ===================================================================
# 5. Frame extraction from mp4 (for ReasonVOS)
# ===================================================================

def extract_frames_from_video(video_path, output_dir, fps=6.0):
    """Extract frames from mp4 video to numbered JPG files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(out / "%05d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return sorted(out.glob("*.jpg"))


# ===================================================================
# 6. J&F metrics (self-contained copy)
# ===================================================================

def _seg2bmap(seg, width=None, height=None):
    """Compute binary boundary map from segmentation mask."""
    seg = seg.astype(bool)
    h, w = seg.shape[:2]
    width = w if width is None else width
    height = h if height is None else height

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        return b
    bmap = np.zeros((height, width))
    for x in range(w):
        for y in range(h):
            if b[y, x]:
                j = 1 + math.floor((y - 1) + height / h)
                i = 1 + math.floor((x - 1) + width / h)
                bmap[j, i] = 1
    return bmap


def f_measure(foreground_mask, gt_mask, bound_th=0.008):
    """Compute boundary F-measure between foreground_mask and gt_mask."""
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    bound_pix = bound_th if bound_th >= 1 else np.ceil(
        bound_th * np.linalg.norm(foreground_mask.shape)
    )

    fg_boundary = _seg2bmap(foreground_mask)
    gt_boundary = _seg2bmap(gt_mask)

    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    if n_fg == 0 and n_gt > 0:
        precision, recall = 1, 0
    elif n_fg > 0 and n_gt == 0:
        precision, recall = 0, 1
    elif n_fg == 0 and n_gt == 0:
        precision, recall = 1, 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def db_eval_iou(annotation, segmentation):
    """Compute Jaccard Index (IoU) per frame. Shape: (T, H, W)."""
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)
    inters = np.sum(segmentation & annotation, axis=(-2, -1))
    union = np.sum(segmentation | annotation, axis=(-2, -1))
    j = inters / union
    if j.ndim == 0:
        j = 1.0 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1.0
    return j


def db_eval_boundary(annotation, segmentation, bound_th=0.008):
    """Compute boundary F-measure per frame. Shape: (T, H, W)."""
    assert annotation.shape == segmentation.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for i in range(n_frames):
            f_res[i] = f_measure(segmentation[i], annotation[i], bound_th=bound_th)
        return f_res
    elif annotation.ndim == 2:
        return f_measure(segmentation, annotation, bound_th=bound_th)
    else:
        raise ValueError(f"Unsupported ndim: {annotation.ndim}")


def compute_hota_single_object(gt_masks, pred_masks):
    """Compute HOTA for single-object mask-level tracking."""
    num_frames = gt_masks.shape[0]
    num_alphas = len(ALPHA_THRESHOLDS)

    frame_ious = np.zeros(num_frames)
    for t in range(num_frames):
        intersection = np.logical_and(gt_masks[t] > 0, pred_masks[t] > 0).sum()
        union = np.logical_or(gt_masks[t] > 0, pred_masks[t] > 0).sum()
        frame_ious[t] = intersection / union if union > 0 else 1.0

    gt_exists = np.array([gt_masks[t].sum() > 0 for t in range(num_frames)])
    pred_exists = np.array([pred_masks[t].sum() > 0 for t in range(num_frames)])

    hota_per_alpha = np.zeros(num_alphas)
    for a, alpha in enumerate(ALPHA_THRESHOLDS):
        tp, fn, fp = 0, 0, 0
        for t in range(num_frames):
            has_gt, has_pred = gt_exists[t], pred_exists[t]
            if not has_gt and not has_pred:
                continue
            if has_gt and has_pred and frame_ious[t] >= alpha:
                tp += 1
            else:
                if has_gt:
                    fn += 1
                if has_pred:
                    fp += 1
        if tp + fn + fp == 0:
            hota_per_alpha[a] = 1.0
        else:
            det_a = tp / (tp + fn + fp)
            hota_per_alpha[a] = np.sqrt(det_a)  # AssA=1 for single object
    return float(np.mean(hota_per_alpha))


def evaluate_masks(gt_masks, pred_masks):
    """Compute J, F, J&F, HOTA for a video-expression pair."""
    j = db_eval_iou(gt_masks, pred_masks).mean()
    f = db_eval_boundary(gt_masks, pred_masks).mean()
    hota = compute_hota_single_object(gt_masks, pred_masks)
    return {"J": float(j), "F": float(f), "J&F": float((j + f) / 2), "HOTA": float(hota)}


# ===================================================================
# 7. GT mask loading
# ===================================================================

def load_gt_masks_from_rle(mask_rle_path, num_frames, height, width):
    """
    Load GT masks from per-expression MasksRLE JSON.
    Format: {anno_id: [rle_or_null_per_frame, ...]}
    Returns: np.ndarray (num_frames, H, W) uint8
    """
    with open(mask_rle_path, "r") as f:
        mask_data = json.load(f)

    gt_masks = np.zeros((num_frames, height, width), dtype=np.uint8)
    for anno_id, rle_list in mask_data.items():
        for frame_idx, rle in enumerate(rle_list):
            if frame_idx >= num_frames:
                break
            if rle is None:
                continue
            decoded = mask_utils.decode(rle)
            if decoded.shape != (height, width):
                decoded = cv2.resize(decoded, (width, height), interpolation=cv2.INTER_NEAREST)
            gt_masks[frame_idx] = np.maximum(gt_masks[frame_idx], decoded)
    return gt_masks


# ===================================================================
# 8. Frame directory resolution
# ===================================================================

def get_frame_dir_for_item(task_cfg, video_id, data_dir, tmp_root=None, video_path=None, video_fps=6.0):
    """
    Get frame directory for SAM2 input.
    For JPEG datasets: return existing JPEGImages dir.
    For mp4 datasets (ReasonVOS): extract frames to temp dir at fps=6.
    For videotrack datasets: extract all frames from the provided video_path at its native fps.
    Returns: (frame_dir_path, tmpdir_to_cleanup_or_None)
    """
    if task_cfg["source"] == "jpeg":
        frame_dir = Path(task_cfg["frames_dir"]) / video_id
        if not frame_dir.exists():
            raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
        return str(frame_dir), None
    elif task_cfg["source"] == "videotrack":
        if video_path is None or not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        tmpdir = tempfile.mkdtemp(prefix=f"sam2_{video_id}_", dir=tmp_root)
        extract_frames_from_video(str(video_path), tmpdir, fps=float(video_fps))
        return tmpdir, tmpdir
    else:
        videos_dir = data_dir / task_cfg["videos_dir"]
        video_path = videos_dir / f"{video_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        tmpdir = tempfile.mkdtemp(prefix=f"sam2_{video_id}_", dir=tmp_root)
        extract_frames_from_video(str(video_path), tmpdir, fps=6.0)
        return tmpdir, tmpdir


# ===================================================================
# 9. Main
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(description="SAM2 segmentation evaluation for tracking predictions")
    p.add_argument("--task", required=True, choices=list(TASK_CONFIGS.keys()),
                   help="Dataset to evaluate")
    p.add_argument("--data-dir", default=None,
                   help="Override data root (default: simple_repo/data)")
    p.add_argument("--predictions", default=None,
                   help="Path to predictions.json (default: eval_output/{task}/predictions.json)")
    p.add_argument("--output-dir", default=None,
                   help="Output dir (default: eval_output/{task}/sam2_results)")
    p.add_argument("--sam2-code-dir", default=DEFAULT_SAM2_CODE_DIR)
    p.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG_PATH)
    p.add_argument("--sam2-checkpoint", default=DEFAULT_SAM2_CHECKPOINT_PATH)
    p.add_argument("--video-fps", type=float, default=6.0,
                   help="Video FPS for timestamp-to-frame mapping")
    p.add_argument("--skip-errors", action="store_true",
                   help="Skip failed items instead of crashing")
    return p.parse_args()


def main():
    args = parse_args()
    task_cfg = TASK_CONFIGS[args.task]

    # ---- Distributed setup ----
    rank, world_size, device = setup_distributed()
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    masks_dir = (data_dir / task_cfg["masks_subpath"]) if task_cfg.get("masks_subpath") else None

    # ---- Input / output paths ----
    pred_path = (Path(args.predictions) if args.predictions
                 else EVAL_OUTPUT_DIR / args.task / "predictions.json")
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    output_dir = (Path(args.output_dir) if args.output_dir
                  else EVAL_OUTPUT_DIR / args.task / "sam2_results")
    if is_main():
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    # ---- Load predictions ----
    with open(pred_path, "r") as f:
        predictions = json.load(f)
    total = len(predictions)

    # ---- Load HF dataset for metadata (fps, width, height, n_frames) ----
    import datasets
    is_vt = task_cfg.get("schema") == "videotrack"
    if is_vt:
        hf_ds = datasets.load_from_disk(VIDEOTRACK_HF_PATH)
        hf_ds = hf_ds.filter(lambda ex, _f=task_cfg["filter_dataset"]: ex["video_dataset"] == _f)
    else:
        hf_path = data_dir / task_cfg["hf_subpath"]
        hf_ds = datasets.load_from_disk(str(hf_path))
    meta_lookup = {}
    for i in range(len(hf_ds)):
        ex = hf_ds[i]
        if is_vt:
            key = (ex["clip"], str(ex["id"]))
            try:
                vpath = _resolve_videotrack_video_path(ex)
            except KeyError as _e:
                log.warning(f"{_e}")
                continue
            meta_lookup[key] = {
                "fps": float(ex["fps"]), "width": int(ex["w"]),
                "height": int(ex["h"]), "n_frames": int(ex["n_frames"]),
                "video_path": vpath, "masks_raw": ex["masks"],
            }
        else:
            meta_lookup[(ex["video"], str(ex["qid"]))] = {
                "fps": ex["fps"], "width": ex["width"],
                "height": ex["height"], "n_frames": ex["n_frames"],
            }

    if is_main():
        log.info(f"Task: {task_cfg['display_name']}")
        log.info(f"Predictions: {pred_path} ({total} items)")
        log.info(f"Masks dir: {masks_dir}")
        log.info(f"World size: {world_size}")

    # ---- Build SAM2 predictor ----
    predictor = build_sam2_video_predictor(
        sam2_code_dir=args.sam2_code_dir,
        config_path=args.sam2_config,
        checkpoint_path=args.sam2_checkpoint,
        device=str(device),
    )
    if is_main():
        log.info("SAM2 predictor loaded")

    # ---- Group predictions by video for efficiency ----
    # init_state (loading frames) is expensive; grouping amortizes the cost
    video_groups = defaultdict(list)
    for pred_item in predictions:
        video_groups[pred_item["video"]].append(pred_item)
    video_ids = sorted(video_groups.keys())

    # Shard *videos* (not items) across GPUs so same-video items stay together
    my_video_ids = [video_ids[i] for i in range(rank, len(video_ids), world_size)]
    my_total_items = sum(len(video_groups[v]) for v in my_video_ids)

    if is_main():
        log.info(f"Grouped into {len(video_ids)} videos; this rank handles "
                 f"{len(my_video_ids)} videos ({my_total_items} items)")

    local_results = []
    t0 = time.time()
    items_done = 0

    for vid_step, video_id in enumerate(my_video_ids):
        items = video_groups[video_id]
        tmpdir_to_cleanup = None

        try:
            # Get frame directory (once per video)
            if task_cfg.get("schema") == "videotrack":
                _any_qid = next((str(it["qid"]) for it in items if (video_id, str(it["qid"])) in meta_lookup), None)
                _vm = meta_lookup.get((video_id, _any_qid)) if _any_qid else None
                if _vm is None:
                    raise FileNotFoundError(f"No videotrack metadata for {video_id}")
                frame_dir, tmpdir_to_cleanup = get_frame_dir_for_item(
                    task_cfg, video_id, data_dir,
                    tmp_root=os.environ.get("TMPDIR"),
                    video_path=_vm["video_path"],
                    video_fps=_vm["fps"],
                )
            else:
                frame_dir, tmpdir_to_cleanup = get_frame_dir_for_item(
                    task_cfg, video_id, data_dir,
                    tmp_root=os.environ.get("TMPDIR"),
                )

            # Init SAM2 state ONCE per video (expensive: loads & preprocesses all frames)
            inference_state = predictor.init_state(video_path=str(frame_dir))

            n_frame_files = len([
                f for f in os.listdir(frame_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

            # Process each expression for this video
            for item_step, pred_item in enumerate(items):
                qid = str(pred_item["qid"])
                prediction_text = pred_item["prediction"]

                meta = meta_lookup.get((video_id, qid))
                if meta is None:
                    log.warning(f"[rank {rank}] No metadata for {video_id}/{qid}, skip")
                    continue

                video_fps = meta["fps"]
                vid_w, vid_h = meta["width"], meta["height"]

                if task_cfg.get("schema") == "videotrack":
                    mask_path = None
                else:
                    mask_path = masks_dir / video_id / f"{qid}.json"
                    if not mask_path.exists():
                        log.warning(f"[rank {rank}] Missing GT mask: {mask_path}, skip")
                        continue

                try:
                    # Reset tracking state (keeps frame features cached)
                    predictor.reset_state(inference_state)

                    # Parse prediction -> points
                    parsed_points = parse_prediction_points(prediction_text)

                    # Build SAM2 prompt map
                    prompt_map = build_sam2_prompt_map(
                        parsed_points=parsed_points,
                        video_width=vid_w,
                        video_height=vid_h,
                        video_fps=video_fps,
                        num_frames=n_frame_files,
                    )

                    # Run SAM2 (reuses inference_state, no re-init)
                    seg_result = run_sam2_segmentation(
                        predictor=predictor,
                        prompt_map=prompt_map,
                        inference_state=inference_state,
                    )
                    pred_masks_list = seg_result["mask_prediction"]
                    num_frames = seg_result["num_video_frames"]
                    seg_h = seg_result["video_height"]
                    seg_w = seg_result["video_width"]

                    # Load GT masks and compute J&F
                    if task_cfg.get("schema") == "videotrack":
                        gt_masks = load_gt_masks_videotrack(meta.get("masks_raw"), num_frames, seg_h, seg_w)
                    else:
                        gt_masks = load_gt_masks_from_rle(str(mask_path), num_frames, seg_h, seg_w)
                    pred_masks = np.stack(pred_masks_list, axis=0)
                    metrics = evaluate_masks(gt_masks, pred_masks)

                    local_results.append({
                        "idx": pred_item["idx"],
                        "video": video_id,
                        "qid": qid,
                        "expression": pred_item.get("expression", ""),
                        "num_frames": num_frames,
                        "num_prompt_points": seg_result["num_prompt_points"],
                        "num_prompt_frames": seg_result["num_prompt_frames"],
                        "J": metrics["J"],
                        "F": metrics["F"],
                        "J&F": metrics["J&F"],
                        "HOTA": metrics["HOTA"],
                    })

                    items_done += 1
                    elapsed = time.time() - t0
                    eta = elapsed / items_done * (my_total_items - items_done)
                    if items_done % 10 == 0 or items_done == 1:
                        log.info(
                            f"[rank {rank}] [{items_done}/{my_total_items}] "
                            f"{video_id}/{qid} J={metrics['J']:.3f} F={metrics['F']:.3f} "
                            f"J&F={metrics['J&F']:.3f} HOTA={metrics['HOTA']:.3f} ETA={eta:.0f}s"
                        )

                except Exception as e:
                    if args.skip_errors:
                        log.warning(f"[rank {rank}] Error on expression {video_id}/{qid}: {e}")
                        items_done += 1
                        continue
                    raise

        except Exception as e:
            if args.skip_errors:
                log.warning(f"[rank {rank}] Error loading video {video_id}: {e}")
                items_done += len(items)
                continue
            raise
        finally:
            if tmpdir_to_cleanup is not None:
                shutil.rmtree(tmpdir_to_cleanup, ignore_errors=True)

    barrier()

    # ---- Gather results from all ranks ----
    if world_size > 1:
        rank_file = output_dir / f"results_rank{rank}.json"
        with open(rank_file, "w") as f:
            json.dump(local_results, f)
        barrier()
        if is_main():
            all_results = []
            for r in range(world_size):
                rf = output_dir / f"results_rank{r}.json"
                with open(rf) as f:
                    all_results.extend(json.load(f))
                rf.unlink()
            all_results.sort(key=lambda x: x["idx"])
    else:
        all_results = local_results

    # ---- Save final metrics ----
    if is_main():
        if not all_results:
            log.warning("No results!")
            return

        j_scores = [r["J"] for r in all_results]
        f_scores = [r["F"] for r in all_results]
        jf_scores = [r["J&F"] for r in all_results]
        hota_scores = [r["HOTA"] for r in all_results]

        summary = {
            "task": task_cfg["display_name"],
            "n_evaluated": len(all_results),
            "n_total": total,
            "J": round(100 * float(np.mean(j_scores)), 2),
            "F": round(100 * float(np.mean(f_scores)), 2),
            "J&F": round(100 * float(np.mean(jf_scores)), 2),
            "HOTA": round(100 * float(np.mean(hota_scores)), 2),
        }

        log.info("=" * 70)
        log.info(f"{task_cfg['display_name']} SAM2 Segmentation Results ({len(all_results)}/{total}):")
        log.info(f"  J={summary['J']:.2f}  F={summary['F']:.2f}  "
                 f"J&F={summary['J&F']:.2f}  HOTA={summary['HOTA']:.2f}")
        log.info("=" * 70)

        with open(output_dir / "sam2_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(output_dir / "sam2_predictions.json", "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"Saved to {output_dir}")

    if dist.is_initialized():
        barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
