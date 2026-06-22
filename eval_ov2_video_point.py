"""
Video Point evaluation for OV2-4B (LLaVA-OneVision-2.0).

Pipeline mirrors eval_video_point.py:
  Stage 1 (LLM): OV2 -> predict point coordinates (XML <points coords="..."/>)
  Stage 2 (SAM2): SAM2.1 Hiera Large -> propagate masks -> J&F vs GT

Differences from Molmo2 version:
  - Loads model via AutoModelForCausalLM with flash_attention_2
  - Frame extraction via decord with smart_resize + stride-based sampling
  - Injects <X.XX seconds> timestamps before each <|vision_start|>
  - Builds patch_positions for the OV2 vision tower

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 --master-port 29560 \
      eval_ov2_video_point.py \
      --model-path /ov2/feilong/LLaVA-OneVision-2.0/examples/llava_onevision2/convert/ax_instruct_video_8gpus_point_iter_0000853_hf
"""

import argparse
import copy
import gc
import importlib
import json
import logging
import math
import os
import random
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

import cv2
import decord
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from PIL import Image
from pycocotools import mask as mask_utils


def _disk(radius):
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    return (X**2 + Y**2 <= radius**2).astype(np.uint8)


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path("/ov2/feilong/video_VC_VP/data_for_test/Molmo2-VideoPointEval")
DEFAULT_SAM2_CODE_DIR = "/ov2/zwk/lmms-eval-ov2/extension/sam2_1"
DEFAULT_SAM2_CONFIG = "/ov2/zwk/lmms-eval-ov2/extension/sam2.1_hiera_large.yaml"
DEFAULT_SAM2_CHECKPOINT = "/ov2/zwk/lmms-eval-ov2/extension/sam2.1_hiera_large.pt"

GT_FPS = 2  # GT masks are indexed at 2fps

COORD_REGEX = re.compile(r'<(?:points|tracks).*? coords="([0-9\t:;, .]+)"/?>')
FRAME_REGEX = re.compile(r"(?:^|\t|:|,|;)([0-9.]+) ([0-9. ]+)")
POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{1,4}) ([0-9]{1,4})")

POINTING_PROMPTS = [
    "Find the {label}",
]

POINT_COUNT_PROMPTS = [
    "How many {label}?",
]


# ===================================================================
# 1. Distributed helpers
# ===================================================================

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=120))
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        return rank, world, torch.device(f"cuda:{local}")
    return 0, 1, None


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


# ===================================================================
# 2. OV2 model loading and inference
# ===================================================================

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
    if max_resolution and (resized_h > max_resolution or resized_w > max_resolution):
        clamp_scale = min(max_resolution / resized_h, max_resolution / resized_w)
        resized_h = max(align_size, int(round(resized_h * clamp_scale / align_size) * align_size))
        resized_w = max(align_size, int(round(resized_w * clamp_scale / align_size) * align_size))
    return resized_h, resized_w


def _format_dense_seconds(seconds: float) -> str:
    return f"{float(seconds):.2f}"


def _inject_timestamps_to_chat_text(text, visual_timestamps):
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


def _recommended_max_new_tokens(num_frames, requested_max_new_tokens):
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


def extract_video_frames_pil(video_path, max_frames=128, patch_size=14,
                             min_pixels=None, max_pixels=None, max_resolution=None,
                             fixed_num_frames=None, target_fps=None,
                             source_fps=None):
    """Extract frames from full video with stride-based sampling (matches OV2 ref pipeline)."""
    vr = decord.VideoReader(video_path)
    frame_count = len(vr)
    fps = vr.get_avg_fps()
    if not fps or fps <= 0:
        fps = 30.0

    effective_source_fps = source_fps if (source_fps and source_fps > 0) else fps

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
        model_path, trust_remote_code=True, dtype="auto",
        attn_implementation="flash_attention_2",
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(
        model_path, max_pixels=max_pixels, min_pixels=min_pixels, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, processor, tokenizer


def run_inference(model, processor, tokenizer, device, video_path, prompt,
                  max_pixels=313600, min_pixels=56*56, max_resolution=None,
                  fixed_num_frames=128, max_new_tokens=2048, temperature=0.0,
                  target_fps=None, source_fps=None, auto_scale_max_new_tokens=True):
    """Run OV2 inference using chat mode with patch_positions."""
    from qwen_vl_utils import process_vision_info

    frames_pil, frame_indices, vid_fps = extract_video_frames_pil(
        video_path, max_frames=fixed_num_frames,
        patch_size=14, min_pixels=min_pixels, max_pixels=max_pixels,
        max_resolution=max_resolution,
        fixed_num_frames=None if target_fps else fixed_num_frames,
        target_fps=target_fps,
        source_fps=source_fps,
    )

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    vid_h, vid_w = vr[0].shape[:2]
    del vr

    num_frames = len(frames_pil)
    system_prompt = "You are a helpful assistant."
    image_content = [{"type": "image", "image": img} for img in frames_pil]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": image_content + [{"type": "text", "text": prompt}]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    ts_fps = source_fps if (source_fps and source_fps > 0) else vid_fps
    visual_timestamps = [
        f"<{_format_dense_seconds(float(sel_idx) / ts_fps)} seconds>"
        for sel_idx in frame_indices
    ]
    text = _inject_timestamps_to_chat_text(text, visual_timestamps)

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    inputs = inputs.to(device)

    image_grid_thw = inputs.get("image_grid_thw", None)
    patch_positions = None
    if image_grid_thw is not None:
        target_device = inputs.input_ids.device
        seq_frame_indices = torch.arange(num_frames, device=target_device, dtype=torch.long)
        patch_positions = build_patch_positions(
            num_frames=num_frames, total_frames=num_frames,
            h=image_grid_thw[0][1], w=image_grid_thw[0][2],
            frame_indices=seq_frame_indices, device=target_device,
        )

    unsupported_keys = ["second_per_grid_ts", "mm_token_type_ids"]
    filtered_inputs = {k: v for k, v in inputs.items() if k not in unsupported_keys}

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eff_max_new = (
        _recommended_max_new_tokens(num_frames, max_new_tokens)
        if auto_scale_max_new_tokens else int(max_new_tokens)
    )
    gen_args = {
        **filtered_inputs,
        "max_new_tokens": eff_max_new,
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
    answer = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()
    return answer, vid_w, vid_h


# ===================================================================
# 3. Point parsing (from model output) — identical to Molmo2 version
# ===================================================================

def parse_prediction_points(text):
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
    prompt_map = defaultdict(list)
    for timestamp, point_id, x_norm, y_norm in parsed_points:
        frame_idx = int(round(timestamp * video_fps))
        frame_idx = max(0, min(frame_idx, num_frames - 1))
        x_abs = max(0.0, min(x_norm / 1000.0 * video_width, float(video_width - 1)))
        y_abs = max(0.0, min(y_norm / 1000.0 * video_height, float(video_height - 1)))
        prompt_map[(frame_idx, str(point_id))].append((x_abs, y_abs))
    return dict(prompt_map)


def compute_point_metrics(parsed_points, gt_by_frame, video_width, video_height,
                          gt_fps, gt_count):
    if not parsed_points:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'n_pred': 0, 'n_correct': 0}

    n_pred = len(parsed_points)
    n_correct = 0

    for timestamp, point_id, x_norm, y_norm in parsed_points:
        frame_id = int(round(timestamp * gt_fps))
        if gt_by_frame:
            closest_fid = min(gt_by_frame.keys(), key=lambda fid: abs(fid - frame_id))
        else:
            continue
        gt_mask = gt_by_frame[closest_fid]
        mask_h, mask_w = gt_mask.shape[:2]
        px = int(round(x_norm / 1000.0 * mask_w))
        py = int(round(y_norm / 1000.0 * mask_h))
        px = max(0, min(px, mask_w - 1))
        py = max(0, min(py, mask_h - 1))
        if gt_mask[py, px] > 0:
            n_correct += 1

    precision = n_correct / n_pred if n_pred > 0 else 0.0
    recall = min(n_correct, gt_count) / gt_count if gt_count > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'n_pred': n_pred, 'n_correct': n_correct}


# ===================================================================
# 4. SAM2 predictor building (identical to Molmo2 version)
# ===================================================================

def _setup_sam2_import_path(sam2_code_dir):
    parent = str(Path(sam2_code_dir).resolve().parent)
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


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _coerce_config_scalars(cfg):
    if isinstance(cfg, dict):
        return {k: _coerce_config_scalars(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_coerce_config_scalars(v) for v in cfg]
    if isinstance(cfg, str):
        v = cfg.strip()
        if _INT_RE.fullmatch(v):
            try:
                return int(v)
            except ValueError:
                pass
        if _FLOAT_RE.fullmatch(v):
            try:
                return float(v)
            except ValueError:
                pass
    return cfg


def _resolve_target(target):
    module_name, class_name = target.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


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
    _setup_sam2_import_path(sam2_code_dir)
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

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
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
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
# 5. SAM2 segmentation (identical)
# ===================================================================

def mask_logits_to_uint8(mask_logits):
    logits = mask_logits.detach()
    if logits.ndim == 4:
        logits = logits.squeeze(1)
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    merged = torch.any(logits > 0, dim=0).to(torch.uint8)
    return merged.cpu()


def run_sam2_segmentation(predictor, prompt_map, inference_state):
    num_frames = int(inference_state["num_frames"])
    video_h = int(inference_state["video_height"])
    video_w = int(inference_state["video_width"])

    frame_to_mask = {}
    sorted_prompts = sorted(prompt_map.items(), key=lambda item: (item[0][0], item[0][1]))

    if sorted_prompts:
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
            for idx, frame_idx in enumerate(prompt_frames):
                for obj_id, points in prompts_by_frame[frame_idx]:
                    pts_t = torch.tensor(points, dtype=torch.float32)
                    lbl_t = torch.ones((len(points),), dtype=torch.int32)
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=pts_t,
                        labels=lbl_t,
                        clear_old_points=True,
                        normalize_coords=True,
                    )

                next_pf = prompt_frames[idx + 1] if idx + 1 < len(prompt_frames) else num_frames
                max_track = max(next_pf - frame_idx - 1, 0)
                for out_fi, _, out_ml in predictor.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=frame_idx,
                    max_frame_num_to_track=max_track,
                    reverse=False,
                ):
                    frame_to_mask[int(out_fi)] = mask_logits_to_uint8(out_ml)

            first_pf = prompt_frames[0]
            if first_pf > 0:
                for out_fi, _, out_ml in predictor.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=first_pf,
                    max_frame_num_to_track=first_pf,
                    reverse=True,
                ):
                    frame_to_mask[int(out_fi)] = mask_logits_to_uint8(out_ml)

    empty = torch.zeros((video_h, video_w), dtype=torch.uint8)
    masks = [frame_to_mask.get(fi, empty).numpy().astype(np.uint8) for fi in range(num_frames)]
    return masks, video_h, video_w


# ===================================================================
# 6. Frame extraction at GT fps for SAM2 (identical)
# ===================================================================

def extract_frames_for_sam2(video_path, output_dir, fps=2.0):
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
# 7. J&F metrics (identical)
# ===================================================================

def _seg2bmap(seg):
    seg = seg.astype(bool)
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
    return b


def f_measure(foreground_mask, gt_mask, bound_th=0.008):
    fg_boundary = _seg2bmap(foreground_mask)
    gt_boundary = _seg2bmap(gt_mask)

    bound_pix = max(1, round(bound_th * np.sqrt(gt_mask.shape[0]**2 + gt_mask.shape[1]**2)))
    se = _disk(bound_pix).astype(np.uint8)
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), se)
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), se)

    fg_match = fg_boundary * gt_dil
    gt_match = gt_boundary * fg_dil
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    if n_fg == 0 and n_gt == 0:
        return 1.0
    if n_fg == 0 or n_gt == 0:
        return 0.0

    precision = np.sum(fg_match) / n_fg
    recall = np.sum(gt_match) / n_gt
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def db_eval_iou(annotation, segmentation):
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
    assert annotation.shape == segmentation.shape
    if annotation.ndim == 3:
        return np.array([f_measure(segmentation[i], annotation[i], bound_th)
                         for i in range(annotation.shape[0])])
    return f_measure(segmentation, annotation, bound_th)


def evaluate_masks(gt_masks, pred_masks):
    j = db_eval_iou(gt_masks, pred_masks).mean()
    f = db_eval_boundary(gt_masks, pred_masks).mean()
    return {"J": float(j), "F": float(f), "J&F": float((j + f) / 2)}


# ===================================================================
# 8. GT mask decoding (identical)
# ===================================================================

def decode_gt_masks(masks_data, height, width):
    gt_by_frame = {}
    for instance_masks in masks_data:
        for mask_dict in instance_masks:
            frame_id = int(mask_dict['frame_id'])
            rle = mask_dict['rle']
            rle_input = {
                'counts': rle['counts'].encode('utf-8') if isinstance(rle['counts'], str) else rle['counts'],
                'size': [int(s) for s in rle['size']],
            }
            decoded = mask_utils.decode(rle_input).astype(np.uint8)
            if decoded.shape != (height, width):
                decoded = cv2.resize(decoded, (width, height), interpolation=cv2.INTER_NEAREST)
            if frame_id in gt_by_frame:
                gt_by_frame[frame_id] = np.maximum(gt_by_frame[frame_id], decoded)
            else:
                gt_by_frame[frame_id] = decoded
    return gt_by_frame


# ===================================================================
# 9. Args & Main
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(description="OV2 Video Point Evaluation (LLM + SAM2 -> J&F)")
    p.add_argument("--model-path", default=None,
                   help="Path to OV2 model (required unless --stage2-only)")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--sampling-fps", type=float, default=1.0,
                   help="Stride-based sampling fps (step = source_fps / sampling_fps)")
    p.add_argument("--fixed-num-frames", type=int, default=128,
                   help="Max number of frames cap (per OV2 README recommendation)")
    p.add_argument("--max-pixels", type=int, default=313600)
    p.add_argument("--min-pixels", type=int, default=3136)
    p.add_argument("--max-resolution", type=int, default=None,
                   help="Max H/W per frame (overrides max-pixels if set)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--sam2-code-dir", default=DEFAULT_SAM2_CODE_DIR)
    p.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG)
    p.add_argument("--sam2-checkpoint", default=DEFAULT_SAM2_CHECKPOINT)
    p.add_argument("--stage2-only", action="store_true",
                   help="Skip LLM inference, load existing predictions.json")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.stage2_only and args.model_path is None:
        raise ValueError("--model-path is required unless --stage2-only is set")
    if args.max_resolution is not None:
        args.max_pixels = args.max_resolution * args.max_resolution

    rank, world_size, device = setup_distributed()
    if device is None:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)

    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    video_dir = data_dir / "youtube_vedio" / "val-00000-of-00001"
    parquet_path = data_dir / "data" / "val-00000-of-00001.parquet"
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "eval_output_ov2" / "video_point"
    log_dir = SCRIPT_DIR / "logs"

    if is_main():
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "ov2_video_point.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        log.addHandler(fh)

    df = pd.read_parquet(str(parquet_path))
    if args.smoke_test:
        df = df.head(max(8, world_size))
    total = len(df)

    pred_path = output_dir / "predictions.json"

    # ================================================================
    # Stage 1: LLM Inference (OV2)
    # ================================================================
    if not args.stage2_only:
        if is_main():
            log.info(f"=== Stage 1: OV2 LLM Inference ({total} examples) ===")
            log.info(f"Loading model from {args.model_path}")

        model, processor, tokenizer = load_model_and_processor(
            args.model_path, device,
            max_pixels=args.max_pixels, min_pixels=args.min_pixels,
        )
        if is_main():
            log.info("Model loaded")
        barrier()

        indices = list(range(rank, total, world_size))
        local_preds = []
        t0 = time.time()

        for i, idx in enumerate(indices):
            row = df.iloc[idx]
            vid_id = str(row['video_id'])
            label = str(row['label'])
            w, h = int(row['width']), int(row['height'])

            vpath = str(video_dir / f"{vid_id}.mp4")
            if not os.path.exists(vpath):
                log.warning(f"[rank {rank}] Missing video: {vid_id}")
                continue

            category = str(row['category'])
            if category == 'animal':
                prompt = f"Track the {label}"
            else:
                random.seed(idx)
                count = int(row["count"])
                if count > 1:
                    prompt = random.choice(POINT_COUNT_PROMPTS).format(label=label)
                else:
                    prompt = random.choice(POINTING_PROMPTS).format(label=label)

            try:
                answer, vw, vh = run_inference(
                    model, processor, tokenizer, device, vpath, prompt,
                    max_pixels=args.max_pixels, min_pixels=args.min_pixels,
                    max_resolution=args.max_resolution,
                    fixed_num_frames=args.fixed_num_frames,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    target_fps=args.sampling_fps,
                    source_fps=None,  # use video's native fps
                    auto_scale_max_new_tokens=True,
                )

                parsed_points = parse_prediction_points(answer)
                local_preds.append({
                    'idx': int(idx), 'video_id': vid_id, 'label': label,
                    'category': str(row['category']),
                    'count': int(row['count']),
                    'width': int(vw or w), 'height': int(vh or h),
                    'prediction': answer,
                    'num_parsed_points': len(parsed_points),
                })

                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(indices) - i - 1)
                log.info(
                    f"[rank {rank}] [{i+1}/{len(indices)}] {vid_id} "
                    f"pts={len(parsed_points)} ETA={eta:.0f}s"
                )
            except Exception:
                log.exception(f"[rank {rank}] Error: {vid_id}")
                continue

        barrier()

        if world_size > 1:
            rank_file = output_dir / f"preds_rank{rank}.json"
            with open(rank_file, "w") as f:
                json.dump(local_preds, f)
            barrier()
            if is_main():
                all_preds = []
                for r in range(world_size):
                    rf = output_dir / f"preds_rank{r}.json"
                    with open(rf) as f:
                        all_preds.extend(json.load(f))
                    rf.unlink()
                all_preds.sort(key=lambda x: x['idx'])
                with open(pred_path, "w") as f:
                    json.dump(all_preds, f, indent=2)
                log.info(f"Stage 1 done: {len(all_preds)} predictions saved")
        else:
            with open(pred_path, "w") as f:
                json.dump(local_preds, f, indent=2)
            if is_main():
                log.info(f"Stage 1 done: {len(local_preds)} predictions saved")

        del model, processor, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    barrier()

    # ================================================================
    # Stage 2: SAM2 -> J&F
    # ================================================================
    if is_main():
        log.info("=== Stage 2: SAM2 Evaluation ===")

    if not pred_path.exists():
        if is_main():
            log.error(f"Predictions not found: {pred_path}")
        return

    with open(pred_path) as f:
        predictions = json.load(f)

    if is_main():
        log.info(f"Loaded {len(predictions)} predictions")
        log.info("Building SAM2 predictor...")

    predictor = build_sam2_video_predictor(
        sam2_code_dir=args.sam2_code_dir,
        config_path=args.sam2_config,
        checkpoint_path=args.sam2_checkpoint,
        device=str(device),
    )
    if is_main():
        log.info("SAM2 loaded")

    df_lookup = {}
    for i in range(len(df)):
        row = df.iloc[i]
        df_lookup[(str(row['video_id']), str(row['label']))] = i

    video_groups = defaultdict(list)
    for pred in predictions:
        video_groups[pred['video_id']].append(pred)
    video_ids = sorted(video_groups.keys())
    my_video_ids = [video_ids[i] for i in range(rank, len(video_ids), world_size)]
    my_total = sum(len(video_groups[v]) for v in my_video_ids)

    if is_main():
        log.info(f"Grouped into {len(video_ids)} videos; rank {rank}: "
                 f"{len(my_video_ids)} videos ({my_total} items)")

    local_results = []
    t0 = time.time()
    items_done = 0
    sam2_output_dir = output_dir / "sam2_results"
    if is_main():
        sam2_output_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    for vid_id in my_video_ids:
        items = video_groups[vid_id]
        tmpdir = None

        try:
            vpath = str(video_dir / f"{vid_id}.mp4")
            if not os.path.exists(vpath):
                log.warning(f"[rank {rank}] Missing video for SAM2: {vid_id}")
                items_done += len(items)
                continue

            tmpdir = tempfile.mkdtemp(prefix=f"sam2_vp_ov2_{vid_id}_")
            extract_frames_for_sam2(vpath, tmpdir, fps=GT_FPS)
            n_frames = len([f for f in os.listdir(tmpdir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

            inference_state = predictor.init_state(video_path=tmpdir)

            first_frame_path = sorted(os.listdir(tmpdir))[0]
            ff_img = cv2.imread(os.path.join(tmpdir, first_frame_path))
            sam2_h, sam2_w = ff_img.shape[:2]

            for pred_item in items:
                label = pred_item['label']
                vid_w = pred_item['width']
                vid_h = pred_item['height']

                df_idx = df_lookup.get((vid_id, label))
                if df_idx is None:
                    log.warning(f"[rank {rank}] No GT for {vid_id}/{label}")
                    items_done += 1
                    continue

                gt_row = df.iloc[df_idx]

                try:
                    predictor.reset_state(inference_state)
                    parsed_points = parse_prediction_points(pred_item['prediction'])
                    prompt_map = build_sam2_prompt_map(
                        parsed_points, vid_w, vid_h, GT_FPS, n_frames,
                    )

                    pred_masks_list, seg_h, seg_w = run_sam2_segmentation(
                        predictor, prompt_map, inference_state)

                    gt_by_frame = decode_gt_masks(gt_row['masks'], sam2_h, sam2_w)
                    eval_fids = sorted(fid for fid in gt_by_frame if 0 <= fid < len(pred_masks_list))

                    if not eval_fids:
                        items_done += 1
                        continue

                    gt_stack = np.stack([gt_by_frame[fid] for fid in eval_fids])
                    pred_stack = np.stack([pred_masks_list[fid] for fid in eval_fids])

                    if pred_stack.shape[1:] != gt_stack.shape[1:]:
                        resized = [cv2.resize(pm, (gt_stack.shape[2], gt_stack.shape[1]),
                                              interpolation=cv2.INTER_NEAREST) for pm in pred_stack]
                        pred_stack = np.stack(resized)

                    metrics = evaluate_masks(gt_stack, pred_stack)

                    pt_metrics = compute_point_metrics(
                        parsed_points, gt_by_frame,
                        vid_w, vid_h, GT_FPS,
                        pred_item.get('count', 1),
                    )

                    local_results.append({
                        'idx': pred_item['idx'],
                        'video_id': vid_id,
                        'label': label,
                        'category': pred_item.get('category', ''),
                        'count': pred_item.get('count', 1),
                        'n_eval_frames': len(eval_fids),
                        'n_parsed_points': len(parsed_points),
                        'J': metrics['J'],
                        'F': metrics['F'],
                        'J&F': metrics['J&F'],
                        'precision': pt_metrics['precision'],
                        'recall': pt_metrics['recall'],
                        'f1': pt_metrics['f1'],
                        'n_correct_points': pt_metrics['n_correct'],
                    })

                    items_done += 1
                    elapsed = time.time() - t0
                    eta = elapsed / items_done * (my_total - items_done) if items_done else 0
                    if items_done % 5 == 0 or items_done == 1:
                        log.info(
                            f"[rank {rank}] [{items_done}/{my_total}] {vid_id}/{label[:30]} "
                            f"J={metrics['J']:.3f} F={metrics['F']:.3f} "
                            f"J&F={metrics['J&F']:.3f} ETA={eta:.0f}s"
                        )

                except Exception as e:
                    log.warning(f"[rank {rank}] SAM2 error {vid_id}/{label}: {e}")
                    items_done += 1
                    continue

        except Exception as e:
            log.warning(f"[rank {rank}] Error loading video {vid_id}: {e}")
            items_done += len(items)
            continue
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)

    barrier()

    if world_size > 1:
        rank_file = sam2_output_dir / f"results_rank{rank}.json"
        with open(rank_file, "w") as f:
            json.dump(local_results, f)
        barrier()
        if is_main():
            all_results = []
            for r in range(world_size):
                rf = sam2_output_dir / f"results_rank{r}.json"
                with open(rf) as f:
                    all_results.extend(json.load(f))
                rf.unlink()
            all_results.sort(key=lambda x: x['idx'])
            _save_jf_results(all_results, sam2_output_dir, len(predictions))
    else:
        _save_jf_results(local_results, sam2_output_dir, len(predictions))

    if dist.is_initialized():
        barrier()
        dist.destroy_process_group()


def _save_jf_results(results, output_dir, total):
    if not results:
        log.warning("No SAM2 results!")
        return

    n = len(results)
    j_mean = np.mean([r['J'] for r in results])
    f_mean = np.mean([r['F'] for r in results])
    jf_mean = np.mean([r['J&F'] for r in results])

    cat_stats = defaultdict(lambda: {'J': [], 'F': [], 'J&F': [],
                                      'precision': [], 'recall': [], 'f1': []})
    for r in results:
        c = r.get('category', 'unknown')
        cat_stats[c]['J'].append(r['J'])
        cat_stats[c]['F'].append(r['F'])
        cat_stats[c]['J&F'].append(r['J&F'])
        cat_stats[c]['precision'].append(r.get('precision', 0))
        cat_stats[c]['recall'].append(r.get('recall', 0))
        cat_stats[c]['f1'].append(r.get('f1', 0))

    prec_mean = np.mean([r.get('precision', 0) for r in results])
    rec_mean = np.mean([r.get('recall', 0) for r in results])
    f1_mean = np.mean([r.get('f1', 0) for r in results])

    cat_summary = {}
    for c, s in cat_stats.items():
        cat_summary[c] = {
            'J': round(100 * np.mean(s['J']), 2),
            'F': round(100 * np.mean(s['F']), 2),
            'J&F': round(100 * np.mean(s['J&F']), 2),
            'precision': round(100 * np.mean(s['precision']), 2),
            'recall': round(100 * np.mean(s['recall']), 2),
            'f1': round(100 * np.mean(s['f1']), 2),
            'n': len(s['J']),
        }

    summary = {
        'task': 'VideoPoint',
        'model_family': 'OV2',
        'n_evaluated': n, 'n_total': total,
        'J': round(100 * float(j_mean), 2),
        'F': round(100 * float(f_mean), 2),
        'J&F': round(100 * float(jf_mean), 2),
        'precision': round(100 * float(prec_mean), 2),
        'recall': round(100 * float(rec_mean), 2),
        'f1': round(100 * float(f1_mean), 2),
        'per_category': cat_summary,
    }

    log.info("=" * 70)
    log.info(f"VideoPoint OV2 SAM2 Results ({n}/{total}):")
    log.info(f"  J={summary['J']:.2f}  F={summary['F']:.2f}  J&F={summary['J&F']:.2f}")
    log.info(f"  Precision={summary['precision']:.2f}  Recall={summary['recall']:.2f}  F1={summary['f1']:.2f}")
    for c, cs in cat_summary.items():
        log.info(f"  {c} (n={cs['n']}): J={cs['J']:.2f} F={cs['F']:.2f} J&F={cs['J&F']:.2f}"
                 f"  P={cs['precision']:.2f} R={cs['recall']:.2f} F1={cs['f1']:.2f}")
    log.info("=" * 70)

    with open(output_dir / "sam2_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "sam2_predictions.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
