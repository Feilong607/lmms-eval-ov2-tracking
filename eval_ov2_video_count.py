"""
Video Count evaluation for OV2-4B (LLaVA-OneVision-2.0).

Mirrors eval_video_count.py but uses OV2 chat-mode inference with
patch_positions and timestamp injection.

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 --master-port 29570 \
      eval_ov2_video_count.py \
      --model-path /ov2/feilong/LLaVA-OneVision-2.0/examples/llava_onevision2/convert/ax_instruct_video_8gpus_count_iter_0000830_hf
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

import cv2
import decord
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path("/ov2/feilong/video_VC_VP/data_for_test/Molmo2-VideoCountEval")

POINT_COUNT_PROMPTS = [
    "How many {label} are there? Output the integer number of the count only. The answer is:",
]


# ===================================================================
# Distributed helpers
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
# OV2 model loading & inference (chat mode + patch_positions + timestamps)
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


def _resize_frame(frame_np, patch_size, min_pixels, max_pixels, max_resolution):
    if not (min_pixels or max_pixels):
        return frame_np
    rh, rw = smart_resize(frame_np.shape[0], frame_np.shape[1], patch_size,
                          min_pixels, max_pixels, max_resolution)
    if (rh, rw) == (frame_np.shape[0], frame_np.shape[1]):
        return frame_np
    interp = cv2.INTER_AREA if rh < frame_np.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(frame_np, (rw, rh), interpolation=interp)


def extract_frames_pil(video_path, max_frames=128, patch_size=14,
                       min_pixels=None, max_pixels=None, max_resolution=None,
                       fixed_num_frames=None, target_fps=None,
                       clip_start=None, clip_end=None):
    """Extract frames from full video (or clip range) with stride-based sampling.

    Returns: (frames_pil, selected_indices, fps, vid_h, vid_w)
    selected_indices are absolute indices into the original VideoReader.
    """
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    frame_count = len(vr)
    fps = vr.get_avg_fps()
    if not fps or fps <= 0:
        fps = 30.0
    vid_h, vid_w = vr[0].shape[:2]

    # Determine [start, end) frame range. Fall back to full video if the clip
    # range is invalid (clip_start beyond video length, etc).
    s_idx, e_idx = 0, frame_count
    if clip_start is not None and clip_end is not None and clip_end > clip_start:
        cs = max(0, int(round(clip_start * fps)))
        ce = min(frame_count, int(round(clip_end * fps)))
        if cs < frame_count and ce > cs:
            s_idx, e_idx = cs, ce

    range_count = e_idx - s_idx

    if target_fps is not None and target_fps > 0:
        step = max(float(fps) / float(target_fps), 1.0)
        selected = []
        position = float(s_idx)
        while len(selected) < max_frames:
            frame_idx = int(round(position))
            if frame_idx >= e_idx:
                break
            if not selected or frame_idx != selected[-1]:
                selected.append(frame_idx)
            position += step
        if not selected:
            selected = [s_idx]
    elif fixed_num_frames is not None:
        target_count = min(fixed_num_frames, max_frames)
        if range_count <= target_count:
            selected = list(range(s_idx, e_idx))
        else:
            selected = np.linspace(s_idx, e_idx - 1, target_count, dtype=int).tolist()
    else:
        duration = range_count / fps
        if duration < 10:
            target_count = 8
        elif duration < 30:
            target_count = 16
        else:
            target_count = max_frames
        if range_count <= target_count:
            selected = list(range(s_idx, e_idx))
        else:
            selected = np.linspace(s_idx, e_idx - 1, target_count, dtype=int).tolist()

    frames_pil = []
    for idx in selected:
        frame = vr[idx].asnumpy()
        frame = _resize_frame(frame, patch_size, min_pixels, max_pixels, max_resolution)
        frames_pil.append(Image.fromarray(frame))

    del vr
    return frames_pil, selected, fps, vid_h, vid_w


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
                  fixed_num_frames=128, max_new_tokens=256, temperature=0.0,
                  target_fps=None, clip_start=None, clip_end=None):
    from qwen_vl_utils import process_vision_info

    frames_pil, frame_indices, vid_fps, vid_h, vid_w = extract_frames_pil(
        video_path, max_frames=fixed_num_frames,
        patch_size=14, min_pixels=min_pixels, max_pixels=max_pixels,
        max_resolution=max_resolution,
        fixed_num_frames=None if target_fps else fixed_num_frames,
        target_fps=target_fps,
        clip_start=clip_start, clip_end=clip_end,
    )

    num_frames = len(frames_pil)
    system_prompt = "You are a helpful assistant."
    image_content = [{"type": "image", "image": img} for img in frames_pil]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": image_content + [{"type": "text", "text": prompt}]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    visual_timestamps = [
        f"<{_format_dense_seconds(float(sel_idx) / vid_fps)} seconds>"
        for sel_idx in frame_indices
    ]
    text = _inject_timestamps_to_chat_text(text, visual_timestamps)

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

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
    gen_args = {
        **filtered_inputs,
        "max_new_tokens": int(max_new_tokens),
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
    return tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()


# ===================================================================
# Count extraction
# ===================================================================

_NUM_WORDS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
    'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
    'thirty': 30, 'forty': 40, 'fifty': 50,
}


def extract_count(text):
    if not text:
        return None
    nums = re.findall(r'\b(\d+)\b', text)
    if nums:
        return int(nums[0])
    text_l = text.lower()
    for word, num in sorted(_NUM_WORDS.items(), key=lambda x: -x[1]):
        if word in text_l:
            return num
    return None


# ===================================================================
# Args & Main
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(description="OV2 Video Count Evaluation")
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--sampling-fps", type=float, default=1.0,
                   help="Stride-based sampling fps (step = source_fps / sampling_fps)")
    p.add_argument("--fixed-num-frames", type=int, default=128)
    p.add_argument("--max-pixels", type=int, default=313600)
    p.add_argument("--min-pixels", type=int, default=3136)
    p.add_argument("--max-resolution", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
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
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "eval_output_ov2" / "video_count"
    log_dir = SCRIPT_DIR / "logs"

    if is_main():
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "ov2_video_count.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        log.addHandler(fh)

    df = pd.read_parquet(str(parquet_path))
    if args.smoke_test:
        df = df.head(max(8, world_size))
    total = len(df)

    indices = list(range(rank, total, world_size))
    if is_main():
        log.info(f"Total: {total}, world_size: {world_size}")
        log.info(f"Loading model from {args.model_path}")

    model, processor, tokenizer = load_model_and_processor(
        args.model_path, device,
        max_pixels=args.max_pixels, min_pixels=args.min_pixels,
    )
    if is_main():
        log.info("Model loaded")
    barrier()

    local_results = []
    t0 = time.time()

    for i, idx in enumerate(indices):
        row = df.iloc[idx]
        vid_id = str(row['video_id'])
        label = str(row['label']) if 'label' in row else None
        if label:
            question = POINT_COUNT_PROMPTS[0].format(label=label)
        else:
            question = str(row['question'])
        gt_count = int(row['count'])
        clip_start = float(row['clip_start']) if pd.notna(row.get('clip_start')) else None
        clip_end = float(row['clip_end']) if pd.notna(row.get('clip_end')) else None

        vpath = str(video_dir / f"{vid_id}.mp4")
        if not os.path.exists(vpath):
            log.warning(f"[rank {rank}] Missing video: {vid_id}")
            continue

        try:
            answer = run_inference(
                model, processor, tokenizer, device, vpath, question,
                max_pixels=args.max_pixels, min_pixels=args.min_pixels,
                max_resolution=args.max_resolution,
                fixed_num_frames=args.fixed_num_frames,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                target_fps=args.sampling_fps,
                clip_start=clip_start, clip_end=clip_end,
            )
            pred_count = extract_count(answer)
            correct = (pred_count == gt_count) if pred_count is not None else False
            close = (abs(pred_count - gt_count) <= 1) if pred_count is not None else False
            mae = abs(pred_count - gt_count) if pred_count is not None else gt_count

            local_results.append({
                'idx': int(idx), 'video_id': vid_id, 'question': question,
                'category': str(row['category']),
                'gt_count': gt_count, 'pred_count': pred_count,
                'prediction': answer, 'correct': bool(correct),
                'close': bool(close), 'mae': int(mae),
            })

            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(indices) - i - 1)
            sym = 'OK' if correct else 'X'
            log.info(
                f"[rank {rank}] [{i+1}/{len(indices)}] {vid_id} "
                f"gt={gt_count} pred={pred_count} {sym} ETA={eta:.0f}s"
            )
        except Exception:
            log.exception(f"[rank {rank}] Error: {vid_id}")
            continue

    barrier()

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
            all_results.sort(key=lambda x: x['idx'])
            _save_results(all_results, output_dir, total)
    else:
        _save_results(local_results, output_dir, total)


def _save_results(results, output_dir, total):
    if not results:
        log.warning("No results!")
        return

    n = len(results)
    acc = sum(1 for r in results if r['correct']) / n * 100
    close_acc = sum(1 for r in results if r.get('close', False)) / n * 100
    mae = sum(r['mae'] for r in results) / n

    cat_stats = {}
    for r in results:
        c = r['category']
        if c not in cat_stats:
            cat_stats[c] = {'correct': 0, 'close': 0, 'total': 0, 'mae_sum': 0}
        cat_stats[c]['total'] += 1
        cat_stats[c]['mae_sum'] += r['mae']
        if r['correct']:
            cat_stats[c]['correct'] += 1
        if r.get('close', False):
            cat_stats[c]['close'] += 1

    cat_acc = {k: round(100 * v['correct'] / v['total'], 2) for k, v in cat_stats.items()}
    cat_close = {k: round(100 * v['close'] / v['total'], 2) for k, v in cat_stats.items()}
    cat_mae = {k: round(v['mae_sum'] / v['total'], 3) for k, v in cat_stats.items()}

    summary = {
        'task': 'VideoCount', 'model_family': 'OV2',
        'n_evaluated': n, 'n_total': total,
        'accuracy': round(acc, 2), 'close_acc': round(close_acc, 2),
        'mae': round(mae, 3),
        'per_category_accuracy': cat_acc,
        'per_category_close_acc': cat_close,
        'per_category_mae': cat_mae,
    }

    log.info("=" * 70)
    log.info(f"VideoCount OV2 Results ({n}/{total}):")
    log.info(f"  Accuracy={acc:.2f}%  Close-ACC={close_acc:.2f}%  MAE={mae:.3f}")
    for c, a in cat_acc.items():
        log.info(f"  {c}: acc={a}% close_acc={cat_close[c]}% mae={cat_mae[c]}")
    log.info("=" * 70)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
