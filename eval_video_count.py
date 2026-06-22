"""
Video Count evaluation for Molmo2 models.

Dataset: Molmo2-VideoCountEval (533 examples, 501 videos)
  - Given a video clip + counting question, predict the count
  - Metrics: exact-match accuracy, MAE

Usage:
  # 8-GPU
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 --master-port 29550 \
      eval_video_count.py --model-path ./Molmo2-4B

  # Single-GPU
  python eval_video_count.py --model-path ./Molmo2-4B --gpu 0
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path("/ov2/feilong/video_VC_VP/data_for_test/Molmo2-VideoCountEval")


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
# Model loading and inference (Molmo2 native approach)
# ===================================================================

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
                  sampling_fps=1, max_new_tokens=256, temperature=0.0,
                  clip_start=None, clip_end=None):
    """Run Molmo2 inference on video with optional clip support."""
    import decord as _decord

    if clip_start is not None and clip_end is not None and clip_end > clip_start:
        # Extract clip frames manually, pass as images
        vr = _decord.VideoReader(video_path, ctx=_decord.cpu(0))
        fps = vr.get_avg_fps() or 30.0
        total = len(vr)
        s_idx = max(0, int(clip_start * fps))
        e_idx = min(total, int(clip_end * fps))
        n_avail = max(1, e_idx - s_idx)
        dur = n_avail / fps
        n_frames = max(1, min(int(dur * sampling_fps), 32))
        if n_avail <= n_frames:
            sel = list(range(s_idx, e_idx))
        else:
            sel = np.linspace(s_idx, e_idx - 1, n_frames, dtype=int).tolist()

        from PIL import Image
        frames = [Image.fromarray(vr[i].asnumpy()) for i in sel]
        del vr
        image_content = [{"type": "image", "image": img} for img in frames]
        messages = [
            {"role": "user", "content": image_content + [{"type": "text", "text": prompt}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info([messages[-1]])
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(device)
    else:
        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": f"file://{video_path}"},
                {"type": "text", "text": prompt},
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=text, videos=[video_path],
            frame_sample_mode="fps", sampling_fps=sampling_fps,
            return_tensors="pt",
        ).to(device)

    gen_args = dict(**inputs, max_new_tokens=max_new_tokens)
    if temperature > 0:
        gen_args.update(do_sample=True, temperature=temperature, top_p=0.9)
    with torch.inference_mode():
        outputs = model.generate(**gen_args)
    gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()


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
# Main
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Video Count Evaluation")
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--sampling-fps", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()



def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()
    if device is None:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)

    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    video_dir = data_dir / "youtube_vedio" / "val-00000-of-00001"
    parquet_path = data_dir / "data" / "val-00000-of-00001.parquet"
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "eval_output" / "video_count"
    log_dir = SCRIPT_DIR / "logs"

    if is_main():
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "video_count.log")
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

    model, processor = load_model_and_processor(args.model_path, device)
    if is_main():
        log.info("Model loaded")
    barrier()

    local_results = []
    t0 = time.time()

    for i, idx in enumerate(indices):
        row = df.iloc[idx]
        vid_id = str(row['video_id'])
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
                model, processor, device, vpath, question,
                sampling_fps=args.sampling_fps, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
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
        'task': 'VideoCount', 'n_evaluated': n, 'n_total': total,
        'accuracy': round(acc, 2), 'close_acc': round(close_acc, 2),
        'mae': round(mae, 3),
        'per_category_accuracy': cat_acc,
        'per_category_close_acc': cat_close,
        'per_category_mae': cat_mae,
    }

    log.info("=" * 70)
    log.info(f"VideoCount Results ({n}/{total}):")
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
