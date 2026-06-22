"""Prepare video and mask data for ref-yt-vos and reasonvos datasets."""
import json, os, subprocess, sys
from pathlib import Path
import cv2
import numpy as np

DATA_DIR = Path("/ov2/feilong/simple_repo/data")
VIDEO_FPS = 6

DATASETS = {
    "ref-yt-vos": {
        "hf_path": DATA_DIR / "tracking" / "ref-yt-vos",
        "frames_root": Path("/ov2/feilong/reproduce/molmo_data/video_datasets/video_track/Ref-YT-VOS/valid/JPEGImages"),
        "videos_dir": DATA_DIR / "Ref-YT-VOS" / "valid" / "videos",
        "masks_dir": DATA_DIR / "Ref-YT-VOS" / "valid" / "MasksRLE",
        "annotations_dir": Path("/ov2/feilong/reproduce/molmo_data/video_datasets/video_track/Ref-YT-VOS/valid/Annotations"),
    },
    "reasonvos": {
        "hf_path": DATA_DIR / "tracking" / "reasonvos",
        "frames_root": DATA_DIR / "ReasonVOS_raw" / "ReasonVOS" / "JPEGImages",
        "videos_dir": DATA_DIR / "ReasonVOS" / "videos",
        "masks_dir": DATA_DIR / "ReasonVOS" / "MasksRLE",
        "annotations_dir": DATA_DIR / "ReasonVOS_raw" / "ReasonVOS" / "Annotations",
    },
}


def load_hf_dataset(path):
    import datasets
    return datasets.load_from_disk(str(path))


def encode_videos(name, cfg, ds):
    frames_root = cfg["frames_root"]
    videos_dir = cfg["videos_dir"]
    videos_dir.mkdir(parents=True, exist_ok=True)
    unique_vids = sorted(set(ds["video"]))
    encoded, skipped, failed = 0, 0, 0
    for vid in unique_vids:
        out_path = videos_dir / f"{vid}.mp4"
        if out_path.exists() and out_path.stat().st_size > 100:
            skipped += 1
            continue
        frame_dir = frames_root / vid
        if not frame_dir.is_dir():
            print(f"  WARN: no frames for {vid}")
            failed += 1
            continue
        frames = sorted(frame_dir.glob("*.jpg"))
        if not frames:
            frames = sorted(frame_dir.glob("*.png"))
        if not frames:
            print(f"  WARN: no frame files for {vid}")
            failed += 1
            continue
        pattern = str(frame_dir / "%05d.jpg")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-framerate", str(VIDEO_FPS),
            "-pattern_type", "glob", "-i", str(frame_dir / "*.jpg"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "fast",
            str(out_path)
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(f"  ffmpeg fail {vid}: {result.stderr.decode()[-200:]}")
            failed += 1
        else:
            encoded += 1
        if (encoded + skipped) % 20 == 0:
            print(f"  [{name}] videos progress: {encoded} new, {skipped} exist, {failed} fail / {len(unique_vids)} total")
    print(f"[{name}] Videos done: {encoded} new, {skipped} existing, {failed} failed / {len(unique_vids)} total")


def build_masks(name, cfg, ds):
    annotations_dir = cfg["annotations_dir"]
    output_dir = cfg["masks_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    from pycocotools import mask as mask_utils
    n_new, n_skip, n_fail = 0, 0, 0
    for i in range(len(ds)):
        ex = ds[i]
        video_id = ex["video"]
        qid = ex["qid"]
        w, h = ex["width"], ex["height"]
        query_out = output_dir / video_id / f"{qid}.json"
        if query_out.exists():
            n_skip += 1
            continue
        if name == "ref-yt-vos":
            mask_dir = annotations_dir / video_id / str(qid)
            obj_key = ex["mask_id"][0]
        elif name == "reasonvos":
            anno_ids = ex["anno_id"]
            mask_dir = annotations_dir / anno_ids[0]
            obj_key = ex["obj_id"][0]
        else:
            continue
        if not mask_dir.exists():
            n_fail += 1
            if n_fail <= 5:
                print(f"  WARN: mask dir not found: {mask_dir}")
            continue
        png_paths = sorted(mask_dir.glob("*.png"))
        if not png_paths:
            n_fail += 1
            continue
        rle_masks = []
        for png_path in png_paths:
            img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                rle_masks.append(None)
                continue
            if img.shape != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
            binary = np.asfortranarray((img > 128).astype(np.uint8))
            rle = mask_utils.encode(binary)
            rle["counts"] = rle["counts"].decode("utf-8")
            rle_masks.append(rle)
        mask_annot = {obj_key: rle_masks}
        query_out.parent.mkdir(parents=True, exist_ok=True)
        with open(query_out, "w") as f:
            json.dump(mask_annot, f)
        n_new += 1
        if (n_new + n_skip) % 100 == 0:
            print(f"  [{name}] masks progress: {n_new} new, {n_skip} exist, {n_fail} fail")
    print(f"[{name}] MasksRLE: {n_new} new, {n_skip} existing, {n_fail} failed / {len(ds)} total")


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())
    for name in targets:
        cfg = DATASETS[name]
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")
        ds = load_hf_dataset(cfg["hf_path"])
        print(f"Loaded {len(ds)} examples")
        encode_videos(name, cfg, ds)
        build_masks(name, cfg, ds)
    print("\nAll done!")


if __name__ == "__main__":
    main()
