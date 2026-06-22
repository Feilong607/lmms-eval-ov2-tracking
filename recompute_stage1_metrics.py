'''Recompute Stage-1 metrics offline from predictions.json, using the fixed
gt_masks id-remap logic. Does NOT call the model.'''
import argparse, json, sys
from pathlib import Path
import numpy as np
from datasets import load_from_disk

sys.path.insert(0, '/ov2/feilong/simple_repo')
import eval_ov2_tracking as E  # use patched functions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=list(E.TASK_CONFIGS.keys()))
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--data-dir', default='/ov2/feilong/simple_repo/data')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = E.TASK_CONFIGS[args.task]
    data_dir = Path(args.data_dir)
    masks_dir = data_dir / cfg['masks_subpath']
    hf_data = data_dir / cfg['hf_subpath']
    ds = load_from_disk(str(hf_data))

    # Index dataset by unique id
    by_key = {ex['id']: ex for ex in ds}

    preds = json.load(open(args.predictions))
    results = []
    skipped = 0
    for r in preds:
        vname, qid = r['video'], str(r['qid'])
        ex = by_key.get(r['id'])
        if ex is None:
            skipped += 1
            continue
        mask_path = masks_dir / vname / f'{qid}.json'
        if not mask_path.exists():
            skipped += 1
            continue
        h, w = int(ex['height']), int(ex['width'])
        vfps = ex.get('fps', 3)
        try:
            pred_tracks = E.extract_tracks(r['prediction'], w, h, vfps)
            gt_tracks = E.build_gt_tracks(ex)
            raw = json.load(open(mask_path))
            raw_keys = list(raw.keys()); mids = ex['mask_id']
            gt_masks = {}
            for i, mid in enumerate(mids):
                if mid in raw: gt_masks[str(i)] = raw[mid]
                elif i < len(raw_keys): gt_masks[str(i)] = raw[raw_keys[i]]
            metrics = E.evaluate_video_object_tracking(pred_tracks, gt_tracks, gt_masks, h, w)
        except Exception as e:
            if skipped < 5: import traceback; traceback.print_exc()
            skipped += 1
            continue
        results.append({**r, 'metrics': {k: v for k, v in metrics.items() if not isinstance(v, (list, np.ndarray))}})

    avg = {}
    for k in ['precision', 'recall', 'f1', 'HOTA', 'DetA', 'AssA']:
        vals = [x['metrics'][k] for x in results if k in x['metrics']]
        avg[k] = float(np.mean(vals)) if vals else 0.0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump({'task': cfg['display_name'], 'metrics': avg,
                   'n': len(results), 'total': len(preds), 'skipped': skipped}, f, indent=2)
    print(f'{cfg["display_name"]}: n={len(results)}/{len(preds)} skipped={skipped}')
    print(json.dumps(avg, indent=2))

if __name__ == '__main__':
    main()
