# Molmo2 Tracking Evaluation

Unified evaluation script for Molmo2-4B on object tracking benchmarks.

## Supported Datasets

| Dataset | Split | Queries | Description |
|---------|-------|---------|-------------|
| `ref-davis17` | valid | 244 | Ref-DAVIS 2017 |
| `mevis` | valid_u | 793 | MeViS |
| `ref-yt-vos` | valid | 834 | Ref-YouTube-VOS |
| `reasonvos` | test | 458 | ReasonVOS |

## Setup

### 1. Install dependencies

```bash
conda create -n molmo2 python=3.11
conda activate molmo2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets decord scipy pycocotools
```

### 2. Download model

```bash
# Download Molmo2-4B from HuggingFace
huggingface-cli download allenai/Molmo-2-0425-4B --local-dir ./Molmo2-4B
```

### 3. Prepare data

Download and prepare the evaluation data:

```bash
# Download HuggingFace evaluation datasets
python -c "
from datasets import load_dataset
for name in ['ref-davis17', 'mevis', 'ref-yt-vos', 'reasonvos']:
    src = f'allenai/molmo2-{name}'
    ds = load_dataset(src, split='test')
    ds.save_to_disk(f'data/tracking/{name}')
"
```

Download raw frames and annotations for each dataset, then run `prep_all_datasets.py` to encode videos and build mask annotations:

```bash
python prep_all_datasets.py ref-davis17 mevis ref-yt-vos reasonvos
```

### Data directory structure

```
data/
  tracking/                  # HuggingFace arrow datasets
    ref-davis17/track/valid/
    mevis/
    ref-yt-vos/
    reasonvos/
  Ref-DAVIS17/valid/         # Encoded videos and mask RLEs
    videos/                  # {video_id}.mp4 at 6fps
    MasksRLE/                # {video_id}/{query_id}.json
  MeViS/valid_u/
    videos/
    MasksRLE/
  Ref-YT-VOS/valid/
    videos/
    MasksRLE/
  ReasonVOS/
    videos/
    MasksRLE/
```

## Evaluation

### Multi-GPU (recommended)

```bash
# Ref-DAVIS17
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 --master-port 29520 \
    eval_tracking.py --model-path ./Molmo2-4B --task ref-davis17

# MeViS
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 --master-port 29520 \
    eval_tracking.py --model-path ./Molmo2-4B --task mevis

# Ref-YouTube-VOS
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 --master-port 29520 \
    eval_tracking.py --model-path ./Molmo2-4B --task ref-yt-vos

# ReasonVOS
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 --master-port 29520 \
    eval_tracking.py --model-path ./Molmo2-4B --task reasonvos
```

### Single GPU

```bash
python eval_tracking.py --model-path ./Molmo2-4B --task ref-davis17 --gpu 0
```

### Options

```
--model-path PATH     Path to Molmo2 model (required)
--task TASK           Dataset: ref-davis17, mevis, ref-yt-vos, reasonvos (required)
--data-dir DIR        Root data directory (default: ./data)
--sampling-fps N      Sampling FPS in prompt (default: 1)
--template-id N       Prompt template 0-9, -1=random (default: -1)
--max-examples N      Limit number of examples (default: all)
--output-dir DIR      Output directory (default: ./eval_output_{task})
--smoke-test          Quick test with 5 examples
```

## Stage 1 输出

Results are saved to `eval_output/{task}/`:
- `metrics.json`: Average precision, recall, F1, HOTA, DetA, AssA
- `predictions.json`: Per-query predictions and metrics

---

## Stage 2: SAM2 分割评估 (`eval_sam2_tracking.py`)

读取 Stage 1 的 `predictions.json`，将预测的跟踪点送入 SAM2.1 生成分割 mask，然后与 GT mask 计算 J&F 指标。

### 额外依赖

```bash
pip install opencv-python pyyaml scikit-image
```

SAM2 代码和权重路径（已部署在共享存储）：
- 代码: `/ov2/zwk/lmms-eval-ov2/extension/sam2_1/`
- 配置: `/ov2/zwk/lmms-eval-ov2/extension/sam2.1_hiera_large.yaml`
- 权重: `/ov2/zwk/lmms-eval-ov2/extension/sam2.1_hiera_large.pt` (898 MB)

### JPEG 帧路径

Stage 2 需要原始 JPEG 帧（而非 mp4），各数据集帧路径如下：

```
/video_vit/tracking/vosdata/
├── Ref-DAVIS17/valid/JPEGImages/{video}/
├── MeViS/valid_u/JPEGImages/{video}/
└── Ref-YTB-VOS/valid/JPEGImages/{video}/
```

ReasonVOS 没有预提取帧，脚本自动从 mp4 用 ffmpeg 提取到临时目录。

### 运行命令

```bash
# 单个数据集 (8 GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 --master-port 29600 \
    eval_sam2_tracking.py --task ref-davis17 --skip-errors

# 全部 4 个数据集
for task in ref-davis17 mevis ref-yt-vos reasonvos; do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 --master-port 29600 \
        eval_sam2_tracking.py --task $task --skip-errors
done

# 使用自定义 predictions 路径（如评估 OV2 模型的结果）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 --master-port 29600 \
    eval_sam2_tracking.py --task ref-davis17 \
    --predictions eval_output_ov2/ref-davis17/predictions.json \
    --output-dir eval_output_ov2/ref-davis17/sam2_results \
    --skip-errors
```

### 参数说明

```
--task TASK              数据集: ref-davis17, mevis, ref-yt-vos, reasonvos (必填)
--predictions PATH       predictions.json 路径 (默认: eval_output/{task}/predictions.json)
--output-dir DIR         输出目录 (默认: eval_output/{task}/sam2_results)
--sam2-code-dir DIR      SAM2 代码目录
--sam2-config PATH       SAM2 配置文件
--sam2-checkpoint PATH   SAM2 权重文件
--video-fps FLOAT        时间戳→帧映射的 FPS (默认: 6.0)
--skip-errors            跳过失败项继续运行
```

### Stage 2 输出

```
eval_output/{task}/sam2_results/
├── sam2_metrics.json        # J, F, J&F, HOTA 汇总
└── sam2_predictions.json    # 逐条详细结果
```

### 性能优化

脚本按视频分组处理：同一视频的多个 expression 只调用一次 `init_state`（加载帧），
后续 expression 通过 `reset_state` 复用帧特征，大幅减少 I/O 开销。

---

## 完整评估流程

```
Stage 1 (eval_tracking.py):
  Video (.mp4) + Expression → Molmo2 → <tracks coords="ts pid x y;..."> → predictions.json
  指标: precision, recall, F1, HOTA (点级别, Hungarian 匹配)

Stage 2 (eval_sam2_tracking.py):
  predictions.json + JPEG 帧 → SAM2.1 Hiera Large → 分割 mask → J&F vs GT
  指标: J (IoU), F (boundary F-measure), J&F, HOTA (mask 级别)
```

---

## Molmo2-4B 评估结果

### Stage 1: 点跟踪 (eval_tracking.py)

| Dataset | n/total | Precision | Recall | F1 | HOTA | DetA | AssA |
|---------|---------|-----------|--------|----|------|------|------|
| Ref-DAVIS17 | 244/244 | 83.2 | 83.6 | 83.3 | 81.0 | 81.0 | 81.0 |
| Ref-YouTube-VOS | 834/834 | 83.7 | 83.7 | 83.7 | 82.1 | 82.1 | 82.1 |
| MeViS | 781/793 | 76.3 | 76.0 | 75.8 | 72.8 | 72.5 | 73.6 |
| ReasonVOS | 458/458 | 65.6 | 67.9 | 66.3 | 63.9 | 63.1 | 65.3 |

### Stage 2: SAM2 分割 (eval_sam2_tracking.py)

| Dataset | n/total | J | F | **J&F** | HOTA |
|---------|---------|------|------|---------|------|
| Ref-DAVIS17 | 244/244 | 74.11 | 80.57 | **77.34** | 77.15 |
| Ref-YouTube-VOS | 834/834 | 69.96 | 74.00 | **71.98** | 72.18 |
| MeViS | 781/781 | 69.57 | 75.58 | **72.57** | 71.60 |
| ReasonVOS | 458/458 | 57.92 | 64.48 | **61.20** | 60.31 |
