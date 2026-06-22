# OV2-4B Tracking Evaluation

Unified evaluation pipeline for the OV2-4B (LLaVA-OneVision-2.0) tracking model on
referring video object segmentation / tracking benchmarks. Stage 1 runs the OV2
model to predict referring-object point tracks; Stage 2 feeds those points into
SAM2.1 Hiera Large to propagate masks and compute J&F.

## Supported Datasets

| Dataset | Split | Queries | Description |
|---------|-------|---------|-------------|
| `ref-davis17` | valid   | 244 | Ref-DAVIS 2017 |
| `mevis`       | valid_u | 793 | MeViS |
| `ref-yt-vos`  | valid   | 834 | Ref-YouTube-VOS |
| `reasonvos`   | test    | 458 | ReasonVOS |

## Environments

Two equivalent ways to run the evaluation on server-35 (`172.16.5.35`, 8×A800).

### Option A — Host conda env `molmo2` (推荐，可直接访问 `/video_vit` JPEG 帧)

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate molmo2
# torch 2.10.0+cu128, datasets 4.6.1, python 3.11
# iopath / hydra / omegaconf / skimage / sam2_1 已就绪
```

### Option B — Docker 容器 (`lmms-eval-ov2:latest` 镜像)

服务器 35 上已创建的容器：

| 容器名 | 用途 |
|--------|------|
| `tracking_eval_ov2_iter2400_on35`           | OV2 tracking 评估 (已挂 `/video_vit`) |
| `lmms_eval_ov2_container_feilong_qwen3vl`   | OV2 / Qwen3-VL 通用 lmms-eval 环境 |
| `lmms_eval_ov2_container_zwk`               | 原始参考容器 |

进入容器：

```bash
docker exec -it tracking_eval_ov2_iter2400_on35 bash
```

注意：Stage 2 需要 `/video_vit/tracking/vosdata/...` 下的 JPEG 帧，请确保所选容器已挂载
该路径（`tracking_eval_ov2_iter2400_on35` 已挂载）。

## Model & Data Paths (server-35)

### 可用 OV2-4B 权重 (按训练步数升序)

均位于 `/ov2/feilong/LLaVA-OneVision-2.0/examples/llava_onevision2/convert/`：

| 权重目录 | 备注 |
|----------|------|
| `tracking_molmo2_200k_iter_0001000_hf` | baseline 200k+iter1000 |
| `ax_instruct_video_8gpus_tracking_iter_0000650_hf` | ax_instruct iter650 |
| `ax_instruct_video_8gpus_tracking_iter_0000900_hf` | ax_instruct iter900 |
| `ax_instruct_video_8gpus_tracking_iter_0001610_hf` | ax_instruct iter1610 |
| `ax_instruct_video_8gpus_tracking_iter_0002000_hf` | ax_instruct iter2000 |
| `ax_instruct_video_8gpus_tracking_iter_0002640_hf` | ax_instruct iter2640 (最新) |

> **推荐**：`ax_instruct_video_8gpus_tracking_iter_0001610_hf` / `iter_0002000_hf` / `iter_0002640_hf`（综合 4 数据集 J&F 最佳）。
> **推荐评测配置**：`--sampling-fps 1 --fixed-num-frames 128 --template-id 0`。
>
> ⚠️ 首次用 8 卡 torchrun 加载新权重时，`~/.cache/huggingface/modules/transformers_modules/<model>/`
> 可能因并发写入损坏（典型报错 `module ... has no attribute 'LlavaOnevision2Config'`）。
> 解决：`rm -rf ~/.cache/huggingface/modules/transformers_modules/<model>` 后用单进程
> `python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('<path>', trust_remote_code=True)"`
> 预热 cache，再启 torchrun。

### 通用资源路径

| 资源 | 路径 |
|------|------|
| Stage 1 脚本 | `/ov2/feilong/simple_repo/eval_ov2_tracking.py` |
| Stage 2 脚本 | `/ov2/feilong/simple_repo/eval_sam2_tracking.py` （**不要修改**） |
| 批处理脚本 | `/ov2/feilong/simple_repo/logs/run_sam2_all4_ov2.sh` |
| SAM2 代码 | `/ov2/zwk/lmms-eval-ov2/extension/sam2_1/` |
| SAM2 配置 | `/ov2/zwk/lmms-eval-ov2/extension/sam2.1_hiera_large.yaml` |
| SAM2 权重 | `/ov2/zwk/lmms-eval-ov2/extension/sam2.1_hiera_large.pt` (898 MB) |
| 视频 (mp4) | `/ov2/feilong/simple_repo/data/{Ref-DAVIS17/valid,MeViS/valid_u,Ref-YT-VOS/valid,ReasonVOS}/videos/{video}.mp4` |
| Mask RLE | `/ov2/feilong/simple_repo/data/{…}/MasksRLE/{video}/{qid}.json` |
| JPEG 帧 (Stage 2) | `/video_vit/tracking/vosdata/{Ref-DAVIS17/valid,MeViS/valid_u,Ref-YTB-VOS/valid}/JPEGImages/{video}/` |
| ReasonVOS Stage 2 帧 | 自动从 `.mp4` 用 ffmpeg 临时抽帧 |
| 输出根目录 | `/ov2/feilong/simple_repo/eval_output_ov2/{task}/` |

---

## Stage 1: OV2 点跟踪 (`eval_ov2_tracking.py`)

读取 HuggingFace arrow 数据集和 mp4 视频，调用 OV2-4B 预测每一帧上对应目标的像素
坐标（归一化到 0–1000），输出 `<tracks coords="ts id x y;...">` 格式。然后与 GT
mask 采样出的参考点做 Hungarian 匹配，得到 precision / recall / F1 / HOTA。

### 运行命令

```bash
MODEL=/ov2/feilong/LLaVA-OneVision-2.0/examples/llava_onevision2/convert/ax_instruct_video_8gpus_tracking_iter_0002000_hf

# 多 GPU (8 卡) — 推荐配置 fps=1, fnf=128, tpl0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 --master-port 29521 \
    eval_ov2_tracking.py --model-path "$MODEL" --task ref-davis17 \
        --sampling-fps 1 --fixed-num-frames 128 --template-id 0

# 单 GPU
python eval_ov2_tracking.py --model-path "$MODEL" --task ref-davis17 --gpu 0
```

### 关键参数

```
--model-path PATH        OV2 模型目录 (必填)
--task TASK              ref-davis17 | mevis | ref-yt-vos | reasonvos (必填)
--sampling-fps N         帧采样 fps (默认 1)
--fixed-num-frames N     固定帧数上限 (默认 128)
--max-pixels N           每帧最大像素数 (默认 313600 ≈ 400×784)
--min-pixels N           每帧最小像素数 (默认 3136 = 256*28*28 ≈)
--max-new-tokens N       生成最大 token (默认 2048，会按帧数自适应)
--template-id N          prompt 模板 0–9, -1=随机 (默认 0; -1 即 promptsweep)
--output-dir DIR         输出目录 (默认 eval_output_ov2/{task})
--smoke-test             小批量快速跑
```

### Stage 1 输出

```
eval_output_ov2/{task}/
├── metrics.json       # precision, recall, f1, HOTA, DetA, AssA
└── predictions.json   # 每个 query 的 tracks 和指标
```

---

## Stage 2: SAM2 分割 (`eval_sam2_tracking.py`)

读取 Stage 1 的 `predictions.json`，将预测点作为 prompt 送入 SAM2.1 Hiera Large，
在整段视频上 propagate 得到 mask，然后与 GT mask 计算 J、F、J&F、HOTA。

**该脚本不可修改，通过 CLI 覆盖路径即可。**

### 额外依赖 (Option A 已安装)

```bash
pip install opencv-python pyyaml scikit-image iopath hydra-core omegaconf
```

### 单个数据集

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 --master-port 29600 \
    eval_sam2_tracking.py \
      --task ref-davis17 \
      --predictions eval_output_ov2/ref-davis17/predictions.json \
      --output-dir  eval_output_ov2/ref-davis17/sam2_results \
      --skip-errors
```

### 全部 4 个数据集 (批处理)

`logs/run_sam2_all4_ov2.sh` 按顺序跑 4 个数据集（已有 `sam2_metrics.json` 会跳过）：

```bash
cd /ov2/feilong/simple_repo
nohup bash logs/run_sam2_all4_ov2.sh > logs/run_sam2_all4_ov2.out 2>&1 &
```

脚本内容：

```bash
TASKS=(ref-davis17 mevis ref-yt-vos reasonvos)
for TASK in "${TASKS[@]}"; do
  OUT_DIR=eval_output_ov2/${TASK}/sam2_results
  PRED=eval_output_ov2/${TASK}/predictions.json
  LOG=eval_output_ov2/${TASK}/sam2_eval.log
  mkdir -p "$OUT_DIR"
  [[ -f "$OUT_DIR/sam2_metrics.json" ]] && { echo "[skip] $TASK"; continue; }
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    torchrun --nproc-per-node 8 --master-port 29600 \
      eval_sam2_tracking.py \
      --task "$TASK" \
      --predictions "$PRED" \
      --output-dir "$OUT_DIR" \
      --skip-errors \
    > "$LOG" 2>&1
done
```

### 关键参数

```
--task TASK              ref-davis17 | mevis | ref-yt-vos | reasonvos (必填)
--predictions PATH       Stage 1 predictions.json (默认 eval_output/{task}/predictions.json)
--output-dir DIR         输出目录 (默认 eval_output/{task}/sam2_results)
--sam2-code-dir DIR      SAM2 代码目录
--sam2-config PATH       SAM2 yaml
--sam2-checkpoint PATH   SAM2 .pt
--video-fps FLOAT        时间戳→帧的 FPS (默认 6.0)
--skip-errors            出错跳过继续
```

### Stage 2 输出

```
eval_output_ov2/{task}/sam2_results/
├── sam2_metrics.json        # J, F, J&F, HOTA 汇总
└── sam2_predictions.json    # 每 query 的 mask 指标
```

### 性能优化

脚本按视频分组：同一视频的多个 expression 只 `init_state`（加载帧） 一次，后续
expression 通过 `reset_state` 复用帧特征，显著降低 I/O。

---

## 完整评估流程

```
Stage 1  eval_ov2_tracking.py:
  mp4 + expression → OV2-4B → <tracks coords="ts pid x y;..."> → predictions.json
  指标: precision / recall / F1 / HOTA (点级, Hungarian 匹配)

Stage 2  eval_sam2_tracking.py:
  predictions.json + JPEG 帧 → SAM2.1 Hiera Large → mask → J&F vs GT
  指标: J (IoU), F (boundary F-measure), J&F, HOTA (mask 级)
```

---

## OV2-4B 评估结果

Model: `tracking_molmo2_200k_iter_0001000_hf` (200k step, iter 1000)

### Stage 1: 点跟踪 (`eval_ov2_tracking.py`)

| Dataset          | n/total | Precision | Recall | F1   | HOTA | DetA | AssA |
|------------------|---------|-----------|--------|------|------|------|------|
| Ref-DAVIS17      | 244/244 | 65.47     | 65.53  | 65.47 | 59.17 | 59.16 | 59.21 |
| Ref-YouTube-VOS  | 834/834 | 78.03     | 78.03  | 78.03 | 75.00 | 75.00 | 75.00 |
| MeViS            | 774/793 | 53.34     | 52.09  | 52.44 | 48.24 | 47.42 | 49.60 |
| ReasonVOS        | 457/458 | 44.94     | 45.37  | 45.07 | 41.13 | 41.39 | 40.95 |

### Stage 2: SAM2 分割 (`eval_sam2_tracking.py`)

| Dataset          | n/total | J     | F     | **J&F**   | HOTA  |
|------------------|---------|-------|-------|-----------|-------|
| Ref-DAVIS17      | 244/244 | 61.52 | 69.97 | **65.74** | 66.16 |
| MeViS            | 774/774 | 50.68 | 57.84 | **54.26** | 54.17 |
| Ref-YouTube-VOS  | —       | —     | —     | running   | —     |
| ReasonVOS        | —       | —     | —     | queued    | —     |

> Stage 2 正在 server-35 上运行，完成后从
> `eval_output_ov2/{task}/sam2_results/sam2_metrics.json` 读取更新。

---

## 常见问题

- **Port 29600 被占用**：上一次 torchrun 没退干净。`pkill -f 'torchrun.*eval_sam2_tracking'`
  或换一个端口（`--master-port 29601`）。
- **`ModuleNotFoundError: sam2_1`**：确认 `sys.path` 包含
  `/ov2/zwk/lmms-eval-ov2/extension`（脚本内已自动处理）。
- **容器找不到 `/video_vit`**：使用 `tracking_eval_ov2_iter2400_on35` 容器或
  直接用宿主机 `molmo2` conda 环境。
- **只跑 Stage 2**：Stage 1 的 `predictions.json` 是 Stage 2 的唯一输入，可以
  通过 `--predictions` / `--output-dir` 指向任意目录（例如 `eval_output_ov2/...`）。
