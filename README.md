# lmms-eval-ov2-tracking

Two-stage evaluation harness for **referring video object tracking / segmentation (RVOS)** on the
**OV2-4B** (LLaVA-OneVision-2.0) and **Molmo2-4B** vision-language models, plus auxiliary
**video-point** and **video-count** evaluations.

## 两阶段测试 (core pipeline)

RVOS 评测分两阶段，**Stage 2 只依赖 Stage 1 的 `predictions.json`，二者解耦、可分别运行，Stage 2 与模型无关**：

```
Stage 1  (VLM 点预测)                                Stage 2  (SAM2 掩码)
  mp4 + 指代表达                                       predictions.json + JPEG 帧
        │                                                    │
  eval_ov2_tracking.py    ──predictions.json──>   eval_sam2_tracking.py
  eval_molmo2_tracking.py                                    │
        │                                                    ▼
  每帧预测目标像素坐标                                 SAM2.1 Hiera-L 传播整段视频掩码
  <tracks coords="ts id x y;...">                     J / F / J&F / HOTA
  指标: P / R / F1 / HOTA (点级, Hungarian)           指标: J&F (掩码级)
```

| | **Stage 1 — 点轨迹** | **Stage 2 — 掩码** |
|---|---|---|
| 脚本 | `eval_ov2_tracking.py` (OV2) / `eval_molmo2_tracking.py` (Molmo2) | `eval_sam2_tracking.py`（模型无关，**勿改，用 CLI 覆盖路径**）|
| 输入 | `.mp4` + 指代表达 | Stage-1 `predictions.json` + JPEG 帧 |
| 输出 | `predictions.json` + `metrics.json` | `sam2_metrics.json` |
| 指标 | precision / recall / F1 / HOTA（点级）| J, F, **J&F**, HOTA（掩码级）|

## 脚本总览

| 脚本 | 阶段 | 任务 | 说明 |
|---|---|---|---|
| `eval_ov2_tracking.py` | **Stage 1** | RVOS | OV2-4B → 点轨迹 |
| `eval_molmo2_tracking.py` | **Stage 1** | RVOS | Molmo2-4B → 点轨迹 |
| `eval_sam2_tracking.py` | **Stage 2** | RVOS | 点 → SAM2 掩码 → J&F（模型无关，勿改）|
| `eval_ov2_video_point.py` | Stage 1+2 | Video-Point | OV2，单脚本内含两阶段（`--stage2-only` 仅重跑 SAM2）|
| `eval_video_point.py` | Stage 1+2 | Video-Point | Molmo2，同上 |
| `eval_ov2_video_count.py` | 单阶段 | Video-Count | OV2，计数 accuracy + MAE（**不走 SAM2**）|
| `eval_video_count.py` | 单阶段 | Video-Count | Molmo2，同上 |
| `prep_all_datasets.py` | 数据准备 | — | 抽帧编码 mp4(6fps) + 构建 mask RLE |
| `recompute_stage1_metrics.py` | 离线 | RVOS | 从 `predictions.json` 重算 Stage-1 指标（不跑模型/GPU）|
| `recompute_videotrack_metrics.py` | 离线 | VideoTrack | 重算 videotrack 基准（animal/dance/misc/person/sports）指标 |
| `demo_pipeline.py` | 可视化 | — | 挑样本 + 渲染 demo mp4（原图 / 点 / 掩码）|

> 命名规律：`eval_ov2_*` = OV2-4B（chat 模式 + 时间戳注入 + patch_positions）；无 `ov2_` 前缀 = Molmo2-4B。

## 数据集 (RVOS)

| Dataset | Split | Queries |
|---|---|---|
| `ref-davis17` | valid | 244 |
| `mevis` | valid_u | 793 |
| `ref-yt-vos` | valid | 834 |
| `reasonvos` | test | 458 |

## 环境

```bash
# 宿主 conda 环境 (server-35, 8×A800；可直接访问 /video_vit JPEG 帧)
source /root/miniconda3/etc/profile.d/conda.sh && conda activate molmo2
# torch 2.10+cu128, transformers, datasets, decord, sam2_1, iopath/hydra/omegaconf/skimage 已就绪
```

SAM2.1 Hiera-Large（Stage 2 用，已在共享盘）：
- 代码 `/ov2/zwk/lmms-eval-ov2/extension/sam2_1/`
- 配置 `/ov2/zwk/lmms-eval-ov2/extension/sam2.1_hiera_large.yaml`
- 权重 `/ov2/zwk/lmms-eval-ov2/extension/sam2.1_hiera_large.pt` (898 MB)

---

## Stage 1 — VLM 点轨迹

**OV2-4B**（推荐配置 `--sampling-fps 1 --fixed-num-frames 128 --template-id 0`）：
```bash
MODEL=/path/to/ov2_4b_hf
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 --master-port 29521 \
    eval_ov2_tracking.py --model-path "$MODEL" --task ref-davis17 \
        --sampling-fps 1 --fixed-num-frames 128 --template-id 0
# 单卡: python eval_ov2_tracking.py --model-path "$MODEL" --task ref-davis17 --gpu 0
```

**Molmo2-4B**：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 --master-port 29520 \
    eval_molmo2_tracking.py --model-path ./Molmo2-4B --task ref-davis17
```

主要参数：
```
--model-path PATH      模型目录 (必填)
--task TASK            ref-davis17 | mevis | ref-yt-vos | reasonvos (必填)
--sampling-fps N       帧采样 fps (默认 1)
--fixed-num-frames N   固定帧数上限 (默认 128)
--template-id N        prompt 模板 0-9, -1=随机 (默认 0)
--output-dir DIR       输出目录 (默认 eval_output_ov2/{task} 或 eval_output_{task})
--smoke-test           小批量快速验证
```
**输出**：`{output-dir}/predictions.json`（点轨迹）+ `metrics.json`（P/R/F1/HOTA）。

## Stage 2 — SAM2 掩码 + J&F

单个数据集：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 --master-port 29600 \
    eval_sam2_tracking.py --task ref-davis17 \
        --predictions eval_output_ov2/ref-davis17/predictions.json \
        --output-dir  eval_output_ov2/ref-davis17/sam2_results \
        --skip-errors
```
全部 4 个数据集：
```bash
for task in ref-davis17 mevis ref-yt-vos reasonvos; do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 --master-port 29600 \
    eval_sam2_tracking.py --task "$task" \
      --predictions eval_output_ov2/$task/predictions.json \
      --output-dir  eval_output_ov2/$task/sam2_results --skip-errors
done
```
主要参数：
```
--task TASK            (必填)
--predictions PATH     Stage-1 predictions.json (默认 eval_output/{task}/predictions.json)
--output-dir DIR       输出目录 (默认 eval_output/{task}/sam2_results)
--sam2-code-dir / --sam2-config / --sam2-checkpoint   SAM2 路径覆盖
--video-fps FLOAT      时间戳→帧 fps (默认 6.0)
--skip-errors          出错跳过继续
```
**输出**：`{output-dir}/sam2_metrics.json`（J/F/J&F/HOTA 汇总）+ `sam2_predictions.json`（逐条）。
JPEG 帧：`/video_vit/tracking/vosdata/{Ref-DAVIS17/valid,MeViS/valid_u,Ref-YTB-VOS/valid}/JPEGImages/{video}/`；ReasonVOS 自动从 mp4 抽帧。

---

## 其他评测

### Video-Point（单脚本两阶段）
`eval_ov2_video_point.py`（OV2）/ `eval_video_point.py`（Molmo2）：Stage 1 预测点 + Stage 2 SAM2 在**同一脚本**内完成（数据集 Molmo2-VideoPointEval, 181 例）。
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 --master-port 29540 \
    eval_video_point.py --model-path ./Molmo2-4B         # 全流程
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 --master-port 29540 \
    eval_video_point.py --stage2-only                    # 复用 predictions.json，仅重跑 SAM2
```

### Video-Count（单阶段，计数）
`eval_ov2_video_count.py`（OV2）/ `eval_video_count.py`（Molmo2）：给视频 + 计数问题预测数量；指标 = exact-match accuracy + MAE，**不走 SAM2**（数据集 Molmo2-VideoCountEval, 533 例）。
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node 8 --master-port 29550 \
    eval_video_count.py --model-path ./Molmo2-4B
```

## 辅助脚本

```bash
# 数据准备：从原始帧+标注编码 mp4(6fps) 并构建 mask RLE
python prep_all_datasets.py ref-yt-vos reasonvos

# 离线重算 Stage-1 指标（不跑模型）
python recompute_stage1_metrics.py --task ref-davis17 \
    --predictions eval_output_ov2/ref-davis17/predictions.json --out metrics_fixed.json

# 离线重算 videotrack 基准指标
python recompute_videotrack_metrics.py --root eval_output_ov2/videotrack [--write]

# 渲染可视化 demo (original/points/mask 三个 mp4)
python demo_pipeline.py --phase select                          # 单进程，挑样本写 selection.json
torchrun --nproc-per-node 8 demo_pipeline.py --phase render     # 分布式，跑 SAM2 + 出 demo mp4
```

## 输出目录

```
eval_output_ov2/{task}/            (Molmo2 默认 eval_output_{task}/)
├── predictions.json     Stage 1 点轨迹
├── metrics.json         Stage 1 指标 (P/R/F1/HOTA)
└── sam2_results/
    ├── sam2_metrics.json      Stage 2 J&F 汇总
    └── sam2_predictions.json  Stage 2 逐条
```

## 常见问题

- **端口被占**：`pkill -f 'torchrun.*eval_sam2_tracking'` 或换 `--master-port`。
- **`ModuleNotFoundError: sam2_1`**：确认 `sys.path` 含 SAM2 extension 目录（脚本已自动加）。
- **只重跑 Stage 2**：Stage 1 的 `predictions.json` 是 Stage 2 唯一输入，用 `--predictions` / `--output-dir` 指向任意目录即可。
- **OV2 首次加载报 `LlavaOnevision2Config`**：并发写坏 HF modules cache → `rm -rf ~/.cache/huggingface/modules/transformers_modules/<model>` 后单进程预热再 torchrun。

## 详细文档与结果

逐模型的环境、权重清单、完整 J&F 结果见：
- **OV2-4B** → [README_ov2.md](README_ov2.md)
- **Molmo2-4B** → [README_molmo2.md](README_molmo2.md)
