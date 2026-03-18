# onset_detection (v2: Two-Stage Password Segment Detection)

## 新方案：两阶段 + Classifier

```
mixed2 连续流
  → Stage 1: binary segment classifier → 粗定位 password region
  → Stage 2: onset detector + IKI 节奏分析 → 精修边界 + per-password onset groups
  → Stage 3: password classifier → char top-k / sequence_topN / CER
```

---

## 训练链路（可直接运行）

### Step 1: 构建二分类数据集

```bash
python3 onset_detection/password_segment_preprocessor.py \
  --project-root . \
  --password-dirs data/raw/password/len_8 \
  --keyboard-dirs data/raw/single_key data/raw/boost \
  --negative-dirs data/raw/onset_negative \
  --mixed2-dirs data/raw/onset_mixed2 \
  --output data/processed/password_segment_dataset.npz
```

数据来源：
- **正样本 (label=1)**：`password/len_8` 全部 session
- **负样本 (label=0)**：
  - `onset_negative`（idle / trackpad / shake / freetyping）
  - `single_key` + `boost`（keyboard but not password）
  - **`mixed2` 的 `typing_1` 段**（free typing hard negative）← 关键新增

当前已知问题：
- standalone split 上该任务非常容易，不代表 mixed2 上一定可用
- 当前真正困难的不是 `password` vs `idle/trackpad/shake`
- 而是 **`password` vs `free typing`**
- 因此当前最重要的补采不是继续加 `single_key`，而是补更多 `freetyping`

### Step 2: 训练 binary segment detector

```bash
python3 onset_detection/train_onset.py \
  --task password_segment \
  --project-root . \
  --dataset data/processed/password_segment_dataset.npz \
  --model cnn \
  --checkpoint results/password_segment_detector.pt \
  --scaler results/password_segment_scaler.npz \
  --report results/password_segment_training_report.json \
  --epochs 80 \
  --batch-size 64 \
  --device cuda
```

`train_onset.py` 已支持 `--task password_segment`，自动映射 default paths 和 dataset。

当前最新状态：
- Stage 1 在 mixed2 上已经能稳定圈住真实 password 大段
- 代表性结果：`Episode IoU = 0.967`
- 当前路线已经证明“先 coarse localization，再 onset/grouping 精修”是可行的
- 当前主瓶颈已经转移到 Stage 2：
  - onset 过多
  - grouping 过碎
  - 导致 E2E Full 仍明显落后于 GT baseline

### Step 3: 确保 onset detector 已训练

```bash
python3 onset_detection/train_onset.py \
  --task onset \
  --project-root . \
  --dataset data/processed/onset_dataset.npz \
  --checkpoint results/onset_detector.pt \
  --scaler results/onset_scaler.npz \
  --device cuda
```

### Step 4: 端到端评估（含 classifier 输出 top-k / CER）

```bash
python3 onset_detection/password_segment_detector.py \
  --project-root . \
  --segment-checkpoint results/password_segment_detector.pt \
  --segment-scaler results/password_segment_scaler.npz \
  --onset-checkpoint results/onset_detector.pt \
  --onset-scaler results/onset_scaler.npz \
  --classifier-checkpoint results/inception_password_final.pt \
  --classifier-scaler results/inception_password_scaler.npz \
  --mixed2-dirs data/raw/onset_mixed2 \
  --report results/password_segment_e2e_report.json \
  --device cuda
```

输出指标：
- **Boundary**: episode_iou / episode_precision / episode_recall / start_error_ms / end_error_ms
- **E2E Full**: char_top1 / char_top3 / char_top5 / sequence_top10 / sequence_top50 / sequence_top100 / CER
- **GT Baseline**: 同上（作为 oracle 参照）

当前 mixed2 最新代表性结果：
- `Episode IoU = 0.967`
- `E2E Full` 仍较弱，说明 Stage 2 onset/grouping 还需要收紧
- `GT Baseline` 已提升到：
  - `char_top1 = 57.5%`
  - `char_top3 = 82.5%`
  - `char_top5 = 87.5%`
  - `CER = 42.5%`

---

## 关键文件变更

| 文件 | 变更 |
|---|---|
| `password_segment_preprocessor.py` | **新建** 构建二分类数据集，含 mixed2 typing_1 hard negative |
| `password_segment_detector.py` | **新建** 两阶段检测 + classifier 全链路评估 |
| `train_onset.py` | **修改** 新增 `password_segment` task 支持 |
| `README_v2.md` | **新建** 本文件 |

保留不变：`onset_model.py` / `onset_dataset.py` / `onset_utils.py` / `onset_preprocessor.py` / `eval_onset.py` / `eval_onset_e2e.py`

---

## 设计说明

### Stage 1 为什么加 free typing hard negative

`typing_1`（free typing）是 mixed2 协议中最容易和 password typing 混淆的段。
如果 Stage 1 只用 idle/trackpad/shake 作为负样本，模型会把所有 keyboard 活动都判成 positive。

建议额外采集：

```bash
python3 onset_detection/onset_collector.py \
  --mode negative \
  --activity freetyping \
  --duration 60 \
  --project-root .
```

数据会保存到：
- `data/raw/onset_negative/freetyping/`

该模式会同时记录：
- `*_sensor.csv`
- `*_events.csv`
- `*_meta.json`

当前判断：
- 如果没有足够的 `freetyping`，Stage 1 往往只会学会区分
  - `password`
  vs
  - 明显非 password 的背景
- 却学不会真正关键的
  - `password`
  vs
  - `free typing`

加入 `typing_1` 作为 hard negative 迫使模型学到 password typing 和 free typing 的区别。

此外，`password_segment_preprocessor.py` 现在会对 source 做 balancing：
- 压低 `single_key_neg`
- 抬高 `negative_freetyping`
- 抬高 `mixed2_free_typing`

这样 Stage 1 才不会继续被明显的 non-password 背景主导。

### Stage 2 IKI 节奏分析

Password 打字节奏特征：
- ~8 个键，IKI 相对规律（CV < 0.8）
- 每条 password 后有 Enter gap（> median_iki × 2.5）
- 整体 group 内 IKI 在 0.15s ~ 2.0s 范围

据此可以：
1. Group by gap → 筛选 CV 合格的节奏簇
2. 在簇内按 Enter gap 分割出单条 password
3. 每条 password 的 onset 列表 → classifier → top-k 评估

### Stage 3 接 classifier 的方式

复用 `eval_onset_e2e.py` 已有的：
- `cut_classifier_windows()`: onset 时间 → IMU 窗口
- `classify_windows()`: 窗口 → classifier 概率向量
- `topk_strings_from_prob_vectors()`: beam search → candidate strings

评估同时输出 e2e_full（完全预测链路）和 gt_baseline（GT onset）两组结果，
方便看 Stage 1+2 引入了多少降级。

### 当前真正的剩余问题

现在已经不是“这条路线对不对”的问题，而是：

1. Stage 1 已经基本成立
2. classifier 也已经重新对齐到 `36` 类并支持 adaptation
3. 当前主要瓶颈集中在 Stage 2：
   - onset threshold 偏松
   - NMS 偏松
   - rhythm/grouping 还会拆出过多 password groups
