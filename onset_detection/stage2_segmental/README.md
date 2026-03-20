# Stage 2 Segmental Prototype (GT episode only)

这是第一版 **monotonic segment / sequence-level learned cutting** 原型。

它不再把单键建模为：
- onset point
- frame spike
- 固定 pre/post fixed window

而是把一个 password episode 内的每个 key 建模为：
- 一个**有序的、共享边界的 segment**
- 相邻 key 之间通过**单调边界**共同决定切窗
- 然后再把学到的 segment 重采样到 classifier 所需长度进行字符识别

## 核心形式

给定 GT episode 和已知 per-key timestamps：

1. 用 episode encoder 编码整个 IMU 序列
2. 在相邻 key timestamps 之间预测共享 boundary
3. 得到一串单调递增的边界 `b0 < b1 < ... < bK`
4. 每个 key 的 segment 为 `[b{i-1}, b{i}]`
5. 将 segment 微分可传播地重采样为固定长度窗口
6. 送入 Inception classifier 进行识别

## 这一版的目的

这是一个 **GT episode-only prototype**，主要回答两个问题：

1. 在 **已知 episode** + **已知 per-key timestamps** 的条件下，
   learned cutting / shared-boundary segmentation 是否能明显优于
   `GT timestamp -> fixed 100/200ms window -> classify` baseline？
2. 如果答案是肯定的，那么下一步再把 GT timestamps 拿掉，转向更完整的
   sequence alignment / monotonic transduction 就更有把握。

## 训练脚本

```bash
python onset_detection/stage2_segmental/scripts/train_gt_segmental.py \
  --input_dir data/raw/mixed_training \
  --output_dir runs/stage2_segmental_gt \
  --classifier_checkpoint results/inception_password_final.pt \
  --classifier_scaler results/inception_password_scaler.npz \
  --device mps
```

如果没有现成 classifier checkpoint：
- 脚本会先用 train split 上的 **GT fixed windows** 训练一个本地 Inception classifier
- 然后冻结这个 classifier，再训练 segmental cutter
- 这样 baseline 与 segmental 的比较仍然公平（同一个 classifier）

## 输出

训练后会在 output_dir 下生成：
- `best_segmental.pt`
- `training_report.json`
- `baseline_results.json`
- `segmental_results.json`
- `segmental_debug.json`
- `local_classifier.pt`（若本地预训练 classifier）

`training_report.json` 里会直接给出：
- fixed-window baseline
- segmental learned-cutting
- 两者 top1 / top5 / CER 的差值

## Smoke test

```bash
python onset_detection/stage2_segmental/scripts/smoke_test.py
```

这个 smoke test 使用 bundle 自带的 `data_samples/` 两个 mixed_training session，
仅用于验证原型代码能跑通，并给出一个非常小规模的 sanity-check 结果。
