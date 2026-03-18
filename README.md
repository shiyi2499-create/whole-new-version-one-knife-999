# Apple Internal IMU Keystroke Side-Channel

本仓库当前最重要的目标是：

- 在 Apple 设备内部 IMU 信号上，完成一条可验证的键盘侧信道恢复链路
- 先在单人、单设备、受控协议下把闭环跑通
- 再逐步讨论更弱先验、更强泛化

## 当前总体结论（2026-03-19）

- Stage 1 `password vs non_password` 粗定位已经成立
- Stage 3 `36` 类 password classifier 已成立，且 `len=8` adaptation 有效
- 当前真正卡住的是 Stage 2
- 最近的主要教训不是“某个阈值没调好”，而是：
  - Stage 2 的训练任务定义和 mixed-style 连续流测试分布没有真正对齐
  - clean `password/len_8` 数据不能自然替代 mixed-style Stage 2 训练数据

因此，当前 onset 主线已经从“继续补 heuristic”转向：

- 保留 Stage 1
- 保留 Stage 3
- 重建 Stage 2 的数据与任务定义

## 与 onset 相关的当前可靠结论

### Stage 1
- `onset_detection/password_segment_preprocessor.py`
- `onset_detection/password_segment_detector.py`

在 mixed2 上，粗定位已经能稳定圈住真实 password 大段。代表性结果：
- `Episode IoU = 0.967`

### Stage 3
- `adapt_password_len8_inception.py`
- `phase3_password_inception/run_password_closure_inception.py`

当前 `36` 类 adapted classifier 的代表性结果：
- 固定 `160/40` split：
  - `char_top1 = 73.8%`
  - `char_top3 = 96.9%`
  - `char_top5 = 98.8%`
  - `sequence_top100 = 67.5%`
  - `CER = 26.2%`

mixed2 上的 GT baseline：
- `char_top1 = 57.5%`
- `char_top3 = 82.5%`
- `char_top5 = 87.5%`
- `CER = 42.5%`

关于 password classifier，有 3 个当前必须记住的事实：
- 目前真正被系统验证跑通的是 `InceptionTime` 这条线，不是“所有模型都验证过后的最终最优版”
- 在现有实验里，`single_key / merged baseline + password adaptation` 明显强于 `password-only` 直接训练；这件事至少在 Inception 路线上已经被验证
- 还没有做完的事情包括：多模型系统比较、更强 adaptation 设计、以及把 password classifier 做到“满血版”

换句话说：
- password classifier 路线已经证明可用
- 但它还没有被证明已经到达天花板
- 当前 mixed2 卡住的主因仍然更像 Stage 2，而不是 classifier 已经无路可走

### Stage 2（现状）
我们已经走过 4 条线：
- heuristic / energy-valley
- Claude `dp_classifier`
- Claude `dense_ctc`
- GPT `dense_structured`

当前结论：
- Claude CTC 可运行，但 mixed2 很差，只保留为 baseline
- GPT dense structured 比 Claude 线更好，但仍未解决 mixed2
- 当前最可信的新方向是：
  - 重新设计 mixed-style Stage 2 训练数据
  - 把 Stage 2 拆成 `2A group segmentation + 2B onset detection`

## onset 相关入口

- `onset_detection/README.md`
  - onset 主模块状态说明
- `onset_detection/README_password_segment.md`
  - 历史两阶段 password-segment 路线与教训
- `onset_detection/stage2_claude/README.md`
  - Claude 分支定位与当前结果
- `onset_detection/stage2_gpt54/README.md`
  - GPT 分支定位与当前结果
- `ONSET_CODEX_HANDOFF.md`
  - 下一个 Codex 会话的快速接手说明

## 当前建议

如果要继续 onset 主线，不建议再主要围绕旧 heuristic 调参。
建议先阅读：
- `ONSET_CODEX_HANDOFF.md`
- `onset_detection/README.md`
- `onset_detection/stage2_claude/README.md`
- `onset_detection/stage2_gpt54/README.md`

然后决定：
- 是先补 `mixed_training` / pseudo mixed training 数据
- 还是先把新的 `Stage 2A / 2B` 重建方案正式落到代码里
