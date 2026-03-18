# onset_detection

这个模块现在最重要的结论很简单：

- Stage 1 粗定位已经成立
- Stage 3 classifier 已经成立
- 当前真正的主瓶颈是 Stage 2

## 1. 当前可靠事实

### Stage 1（coarse localization）
当前 `password_segment` 路线在 mixed2 上已经能稳定圈住真实 password 大段。
代表性结果：
- `Episode IoU = 0.967`

这说明：
- `password_typing vs non_password` 的 coarse region extraction 是有效的
- 至少在受控 mixed2 协议里，password block 可以先被稳定提出来

### Stage 3（classifier）
现有 `36` 类 adapted classifier 在 mixed2 上的 GT baseline：
- `char_top1 = 57.5%`
- `char_top3 = 82.5%`
- `char_top5 = 87.5%`
- `CER = 42.5%`

这说明：
- 只要分段/对齐合理，Stage 3 是能工作的

## 2. 当前失败经验

我们已经系统试过以下 Stage 2 方向：

### 历史 heuristic / energy-valley
入口：
- [password_segment_detector.py](/Users/shiyi/备份（mac_vs专用）/onset_detection/password_segment_detector.py)

结果：
- 能给出一些 coarse-to-fine baseline
- 但始终停留在 heuristic / proposal + grouping 的瓶颈里

### Claude branch
目录：
- [stage2_claude](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_claude)

尝试过：
- `dp_classifier`
- `dense_ctc`

当前代表性结论：
- CTC 路线已经能跑通 Path B
- 但 mixed2 上结果很差，约：
  - `char_top1 ≈ 2.5%`
  - `CER ≈ 97.5%`
- 因此 Claude 线目前保留作 baseline / side branch

### GPT branch
目录：
- [stage2_gpt54](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_gpt54)

尝试过：
- dense `key / boundary / inside` modeling
- structured decode
- top-K hypothesis + global classifier rerank

当前代表性结论：
- 结构上能稳定输出 `5 groups / 40 onsets`
- 但 mixed2 E2E 仍然不够好
- 代表性结果曾到：
  - `char_top1 = 7.5%`
  - `top3 = 15.0%`
  - `top5 = 27.5%`
  - `CER = 90.0%`
- 后续 top-K global rerank 也没有根本救回来

## 3. 当前最可信的总判断

当前最可能的问题不是：
- 某个阈值没调对
- 某个 decoder 再补一点就行

而是：
- **Stage 2 的训练任务定义和 mixed2 连续流测试分布没有真正对齐**

更直白地说：
- clean `password/len_8` 数据不能自然替代 mixed-style Stage 2 训练数据
- 这和 earlier classifier 路线里“single_key 不做 password adaptation 就直接测 password 会失败”是同一逻辑

## 4. 当前建议的新方向

保留：
- Stage 1
- Stage 3

重建：
- Stage 2

当前最被认可的重建思路是：
- mixed-style Stage 2 训练数据 / pseudo mixed training
- 把 Stage 2 拆成：
  - `2A`: password group segmentation
  - `2B`: onset detection within each group

也就是说，不再把一个模型硬逼成一次性恢复 `5 × 8` 全结构。

## 5. 当前关键文件

- [password_segment_preprocessor.py](/Users/shiyi/备份（mac_vs专用）/onset_detection/password_segment_preprocessor.py)
- [password_segment_detector.py](/Users/shiyi/备份（mac_vs专用）/onset_detection/password_segment_detector.py)
- [stage2_claude/README.md](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_claude/README.md)
- [stage2_gpt54/README.md](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_gpt54/README.md)
- [ONSET_CODEX_HANDOFF.md](/Users/shiyi/备份（mac_vs专用）/ONSET_CODEX_HANDOFF.md)

如果是新接手这个模块，建议先读：
1. 本文件
2. `stage2_claude/README.md`
3. `stage2_gpt54/README.md`
4. `ONSET_CODEX_HANDOFF.md`
