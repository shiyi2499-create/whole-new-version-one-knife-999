# Onset Handoff For Next Codex

如果你是新开的 Codex 会话，请把自己当作上一位接手者的直接延续。

## 先看什么
按这个顺序读：
1. [README.md](/Users/shiyi/备份（mac_vs专用）/README.md)
2. [onset_detection/README.md](/Users/shiyi/备份（mac_vs专用）/onset_detection/README.md)
3. [onset_detection/stage2_claude/README.md](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_claude/README.md)
4. [onset_detection/stage2_gpt54/README.md](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_gpt54/README.md)
5. [CODE_MAP.md](/Users/shiyi/备份（mac_vs专用）/CODE_MAP.md)

## 当前目标
我们在做的是 IMU password side-channel。
当前 onset 方向不是泛化 activity recognition，而是：
- Stage 1: 从 mixed2 连续流里圈出 password coarse block
- Stage 2: 在 coarse block 内恢复 5 条 password 的结构与 onset
- Stage 3: 用现有 classifier 恢复字符

## 当前已经确认成立的事
### Stage 1
- coarse localization 已成立
- mixed2 代表性结果：`Episode IoU = 0.967`

### Stage 3
- 36 类 adapted classifier 已成立
- mixed2 GT baseline：
  - `char_top1 = 57.5%`
  - `char_top3 = 82.5%`
  - `char_top5 = 87.5%`
  - `CER = 42.5%`

## 当前真正的瓶颈
- Stage 2

## 我们已经试过什么
### 历史 heuristic / energy-valley
入口：
- [onset_detection/password_segment_detector.py](/Users/shiyi/备份（mac_vs专用）/onset_detection/password_segment_detector.py)

结论：
- 证明了 coarse-to-fine 思路有一定价值
- 但没有真正解决 mixed2 Stage 2

### Claude branch
目录：
- [onset_detection/stage2_claude](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_claude)

结论：
- 已真正接进 Path B
- CTC 路线在 mixed2 上很差
- 当前保留作 baseline

### GPT branch
目录：
- [onset_detection/stage2_gpt54](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_gpt54)

结论：
- dense structured 路线比 Claude 更有希望
- 但仍未解决 mixed2
- 后续 top-K global rerank 也没有根本救回来

## 当前最可信的新判断
不要再默认问题只是 decoder 或阈值。
当前最可能的根因是：
- Stage 2 的训练任务定义不对
- Stage 2 的训练分布和 mixed2 测试分布没有真正对齐

这和 earlier classifier 路线里：
- single_key 不做 password adaptation
- 直接测 password 会失败
是同一逻辑

## 当前最被认可的新方向
保留：
- Stage 1
- Stage 3

重建：
- Stage 2

更具体地说：
- mixed-style Stage 2 训练数据 / pseudo mixed training
- Stage 2 拆成：
  - `2A`: password group segmentation
  - `2B`: onset detection within each group

## 你接下来应该怎么做
### 第一步
先浏览：
- `onset_detection/`
- `onset_detection/stage2_claude/`
- `onset_detection/stage2_gpt54/`

不要急着修旧 heuristic。

### 第二步
先回答：
- 现有数据结构能不能支持 mixed-style Stage 2 重建？
- 是先做 pseudo mixed training，还是直接定义新的 `mixed_training` 采集协议？
- `2A / 2B` 的数据构造和训练入口应该如何落到当前代码树？

### 第三步
如果你要写代码，优先做：
- 新的 Stage 2 数据构造
- 新的 `2A / 2B` 训练脚手架
- 不要覆盖现有 Claude / GPT 分支，把它们保留作 baseline

## 注意
- `mixed2` 当前应视为 held-out 连续流测试目标，不要直接混进训练
- 不要再把 Stage 1 和 Stage 3 当主问题去改
- 当前最值得怀疑和重建的是 Stage 2
