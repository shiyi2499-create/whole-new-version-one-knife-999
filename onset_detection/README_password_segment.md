# onset_detection (historical `password_segment` branch)

这个分支记录的是我们最早那条“先 coarse localization，再 onset/grouping 精修”的两阶段路线。

它仍然有价值，但现在更适合被看作：
- 一个重要历史 baseline
- 一个已经帮助我们定位问题的实验分支

而不是当前唯一推荐主线。

## 核心流程

```text
mixed2 连续流
  -> Stage 1: binary `password_typing vs non_password`
  -> Stage 2: onset + grouping / rhythm / energy-valley
  -> Stage 3: password classifier
```

## 这个分支真正证明了什么

### 已经证明对的
- Stage 1 coarse localization 是可行的
- 在补了 `freetyping` 负样本并做 source balancing 后，mixed2 上代表性结果：
  - `Episode IoU = 0.967`

### 也证明了什么不够
- 单纯靠 onset proposal + grouping heuristic 很难真正解决 mixed2
- 即使加了 energy-valley、auto sweep、protocol prior，最终 Stage 2 仍然没有被根本解决

## 当前 mixed2 上的重要参照

### GT baseline
- `char_top1 = 57.5%`
- `char_top3 = 82.5%`
- `char_top5 = 87.5%`
- `CER = 42.5%`

### 含义
- Stage 3 本身不是主要问题
- Stage 2 的分组/对齐/slot 定位才是主要问题
- 但这不等于 Stage 3 已经到头：
  - 当前最成熟的是 `InceptionTime` adaptation 线
  - `password-only` 没有在已核实实验里超过 `baseline + password adaptation`
  - 多模型和更强 classifier 版本仍然值得后续继续做
  - 当前最成熟的实验口径仍然是 `36` 类、`len=8`；符号与更广长度设置还没有系统展开

## 为什么这个分支不再是最终主线

因为当前更可能的根因已经被定位为：
- Stage 2 训练任务定义不对
- Stage 2 训练分布和 mixed2 测试分布不对齐

所以当前建议是：
- 保留本分支作为历史 baseline
- 把新的主要精力放到 mixed-style Stage 2 重建上

## 相关文件

- [password_segment_preprocessor.py](/Users/shiyi/备份（mac_vs专用）/onset_detection/password_segment_preprocessor.py)
- [password_segment_detector.py](/Users/shiyi/备份（mac_vs专用）/onset_detection/password_segment_detector.py)
