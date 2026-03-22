# Next Chat Brief (2026-03-22)

下面这段可以直接交给下一个有整机权限的聊天。目标是让他像直接接手当前上下文一样工作。

## 你接手的项目是什么

这是一个 Apple 设备内部 IMU 键盘侧信道项目。
当前真正的核心目标是：

> 录一整段连续 IMU 数据流，模型自动找出其中哪一段是 password，并恢复内容。

## 你先不要再花时间重讲什么

下面这些已经站住，不要再把时间花在重复证明它们上：

1. `Stage 3` classifier 已经成立，而且 `len8/9/10` 多长度版本也已经站住。
2. `Stage 2` 的“已知 password 段内找真正 key 峰”已经成立。
   - 代表性报告：
     - `/Users/shiyi/备份（mac_vs专用）/results/stage2_peak_keyness_len8_len9_len10_v2/report.json`
   - 指标：
     - `exact_all_keys = 94.94%`
     - `mean_peak_recall = 99.37%`
     - `mean_peak_precision = 99.37%`
3. 长度/计数是可学的，不必再怀疑这一点本身。

## 当前真正没打通的是什么

不是 classifier，不是段内 keyness，也不是 fixed-window 本身。

当前真正的瓶颈是：

> full-stream 里哪个 candidate burst 才是真正的 password，
> 以及 clean non-GT 条件下如何把 candidate ranking / length coupling 和 downstream recovery 对齐。

也就是说，问题在 `Stage 1 / Stage2` 边界。

## 当前最可信的主线

请沿这条主线继续，不要回头到旧的 segment 二分类主线：

```text
full stream
-> propose peaks
-> peak keyness on all peaks
-> cluster high-keyness peaks into candidate bursts
-> bag/context/recoverability ranking
-> choose top burst
-> within-burst key selection
-> fixed-window / overlap recovery
```

当前关键脚本：
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/train_eval_peak_keyness.py`
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/train_eval_segment_bagrank_ctx_v2.py`
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/length_model.py`

## 当前最新最诚实的结果

- `/Users/shiyi/备份（mac_vs专用）/results/stage2_segment_bagrank_ctx_v2_keynesspool_union_nogthint_targetv2_proxyv3/report.json`
- baseline:
  - `top1 = 39.22%`
  - `top5 = 56.86%`
  - `CER = 60.78%`
- overlap:
  - `top1 = 45.10%`
  - `top5 = 58.82%`
  - `CER = 54.90%`

当前 oracle：
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_segment_bagrank_ctx_v2_keynesspool_union_oracle_check.json`
- `mean_best = 0.7381`
- `ge_0_75 = 0.6667`
- `ge_0_90 = 0.6667`

这说明：
- `keynesspool` proposer 是对的
- `union` 只是局部补 proposal，不是核心突破
- 当前剩下的大头还是 clean ranking / length coupling

## 当前 refined diagnosis

请带着下面这组判断往下做：

1. `peak keyness` 已经学到本质
2. `keynesspool` proposer 是当前最对的 Stage1/Stage2 主方向
3. 当前剩下的问题分两类：
   - candidate completeness：真 password burst 是否完整进池
   - clean ranking / length coupling：好候选已经在池里时，如何让它稳定排第 1

## 已经吸收过但要谨慎使用的外部思路

Claude / GPT Pro 有用的部分已经被吸收：
- bag/listwise ranking
- context
- recoverability target

但不要直接照搬：
- 假设用户是“快速打字”的 proposer
- 偏向大 cluster 的 `score_sum`
- 脱离本地数据分布另起一整套 stage1 代码

## 你接手后优先做什么

1. 不要删任何已有研究分支和旧文档。
2. 不要再回头做 pointwise segment classifier 主线。
3. 优先继续打：
   - candidate completeness
   - clean ranking / length coupling
4. 先把 `single len8/9/10` 打稳，再回到 `retry`。

## 你应该先读这些文件

1. `/Users/shiyi/备份（mac_vs专用）/README.md`
2. `/Users/shiyi/备份（mac_vs专用）/PROJECT_HANDOFF_20260322.md`
3. `/Users/shiyi/备份（mac_vs专用）/ONSET_CODEX_HANDOFF.md`
4. `/Users/shiyi/备份（mac_vs专用）/onset_detection/README.md`
5. `/Users/shiyi/备份（mac_vs专用）/onset_detection/ONSET_METHODS_AND_CONCLUSIONS.md`
6. `/Users/shiyi/备份（mac_vs专用）/onset_detection/STAGE1_SINGLE_NONGT_AUDIT_20260321.md`
7. `/Users/shiyi/备份（mac_vs专用）/onset_detection/LEN9_STAGE3_AND_LENGTH_NOTE_20260321.md`
8. `/Users/shiyi/备份（mac_vs专用）/CODE_MAP.md`
