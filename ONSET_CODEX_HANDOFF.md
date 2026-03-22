# Onset Handoff For Next Codex (2026-03-22)

如果你是新开的 Codex 会话，请把自己当作上一位接手者的直接延续。

## 核心目标

我们当前真正想做的是：

> 录一整段连续 IMU 数据流，模型自动找出其中哪一段是 password，并恢复其中内容。

请注意：
- 不要把 GT 段、GT key timestamp 当作最终结果
- 这些只能用于训练监督、上界分析或模块诊断
- 最终 claim 必须是 non-GT / full-stream automatic

## 当前全局判断

### 已经站住的
1. `Stage 3` 已站住
   - 多长度 classifier (`len8/9/10`) 已经很强
2. `Stage 2` 的“已知 password 段内找 key”已基本站住
   - `exact_all_keys ≈ 94.94%`
   - 这说明模型已经学会“哪个峰是真 key”
3. 长度/计数是可学的
   - `8/9/10` 的 no-time 长度模型仍然能做到强准确率

### 当前真正没打通的
> `Stage 1 / Stage2 边界`：在 full-stream 里，哪一整段 candidate burst 才是真正的 password，并且这段的长度/完整性要和 downstream recoverability 对齐。

## 最重要的当前代码

### 关键主线
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/train_eval_peak_keyness.py`
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/train_eval_segment_bagrank_ctx_v2.py`

### 关键辅助
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/length_model.py`
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/train_length_model.py`
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/visualize_mixed_single_session.py`

### 不要再当主线的旧方向
- pointwise segment binary classification
- 旧 duration-aware coarse detector
- 纯 heuristic valley/open grouping
- 旧 CTC / episode / rebuild 分支（保留做 baseline，但不是当前最值钱的主线）

## 最新最该记住的结果

### 1. 已知 password 段内找 key
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_peak_keyness_len8_len9_len10_v2/report.json`
- `exact_all_keys = 94.94%`
- `mean_peak_recall = 99.37%`
- `mean_peak_precision = 99.37%`

### 2. 当前 clean non-GT 主线
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_segment_bagrank_ctx_v2_keynesspool_union_nogthint_targetv2_proxyv3/report.json`
- `baseline top1 = 39.22%`
- `baseline top5 = 56.86%`
- `baseline CER = 60.78%`
- `overlap top1 = 45.10%`
- `overlap top5 = 58.82%`
- `overlap CER = 54.90%`

### 3. 当前 best candidate oracle（同口径）
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_segment_bagrank_ctx_v2_keynesspool_union_oracle_check.json`
- `mean_best = 0.7381`
- `ge_0_75 = 0.6667`
- `ge_0_90 = 0.6667`

## 当前 refined diagnosis

不要再把问题说成“模型不够大”或“信号不够强”。
现在更准确的判断是：

1. `peak keyness` 已经学到本质
2. `keynesspool` proposer 是当前最对的 Stage1/Stage2 主方向
3. 当前剩下的问题分成两类：
   - candidate completeness：真 password burst 是否完整进池
   - clean ranking / length coupling：好候选已经在池里时，如何让它稳定排第 1

## 已吸收但要谨慎使用的外部思路

来自 Claude / GPT Pro 的思路中，当前确认有价值的只有这些：
- bag/listwise ranking
- context
- recoverability target

当前不应该直接照搬的：
- 假设用户是“快速打字”的 proposer 改动
- 用 `cluster_score_sum` 偏向大簇的打分
- 脱离我们真实代码骨架、另起一整套 stage1 代码

## 接下来最值得做的事情

只继续做下面这条线：

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

更具体说：
1. 固定 `keynesspool` proposer，不再回头折腾旧 coarse detector
2. 继续加强 candidate completeness，尤其是 hard case 的完整 burst 保留
3. 继续加强 clean non-GT 下的 ranking / length coupling
4. 先把 `single len8/9/10` 打稳，再回到 `retry`

## 不要丢掉的旧研究资产

不要删掉任何已有研究内容，包括但不限于：
- `stage2_claude`
- `stage2_gpt54`
- `stage2_episode`
- `stage2_ctc`
- `stage2_open`
- `stage2_rebuild`
- 之前的 mixed2 / classifier / audit / length 文档

这些不是当前主线，但它们记录了哪些方向已经证伪、哪些模块曾经有效。

## 快速阅读顺序

1. `/Users/shiyi/备份（mac_vs专用）/README.md`
2. 本文件
3. `/Users/shiyi/备份（mac_vs专用）/onset_detection/README.md`
4. `/Users/shiyi/备份（mac_vs专用）/onset_detection/ONSET_METHODS_AND_CONCLUSIONS.md`
5. `/Users/shiyi/备份（mac_vs专用）/onset_detection/STAGE1_SINGLE_NONGT_AUDIT_20260321.md`
6. `/Users/shiyi/备份（mac_vs专用）/onset_detection/LEN9_STAGE3_AND_LENGTH_NOTE_20260321.md`
