# onset_detection

这份 README 只讲当前 onset / password-segment 主线，不替代项目根目录的全局 README。

## 当前最重要的结论

现在要把问题拆成三层来看：

1. `Stage 1`: 从整段 mixed/full-stream 里找出真正的 password burst
2. `Stage 2`: 在已知 password burst 内，找出真正的 key 峰，并给出正确的 key 序列/长度
3. `Stage 3`: 给定局部 key 窗口，恢复字符

当前最准确的状态是：

- `Stage 3`：已经成立，且多长度版本也已经站住
- `Stage 2` 的“段内找 key”部分：已经基本成立
- 当前真正的主瓶颈：`Stage 1 / Stage2 边界`，也就是 **full-stream 里哪个 candidate burst 才是真正的 password**

## 1. 已经成立的部分

### 1.1 Stage 3（字符分类）

当前最成熟的 classifier 线仍然是 `InceptionTime` password adaptation。

代表性多长度结果：
- `/Users/shiyi/备份（mac_vs专用）/results/password_len8_len9_len10_quick_adaptation.json`
- 组合 held-out：
  - `top1 = 81.37%`
  - `top3 = 97.65%`
  - `top5 = 99.61%`
  - `CER = 18.63%`

这说明：
- Stage 3 不是当前主问题
- `len8/9/10` 的 standalone password classifier 已经是强正信号
- 但多长度 classifier 不能自动替代所有当前 `len8 mixed` demo 路径，具体要看上游候选质量

### 1.2 Stage 2（已知 password 段内找 key）

这部分已经被单独跑实：
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_peak_keyness_len8_len9_len10_v2/report.json`

关键指标：
- `num_episodes = 356`
- `exact_all_keys = 94.94%`
- `mean_peak_recall = 99.37%`
- `mean_peak_precision = 99.37%`

也就是说：
> 如果真正的 password 段已经给对，模型已经基本能学会“哪个峰是真 key”。

这点非常重要，因为它说明：
- 现在不是“机器学不会 key”
- 更不是“切窗本身完全不行”
- 现在更像是：**full-stream 里先把哪一整段找出来** 这层还没做好

### 1.3 长度 / key-count 是可学的

显式长度头已经在 `8/9/10` 上站住：
- `/Users/shiyi/备份（mac_vs专用）/results/length_model_len8_len9_len10_notime_mixed_cluster_v2_report.json`
- 去掉显式时间特征后，`8/9/10` 长度识别仍然能到约 `90%`

这说明：
- 长度信息不只是“时长作弊”
- 信号形状 / 峰结构里确实包含长度信息
- 但长度头要想在主链里真正 work，前提仍然是候选段本体不能扣错

## 2. 已经证伪或不再主推的方向

### 2.1 旧 coarse detector / duration-aware ranking 不是主线

后续审计表明，旧的 Stage1/coarse 路线一旦去掉 duration bias，就会明显塌掉。
这说明它并没有真正学到 password 段本体，而是部分在利用：
- 这段够长
- 这段时长像 password
- 这段 probability 也还行

这条线不再是当前主线。

### 2.2 segment binary classification / pointwise passwordness 不够对题

已经试过：
- `segment_passwordness_v3_notime`
- `segment_slice_cnn_v1`
- `segment_slice_rankcnn_v1`

这些路线长期卡在：
- `top1 ≈ 33% - 39%`
- `CER ≈ 0.51 - 0.65`

它们的共性问题是：
- 把每个 crop 独立打分
- 没有充分利用同 session 候选之间的竞争关系
- 没有足够贴 downstream recoverability

### 2.3 纯 heuristic / energy valley / 旧 open/episode/ctc 路线都不是当前主线

这些分支仍保留作 baseline / 历史经验：
- `stage2_claude`
- `stage2_gpt54`
- `stage2_open`
- `stage2_episode`
- `stage2_ctc`
- `stage2_rebuild`

它们有探索价值，但当前最值得继续押的并不是这些旧分支。

## 3. 当前主线

当前最可信的 Stage1/Stage2 主线是：

```text
full stream
-> propose peaks on entire stream
-> peak keyness (which peaks look like real keys)
-> cluster high-keyness peaks into candidate bursts
-> bag/context/recoverability ranking among candidates
-> choose top candidate burst
-> within-burst key selection
-> fixed-window / overlap recovery
```

代码主入口：
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/train_eval_peak_keyness.py`
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/train_eval_segment_bagrank_ctx_v2.py`

## 4. 当前最佳 clean non-GT Stage1/Stage2 结果

最新主线结果：
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_segment_bagrank_ctx_v2_keynesspool_union_nogthint_targetv2_proxyv3/report.json`

在 `mixed_single_training + mixed_single_len9` 的 clean non-GT 评估上：

- `baseline_fixed_window`
  - `top1 = 39.22%`
  - `top3 = 52.94%`
  - `top5 = 56.86%`
  - `CER = 60.78%`
- `overlap_refine`
  - `top1 = 45.10%`
  - `top3 = 52.94%`
  - `top5 = 58.82%`
  - `CER = 54.90%`

这还远不是终局，但已经说明：
- `keynesspool + bag/context/recoverability` 比 earlier pointwise segment lines 更对
- `overlap` 仍然是有效的后段 refinement，但不是当前最大的增益来源

## 5. 当前最真实的瓶颈

当前主问题已经被收缩得很清楚：

> 不是“段内 key 学不会”，也不是“字符认不出来”，而是：
> **在 full-stream 里，哪一个 candidate burst 才是真正的 password，并且这个 burst 的长度/完整性要和 downstream recovery 对齐。**

更细一点说，现在还有两类 hard case：

1. **candidate pool completeness 问题**
- 真 password burst 没完整进候选池
- 或只进了一个残缺版

2. **clean ranking / length coupling 问题**
- 候选池里已经有较好候选
- 但最终 top-1 仍然被一个较短或较“假优”的候选赢走

## 6. 当前推荐的下一步

只继续做这条主线，不再回头做旧 segment 分类：

1. 固定 `peak keyness` 作为 full-stream proposer 的核心
2. 继续增强 `keynesspool` 候选池完整性
3. 继续增强 clean non-GT 下的 `ranking / length coupling`
4. 先把 `single 8/9/10` 打稳，再继续回到 `retry / multi-password`

## 7. 你如果是新接手者，应先读什么

按这个顺序：
1. `/Users/shiyi/备份（mac_vs专用）/README.md`
2. `/Users/shiyi/备份（mac_vs专用）/ONSET_CODEX_HANDOFF.md`
3. `/Users/shiyi/备份（mac_vs专用）/onset_detection/ONSET_METHODS_AND_CONCLUSIONS.md`
4. `/Users/shiyi/备份（mac_vs专用）/onset_detection/STAGE1_SINGLE_NONGT_AUDIT_20260321.md`
5. `/Users/shiyi/备份（mac_vs专用）/onset_detection/LEN9_STAGE3_AND_LENGTH_NOTE_20260321.md`

然后重点看：
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/train_eval_peak_keyness.py`
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/train_eval_segment_bagrank_ctx_v2.py`
