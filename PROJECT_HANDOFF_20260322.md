# Project Handoff (2026-03-22)

这份文档给“下一个聊天 / 下一个 agent”用。目标不是只交 onset，而是把整个项目目前真正重要的上下文一次性说清楚。

## 1. 项目在做什么

这是一个 Apple 设备内部 IMU 键盘侧信道项目。

核心目标：
- 先在单人、单设备、受控环境下，跑通 password 恢复闭环
- 再逐步弱化先验、扩大威胁模型

当前最重要的公开表达目标可以概括成：

> 录一整段连续 IMU 数据流，模型自动找出其中哪一段是 password，并恢复内容。

## 2. 当前大的模块划分

### 采集 / 预处理
- `collector.py`
- `onset_detection/onset_collector.py`
- `preprocessor.py`
- `typing_prompt_profiles.py`
- `keyboard_listener.py`

### Stage 3 / classifier
- `phase3_password_inception/`
- `adapt_password_len8_inception.py`
- `adapt_password_multilen_inception.py`

### onset / stage1-stage2
- `onset_detection/`
- 当前最新主线在：
  - `onset_detection/stage2_segmental/scripts/train_eval_peak_keyness.py`
  - `onset_detection/stage2_segmental/scripts/train_eval_segment_bagrank_ctx_v2.py`

## 3. 当前已经站住的事

### 3.1 Stage 3 已经站住
代表性多长度结果：
- `/Users/shiyi/备份（mac_vs专用）/results/password_len8_len9_len10_quick_adaptation.json`
- 组合 held-out：
  - `top1 = 81.37%`
  - `top3 = 97.65%`
  - `top5 = 99.61%`
  - `CER = 18.63%`

### 3.2 Stage 2 的“已知 password 段内找 key”已站住
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_peak_keyness_len8_len9_len10_v2/report.json`
- `exact_all_keys = 94.94%`
- `mean_peak_recall = 99.37%`
- `mean_peak_precision = 99.37%`

这意味着：
> 只要 password 段给对，模型已经基本学会了真正的 key 峰长什么样。

### 3.3 长度/计数是可学的
- `/Users/shiyi/备份（mac_vs专用）/results/length_model_len8_len9_len10_notime_mixed_cluster_v2_report.json`
- `8/9/10` no-time 长度识别仍然较强

## 4. 当前真正没打通的事

不是 Stage 3，不是段内 keyness。

当前最真实的瓶颈是：

> 在 full-stream 里，怎样先把真正的 password burst 提出来，并让 ranking / length / recovery cleanly 对齐。

也就是：
- `Stage 1 / Stage2 边界`
- 或者说 `candidate proposal / candidate ranking / length coupling`

## 5. 当前最可信的 onset 主线

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

当前 clean non-GT 结果：
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_segment_bagrank_ctx_v2_keynesspool_union_nogthint_targetv2_proxyv3/report.json`
- baseline:
  - `top1 = 39.22%`
  - `top5 = 56.86%`
  - `CER = 60.78%`
- overlap:
  - `top1 = 45.10%`
  - `top5 = 58.82%`
  - `CER = 54.90%`

这不是终局，但它是当前最诚实、最干净、也最值得继续押的主线。

## 6. 当前的 refined diagnosis

- `peak keyness` 学到本质了
- `keynesspool` proposer 是对的
- 当前剩下两类 hard case：
  1. 真 password burst 没完整进候选池
  2. 好候选在池里，但 clean ranking / length coupling 仍然会选错 top-1

## 7. 外部模型（Claude / GPT Pro）当前结论

吸收了有用思路，但**没有直接照搬整套代码**。

当前真正吸收并已适配到本地主线里的只有：
- bag/listwise ranking
- context
- recoverability target

没有照搬的：
- 不符合真实 IKI 的“快速打字” proposer 假设
- 和本地代码/数据分布脱节的整套 stage1 重写

## 8. 下一个聊天最该做什么

如果你是下一个 agent：

1. 不要再重讲 Stage 3 和已知 password 段内 keyness
2. 直接沿 `keynesspool + bag/context/recoverability` 主线继续迭代
3. 优先打：
   - candidate completeness
   - clean ranking / length coupling
4. 不要删任何旧研究分支和旧文档

## 9. 你应该先读哪些文件

1. `/Users/shiyi/备份（mac_vs专用）/README.md`
2. `/Users/shiyi/备份（mac_vs专用）/ONSET_CODEX_HANDOFF.md`
3. `/Users/shiyi/备份（mac_vs专用）/onset_detection/README.md`
4. `/Users/shiyi/备份（mac_vs专用）/onset_detection/ONSET_METHODS_AND_CONCLUSIONS.md`
5. `/Users/shiyi/备份（mac_vs专用）/onset_detection/STAGE1_SINGLE_NONGT_AUDIT_20260321.md`
6. `/Users/shiyi/备份（mac_vs专用）/onset_detection/LEN9_STAGE3_AND_LENGTH_NOTE_20260321.md`
7. `/Users/shiyi/备份（mac_vs专用）/CODE_MAP.md`
