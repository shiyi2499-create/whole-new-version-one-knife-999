# Onset Methods And Conclusions

## 0. Status Correction (2026-03-22)

这份文档前半部分保留历史脉络，但请先用下面这组更新后的判断覆盖阅读：

### 0.1 当前已经成立的
- `Stage 3` 已成立，而且 `len8/9/10` 多长度 classifier 已站住。
- `Stage 2` 的“已知 password 段内找真正 key 峰”已成立：
  - `/Users/shiyi/备份（mac_vs专用）/results/stage2_peak_keyness_len8_len9_len10_v2/report.json`
  - `exact_all_keys = 94.94%`
  - `mean_peak_recall = 99.37%`
  - `mean_peak_precision = 99.37%`
- 长度/计数是可学的；no-time 长度头在 `8/9/10` 上仍然有强信号。

### 0.2 当前真正没打通的
现在最真实的瓶颈已经不是：
- classifier
- 段内 keyness
- 固定窗本身

而是：
> full-stream 里哪个 candidate burst 才是真正的 password，
> 以及 clean non-GT 条件下如何把 candidate ranking / length coupling 和 downstream recovery 对齐。

### 0.3 当前最可信主线
当前最值得继续押的主线是：

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

关键脚本：
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/train_eval_peak_keyness.py`
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/train_eval_segment_bagrank_ctx_v2.py`

### 0.4 当前最新 clean non-GT 结果
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_segment_bagrank_ctx_v2_keynesspool_union_nogthint_targetv2_proxyv3/report.json`
- baseline:
  - `top1 = 39.22%`
  - `top5 = 56.86%`
  - `CER = 60.78%`
- overlap:
  - `top1 = 45.10%`
  - `top5 = 58.82%`
  - `CER = 54.90%`

### 0.5 当前最该避免的误读
- 不要再把 `Stage 1` 简化成“已经完全成立”。旧 coarse detector 存在 duration-bias 问题。
- 不要再把问题定义成“单段 passwordness 二分类”。这条线已长期卡在 `top1 ~ 33-39%`。
- 不要再把 `GT` 或 `GT length hint` 结果当作最终故事。

这份文档专门记录 onset 方向到目前为止试过的方法、代表性结果、已经确认的结论，以及当前最可信的下一步判断。

目标不是写宣传稿，而是防止后续接手时遗漏关键事实。

## 1. 问题分解

当前整体链路可以分成三层：

1. `Stage 1`: 从连续 IMU 流里圈出 password 相关的大区间 / episode
2. `Stage 2`: 在 password episode 内完成更细的时序对齐
3. `Stage 3`: 对单键窗口或等价局部表示做字符识别

当前已经很明确：
- `Stage 1` 不是主要瓶颈
- `Stage 3` 不是主要瓶颈
- 当前真正的主瓶颈是 `Stage 2`

## 2. 主要数据集与监督形式

### 2.1 原始文件形态
每个 session 通常包含：
- `*_sensor.csv`: 连续 IMU 流（accel xyz + gyro xyz）
- `*_events.csv`: 按键事件时间戳和 key identity
- `*_activity_log.csv`: 协议活动段边界
- `*_protocol.json`: prompts / 协议 skeleton / 目标 password

### 2.2 目前主要用到的数据
- `data/raw/password/len_8`
  - 干净 password attempts
  - 主要用于 password classifier adaptation，以及部分 onset / synthetic 构造
- `data/raw/onset_negative`
  - 非 password / gap / 背景片段
- `data/raw/onset_mixed2`
  - 受控 mixed 连续流测试目标
  - 主要作为 held-out 连续流评测，不应混入训练
- `data/raw/mixed_training`
  - 当前真实 mixed-style onset 训练集
  - 目前规模仍偏小，但比纯 clean password 更贴最终任务

### 2.3 我们实际上拥有的监督
不是只有弱标签。当前 mixed-style 数据里，我们通常能拿到：
- password episode 区间
- per-key timestamp
- per-key 字符标签
- 协议活动边界

这意味着：
- 既可以做 episode-level 训练
- 也可以做 per-key alignment / frame-level supervision
- 未来也可以做 sequence alignment / CTC / transducer 风格方案

## 3. 已经确认成立的事实

### 3.1 Stage 1 已成立
在历史 `password_segment` 路线上，mixed2 的代表性 coarse localization 结果：
- `Episode IoU = 0.967`

含义：
- 从连续流里先圈出 password 大区间，这件事已经基本成立
- 当前不该再把主要精力放在“有没有 password block”这层问题上

### 3.2 Stage 3 classifier 已成立，但还不是满血版
最成熟的 classifier 线仍然是：
- `single_key / merged baseline + password adaptation`
- 代表模型：`InceptionTime`

本地已重训并核实的 adapted classifier 结果：
- `char_top1 = 73.4%`
- `char_top3 = 97.5%`
- `char_top5 = 99.4%`
- `CER = 26.6%`

重要说明：
- 这证明了“字符信息是存在的，classifier 线是工作的”
- 但这不等于 classifier 已经封顶
- 当前最成熟的实验主要集中在：
  - `36` 类
  - `len=8`
- 还没有系统做完的内容包括：
  - 更强 classifier 架构对比
  - `password-only` 与 `baseline+adaptation` 的更全对比
  - 符号集扩展
  - 不同长度（不只 `len=8`）

## 4. 最关键的上界/下界关系

这一组数字非常关键，它解释了为什么当前问题不是单纯的“数据少”或“峰值没调好”。

### 4.1 Oracle / 理想切窗上界
来自 classifier adaptation 路线：
- `char_top1 = 73.4%`
- `char_top5 = 99.4%`
- `CER = 26.6%`

这对应的是：
- 已知较优的局部窗口
- classifier 只负责认字

### 4.2 GT onset -> fixed window -> classify
在当前 8 组 `mixed_training` episode 评估里，GT baseline 约为：
- `char_top1 = 40.2%`
- `char_top3 = 68.8%`
- `char_top5 = 82.6%`
- `CER = 59.8%`

含义：
- 即使我们拿到了 GT onset 时间戳
- 只要继续使用 `onset -> fixed pre/post window -> classifier`
- 性能也已经从 oracle 大幅下降

这说明：
> 问题不只是 onset detector 准不准，
> 而是 `onset -> fixed window -> classify` 这条中间层 formulation 本身就在丢信息。

### 4.3 Pred onset -> fixed window -> classify
当前最好的自动 Stage 2 + classifier 闭环，仍然远低于 GT baseline。

这说明：
- 当前瓶颈同时包含：
  1. 自动对齐还不够准
  2. 固定切窗这件事本身也有结构性损失

## 5. 已经试过的方法与代表性结果

下面按路线记录。

### 5.1 历史 heuristic / energy-valley / password_segment 分支
目录 / 入口：
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/password_segment_detector.py`
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/password_segment_preprocessor.py`

核心思路：
- Stage 1 先做 `password_typing vs non_password` 粗定位
- Stage 2 再做 onset proposal、grouping、energy valley、gap heuristics
- Stage 3 classifier 做字符恢复

结论：
- 这条线证明了 coarse-to-fine pipeline 有价值
- 也证明了 Stage 1 可以做好
- 但 Stage 2 一直停留在 heuristic / proposal + grouping 的瓶颈中
- 后续不再适合作为唯一主线，但仍然是重要历史 baseline

### 5.2 Claude branch: `stage2_claude`
目录：
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_claude`

试过的方法：
- `dp_classifier`
- `dense_ctc`

代表性结论：
- 已经真正接进过 Path B
- mixed2 上结果很差
- 代表性数值：
  - `char_top1 ≈ 2.5%`
  - `CER ≈ 97.5%`

判断：
- 作为 baseline / exploratory branch 保留
- 不是当前推荐主线

### 5.3 GPT branch: `stage2_gpt54`
目录：
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_gpt54`

试过的方法：
- dense `key / boundary / inside` modeling
- structured decode
- top-K hypothesis
- global classifier rerank

代表性结论：
- 结构上曾能稳定输出 `5 groups / 40 onsets`
- 比 Claude branch 更像样
- 但 mixed2 E2E 仍不够好
- 代表性结果：
  - `char_top1 = 7.5%`
  - `top3 = 15.0%`
  - `top5 = 27.5%`
  - `CER = 90.0%`

判断：
- 证明 dense structured 方向比纯 heuristic 更有希望
- 但 top-K / rerank 也没有根本救回来
- 当前保留作历史重要分支

### 5.4 `stage2_rebuild`
目录：
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_rebuild`

核心思路：
- 重建为更规范的 `2A / 2B`
  - `2A`: group segmentation
  - `2B`: onset detection within group
- synthetic mixed + real mixed fine-tune

代表性 Stage 2-only 指标：
- `e2e_mixed2_v2_best`
  - `avg_group_iou = 0.7289`
  - `avg_onset_f1 = 0.75`
- `e2e_mixed2_real_ft_v1`
  - `avg_group_iou = 0.7775`
  - `avg_onset_f1 = 0.70`

重要说明：
- 这些结果主要证明的是：
  - 在 GT coarse block 条件下，Stage 2A / 2B 自身开始稳定工作
- 它不是完整最终 char-level E2E 成功

判断：
- 这条线把 Stage 2 拆开后，方法上明显比之前更对
- 但它仍然主要服务于固定协议（尤其 `5x8`）
- 对“随手录一段 password”这种更开放目标，不够自然

### 5.5 `stage2_open`
目录：
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_open`

核心思路：
- 摆脱固定 `5x8`
- 用 `gap / keystroke / separator` 一类 open segmentation 思路
- 面向 variable-length / open grouping

在 7 组 `mixed_training` 上的代表性 aggregate：
- `avg_pred_groups = 27.86`
- `avg_group_iou = 0.2822`
- `avg_onset_f1 = 0.4447`
- `avg_onset_recall = 0.2921`

判断：
- 方向上更接近开放任务
- 但过分段非常严重
- 当前这条线没有成为主线

### 5.6 `stage2_episode`
目录：
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_episode`

核心思路：
- 不再死磕固定 `5x8`
- 先找 password episode
- 再在 episode 内找 onset / keypoint
- 强调“长停顿才算新 episode”

这条线试过的子方法很多：
- `typing vs silence` head
- `energy-based onset`
- `stage2b refine`
- `dual-head typing + onset`
- 更稀疏 onset supervision
- classifier-aware pruning
- sequence-level subset selection
- peak DP / local refine

#### 代表性 Stage 2 指标
较早但更差的一版：
- `stage2_episode_dual_eval_v2_decoderfix`
  - `avg_pred_episodes = 12.25`
  - `avg_episode_detection_rate = 1.00`
  - `avg_onset_f1 = 0.1534`
  - `avg_onset_recall = 0.2028`

更健康的一版：
- `stage2_episode_dual_eval_v3_sparse`
  - `avg_pred_episodes = 5.75`
  - `avg_episode_detection_rate = 0.75`
  - `avg_onset_f1 = 0.5767`
  - `avg_onset_recall = 0.7944`

#### 代表性 char-level E2E
较平衡的一版：
- `stage2_episode_dual_char_eval_v11_peakdp_localrefine`
  - `char_top1 = 7.2%`
  - `top3 = 16.5%`
  - `top5 = 21.2%`
  - `CER = 94.4%`

另一版更激进裁点：
- `stage2_episode_dual_char_eval_v7_aggrprune`
  - `char_top1 = 6.2%`
  - `top3 = 17.4%`
  - `top5 = 22.7%`
  - `CER = 98.8%`

判断：
- `stage2_episode` 是旧 onset/window 路线里最认真、也最接近目标的一条
- 它明确证明了：
  - episode 检测可以做得越来越好
  - 但 `per-key onset -> fixed window -> classifier` 这条链仍然很难真正救活
- 这条线的最大贡献是：
  - 把问题逼到了一个更明确的结论上
  - 即“固定切窗 formulation 本身可能有结构性损失”

### 5.7 `stage2_ctc`
目录：
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_ctc`

这是当前最新的新 formulation，核心思路是：
- 不再显式先找 per-key onset 再切固定窗
- 改成：
  - `episode IMU -> frame-level char posterior -> CTC decode`

也就是：
> 不再先把整段信号切成一个一个字的小窗，
> 而是让模型直接看整段 episode，自己把字符序列读出来。

#### v1: 原始 smoke
- `char_top1 = 0.0%`
- `top3 = 4.36%`
- `top5 = 9.66%`
- `CER = 97.5%`

失败形态：
- 大量空串或单字符塌缩

#### v2: 更软一点的 frame target
- `char_top1 = 0.0%`
- `top3 = 5.92%`
- `top5 = 14.0%`
- `CER = 96.9%`

失败形态：
- 仍然多为空串或单字符 / 极少数字符塌缩

#### v3: 更局部的监督（更接近“教模型怎么对齐局部字符”）
- `char_top1 = 1.6%`
- `top3 = 14.6%`
- `top5 = 24.6%`
- `CER = 164.8%`

含义：
- 模型开始学到更多局部字符信息
- 但解码出来的字符串经常过长、重复、失控

#### v5: curriculum + 正确 checkpoint 选择（当前最有价值）
- `char_top1 = 23.7%`
- `char_top3 = 47.0%`
- `char_top5 = 67.0%`
- `CER = 7894.1%`

这个结果表面上很怪，但非常重要。

它说明：
- 这条 CTC 线已经不是“完全没学起来”
- 模型在局部字符层面已经学到相当多的正确字符信息
- 但最终 sequence decode 仍然会输出极长、重复、失控的字符串

换句话说：
> `stage2_ctc` 当前的主要问题已经不是“学不到字符”，
> 而是“怎么把这些局部字符 posterior 校准成一条合理长度的字符串”。

判断：
- 这是当前最值得保留的新主线之一
- 它还远没有成功
- 但它比继续修旧 onset detector 更像结构性正确方向

### 5.7 `stage2_segmental` v1: partition segmental（失败）
目录：
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/model.py`
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/train_gt_segmental.py`

核心思路：
- 把每个 key 建模成一个单调 segment
- 相邻 key 共享边界
- 每个 key 的 segment 不允许重叠

真实工作区结果（GT episode-only + 强 classifier）：
- fixed-window baseline：
  - `top1 = 58.75%`
  - `top3 = 81.25%`
  - `top5 = 93.75%`
  - `CER = 41.25%`
- `stage2_segmental` v1：
  - `top1 = 17.50%`
  - `top3 = 23.75%`
  - `top5 = 31.25%`
  - `CER = 82.50%`

结论：
- 这不是“小输”，而是明显失败
- 后续诊断表明它的结构性问题是：
  - **partition segment** 与现有 classifier 的训练分布不兼容
  - classifier 习惯的是以 key 为中心的 **overlapping windows**
  - partition 版本把快打时的 key 压成极短片段，再强行拉伸，直接破坏信号结构

### 5.8 `stage2_segmental` v2: overlapping learned windows（当前最强 GT / semi-oracle 主线）
目录：
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/model_v2.py`
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/train_gt_overlap.py`
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/eval_overlap_with_episode_candidates.py`

核心思路：
- 仍然是 per-key learned cutting / learned window alignment
- 但每个 key 的窗口是 **独立、可学习、可重叠** 的
- 初始化就等价于旧 baseline 的固定窗（约 `100ms pre + 200ms post`）
- 然后只学习：
  - `offset`
  - `width_scale`

#### GT timestamp 条件下的稳定结果
代表性 held-out 结果：
- `p04+p05`：
  - baseline：
    - `top1 = 60.0%`
    - `top3 = 85.0%`
    - `top5 = 95.0%`
    - `CER = 40.0%`
  - overlap：
    - `top1 = 66.25%`
    - `top3 = 93.75%`
    - `top5 = 97.5%`
    - `CER = 33.75%`
- `p01` held-out：
  - baseline：
    - `top1 = 32.5%`
    - `top3 = 70.0%`
    - `top5 = 92.5%`
    - `CER = 67.5%`
  - overlap：
    - `top1 = 50.0%`
    - `top3 = 87.5%`
    - `top5 = 90.0%`
    - `CER = 50.0%`
- `p02` held-out：
  - baseline：
    - `top1 = 46.34%`
    - `top3 = 80.49%`
    - `top5 = 87.80%`
    - `CER = 53.66%`
  - overlap：
    - `top1 = 68.29%`
    - `top3 = 92.68%`
    - `top5 = 100.0%`
    - `CER = 31.71%`

3 组加权汇总（20 episodes / 161 chars）：
- baseline：
  - `top1 = 50.31%`
  - `top3 = 81.37%`
  - `top5 = 94.41%`
  - `CER = 49.69%`
- overlap：
  - `top1 = 62.73%`
  - `top3 = 92.55%`
  - `top5 = 96.27%`
  - `CER = 37.27%`

结论：
- **用户最初的 learned cutting / learned window alignment 思路是对的**
- 错的不是这条大方向，而是 v1 把它实现成了 non-overlapping partition
- v2 是当前最强的 **GT key timestamp 条件下** 方法

#### 非完美锚点条件：jitter 与 candidate bridge
`train_gt_overlap.py` 现在已支持：
- `--train_anchor_jitter_ms`
- `--eval_anchor_jitter_ms`

代表性结果：
- 对 `30ms` 锚点扰动，jitter-trained overlap 仍然明显优于 fixed-window：
  - baseline：
    - `top1 = 52.5%`
    - `top5 = 93.75%`
    - `CER = 47.5%`
  - overlap：
    - `top1 = 67.5%`
    - `top5 = 96.25%`
    - `CER = 32.5%`

进一步，用 `stage2_episode_dual_eval_v3_sparse` 的自动候选 onset 池做 bridge：
- 先在每个 GT episode 内，从自动候选池里选出 `K` 个 anchors
- 再交给 overlap learned-window 修窗

当前 3 组加权 bridge 结果（161 chars）：
- candidate fixed-window：
  - `top1 = 54.66%`
  - `top3 = 84.47%`
  - `top5 = 88.20%`
  - `CER = 45.34%`
- candidate + overlap：
  - `top1 = 60.87%`
  - `top3 = 84.47%`
  - `top5 = 91.30%`
  - `CER = 37.89%`

说明：
- overlap learned-window 已经不只是“会修 GT 点”
- 在 **semi-oracle / candidate-anchor** 条件下也开始稳定带来正收益
- 当前主问题开始从“切窗怎么学”转向“自动候选锚点怎么生成 / 选择更好”

进一步，使用 `stage2_episode` 候选池直接构造 pseudo-anchor 进行微调后（`runs/stage2_overlap_candidate_finetune_v1`）：
- candidate fixed-window：
  - `top1 = 52.5%`
  - `top3 = 85.0%`
  - `top5 = 93.75%`
  - `CER = 47.5%`
- finetuned overlap：
  - `top1 = 67.5%`
  - `top3 = 95.0%`
  - `top5 = 96.25%`
  - `CER = 32.5%`

这说明：
- overlap 主线在 **真实自动候选 pseudo-anchor** 上仍然可以继续涨
- 它不只是 “GT+jitter 才 work”

#### 新增现实场景验证：single-password mixed stream
为了更贴近真实使用，我们新增了 `mixed_single_training`：
- 每个 session 只有 1 条 password episode
- 其余部分保留 mixed-style 上下文（idle / trackpad / free typing / shake）

这批新数据的形态很干净：
- 当前 3 个 session 都只有 **1 个 password 段**
- 每个 password 段都是：
  - `8 个字符 + 1 次 Enter`
- 中位 IKI 约 `1405.6 ms`

在这 3 个 session 上做 leave-one-session-out（`runs/stage2_overlap_single_folds_cpu`）：
- baseline（固定窗）：
  - `top1 = 45.83%`
  - `top3 = 75.00%`
  - `top5 = 91.67%`
  - `CER = 54.17%`
- overlap learned-window：
  - `top1 = 50.00%`
  - `top3 = 95.83%`
  - `top5 = 100.0%`
  - `CER = 50.00%`

说明：
- 在 **单条 password** 这个更真实的口径下，overlap 线仍然是正向的
- 但当前样本量只有 `3 sessions / 24 chars`
- 这足以说明“方向继续成立”，但还不足以下强统计结论

这条新数据非常有价值，因为它把问题从：
> “怎么从一段里拆出 5 条 password”
拉回到更真实的：
> “怎么从一段 mixed stream 里找到 1 条 password，并把它恢复出来”

#### full-auto-ish 进一步结论
我们又尝试了更进一步的 full-auto-ish 口径：
- 不再使用 GT episode 去裁候选池
- 只使用 `stage2_episode` 自身吐出的候选 onsets
- 自动做候选簇划分、估计每簇 key 数，再交给 overlap refine

结果表明：
- 这一步目前还 **没有打通**
- 当前失败主因不是 overlap learned-window
- 而是：
  - **自动候选簇怎么拆**
  - **每个簇里到底该保留多少个 key**

也就是说，当前主线已经把瓶颈继续前推到了：
> `candidate clustering / key-count estimation`
而不再是：
> `window alignment itself`

## 6. 当前最可信的大结论

### 6.1 不是简单阈值问题
我们已经系统试过：
- energy envelope
- peak picking
- stage2b refine
- dual-head onset heatmap
- sparse loss
- classifier-aware pruning
- sequence DP / subset selection
- CTC frame posterior + curriculum

所以当前结论不是：
- “某个阈值还没扫到”

### 6.2 也不是单纯数据量问题
`mixed_training` 目前确实偏少，这会：
- 限制泛化
- 放大方差
- 让节奏先验容易过拟合

但当前最核心的证据是：
- `Oracle window` 很高
- `GT onset -> fixed window` 已经掉很多
- `Pred onset -> fixed window` 掉得更多

这说明：
> 当前问题不只是“真实 mixed 数据太少”，
> 更像是“中间层问题定义不对”。

### 6.3 当前最关键的结构性判断
旧 formulation：
- `detect onset -> cut fixed window -> classify`

确实存在结构性信息损失，但最新结果说明需要更细分地看：

1. **fixed window 不是最优**
   - overlap learned-window 已经证明，在 GT / semi-oracle timestamp 条件下，
     学习窗口偏移和宽度能明显优于固定窗
2. **onset point / frame spike 仍然不是理想对象**
   - `stage2_ctc` 说明 whole-episode 表述有价值
   - 但当前最好正结果来自：
     - **timestamp candidate -> learned overlapping window -> classifier**

所以当前最可信的判断不再是“只押 sequence formulation”，而是：
> `overlapping learned windows` 已经成为当前主线之一，
> 并且它比 partition segmental 或纯 point detector 更贴当前数据和 classifier 资产。

## 7. 当前最该记住的研究判断

### 7.1 对旧 onset/window 路线的判断
- 旧的固定窗 + heuristic onset 路线已经试到较深，上限有限
- 但 **learned cutting / learned window alignment 本身是成立的**
- 当前最该保留并继续推进的，不是 fixed-window，而是：
  - **overlapping learned windows**

### 7.2 对当前主线的判断
现阶段最值得继续的两条是：
1. `stage2_segmental` v2 的 **overlap learned-window**
2. `stage2_ctc` 作为 whole-episode / alignment 备选主线

其中：
- overlap 线已经打出了更强、更稳定的真实正结果
- CTC 线仍有研究价值，但目前结果不如 overlap 线健康

### 7.3 现实标准
用户当前明确的主观门槛是：
- `top5` 接近 `90%`，才进入“可以接受 / 有说服力”的区间

按这个标准：
- 当前所有自动 Stage 2 / Stage 2+3 路线都还远不够

## 8. 当前建议的推进顺序

1. 把 `stage2_segmental` v2 overlap learned-window 当成当前主线继续推进
2. 下一步重点从 `GT timestamp -> better window` 推到：
   - `candidate anchor -> better window`
3. 当前最自然的问题已经变成：
   - 自动候选锚点怎么生成 / 选择更好
   - overlap 如何对不完美 anchor 做更强修正
4. `stage2_ctc` 保留为备选研究主线，但暂不作为第一主攻方向
5. 不要再把主要精力放回旧 heuristic onset detector 修补上

## 9. 最后一句话

到目前为止，onset 方向最重要的结论不是“某个分支赢了”，而是：

> `Stage 1` 已成立，`Stage 3` 已成立；
> 当前最难、也是决定成败的，是中间层时序对齐。
> 现在最强的正结果来自：
> **overlapping learned windows**，
> 也就是把用户最初的 learned cutting 思路做成正确的 overlapping realization。
> 下一步重点不再是证明 fixed-window 不够，而是把这条 learned-window 主线真正接到更自动的 anchor 候选上。

## 10. 2026-03-21 新进展：single full-stream non-GT 开始真正站起来

这一步是当前 onset 主线里非常关键的新阶段。

### 10.1 新的 Stage 1 结论：旧 coarse detector 不够，新 mixed-aware coarse detector 基本可用

我们重新看了 full-stream single 场景后发现：
- 旧的 `password_segment_detector` 训练分布偏老，放到新的 `mixed_single_training` 上不行
- 单纯扫阈值也救不回来

于是我们专门重建了一版 **mixed-aware Stage 1 coarse detector**：
- 新数据构建脚本：
  - `/Users/shiyi/备份（mac_vs专用）/onset_detection/password_segment_mixed_preprocessor.py`
- 新训练产物：
  - `/Users/shiyi/备份（mac_vs专用）/results/password_segment_mixed_detector.pt`
  - `/Users/shiyi/备份（mac_vs专用）/results/password_segment_mixed_scaler.npz`
  - `/Users/shiyi/备份（mac_vs专用）/results/password_segment_mixed_training_report.json`

它的关键变化是：
- 正样本不再只有 standalone password
- 还加入了 `mixed_single_training` / `mixed_retry_training` 的真实 password blocks
- 负样本不再只是单键和负例流
- 还加入了真实 mixed 场景里的 `typing_1 / idle / trackpad_move / shake / interference` 上下文

在 `mixed_single_training` full stream 上：
- 用简单 threshold + duration-aware coarse region ranking
- 已经可以做到：
  - **3/3 session 都把真正的 single password region 找出来**
  - mean IoU 大约在 `0.87~0.88`

这说明：
> 对 single full-stream 而言，
> **coarse password region detection 已经开始真正 work**。

### 10.2 一个关键转折：不是 Stage 1 坏，而是 Stage 2 anchor source 选错了

我们曾经尝试：
- `full stream -> mixed-aware coarse detector -> stage2_episode -> overlap refine`

结果很差：
- 原因不是 coarse region 不准
- 而是 `stage2_episode` 在 coarse region 里只吐出 `2 / 4 / 5` 个 onsets
- 然后被强行插值成 8 个 local frames，字符恢复就直接坏掉了

所以新的判断变成：
> 在 single full-stream 场景里，
> 当前最弱的不是 coarse detector，
> 而是 **coarse region 内的 anchor proposal**。

### 10.3 新的 non-GT 正结果：coarse region + energy anchors 已经明显可用

我们新增了：
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/eval_overlap_single_coarse_energy.py`

它做的是：
- full stream
- mixed-aware coarse detector 找 single password region
- 不再用 `stage2_episode` 提 anchors
- 直接在 coarse region 里做 energy peak proposal
- 强制选出 8 个单调 anchors
- 再分别送：
  - fixed-window baseline
  - overlap refine

结果：
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_overlap_single_fullstream_energy_v1/report.json`

其中 fixed-window baseline 已经达到：
- `top1 = 66.67%`
- `top3 = 79.17%`
- `top5 = 79.17%`
- `CER = 20.83%`

这个结果很重要，因为它意味着：
> **完全脱离 GT 之后，single full-stream 已经第一次跑出了像样的自动恢复结果。**

而且 debug 显示：
- 3 个 session 里有 2 个，energy anchors 几乎和 GT key frames 对齐
- 第 3 个坏例子本质上是“8 个峰选错了一个”

### 10.4 再往前推一步：energy + classifier-guided anchor selection 更强

我们继续把强 classifier 接回到选峰阶段：
- 新脚本：
  - `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/eval_overlap_single_coarse_energy_cls.py`

它的逻辑是：
- coarse region 里先提一批 raw energy peaks
- 每个 peak 都切一个 fixed window
- 让现有强 classifier 判断“这个峰附近像不像一个真按键窗”
- 再用：
  - energy score
  - classifier confidence
  - monotonic gap prior
  联合选出 8 个 anchors

结果：
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_overlap_single_fullstream_energy_cls_v1/report.json`

fixed-window baseline 达到：
- `top1 = 75.00%`
- `top3 = 83.33%`
- `top5 = 87.50%`
- `CER = 25.00%`

对应逐条结果：
- 两条 session 基本已经接近正确字符串
- 第三条也明显比前面 energy-only 版本更好

这一步说明：
> 在 single full-stream 场景里，
> **最有效的自动 anchor source 不是旧的 `stage2_episode`，而是 `coarse region + energy peaks + classifier-guided subset selection`。**

### 10.5 一个很重要但必须诚实写清楚的发现：当前 overlap refine 在 non-GT single 上还不是必须品

在这条 non-GT single 链里，我们还试了多个 overlap checkpoint：
- `candidate_finetune_v1`
- `gt_run1`
- `gt_run1_jitter30`

结果都表明：
- 当前最好的 non-GT single 结果，实际上来自：
  - **coarse region + better candidate anchors + fixed-window classifier**
- overlap refine 目前并没有在这一步稳定再往上推
- 有时甚至会略伤 top1 / CER

这意味着：
> `overlap learned-window` 仍然是一个很强的 GT / pseudo-anchor alignment 模块，
> 但在当前 single full-stream non-GT setting 下，
> **更先决定成败的是 candidate anchors 本身够不够好。**

### 10.6 对当前主线的更新判断

截至 2026-03-21，single 场景下最新最可信的判断是：

1. **full-stream coarse password region detection 已经开始成立**
2. **真正新的关键瓶颈是 coarse region 内的 anchor proposal**
3. `stage2_episode` 目前不适合作为这个场景里的主 anchor source
4. `coarse region + energy peaks + classifier-guided anchor selection` 是目前 single non-GT 下最强的自动链路
5. overlap learned-window 仍然有研究价值，但在这一步不是最关键增益来源

### 10.7 现在离“安全测试小工具”还有多远

现在还不能诚实地写成：
- “随机采一段连续流，系统已经能稳定自动找出并恢复 password”

但已经可以更有底气地说：
- 对 **single-password mixed stream**，
- 我们已经有了一条开始可工作的 non-GT 自动原型：
  - `full stream -> mixed-aware coarse detector -> energy/classifier-guided anchors -> char recovery`

也就是说，和此前相比，
当前离“安全测试小工具”最近的一步，已经不再是 GT-conditioned overlap，
而是：
> **single full-stream non-GT automatic recovery prototype**。

### 10.8 下一步最合理的推进顺序

1. 继续把 **single non-GT** 做稳
   - 把当前 energy+classifier anchor proposer 再做稳一点
   - 争取把 `top5` 从 `87.5%` 稳定推到接近或超过用户心里的 `90%`
2. 然后再回到 **retry / 多条 password**
   - 当前 retry 的核心问题仍然是：
     - 第二条 password episode 没抓稳
3. GT 以后只保留做上限/诊断，不再当主结果

一句话更新：
> 当前主线已经从“GT timestamp 下 overlap refinement 很强”，
> 推进到了“single full-stream non-GT automatic prototype 开始成立”；
> 现在最关键的自动化问题，不再是 coarse region，而是 region 内 anchor proposal。

### 10.9 关于“适配各种数据流”的最新判断：先去掉采样率硬编码，暂时不要去掉键数先验

为了让当前 non-GT single 原型更贴近“适配各种数据流”，我们额外验证了两件事：

#### A. 采样率自动推断：已经可行

我们把 `eval_overlap_single_coarse_energy_cls.py` 改成了：
- 不再默认写死 `200Hz`
- 而是从 coarse region 内的真实 timestamps 自动估计 sample rate

在当前 `mixed_single_training` 上回归结果：
- 和写死 `200Hz` 的最好结果完全一致
- 说明这条链对采样率已经没有那么脆弱的协议依赖

结论：
> **采样率硬编码可以去掉，而且不伤当前结果。**

#### B. 键数自动推断：现在还不行

我们同时尝试了：
- `expected_keys = 0`
- 让系统根据 energy+classifier anchor scores 自己推断该保留几个 key

结果直接崩掉：
- 它把 8 键常常误判成 `4 / 4 / 5`
- 然后后续 local frame 插值全部被拉坏
- 指标大幅下降到不可接受范围

debug 非常明确：
- `trial000 -> used_expected_keys = 4`
- `trial001 -> used_expected_keys = 4`
- `trial002 -> used_expected_keys = 5`

所以当前最可信的判断是：
> 在 single non-GT 主线里，
> **采样率可以自适应，但键数还不能自适应。**

也就是说，现阶段更现实的策略是：
1. 继续保留一个 soft key-count prior（例如当前场景下仍然知道大约是 8）
2. 把主要精力放在让 anchor proposal 更稳
3. 等 anchor proposal 足够强之后，再考虑真正去掉 key-count 先验

一句话更新：
> 想让系统适配“各种数据流”，
> 当前已经成功去掉的是 `200Hz` 假设；
> 当前还不能去掉的是 `8-key` 这个计数先验。

### 10.10 新 best：single non-GT full-stream 已把 top5 推过 90%

在继续优化 `coarse region + energy/classifier-guided anchors` 后，我们发现一个非常关键的细节：
- 选 8 个峰时，不能简单用 `coarse region 总时长 / (K-1)` 去推理想 gap
- 因为 coarse region 本身带有 password 前后的缓冲，这会把目标 gap 拉得过大

于是我们把 fixed-8 情况下的 gap prior 改成更贴真实输入节奏的固定先验：
- `gap_prior_s = 1.3`

对应结果：
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_overlap_single_fullstream_energy_cls_v4_gap13/report.json`

当前 best 的 **non-GT single full-stream** fixed-window baseline 达到：
- `top1 = 87.50%`
- `top3 = 95.83%`
- `top5 = 95.83%`
- `CER = 12.50%`

逐条看：
- `zmk676kf -> xmk676kf`（只差 1 个字符）
- `f9oxeocr -> f9oxeoxr`（只差 1 个字符）
- `kodtpoxk -> kodtpoxl`（只差 1 个字符）

也就是说：
> 当前 single non-GT automatic prototype 已经把 `top5` 推过了用户心里的 `90%` 门槛，
> 并且 `CER` 已降到 `12.5%`。

但这里必须立刻加上一条 **审计 caveat**：
- 这条 best non-GT single 链的 **Stage 1 coarse detector** 使用的是
  `password_segment_mixed_detector.pt`
- 而这版 mixed-aware detector 的训练集 `password_segment_mixed_dataset.npz`
  里**包含了当前评估的 `mixed_single_training` / `mixed_retry_training` session**
- 用同一脚本、同样的 anchor 选择逻辑，把 coarse detector 换回旧的
  `password_segment_detector.pt`（不包含 mixed_single / retry）后，
  在 `mixed_single_training` 上会直接掉到：
  - `num_episodes = 0`
  - `top1/top5 = 0`
  - `CER = 100%`

所以这组 `top5 = 95.83%`、`CER = 12.50%` 的 single full-stream 结果，
**不能被当作完全干净的“任意未见数据流”泛化结果**。
更准确的表述应该是：

> 这证明了当前链路在 `mixed_single` 这种真实协议下已经具备很强的恢复潜力，
> 但其中 Stage 1 仍然带有明显的 mixed-domain adaptation / same-protocol optimism。

从审稿人视角，这组数字现在最适合被当作：
- **阶段性 prototype milestone**
- **方向成立的强正信号**
- 而不是已经完全 clean 的最终主结果

这个结果非常重要，因为它意味着：
1. **single full-stream 的非 GT 自动恢复已经开始接近“工具可用”区间**
2. 当前关键不是 overlap refine，而是：
   - coarse region 的质量
   - region 内 anchor proposal
   - 合理的 rhythm prior
3. “不写死 200Hz”已经成立；而“完全不写死 key count”还没成立

一句话更新：
> 当前 single non-GT 主线最好的实现，已经在真实 mixed single 数据上达到：
> `top5 = 95.83%`、`CER = 12.50%`。
> 但这条结果目前仍然带有 Stage 1 mixed-aware 训练/评估口径偏乐观的风险，
> 因此更适合作为“prototype milestone”，而不是 clean final claim。

### 10.11 2026-03-21 审计结论：当前 single non-GT 最好的数字还不能当 clean final result

为了避免自欺，我们专门对当前最好的 single non-GT 链做了审计。

#### 审计问题 1：Stage 1 coarse detector 对当前 single 数据存在 optimistic 口径

mixed-aware coarse detector 的训练集：
- `/Users/shiyi/备份（mac_vs专用）/data/processed/password_segment_mixed_dataset.npz`

其中包含：
- `p01_mixed_single_training_trial000_20260321_014800`
- `p01_mixed_single_training_trial001_20260321_014800`
- `p01_mixed_single_training_trial002_20260321_014800`
- `p01_mixed_retry_training_trial000_20260321_015907`
- `p02_mixed_retry_training_trial000_20260321_020321`

按当前 `session_split(seed=42)`：
- `trial002` 进入 train
- `trial000` / `trial001` 进入 val
- `p02_retry` 进入 test

所以当前 single full-stream best run
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_overlap_single_fullstream_energy_cls_v4_gap13/report.json`

并不是建立在“coarse detector 完全没见过这 3 条 single session”的条件上。

#### 审计问题 2：更严格 sanity check 会明显掉

我们做了一个严格 sanity check：
- 保持同样的 `energy + classifier-guided anchors` 后半段不变
- 只把 coarse detector 换回旧的
  `password_segment_detector.pt`
  （即不包含 mixed_single / retry 的 detector）

结果：
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_overlap_single_fullstream_energy_cls_olddet_gap13/report.json`

聚合直接变成：
- `num_episodes = 0`
- `char_top1 = 0`
- `char_top5 = 0`
- `CER = 1.0`

这说明当前 single non-GT best 里，
**真正最脆弱、最可能存在 optimistic bias 的部分是 Stage 1 coarse detection**，
而不是后面的 classifier 或 overlap/refine 模块。

#### 目前仍然可信的部分

即使加入上面的 caveat，下面这些判断仍然成立：

1. **旧的 `stage2_episode` 不是当前 single full-stream 最好的 anchor source**
2. **`coarse region + energy peaks + classifier-guided subset selection`**
   这条结构是对的
3. **采样率硬编码可以去掉**
4. **自动 key-count 目前还不成立**
5. **retry / 多条 password 的主瓶颈是第二条 password 的 episode detection / clustering**

#### 当前最诚实的表述

现在最准确的说法应该是：

> 我们已经得到一条在 `mixed_single` 协议上非常强的 non-GT prototype pipeline，
> 但它当前最好的 full-stream 数字还不能被当作 clean final result，
> 因为 Stage 1 coarse detector 对这类数据流存在明显的 mixed-aware optimistic bias。

因此，后续如果要把这条线写成更强结论，必须至少满足其一：

1. 用**未参与 mixed-aware detector 训练**的 single/retry session 做正式评估
2. 或者重做 Stage 1 的严格 held-out protocol，再重新跑整个 non-GT 链

### 10.12 风险排除后的 strict rerun：single non-GT 仍然成立，但 Stage 1 operating point 需要重调

为了真正排除“same-domain / same-session optimistic bias”，我们重建了一个更严格的 Stage 1 训练集：

- `/Users/shiyi/备份（mac_vs专用）/data/processed/password_segment_mixed_nosingle_retry_dataset.npz`

这版训练集**完全排除了**：
- `mixed_single_training`
- `mixed_retry_training`

对应 detector：
- `/Users/shiyi/备份（mac_vs专用）/results/password_segment_mixed_nosingle_retry_detector.pt`

#### 先看坏消息

如果沿用之前 mixed-aware detector 的阈值：
- `segment_threshold = 0.30`

那么 strict rerun 会明显掉下来：
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_overlap_single_fullstream_energy_cls_noleak_gap13/report.json`

fixed-window baseline：
- `num_episodes = 2`
- `top1 = 0`
- `top5 = 6.25%`
- `CER = 100%`

这说明：
> mixed-aware Stage 1 的最佳 operating point 不能直接照搬到更严格的 no-leak detector 上。

#### 再看关键的好消息

我们随后对 strict detector 做了一个小的 threshold sweep：

- `0.10 / 0.20 / 0.30 / 0.40 / 0.50`

其中最佳是：
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_overlap_single_fullstream_energy_cls_noleak_sweep_0.10/report.json`

fixed-window baseline：
- `top1 = 83.33%`
- `top3 = 95.83%`
- `top5 = 95.83%`
- `CER = 16.67%`

overlap refine：
- `top1 = 75.00%`
- `top3 = 95.83%`
- `top5 = 100.0%`
- `CER = 25.00%`

#### 这轮 strict rerun 的真实含义

这一轮给出的最重要结论不是“风险不存在了”，而是：

1. **之前 `95.83% / 12.50% CER` 的 best single non-GT 确实偏乐观**
2. 但**把 `mixed_single/retry` 从 Stage 1 训练里完全移除之后，这条链并没有倒掉**
3. 真正发生的是：
   - Stage 1 的 operating point 需要重新调
   - 在重新调阈值后，single non-GT 仍然保持很强

所以更准确的最终表述应该是：

> 当前 single non-GT full-stream pipeline 是真的有能力的；
> 之前的 best result 里确实有 mixed-aware optimism；
> 但在更严格的 no-leak Stage 1 下，这条线仍然可以达到
> `top5 = 95.83%`、`CER = 16.67%` 这一量级。

从客观角度看，这已经足以把它当作：
- **可信的阶段性主线**
- 以及明天继续做 `len9/10/11` 的合理基线。

### 10.13 长度/计数接口问题被进一步钉死：不是长度学不会，而是主线喂法不对

在加入 `len10` 后，我们继续把“自动长度/自动计数”这件事往主线里接。

#### 已经确认的新增事实

1. `len10` pilot 数据可用
2. `stage3` 的 `len8 + len9 + len10` 多长度分支继续成立
3. 显式长度头在 `8 / 9 / 10` 上可以达到很强的 held-out 表现

对应结果：

- `/Users/shiyi/备份（mac_vs专用）/results/password_len8_len9_len10_quick_adaptation.json`
- `/Users/shiyi/备份（mac_vs专用）/results/length_model_len8_len9_len10_report.json`

其中：

- multi-length Stage3 combined:
  - `top1 = 81.37%`
  - `top5 = 99.61%`
  - `CER = 18.63%`
- explicit length head (`8/9/10`):
  - `accuracy = 96.67%`

#### 直接失败的做法

最开始我们把长度头直接接到当前 non-GT 主线里，
让它读取 **整个 coarse region** 来预测长度。

这一步失败了：

- `/Users/shiyi/备份（mac_vs专用）/results/stage2_overlap_single_fullstream_energy_cls_lenmodel_v1/report.json`

表现会大幅掉下去，原因不是“长度不能学”，而是：

> **whole coarse region 不是一个合适的长度输入接口。**

对 `mixed_single len8` 来说，whole coarse region 会包含太多额外上下文，
从而把 `len8` 误判成 `len9`。

#### 真正有效的修正

有效方案不是放弃长度头，而是改变它看到的输入：

1. 在 coarse region 内提 raw energy peaks
2. 按 temporal gap 聚成若干 peak clusters
3. 选 strongest cluster
4. 用 “cluster + context padding” 得到一个更紧凑的 subregion
5. 让长度头对这个 subregion 做长度预测

这一步已经接回：

- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/eval_overlap_single_coarse_energy_cls.py`

对应严格 no-leak rerun：

- `/Users/shiyi/备份（mac_vs专用）/results/stage2_overlap_single_fullstream_energy_cls_lenmodel_v2_clusterregion/report.json`

结果：

- fixed-window path:
  - `top1 = 62.50%`
  - `top5 = 95.83%`
  - `CER = 37.50%`
- overlap path:
  - `top1 = 83.33%`
  - `top5 = 95.83%`
  - `CER = 16.67%`

并且 debug 显示：

- 三条 `mixed_single len8`
- 现在都被长度头正确判成了 `8`

#### 当前最诚实的大结论

这一步给出的最重要结论是：

> **在 strict non-GT single 场景下，我们现在已经不再需要硬编码 `expected_keys = 8`。**

更准确地说：

- 仍然需要一个显式长度模块
- 但这个长度模块已经能在当前主线里 work
- 真正关键不是“有没有长度头”，而是“长度头看的是不是正确的 subregion”

所以从现在开始，当前 single 主线的主要未解问题已经不再是：

- windowing
- fixed sample rate
- hard-coded `8`

而会转向：

- `len11` 及更长长度是否继续成立
- retry / 多条 password 时的第二条检测
- 以及未来的跨人泛化

### 10.14 新确认：已知 password 段内，peak-level keyness 基本已经打通

为了验证一个更直接的问题：

> 如果先假设 Stage 1 已经过了，只在 password 段内部看，  
> 模型能不能直接学会“哪个山峰才是真 key 峰”？

补做了一个更直接的监督实验。

对应脚本：
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/train_eval_peak_keyness.py`

对应结果：
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_peak_keyness_len8_len9_v1/dataset_summary.json`
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_peak_keyness_len8_len9_v1/report.json`
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_peak_keyness_len8_len9_v1/episode_rows.json`

训练/评估数据：
- `password/len_8`
- `password/len9`
- `mixed_single_training`
- `mixed_single_len9`

方法非常直接：

1. 在已知 password 段内先提所有候选 peak
2. 用 GT key timestamp 给这些 peak 打标签：
   - 靠近真 key 的 peak = 正样本
   - 其他 peak = 负样本
3. 用峰附近的局部能量形状、prominence、左右 gap 等特征训练 `peak keyness` 模型
4. 在每个 episode 里，从所有 peak 中选出 `K=len(password)` 个最像真 key 的峰

#### 结果

- candidate-level CV accuracy 普遍在 `0.94 ~ 1.00`
- candidate-level AUC 普遍在 `0.99` 左右
- episode-level aggregate：
  - `exact_all_keys = 94.14%`
  - `mean_peak_recall = 99.22%`
  - `mean_peak_precision = 99.22%`

进一步拆开看：
- standalone password episodes：
  - `exact_all_keys = 94.0%`
  - `mean_peak_recall = 99.2%`
  - `mean_peak_precision = 99.2%`
- mixed single password episodes：
  - `exact_all_keys = 100%`
  - `mean_peak_recall = 100%`
  - `mean_peak_precision = 100%`

#### 这条结果真正说明了什么

这不是小修小补，而是把 Stage 2 的问题进一步收缩了：

1. **“password 段里一串山峰对应一串 key” 这件事是可学的，而且已经学得很好**
2. 当前 Stage 2 的主问题 **不是**：
   - 单键 classifier 太弱
   - 也不是已知 password 段内不会找真 key 峰
3. 当前真正没解决的，是更前面的那一步：
   - **full-stream 里，哪一整团峰簇才是真正的 password 段**

所以从更精确的分解看：

- `Stage 2A`: full-stream candidate password segment proposal / ranking  
  还没解决，是当前主瓶颈
- `Stage 2B`: 已知 password 段内的 peak-level keyness / key picking  
  已经基本成立

这条结论很重要，因为它说明：

> 现在不该再怀疑“password 段里找 key 会不会根本学不会”；  
> 这一步已经证明是能学会的。  
> 真正还难的是 **先把整段 password 本体从 full stream 里扣出来**。
