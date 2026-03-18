# Apple Internal IMU Keystroke Side-Channel (Working Plan)

本仓库用于验证一个高价值安全假设：
如果 Apple 设备内部未公开 IMU 传感器的振动信号可被利用来恢复键盘输入，那么这是一个具有顶会潜力的侧信道攻击方向。

当前策略是先完成单人、单设备、受控条件下的完整攻击闭环，再逐步扩大威胁模型。

## 1. 研究背景与阶段目标

### 背景
- 传感器：Apple 设备内部 BMI286 IMU（加速度计 + 陀螺仪）
- 风险点：输入时产生的微振动可能泄露按键信息
- 价值点：若攻击闭环成立，属于高影响侧信道安全问题

### 阶段化目标
- Phase A（当前）：单人、单设备、受控环境、闭环跑通
- Phase B：跨天稳定性 + 速度条件对照
- Phase C：弱化先验（边界检测/更自然输入）
- Phase D：多用户/多设备泛化（后续）

## 2. 当前共识（已更新）

### 采样与建模共识
- 当前训练主线**不做“频率档显式建模”**。
- 频率仅作为**采集质控门控**使用，不作为模型输入特征。
- 统一预处理到 `target_rate_hz=190`（固定窗口长度 57）。

### 数据状态共识（2026-03-15）
- `single_key` 主数据已完成当前轮重采与清洗。
- `boost`（g8 补强）已完成并并入训练可用集。
- 目前 `single_key + boost` 频率扫描为目标域内（无 non-target 会话，按当前容差）。
- `sentence` 型 free_type 数据保留，但暂时不作为当前主攻路线。
- 当前 Phase 3 主线改为：`single_key + boost` baseline -> `password` 数据集测试。
- `password` 当前协议：`a-z0-9`，长度固定 `8`，总池 `200` 条，按 `20 × 10` 采集。
- 当前 `len=8` 的 `200` 条 password 数据已完成采集，并已完成 zero-shot / adaptation / multisplit / password-only 对照。
- `continuous` profile 保留为兼容/桥接层，但不是当前主线。

## 3. 代码入口与职责

- `collector.py`
  - 数据采集入口（single_key / free_type）
  - 频率实时监控 + 采集前频率门控 + 失败自动丢弃
  - `free_type` 现支持 `sentence / continuous / password` 三种 prompt profile
  - 当前主攻 profile 是 `password`
- `preprocessor.py`
  - 按键事件对齐切窗 + 统一重采样（默认 190Hz）
- `train_baseline.py`
  - 传统特征模型基线
- `train_phase2.py`, `run_transformer_only.py`
  - 深度模型与融合（当前主训练链路）
- `run_freetype_closure_eval.py`
  - free_type 独立闭环评估（质量审计/校准/解码）
- `run_freetype_finetune_beam.py`
  - free_type 微调 + beam 解码评估
- `scan_sampling_rates.py`
  - 会话采样率扫描与异常会话定位
- `multisplit_password_len8_inception.py`
  - `len_8` password route 的随机 `16/4` group split 稳定性脚本
- `password_only_len8_inception.py`
  - 纯 `password` 训练/测试的 Inception 对照脚本

### 当前数据目录约定
- `data/raw/single_key/`
  - 单键主训练数据（已清洗）
- `data/raw/boost/`
  - 补强数据（hard keys）
- `data/raw/free_type/`
  - free_type 数据
- `data/raw/password/len_8/`
  - 当前主线 password 数据目录
  - 协议：`a-z0-9`、长度 `8`、总池 `200` 条、`20 × 10`
- [CODE_MAP.md](/Users/shiyi/备份（mac_vs专用）/CODE_MAP.md)
  - 当前代码梳理入口：主线/辅助/历史脚本一览
- `data/raw/free_type_sentence_* / free_type_continuous_* / free_type_password_*`
  - 旧版或过渡型 free_type 路径，保留但不作为当前主测试集
- `data/raw/legacy_round4_ro/`
  - 历史只读备份目录（默认不作为主扫描源）

## 4. 采集器门控策略（已实现）

`collector.py` 已支持采集前频率门控，低频会话会被自动丢弃并删除文件：

- `single_key` 门控（默认）：`--single-gate-rate 190`
- `free_type` 门控（默认）：`--free-gate-rate 150`
- 预检时长：`--precheck-sec 5`

### free_type 采集新增审计能力（2026-03-12 已实现）
- 每次 free_type 会话新增 `*_attempts.csv`，记录每次 attempt 的：
  - `match(YES/NO)`、退格次数、按键数、输入时长、当时采样率统计
- 发生掉速（watchdog）时，保持“**终止整个 session**”策略，不降级为“仅重打一条句子”
- 采集前 gate 未达标时，会话文件会自动删除（不保留脏数据）

### 常用采集命令

```bash
# 单键（写入 data/raw/single_key）
.venv/bin/python3 collector.py \
  --mode single_key --raw-subdir single_key --group 2 --repeats 100 \
  --single-gate-rate 190 --precheck-sec 5

# free_type（句子型，保留但非当前主攻）
.venv/bin/python3 collector.py \
  --mode free_type --raw-subdir free_type --part 1 --free-groups 16 \
  --free-gate-rate 150 --precheck-sec 5

# password len=8 (200 total, 20 groups; continue from part 11 for round 2)
.venv/bin/python3 collector.py \
  --mode free_type --prompt-profile password \
  --raw-subdir password/len_8 --part 11 --free-groups 20 \
  --free-gate-rate 150 --precheck-sec 5

# or use the helper
./run_password_len8_part.sh 1

# hard-key 补强（写入 data/raw/boost）
.venv/bin/python3 collector.py \
  --mode single_key --raw-subdir boost --group 8 --repeats 100 \
  --single-gate-rate 190 --precheck-sec 5
```

## 5. 频率扫描与预处理

### 频率扫描

```bash
# 扫描 single_key + boost（当前主训练源）
python3 scan_sampling_rates.py --mode single_key --sources single_key boost --target-hz 199 --tol 8

# 扫描全部并导出 JSON
python3 scan_sampling_rates.py --mode all --json-out results/rate_scan_all.json
```

### 预处理（190Hz 默认）

```bash
# 单键训练集（主数据+补强）
python3 preprocessor.py --rounds single_key boost --session-type single_key --target-rate 190

# password / free_type 数据
python3 preprocessor.py --rounds password/len_8 --session-type free_type --target-rate 190
```

> 2026-03-12 更新：预处理结果现在会写入 `session_ids/source_dirs/group_tags` 元数据。  
> 请在更新代码后重新运行一次预处理，以启用会话级分组切分评估（避免 session 泄漏）。

### 训练协议（当前）
- Phase1/Phase2 评估默认使用会话级分组切分（优先 `StratifiedGroupKFold`）。
- 每个外层测试折内部再划分训练/验证集（不再把 test fold 当验证集）。
- 结果新增：`accuracy_ci95`、`macro_f1`、`per_key_recall`。
- `train_baseline.py` 已与 Phase2 对齐：优先 group-wise split，输出 `split_protocol`，避免 Phase1 session 泄漏虚高。

### free_type 评估/微调链路（2026-03-12 已实现）

`run_freetype_closure_eval.py`：
- 构建数据集时支持 `--dataset-yes-only`（默认开）
- IKI 物理重叠剔除：`--drop-iki-overlap --iki-overlap-ms 200`（默认开）
- 插值窗口门禁：`--max-imputed-ratio 0.03`（超阈值 session 直接丢弃）
- 报告输出重叠/插值/会话丢弃统计

`run_freetype_finetune_beam.py`：
- 切分粒度开关：`--split-by session|sentence`（默认 `session`，防泄漏）
- 两阶段微调：
  - Stage1 只训分类头（head warm-up）
  - Stage2 全网络解冻微调
- 类平衡采样：`--balanced-sampling`（默认开）
- 支持继承 F1 的数据门禁参数（YES-only / IKI / imputed ratio）

### 模型结构一致性修复（2026-03-13）

- `run_real_freetype.py` 保存最终模型时会写入 Transformer 架构元信息：
  - `d_model / nhead / num_layers / dim_feedforward / cls_hidden / dropout*`
- `run_freetype_closure_eval.py` 加载 checkpoint 时会自动推断并打印架构，避免 64/128 结构不一致导致的加载失败或静默评估偏差。
- 该修复用于保证：训练脚本与 free_type 闭环评估脚本在模型结构上严格一致。

### 训练运行开关（Mac / 服务器）

```bash
# Mac (M4, CPU) - 稳定复现模式
.venv/bin/python3 train_phase2.py --profile mac
.venv/bin/python3 run_transformer_only.py --profile mac

# 服务器 (4090) - GPU优先
.venv/bin/python3 train_phase2.py --profile server --device cuda
.venv/bin/python3 run_transformer_only.py --profile server --device cuda
```

可选覆盖参数：
- `--num-workers`：DataLoader 并行读取
- `--threads`：PyTorch CPU 线程
- `--xgb-jobs`：XGBoost/RandomForest 并行度
- `--nondeterministic`：追求速度时关闭严格确定性

## 6. 当前状态看板

- 🟩【已完成】Step 1: 频率清洗与重采清单
  - 已完成 `single_key + boost` 扫描与非目标会话清理
- 🟩【已完成】Step 2: single_key 补采
  - g1-g6 高质量数据已补齐，g8 补强已完成
- 🟩【已完成】Step 3: `len=8` password 数据集采集
  - 协议：`a-z0-9`、长度 `8`、总池 `200` 条、`20 × 10`
  - 目录：`data/raw/password/len_8`
  - 当前状态：`part 1-20` 已完成，采样率稳定在 `~200Hz`
- 🟩【已完成】Step 4: `len=8` password-route 闭环评估与对照
  - `phase3_password_inception/run_password_closure_inception.py`
  - `adapt_password_len8_inception.py`
  - `multisplit_password_len8_inception.py`
  - `password_only_len8_inception.py`
  - 当前结论：
    - zero-shot 很弱，说明 `single_key -> password` 域偏移明显
    - `single_key + password adaptation` 是当前最强路线
    - `password only` 可行，但更弱、波动更大
  - 当前最好看的稳定结果（`5` 次随机 `16/4` split 均值）：
    - `char_top1 = 67.3% ± 1.0%`
    - `char_top3 = 91.7% ± 1.2%`
    - `char_top5 = 96.8% ± 0.6%`
    - `sequence_top100 = 46.5% ± 5.1%`
    - `CER = 32.7% ± 1.0%`
  - 固定 `160/40` split 最佳单次结果：
    - `char_top1 = 73.4%`
    - `char_top3 = 97.8%`
    - `char_top5 = 99.1%`
    - `sequence_top100 = 65.0%`
    - `CER = 26.6%`
  - 指标：`char top-1/top-3/top-5`、`sequence top-10/top-50/top-100`、`CER`
- 🟨【进行中】Step 5: onset / password-boundary 模块
  - 当前已经有独立 [onset_detection/README.md](/Users/shiyi/备份（mac_vs专用）/onset_detection/README.md) 模块
  - 原始 onset detector 第一轮结果：
    - `AUC = 0.997`
    - `F1 = 0.835`
    - `Precision = 0.726`
    - `Recall = 0.983`
  - 当前 onset 主任务已经从 generic activity recognition 收缩到：
    - `password_boundary`
    - 即：在 `mixed2` 连续流中精准切出真实 password episode
  - 同时新增一条并行实验线：
    - `password_segment`
    - 先做 `password_typing vs non_password` 粗定位
    - 再用 onset + IKI 节奏分析精修边界
    - 最后接现有 password classifier 输出 `top-k / top-N / CER`
  - 当前 `mixed2` 协议是约 `3` 分钟的结构化连续流：
    - `idle -> trackpad_move -> typing_1 -> trackpad_click -> idle -> typing_2 -> shake`
  - `typing_2` 段会接现有 onset detector + password classifier
  - 当前 `e2e_full` 已经去掉 GT-assisted group alignment；`e2e_gt_aligned` 保留为显式 oracle baseline
  - 当前已经定位到的主要问题：
    - `password_segment` 在 standalone split 上很容易做得“过好”
    - 但在 mixed2 上，真正困难的是 **`password typing` vs `free typing`**
    - 因此当前最重要的新补采不是继续加 `single_key`
    - 而是补 `data/raw/onset_negative/freetyping/`
- 🟥【待办】Step 6: `len=9 / len=10` password 扩展
  - 目标：验证长度增长后 top-k / top-N 的退化曲线
- 🟥【待办】Step 7: 符号与大写扩展
  - 第一批目标：`! ? @`
  - 需要明确是按“最终字符”还是“物理组合键”建模
- 🟥【待办】Step 8: cross-device / cross-user 扩展
  - 先补更多人和更多设备
  - 再讨论跨设备迁移与泛化
- 🟥【待办】Step 9: 论文级整理
  - 指标表、消融、威胁模型边界、可复现实验脚本、demo 描述

> 注：`phase3_decoder.py` 当前仍偏向旧的 sentence/word 解码口径，
> 不能直接当作当前 password 主线的 headline。

## 7. 当前主结论

- 预检失败会自动删除该次会话文件，这是预期行为
- 当前主线按“单一高频域”训练，不引入频率档特征
- 当前最支持的攻击路线是：
  - `single_key + boost`
  - 再加 `password-style adaptation`
- `password only` 不差，但当前没有超过上面这条路线
- `sentence` / 自然语言恢复保留，但暂不作为当前 headline
- 当前结果已经足以支撑一个受控 password-style continuous-string 攻击故事
- onset 现在已经进入实现与训练阶段；下一阶段最值钱的是：
  - `mixed2` 上的 `password_boundary` 训练与 boundary evaluation
  - 更多 `freetyping` hard negative
  - 更多长度
  - 更多设备 / 更多用户

## 8. 下一阶段待做

1. `len=9 / len=10` password 扩展
2. `! ? @` 等常见符号扩展
3. `mixed2` 连续流采集、`password_boundary` 训练与 episode-level 评估
4. cross-device / cross-user 采集
5. `2` 分钟混合流 demo：自动识别键盘活动开始/结束，并在 `typing_2` 段恢复 password 内容

## 9. Onset / Password-Boundary 设计思路

当前 password 路线默认依赖已有键盘标签切窗；真正的自动攻击链现在已经开始补 onset 模块。
但现在的核心目标已经不是 generic keyboard activity recognition，而是：

**在连续 IMU 流中尽量精准地切出真实 password episode。**

也就是：
- `password_start` 更靠近第一个 password keystroke
- `password_end` 更靠近最后一个 password keystroke
- 内部允许短暂停顿
- 切出的 episode 再送给现有 onset detector + password classifier

当前 `onset_detection/` 模块分成两层：

1. `password_boundary`
   - 主任务
   - 4 类：
     - `non_password`
     - `password_start`
     - `password_active`
     - `password_end`
   - 主要监督来源：
     - `mixed2` 连续流
     - `events.csv` 收紧后的 refined password episode

2. `keyboard onset`
   - 辅助任务
   - 用于在已经切出的 password episode 内定位单个按键时刻

### 当前新增共识：为什么要补 `freetyping`

目前第一阶段最容易学到的是：
- `password`
vs
- 明显不像 password 的背景（idle / trackpad / shake / single_key）

但 mixed2 里真正最难的不是这些，而是：
- `password typing`
vs
- `typing_1 / free typing`

所以当前最重要的新采集是：
- `data/raw/onset_negative/freetyping/`

推荐直接单独采集，而不是完全依赖 mixed2 里的 `typing_1`。

### 当前 mixed2 demo 协议

当前不把“长时监控 1 小时”作为论文必需项，而是先做一个更可控的约 `3` 分钟 `mixed2` demo：

- `idle`
- `trackpad_move`
- `typing_1`（free typing）
- `trackpad_click`
- `idle`
- `typing_2`（password-style）
- `shake`

当前目标不是恢复自由文本，而是证明：

1. 连续流里能把真实 password episode 边界切出来
2. 不会把前后的 free typing / 干扰动作大量混进 password 段
3. 在切出的 `typing_2` / password episode 上，现有 password classifier 还能恢复内容

### 当前最关键指标

- `password_boundary` 的 segment-level：
  - `macro_f1`
  - `weighted_f1`
  - per-class `P/R/F1`
- episode-level：
  - `episode_precision / recall`
  - `mean_iou`
  - `mean_start_error_ms`
  - `mean_end_error_ms`
- Path B 下游影响：
  - `char_top1/top3/top5`
  - `sequence_top10/top50/top100`
  - `CER`
  - 以及：
    - `e2e_full`
    - `e2e_gt_seg`
    - `e2e_gt_aligned`
    - `gt_baseline`
    之间的退化差异
