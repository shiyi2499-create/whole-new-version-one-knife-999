# Onset Detection 可执行实施方案

## 0. 核心设计理念

在展开每个细节之前，先澄清一个关键架构选择：

当前 password classifier 的窗口协议是 **100ms pre-trigger + 200ms post-trigger = 300ms, 重采样到 190Hz = 57 samples**。onset detector 的唯一使命是：**在连续 IMU 流里找到这些 trigger 时刻**，然后把 trigger ± 窗口交给已有 classifier。

因此 onset detector 和 password classifier 是 **串联** 关系，不是替代关系。onset detector 不需要知道按了哪个键，只需要回答"此刻是否正在发生一次按键冲击"。

---

## 1. Recommended Task Definition

### 1.1 任务形式：滑动窗口二分类 → 峰值提取

不建议直接做 sequence-to-sequence 或 CTC-style 的 proposal generation——复杂度高、调参难、对数据量要求大。

推荐方案：

```
连续 IMU 流
  → 滑动窗口（固定长度, 固定步长）
    → 二分类器：P(keystroke_onset_in_center)
      → 输出连续概率曲线
        → 峰值检测 + NMS
          → 候选 onset 时间戳列表
```

### 1.2 窗口与步长参数

| 参数 | 推荐值 | 理由 |
|------|--------|------|
| 检测窗口长度 | **150ms（≈29 samples @ 190Hz）** | 足够捕捉按键冲击主体（~50-80ms），同时短于 password classifier 窗口（300ms），避免跨键 |
| 滑动步长 | **25ms（≈5 samples @ 190Hz）** | 在时间分辨率和计算量之间平衡；25ms 步长意味着每秒 40 次分类 |
| 标签半径 | **±30ms** | 如果窗口中心距离真实 onset ≤30ms → label=1, 否则 label=0 |

### 1.3 为什么 150ms 而不是 300ms

- 300ms 是 **classifier** 需要的上下文（包含 pre-trigger 和 post-trigger 信息来识别 *是哪个键*）
- onset detector 只需判断 **"此处有没有冲击"**，冲击主峰集中在 ~50-80ms
- 150ms 给出足够上下文又不会让相邻按键（IKI ≈ 300-500ms in slow typing）的信号互相干扰
- 如果实测发现 150ms 太短，可以扩到 200ms，但不建议超过 200ms

### 1.4 正 / 负标签定义

给定一个 150ms 窗口，其中心时刻为 t_center：

- **label = 1**：存在至少一个真实按键时间戳 t_key，使得 |t_center - t_key| ≤ 30ms
- **label = 0**：不存在任何 t_key 满足上述条件

这意味着：
- 对于 25ms 步长，每个按键事件大约产生 2-3 个连续的正窗口
- 正负样本比例可通过负样本的丰富性来自然调节

### 1.5 后处理：峰值检测 + NMS

```
1. 对概率曲线做平滑（可选，3-5 点均值滤波）
2. 找所有局部峰值 > threshold（建议起始 0.5，后续可调）
3. NMS：在每个峰值周围 ±100ms 内，只保留概率最高的一个
4. 输出：候选 onset 时间戳列表
```

NMS 半径 100ms 的理由：当前 slow typing IKI ≈ 300-500ms，100ms 抑制窗口不会合并相邻按键，但能消除同一按键的重复检测。

---

## 2. Data Collection Plan

### 2.1 正样本来源

#### 可以直接复用的数据

| 数据源 | 复用方式 | 注意事项 |
|--------|----------|----------|
| `single_key` | 每个 session 的 `sensor.csv` + `events.csv` 已有精确 onset 时间戳 | 按键前后有人为等待期，天然提供 keystroke vs idle 对比 |
| `boost` | 同上 | 同上 |
| `password/len_8` | 连续 8 键输入，`events.csv` 有每个字符的时间戳 | **这是最有价值的正样本**——它包含真实的连续输入上下文和 inter-key 过渡段 |

#### 复用策略

**单键数据 → segment-level 正负样本**：
- 每个 session 的 sensor.csv 包含完整采集过程
- events.csv 里的时间戳标记了按键时刻
- 按键周围 ±75ms = 正窗口
- 按键之间的等待期 = 负窗口（idle within typing session）

**password 数据 → 连续流正负样本**：
- 每条 password 的 sensor.csv 是一段 ~3-5 秒连续流
- events.csv 有 8 个按键时间戳
- 按键间的 inter-key interval 天然提供"正在打字上下文但当前没有按键"的负窗口
- 这种"打字间隙负样本"是最有价值的 hard negative

#### 需要额外采集的数据

现有数据 **完全缺失** 以下场景的负样本：

| 场景 | 代号 | 采集建议 |
|------|------|----------|
| 完全静止 | `idle` | 不碰电脑，录 3-5 分钟，分 10-15 段 |
| 触控板移动/滚动 | `trackpad_move` | 正常浏览网页，录 3-5 分钟 |
| 触控板点击 | `trackpad_click` | 有节奏地点击，录 2-3 分钟 |
| 搬动/晃动 Mac | `shake` | 拿起、放下、轻推，录 2-3 分钟 |
| 桌面振动 | `desk_bump` | 敲桌子、放水杯，录 2-3 分钟 |
| 外接键盘打字 | `external_kb`（可选） | 如果有外接键盘的话：在外接键盘上打字，内置 IMU 可能会拾取传导振动 |

**每类负样本建议总量**：至少 2-3 分钟有效数据。总计约 15-20 分钟纯负样本。

### 2.2 是否需要单独采 continuous keyboard-only 流？

**建议采，但优先级是 P1 而非 P0。**

理由：
- `password/len_8` 已经有连续输入，但每条只有 8 键，总时长短
- 一段 30-60 秒的纯键盘连续输入流（比如打一段已知文本）可以提供更丰富的 IKI 分布和节奏变化
- 但对 MVP 来说，复用 password 数据的连续段已经足够

如果采：
- 建议录 10-15 段，每段 30-60 秒
- 可以用 sentence prompt 或者让用户连续打随机字符
- 关键是保持 IMU 连续采集 + events.csv 精确标签

### 2.3 混合长时流设计

**这是 onset detection 评估的核心测试集。** 必须单独采。

#### 协议

每段混合流 **30-45 秒**，内含 **随机排列** 的以下动作块：

| 动作块 | 时长范围 | 说明 |
|--------|----------|------|
| idle | 3-5s | 静止不动 |
| trackpad_scroll | 3-5s | 正常滑动浏览 |
| trackpad_click | 2-3s | 点击若干次 |
| keyboard_password | 3-5s | 打一个 8 字符 password |
| keyboard_sentence | 5-8s | 打一段短文本（可选） |
| shake / bump | 2-3s | 轻晃或桌面碰撞 |

#### 录制要求

- 每段的动作顺序 **随机排列**（不要每段都 idle → scroll → type → ...）
- 建议用一个简单脚本在屏幕上显示当前应该做的动作和剩余时间
- 全程 IMU 连续采集
- 键盘部分需要 events.csv 标签
- 非键盘部分用脚本记录动作切换时间戳（后处理用）

#### 建议数量

- **最少 20 段**，总计 ~10-15 分钟
- 理想 30-40 段
- 分成 train/val/test 或者全部作为 test set（如果只用于评估）

#### 为什么建议全部作为 test set

混合流的主要目的是 **评估**，不是训练。训练用 segment-level 数据（复用的 single_key + password + 负样本 segments）就够了。混合流用来测试模型在"真实监听场景"下的表现，所以应该全部留作 held-out 评估集。

如果数据充裕（>30 段），可以拆 5 段做 validation（调 threshold/NMS 参数），剩余做 test。

### 2.4 数据量估算

| 来源 | 正窗口数（估） | 负窗口数（估） |
|------|----------------|----------------|
| single_key 复用 | ~3000-5000 | ~3000-5000（按键间空档） |
| boost 复用 | ~500-1000 | ~500-1000 |
| password/len_8 复用 | ~1600（200×8） | ~3000-5000（inter-key gaps） |
| 新采负样本 segments | 0 | ~15000-20000（15-20 分钟） |
| **总计** | **~5000-7000** | **~20000-30000** |

正负比约 1:3 到 1:5，合理。如果需要平衡，可以对正样本做时间偏移增强（±5-10ms jitter）。

---

## 3. Labeling And Evaluation Plan

### 3.1 训练数据标签

**已有标签数据（零标注成本）**：
- `single_key` / `boost` / `password` 的 `events.csv` 已有精确的按键时间戳
- onset preprocessor 只需要读取 `sensor.csv` + `events.csv`，用滑动窗口 + 时间距离规则自动生成 label

**新采负样本标签**：
- 整段标记为 label=0
- 无需逐窗口标注

**混合流标签**：
- 键盘部分：events.csv 自动提供
- 非键盘部分：整段 label=0
- 需要记录每个动作块的起止时间戳，用于后处理和分析

### 3.2 训练/验证/测试 切分

```
训练集：
  - single_key 的 ~70% sessions（按 session 切分，防泄漏）
  - boost 全部（或 70%）
  - password/len_8 的 16 parts（和现有 adaptation 协议对齐）
  - 新采负样本 segments 的 ~70%

验证集：
  - single_key 的 ~15% sessions
  - password/len_8 的 2 parts
  - 新采负样本 segments 的 ~15%
  - 混合流的 5 段（仅用于调 threshold / NMS）

测试集：
  - single_key 的 ~15% sessions
  - password/len_8 的 2 parts
  - 新采负样本 segments 的 ~15%
  - 混合流的 15-25 段（核心评估集）
```

**关键原则**：session-level split，不允许同一 session 的窗口出现在不同集合中。

### 3.3 Segment-Level 指标（在独立窗口上）

| 指标 | 定义 |
|------|------|
| Window Precision | 预测为 onset 的窗口中，真正包含 onset 的比例 |
| Window Recall | 真正的 onset 窗口中，被正确预测的比例 |
| Window F1 | 上述两者的调和均值 |
| AUC-ROC | 概率阈值无关的整体判别力 |

### 3.4 Event-Level 指标（在连续流上，核心指标）

这些指标需要先做峰值提取 + NMS，然后和 ground truth 事件列表做匹配。

**匹配规则**：
- 对每个 predicted onset，在 ground truth 中找最近的未匹配 onset
- 如果时间距离 ≤ tolerance（建议 **±50ms**），算作 True Positive
- 一个 ground truth onset 只能被匹配一次（贪心匹配，按时间距离排序）

| 指标 | 定义 | 说明 |
|------|------|------|
| Event Precision | TP / (TP + FP) | 预测的 onset 中有多少是真的 |
| Event Recall | TP / (TP + FN) | 真实的 onset 中有多少被找到 |
| Event F1 | 2 × P × R / (P + R) | 主 headline 指标 |
| Timing Error (mean ± std) | 匹配到的 TP 对的 |t_pred - t_true| 分布 | 评估时间精度 |
| Timing Error (median) | 同上，取中位数 | 更鲁棒 |
| False Alarms / Minute | FP 总数 / 总监听时长（分钟） | 实际部署视角的指标 |

### 3.5 Tolerance 敏感性分析

建议在论文中报告不同 tolerance 下的 Event F1：

| Tolerance | 含义 |
|-----------|------|
| ±25ms | 严格——要求近乎完美对齐 |
| ±50ms | 主报告值——合理的切窗误差 |
| ±75ms | 宽松——仍在 classifier 窗口容忍范围内 |

因为 password classifier 的窗口是 100ms pre + 200ms post，即使 onset 估计偏差 ±50ms，classifier 窗口仍然覆盖大部分有效信号。可以在论文里用此论据来 justify ±50ms tolerance 的合理性。

### 3.6 端到端指标（onset → classifier pipeline）

这是最终验证攻击链完整性的指标：

```
连续混合流 IMU
  → onset detector → 候选 onset 列表
    → 以每个候选 onset 为中心，切 300ms 窗口
      → password classifier → per-position top-k 预测
        → 拼接 → sequence-level 评估
```

端到端指标直接复用现有 password 评估指标体系：
- `char_top1 / top3 / top5`
- `sequence_top10 / top50 / top100`
- `CER`

但额外报告：
- 因 onset miss 导致的 **字符丢失率**（missed characters / total characters）
- 因 onset false alarm 导致的 **多余字符插入率**
- 与 ground-truth onset 切窗结果的 **性能退化幅度**（Δ char_top1 等）

---

## 4. Code Integration Plan

### 4.1 新增文件清单

```
onset_detection/
├── onset_collector.py          # 负样本 + 混合流采集器
├── onset_preprocessor.py       # 连续流 → 滑动窗口数据集构建
├── onset_dataset.py            # PyTorch Dataset（窗口级 + 连续流级）
├── onset_model.py              # 检测器模型定义
├── train_onset.py              # 训练脚本
├── eval_onset.py               # segment + event-level 评估
├── eval_onset_e2e.py           # onset → classifier 端到端 pipeline
├── onset_utils.py              # NMS / peak detection / matching 工具函数
└── README.md                   # onset detection 子模块文档
```

### 4.2 每个文件的职责

#### `onset_collector.py`

**职责**：采集负样本和混合流数据

功能：
- `--mode negative`：纯负样本采集
  - 屏幕提示当前动作（idle / trackpad_move / trackpad_click / shake / desk_bump）
  - 每个动作录固定时长后自动切换
  - 输出 `sensor.csv` + `activity_log.csv`（记录每个动作块的起止时间）
  - 复用现有 `sensor_reader.py` / `spu_backend.py`
  - 复用现有频率门控逻辑

- `--mode mixed`：混合流采集
  - 从预定义的动作块池中随机排列一个序列
  - 屏幕实时提示"现在做：idle"→ "现在做：打字 `a8k3m2p9`"→ ...
  - 键盘部分激活 `keyboard_listener.py` 记录 events.csv
  - 输出 `sensor.csv` + `events.csv` + `activity_log.csv` + `script.json`（计划的动作序列）

**不建议直接改 collector.py 的理由**：
- collector.py 已经够复杂（single_key / free_type / password 三种模式）
- onset 采集的交互流程完全不同（动作提示 → 计时 → 自动切换 vs 等待用户打字）
- 但 onset_collector.py **应该复用** collector.py 的底层组件：`sensor_reader.py`, `spu_backend.py`, `keyboard_listener.py`, 频率门控逻辑

#### `onset_preprocessor.py`

**职责**：从原始数据构建训练用的滑动窗口数据集

功能：
- 读取 `single_key` / `boost` / `password` 的 `sensor.csv` + `events.csv`
- 读取新采的负样本 `sensor.csv` + `activity_log.csv`
- 统一重采样到 190Hz（复用 preprocessor.py 的重采样逻辑）
- 生成滑动窗口 + label
- 输出 `.npz` 格式：windows, labels, timestamps, session_ids, source_types
- 支持 `--window-ms 150 --stride-ms 25 --label-radius-ms 30`

#### `onset_dataset.py`

**职责**：PyTorch Dataset 封装

- `OnsetWindowDataset`：窗口级分类用（训练/验证）
- `OnsetStreamDataset`：完整连续流用（推理/评估时逐段送入）
- 支持增强：时间 jitter（±5-10ms 窗口偏移）、高斯噪声、通道 dropout

#### `onset_model.py`

**职责**：onset detector 模型定义

MVP 推荐：**1D-CNN（3-4 层）**

理由：
- 输入只有 29 samples × 6 channels，极短
- 不需要 InceptionTime 级别的复杂度
- 推理速度快，适合实时滑动窗口场景
- 后续可以换成更强的模型做消融

备选：
- 小型 InceptionTime（1 个 Inception block 而非完整 stack）
- 简单阈值检测器（作为 baseline 对比）

```
模型结构草案（MVP）：
  Conv1d(6, 32, kernel=5, padding=2) → BN → ReLU
  Conv1d(32, 64, kernel=5, padding=2) → BN → ReLU
  Conv1d(64, 64, kernel=3, padding=1) → BN → ReLU
  GlobalAvgPool → FC(64, 1) → Sigmoid
```

#### `train_onset.py`

**职责**：训练 onset detector

- 读取 `onset_preprocessor.py` 的输出
- Session-level split
- 支持类平衡采样（或 focal loss）处理正负不均衡
- 输出：best model checkpoint + 训练日志
- 评估：每个 epoch 报告 window-level precision/recall/F1/AUC

#### `eval_onset.py`

**职责**：onset detector 的独立评估

- Segment-level 指标：在 held-out 窗口上报告 P/R/F1/AUC
- Event-level 指标：在连续流（混合流测试集）上报告 Event P/R/F1 + timing error + false alarms/min
- 支持 tolerance sweep（±25/50/75ms）
- 输出 JSON + 可读报告

#### `eval_onset_e2e.py`

**职责**：端到端攻击链演示

```
流程：
1. 加载混合流测试数据
2. 运行 onset detector → 候选 onset 列表
3. 在每个候选 onset 周围切 300ms 窗口
4. 加载 password classifier（现有 Inception checkpoint）
5. 对每个窗口做 top-k 字符预测
6. 找到连续打字片段（onset 间距 < 1s 的聚类）
7. 在每个打字片段内拼接预测
8. 报告 char_top1/3/5 + sequence_top10/50/100 + CER
9. 与 ground-truth onset 切窗结果做 Δ 对比
```

#### `onset_utils.py`

**职责**：共用工具函数

- `peak_detect(probs, threshold)`：局部峰值检测
- `nms_1d(peaks, probs, radius_ms)`：1D NMS
- `match_events(predicted, ground_truth, tolerance_ms)`：贪心事件匹配
- `compute_event_metrics(matches, n_pred, n_true)`：P/R/F1 计算
- `compute_timing_errors(matches)`：timing error 统计

### 4.3 需要修改的现有文件

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| 无 | onset detection 作为独立子模块，不修改现有主线代码 | — |
| `preprocessor.py` | 可选：抽取重采样函数为公共 util，供 onset_preprocessor.py 调用 | P2 |
| `adapt_password_len8_inception.py` | 可选：暴露 classifier 加载接口供 eval_onset_e2e.py 调用 | P1 |

**设计原则**：onset detection 不侵入现有 password 主线。所有依赖通过导入或加载 checkpoint 实现。

### 4.4 目录结构

```
项目根目录/
├── onset_detection/           # 新增：onset detection 子模块
│   ├── onset_collector.py
│   ├── onset_preprocessor.py
│   ├── onset_dataset.py
│   ├── onset_model.py
│   ├── train_onset.py
│   ├── eval_onset.py
│   ├── eval_onset_e2e.py
│   ├── onset_utils.py
│   └── README.md
├── data/
│   └── raw/
│       ├── onset_negative/    # 新增：负样本数据
│       │   ├── idle/
│       │   ├── trackpad_move/
│       │   ├── trackpad_click/
│       │   ├── shake/
│       │   └── desk_bump/
│       └── onset_mixed/       # 新增：混合流数据
│           ├── stream_001/
│           ├── stream_002/
│           └── ...
├── data/
│   └── processed/
│       └── onset_dataset.npz  # 新增：onset 训练数据
├── collector.py               # 不修改
├── preprocessor.py            # 可选小改
├── adapt_password_len8_inception.py  # 可选小改
└── ...
```

---

## 5. MVP Plan

### 5.1 MVP 定义

**最小可行系统 = 能在混合流上报出 event-level P/R/F1 + 能串联 classifier 跑出端到端 top-k 的版本。**

### 5.2 MVP 包含什么

| 组件 | MVP 范围 | 非 MVP（后续） |
|------|----------|----------------|
| 正样本 | 复用 single_key + password/len_8 | 单独采的长时键盘流 |
| 负样本 | idle + trackpad_move + trackpad_click（3 类） | shake + desk_bump + external_kb |
| 混合流 | 10-15 段 × 30s | 30-40 段 |
| 模型 | 3 层 1D-CNN | InceptionTime ablation, energy baseline |
| 评估 | Event P/R/F1 @ ±50ms + timing error + 端到端 top-k | tolerance sweep, false alarms/min 详细分析 |
| 论文 | "onset detection prototype" 一节足够 | 完整消融 + 多条件对比 |

### 5.3 MVP 时间估算

| 阶段 | 预估时间 | 产出 |
|------|----------|------|
| 负样本采集（3 类） | 0.5 天 | ~10 分钟负样本 |
| 混合流采集 | 0.5 天 | 10-15 段混合流 |
| onset_collector.py | 0.5 天 | 采集脚本 |
| onset_preprocessor.py + dataset | 0.5 天 | 训练数据 |
| onset_model.py + train_onset.py | 0.5 天 | 训练 pipeline |
| eval_onset.py | 0.5 天 | event-level 评估 |
| eval_onset_e2e.py | 0.5 天 | 端到端 demo |
| 调参 + 修 bug | 0.5-1 天 | 稳定结果 |
| **总计** | **~4 天** | 完整 MVP |

### 5.4 论文里 onset detection 一节需要什么

一个能通过审稿的 "onset detection prototype" 节至少需要：

1. 明确说明检测任务定义（滑动窗口二分类 + NMS）
2. 报告 segment-level AUC（说明模型确实能区分 keystroke vs non-keystroke）
3. 报告 event-level P/R/F1 @ ±50ms（在混合流上）
4. 报告 timing error 分布（至少 median + std）
5. 报告端到端 password inference 性能，与 ground-truth onset 做 Δ 对比
6. 一张概率曲线可视化图（连续流 + predicted onsets + ground truth onsets 叠加）

**不需要**：完美的 onset detection、多模型对比消融、跨场景泛化。论文可以明确说"这是 controlled prototype，表明端到端攻击链可行"。

---

## 6. Main Risks

### 6.1 技术风险（按严重程度排序）

**风险 1（中等）：trackpad 冲击和 keystroke 冲击在 IMU 上太相似**

触控板点击也会产生机械冲击，可能在 IMU 上产生类似信号。如果 onset detector 无法区分，会导致 false alarm 率偏高。

缓解策略：
- 先看数据：采了 trackpad_click 负样本后，和 keystroke 窗口做特征分布对比
- 如果确实相似，考虑把问题改成两级：先检测"机械冲击"，再分类"键盘 vs 触控板"
- 论文中可以诚实报告这个混淆来源

**风险 2（低-中）：自然 typing 节奏下相邻按键窗口重叠**

在较快的打字速度下（IKI < 250ms），150ms 检测窗口可能跨越相邻按键，导致 onset 定位不准。

缓解策略：
- 当前协议是 slow controlled typing（IKI ≈ 300-500ms），这个风险在 MVP 阶段不大
- 论文中标注这是 controlled condition 的限制
- 后续可以缩短检测窗口或引入更精细的 peak localization

**风险 3（低）：正负样本分布偏移**

训练时用的负样本（干净的 idle/trackpad segments）和测试时的混合流（上下文不同）可能存在分布差异。

缓解策略：
- 确保负样本来源多样化（不要全是 idle）
- password 数据中的 inter-key gap 是最好的 hard negative，一定要用
- 混合流只用于测试，不用于训练——这本身就是对泛化能力的测试

### 6.2 非技术风险

**采集疲劳**：混合流采集需要频繁切换动作，可能导致后期数据质量下降。建议分多次采、每次 15-20 分钟。

---

## 7. Recommended Execution Order

```
Phase 1: 数据准备（Day 1-2）
├── Step 1.1: 写 onset_collector.py（negative + mixed 模式）
├── Step 1.2: 采集 3 类负样本（idle / trackpad_move / trackpad_click）
├── Step 1.3: 采集 10-15 段混合流
└── Step 1.4: 写 onset_preprocessor.py，从现有数据 + 新数据构建训练集

Phase 2: 训练与评估（Day 2-3）
├── Step 2.1: 写 onset_model.py（3 层 1D-CNN）
├── Step 2.2: 写 onset_dataset.py + train_onset.py
├── Step 2.3: 训练 onset detector，报告 window-level 指标
├── Step 2.4: 写 onset_utils.py（NMS / matching）
└── Step 2.5: 写 eval_onset.py，在混合流上报告 event-level 指标

Phase 3: 端到端串联（Day 3-4）
├── Step 3.1: 写 eval_onset_e2e.py
├── Step 3.2: 串联 onset → classifier，报告 password-level top-k
├── Step 3.3: 与 ground-truth onset 做 Δ 对比
└── Step 3.4: 生成概率曲线可视化图

Phase 4: 补强（Day 4+，可选）
├── Step 4.1: 补采 shake + desk_bump 负样本，重新训练
├── Step 4.2: 增加混合流段数，提升评估统计显著性
├── Step 4.3: energy-based baseline 作为对比
└── Step 4.4: tolerance sweep 消融
```

### 建议先做什么

**最值得先做的是 Step 1.4（onset_preprocessor.py）**——因为你可以在不采任何新数据的情况下，先用 single_key + password 现有数据验证整个 pipeline 是否跑通。仅用 inter-key gap 作为负样本，先看 window-level AUC。如果 AUC > 0.9，说明信号本身是强的，值得继续投入采集。

### 可以晚做的

- shake / desk_bump / external_kb 负样本 → 对 MVP 不是必需的
- 多模型对比（InceptionTime vs CNN vs energy baseline）→ 论文修改阶段再做
- tolerance sweep → 主结果出来后补
- 长时连续键盘流单独采集 → password 数据复用已经足够

---

## 8. 当前文档中缺失的信息

为了更精确地设计方案，以下信息如果你能补充会更好：

1. **single_key session 的典型时长**——每个 session 的 sensor.csv 大概多长？（秒级 / 分钟级？）这决定了从 single_key 能获取多少 inter-key 负样本。

2. **password session 的典型时长**——每条 8 字符 password 从开始到结束大概多少秒？IKI 大概在什么范围？

3. **events.csv 的时间戳精度**——是毫秒级还是微秒级？和 sensor.csv 的时间戳在同一个时钟域下吗？（ROOT_README 提到了"same monotonic clock domain"，但想确认实际精度。）

4. **sensor.csv 的具体列格式**——是 `timestamp, ax, ay, az, gx, gy, gz` 还是其他格式？

5. **目标投稿 venue 和 deadline**——这决定了 MVP 够不够还是需要做更完整的消融。

6. **当前 Mac 型号**——这影响 IMU 信号特征的描述。

---

## 9. 一句话总结

onset detection 的核心不难——你已经有了大量精确标注的 keystroke 数据和强 classifier，只需要补上 15-20 分钟的负样本 + 10-15 段混合流，训练一个轻量 1D-CNN 滑动窗口检测器，就足以完成攻击链闭环。**最大的不确定性不在 onset detection 本身，而在 trackpad click 与 keystroke 的区分度上——建议采完第一批数据后先看这个。**
