# Apple Internal IMU Keystroke Side-Channel (Working Plan)

本仓库用于验证一个高价值安全假设：
如果 Apple 设备内部未公开 IMU 传感器的振动信号可被利用来恢复键盘输入，那么这是一个具有顶会潜力的侧信道攻击方向。

当前目标不是一次性解决所有泛化问题，而是先完成一个严谨的单人闭环攻击原型，再逐步扩展威胁模型和实验难度。

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

## 2. 当前已知关键事实

### 采样频率事实（基于已有数据）
- 采样率不是稳定可控变量，历史数据呈多档波动
- `single_key` 与 `free_type` 都出现两种主档：
  - 中位频率约 `131 Hz`
  - 中位频率约 `197 Hz`
- 实际有效频率常见约 `146 Hz` 与 `199 Hz`

这意味着：不做频率门控会引入强域漂移，影响模型可解释性和复现实验质量。

### 数据策略结论（当前共识）
- 现有 `single_key` 暂不推倒重采
- `free_type` 重新采集，优先保障高频档和可控输入质量
- 先做“慢速输入闭环”，并将其明确写为受控实验条件（不是隐藏处理）

## 3. 代码入口与职责

- `collector.py`
  - 数据采集入口（single_key / free_type）
  - 频率实时监控 + 采集前频率门控
- `preprocessor.py`
  - 按键事件对齐切窗 + 重采样
- `train_baseline.py`
  - 传统特征模型基线
- `train_phase2.py`, `run_transformer_only.py`
  - 深度模型与融合
- `run_freetype_closure_eval.py`
  - free_type 独立闭环评估（质量审计/校准/解码）
- `run_freetype_finetune_beam.py`
  - free_type 微调 + beam 解码评估
- `scan_sampling_rates.py`
  - 扫描并标记非目标频率会话（新加）

### 当前数据目录约定
- `data/raw/single_key/`
  - 统一单键主数据（原 round2 + round4）
- `data/raw/boost/`
  - 补强数据（hard keys 等）
- `data/raw/free_type/`
  - free_type 数据
- `data/raw/round4/`（若仍存在）
  - 只读历史副本，默认脚本已不再作为主扫描源

## 4. 采集器门控策略（已实现）

`collector.py` 已支持采集前频率门控，且低频会话直接丢弃不保存：

- single_key 门控（默认）：
  - `--single-gate-rate 190`
- free_type 门控（默认）：
  - `--free-gate-rate 150`
- 预检时长：
  - `--precheck-sec 5`

若预检失败：
- 会话标记为 discard
- 删除该次 `sensor/events/meta/prompts` 文件
- 终端打印丢弃原因

### 常用命令

```bash
# 单键（默认要求接近 199Hz 档，写入 data/raw/single_key）
sudo .venv/bin/python3 collector.py \
  --mode single_key --raw-subdir single_key --group 2 --repeats 100 \
  --single-gate-rate 190 --precheck-sec 5

# free_type（先跑闭环，可慢速，写入 data/raw/free_type）
sudo .venv/bin/python3 collector.py \
  --mode free_type --raw-subdir free_type --part 1 --free-groups 16 \
  --free-gate-rate 150 --precheck-sec 5

# 补强 hard keys（写入 data/raw/boost）
sudo .venv/bin/python3 collector.py \
  --mode single_key --raw-subdir boost --group 8 --repeats 100 \
  --single-gate-rate 190 --precheck-sec 5
```

## 5. 频率扫描与重采定位

使用脚本扫描哪些会话不在目标频率附近：

```bash
# 默认扫描 single_key，目标 199±5Hz（按 median_hz 判定）
python3 scan_sampling_rates.py

# 扫描全部并导出 JSON
python3 scan_sampling_rates.py --mode all --json-out results/rate_scan_all.json
```

用途：
- 找出需重采的 single_key 组
- 明确 `single_key` 与 `boost` 的频率分布
- 为下一轮重采提供精确列表

### 预处理推荐命令（190Hz 默认）

```bash
# 单键主数据
python3 preprocessor.py --rounds single_key --session-type single_key

# free_type 数据
python3 preprocessor.py --rounds free_type --session-type free_type
```

## 6. 当前推荐实验计划（供后续聊天直接执行）

### Step 1: 频率清洗与重采清单
- 跑 `scan_sampling_rates.py`
- 导出 `non_target_sessions`
- 形成需补采组列表（尤其 `boost` 中的 hard-key 额外组）

### Step 2: single_key 补采（只补非目标频率组）
- 每个需补采组追加 `100` 次/键
- 采集必须通过 single_key 频率门控

### Step 3: free_type 重采（慢速闭环优先）
- 先采受控慢速（建议 12-20 WPM）
- 记录 YES/NO，保留 prompts 对齐
- 每天分段采，避免单次疲劳偏置

### Step 4: 闭环评估
- 先跑 `run_freetype_closure_eval.py`（零样本基线）
- 再跑 `run_freetype_finetune_beam.py`（微调前后对比）
- 报告核心指标：Top1/3/5, CER, WER, sentence exact match

### Step 5: 论文级整理
- 把“慢速输入”定义为明确实验条件（low-overlap regime）
- 结论限定在受控威胁模型，不夸大到自然场景

## 7. 开放技术问题：能否强制跑到最高频档？

当前结论：尚未证明可直接“软件强制”固定在最高频档（~199Hz）。

### 下一步调查方向（交给后续聊天）
- 对比不同运行条件下的频率分布：
  - 是否插电/电池
  - 系统负载水平
  - 传感器初始化顺序
  - 采集前静置时长
- 在 `sensor_reader.py` / `macimu` 侧确认：
  - 是否存在可配置 ODR（output data rate）接口
  - 是否可通过 IOKit/HID 参数切换采样档
- 若不可控：
  - 将频率档显式当作 domain，进行域分离训练与评估

## 8. 注意事项

- 低频门控失败会删除该次会话文件，这是预期行为
- 不要混合使用不同频率档数据做主结果，除非明确做 domain 对照
- 下一个聊天优先任务：
  1. 运行频率扫描并生成重采清单
  2. 开始补采 non-199Hz 的 single_key 组（100 次/键）
  3. 启动 free_type 慢速重采闭环
