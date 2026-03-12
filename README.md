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

### 数据状态共识（2026-03-12）
- `single_key` 主数据已完成当前轮重采与清洗。
- `boost`（g8 补强）已完成并并入训练可用集。
- 目前 `single_key + boost` 频率扫描为目标域内（无 non-target 会话，按当前容差）。
- 下一步重点是 `free_type` 受控慢速闭环与深度模型评估。

## 3. 代码入口与职责

- `collector.py`
  - 数据采集入口（single_key / free_type）
  - 频率实时监控 + 采集前频率门控 + 失败自动丢弃
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

### 当前数据目录约定
- `data/raw/single_key/`
  - 单键主训练数据（已清洗）
- `data/raw/boost/`
  - 补强数据（hard keys）
- `data/raw/free_type/`
  - free_type 数据
- `data/raw/legacy_round4_ro/`
  - 历史只读备份目录（默认不作为主扫描源）

## 4. 采集器门控策略（已实现）

`collector.py` 已支持采集前频率门控，低频会话会被自动丢弃并删除文件：

- `single_key` 门控（默认）：`--single-gate-rate 190`
- `free_type` 门控（默认）：`--free-gate-rate 150`
- 预检时长：`--precheck-sec 5`

### 常用采集命令

```bash
# 单键（写入 data/raw/single_key）
sudo .venv/bin/python3 collector.py \
  --mode single_key --raw-subdir single_key --group 2 --repeats 100 \
  --single-gate-rate 190 --precheck-sec 5

# free_type（慢速闭环，写入 data/raw/free_type）
sudo .venv/bin/python3 collector.py \
  --mode free_type --raw-subdir free_type --part 1 --free-groups 16 \
  --free-gate-rate 150 --precheck-sec 5

# hard-key 补强（写入 data/raw/boost）
sudo .venv/bin/python3 collector.py \
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

# free_type 数据
python3 preprocessor.py --rounds free_type --session-type free_type --target-rate 190
```

> 2026-03-12 更新：预处理结果现在会写入 `session_ids/source_dirs/group_tags` 元数据。  
> 请在更新代码后重新运行一次预处理，以启用会话级分组切分评估（避免 session 泄漏）。

### 训练协议（当前）
- 深度模型与 XGBoost 评估默认使用会话级分组切分（优先 `StratifiedGroupKFold`）。
- 每个外层测试折内部再划分训练/验证集（不再把 test fold 当验证集）。
- 结果新增：`accuracy_ci95`、`macro_f1`、`per_key_recall`。

## 6. 计划状态看板（鲜艳标记）

- 🟩【已完成】Step 1: 频率清洗与重采清单
  - 已完成 `single_key + boost` 扫描与非目标会话清理
- 🟩【已完成】Step 2: single_key 补采
  - g1-g6 高质量数据已补齐，g8 补强已完成
- 🟨【进行中】Step 3: free_type 重采（慢速闭环优先）
  - 目标：先在低重叠输入条件下跑通稳定闭环
- 🟥【未开始】Step 4: free_type 闭环评估（新数据）
  - `run_freetype_closure_eval.py`（zero-shot）
  - `run_freetype_finetune_beam.py`（微调前后对比）
- 🟥【未开始】Step 5: 论文级整理
  - 指标表、消融、威胁模型边界、可复现实验脚本

## 7. 开放技术问题：能否强制跑到最高频档？

当前结论：尚未证明可直接“软件强制”固定在最高频档（~199Hz）。

### 调查方向（保留）
- 对比不同运行条件下的频率分布：
  - 是否插电/电池
  - 系统负载水平
  - 传感器初始化顺序
  - 采集前静置时长
- 在 `sensor_reader.py` / `macimu` 侧确认：
  - 是否存在可配置 ODR（output data rate）接口
  - 是否可通过 IOKit/HID 参数切换采样档

## 8. 注意事项

- 预检失败会自动删除该次会话文件，这是预期行为
- 当前主线按“单一高频域”训练，不引入频率档特征
- free_type “慢速输入”属于明确受控条件，后续论文中需显式声明实验边界
