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

### free_type 采集新增审计能力（2026-03-12 已实现）
- 每次 free_type 会话新增 `*_attempts.csv`，记录每次 attempt 的：
  - `match(YES/NO)`、退格次数、按键数、输入时长、当时采样率统计
- 发生掉速（watchdog）时，保持“**终止整个 session**”策略，不降级为“仅重打一条句子”
- 采集前 gate 未达标时，会话文件会自动删除（不保留脏数据）

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

## 6. 计划状态看板（鲜艳标记）

- 🟩【已完成】Step 1: 频率清洗与重采清单
  - 已完成 `single_key + boost` 扫描与非目标会话清理
- 🟩【已完成】Step 2: single_key 补采
  - g1-g6 高质量数据已补齐，g8 补强已完成
- 🟨【进行中】Step 3: free_type 重采（慢速闭环优先）
  - 目标：先在低重叠输入条件下跑通稳定闭环
  - 说明：采集/评估/微调代码已就绪，当前等待明天正式采集新 free_type
- 🟨【进行中】Step 4: free_type 闭环评估（新数据）
  - `run_freetype_closure_eval.py`（zero-shot）
  - `run_freetype_finetune_beam.py`（微调前后对比）
- 🟥【未开始】Step 5: 论文级整理
  - 指标表、消融、威胁模型边界、可复现实验脚本

> 注：`phase3_decoder.py` 当前是“isolated keystroke → synthetic word simulation”评估，
> 不能当作真实 free_type 端到端准确率 headline。
> `results_phase3.json` 已新增 `evaluation_mode` 字段，防止口径误读。

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

## 9. 明天 free_type 采集与训练执行清单

建议新数据放到独立目录（不污染旧数据）：

```bash
# 1) 逐组采集（一次一组，16 组总量，慢速打字）
sudo .venv/bin/python3 collector.py \
  --mode free_type \
  --raw-subdir free_type_slow_v2 \
  --part 1 --free-groups 16 \
  --free-gate-rate 150 --precheck-sec 5

# part 改为 2..16 继续采
```

采完后先跑 closure 质检：

```bash
.venv/bin/python3 run_freetype_closure_eval.py \
  --device auto \
  --rounds free_type_slow_v2 \
  --yes-only \
  --dataset-yes-only \
  --drop-iki-overlap \
  --iki-overlap-ms 200 \
  --max-imputed-ratio 0.03
```

再跑微调 + beam：

```bash
.venv/bin/python3 run_freetype_finetune_beam.py \
  --device auto \
  --rounds free_type_slow_v2 \
  --split-by session \
  --dataset-yes-only \
  --eval-yes-only \
  --drop-iki-overlap \
  --iki-overlap-ms 200 \
  --max-imputed-ratio 0.03 \
  --stage1-epochs 4 \
  --stage2-epochs 12 \
  --stage1-lr 3e-4 \
  --stage2-lr 1e-4 \
  --balanced-sampling \
  --beam 100 --alpha 0.15
```
