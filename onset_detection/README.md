# onset_detection

这个模块当前的主目标是：

**从 `mixed2` 连续流里尽量精准地找出真实 password episode 的开始和结束，**
然后把这一段送入：

`password_boundary -> onset detector -> grouping -> password classifier`

也就是说，主线已经从 generic mixed activity recognition 收缩成：
**password-centric boundary segmentation**。

---

## 当前支持的任务

### 1. `task=onset`
原始二分类按键起点检测：
- 输入：滑窗 IMU
- 标签：`non_onset / onset`
- 输出：后续 password classifier 使用的单击键 onset

### 2. `task=password_boundary`（当前主任务）
4 类 password 边界检测：
- `non_password`
- `password_start`
- `password_active`
- `password_end`

它的目标不是把各种 activity 分得很细，而是：
- 在 `mixed2` 连续流里提取 **真实 password activity episode**
- `password_start` 更靠近第一个 password keystroke，而不是 protocol `typing_2` block 的开始
- `password_end` 更靠近最后一个 password keystroke，而不是 protocol `typing_2` block 的结束
- episode 内允许短暂停顿，不因为短 silence 立刻截断
- 最终服务 Path B 的密码恢复

### 3. `task=activity`
保留旧的 keyboard-active 二分类，仅作兼容，不再是主推荐路线。

---

## mixed2 协议（当前实际版本）

当前 `onset_collector.py --mode mixed2` 使用的是约 3 分钟的结构化协议：

| 阶段 | 活动 | 时长 | 标签 |
|---|---|---:|---|
| 1 | idle | 12s | `idle_1` |
| 2 | trackpad_move | 18s | `trackpad_move_1` |
| 3 | keyboard free typing | 35s | `typing_1` |
| 4 | trackpad_click | 18s | `trackpad_click_1` |
| 5 | idle | 12s | `idle_2` |
| 6 | keyboard password typing | 60s | `typing_2` |
| 7 | shake | 12s | `shake_1` |

逻辑上是：
- 前半段提供 non-password 干扰和 free typing 背景
- `typing_2` 前专门留一个静止段，帮助后续切出更干净的 password start
- `shake` 放在最后，作为收尾干扰，而不是直接接在 password 前面

在 `typing_2` 阶段，采集器会明确显示当前轮要输入的 8 位 `a-z0-9` password 列表，并要求：
- 慢速输入
- 每条输完按一次 `Enter`

---

## 数据来源角色

### mixed2
`mixed2` 是 `password_boundary` 的**主监督来源**：
- 提供真实连续流背景
- 提供 coarse `typing_2` block
- 再结合 `events.csv`，把 coarse block 收紧成 refined password episode：
  - `start ≈ first password key + 少量 pre-roll`
  - `end ≈ last password key + 少量 post-roll`
- 提供 non-password 干扰背景

### password session (`free_type` / `password/len_8`)
作为**补充 `password_active` 正样本**：
- 整段 session 视为 `password_active`
- **不**从 session 首尾造 synthetic `password_start / password_end`
- 避免伪边界污染真实边界学习

### single_key / boost / 其它 keyboard session
作为**hard non-password background**：
- 它们包含键盘动作
- 但不是目标 password episode
- 全部记作 `non_password`

### onset negatives
如：
- `idle`
- `trackpad_move`
- `trackpad_click`
- `shake`

这些都作为普通 `non_password` 背景。

---

## 关键文件

| 文件 | 作用 |
|---|---|
| `onset_collector.py` | 采集 `negative / mixed / mixed2` |
| `onset_preprocessor.py` | 构建 `password_boundary_dataset.npz` / `onset_dataset.npz` |
| `onset_model.py` | `PasswordBoundaryCNN` + 原 onset 模型 |
| `onset_dataset.py` | binary / multiclass dataset + sampler |
| `train_onset.py` | 训练 `password_boundary` / `onset` |
| `onset_utils.py` | episode 解码、gap bridging、matching、grouping |
| `eval_onset.py` | segment-level 多类评估 + mixed2 episode boundary 评估 |
| `eval_onset_e2e.py` | Path A / Path B 端到端评估 |

---

## 训练链路

### Step 1: 构建 `password_boundary` 数据集

```bash
python3 onset_detection/onset_preprocessor.py \
  --task password_boundary \
  --project-root . \
  --mixed2-dirs data/raw/onset_mixed2 \
  --keyboard-dirs data/raw/single_key data/raw/boost \
  --password-dirs data/raw/password/len_8 \
  --negative-dirs data/raw/onset_negative \
  --output data/processed/password_boundary_dataset.npz
```

默认会把 `mixed2` 中 coarse `typing_2` block 结合 `events.csv` 收紧成 refined password episode，再生成 4 类标签：
- `password_start`
- `password_active`
- `password_end`
- 其余全部 `non_password`

当前关键默认参数：
- `window_ms = 500`
- `stride_ms = 40`
- `label_radius_ms = 120`
- `pre_key_ms = 120`
- `post_key_ms = 220`
- `transition_exclusion_ms = 240`

其中 `transition_exclusion_ms` 的作用是：
- 把 refined 边界附近最模糊的 transition shell 剔除
- 避免把 protocol block 边缘硬压成真边界监督

### Step 2: 训练 `password_boundary` 模型

```bash
python3 onset_detection/train_onset.py \
  --task password_boundary \
  --project-root . \
  --dataset data/processed/password_boundary_dataset.npz \
  --model password_boundary_cnn \
  --checkpoint results/password_boundary_detector.pt \
  --scaler results/password_boundary_scaler.npz \
  --report results/password_boundary_training_report.json \
  --device cuda
```

### Step 3: `password_boundary` 的 segment-level 多类评估

```bash
python3 onset_detection/eval_onset.py \
  --task password_boundary \
  --project-root . \
  --checkpoint results/password_boundary_detector.pt \
  --scaler results/password_boundary_scaler.npz \
  --dataset data/processed/password_boundary_dataset.npz \
  --report results/password_boundary_eval_report.json \
  --device cuda
```

会输出：
- `macro_f1`
- `weighted_f1`
- `accuracy`
- 每个类别的 `P / R / F1`

### Step 4: mixed2 上做 password episode 边界评估

```bash
python3 onset_detection/eval_onset.py \
  --task password_boundary \
  --project-root . \
  --checkpoint results/password_boundary_detector.pt \
  --scaler results/password_boundary_scaler.npz \
  --mixed2-dirs data/raw/onset_mixed2 \
  --report results/password_boundary_mixed2_report.json \
  --device cuda
```

会输出：
- `episode_precision`
- `episode_recall`
- `mean_iou`
- `mean_start_error_ms`
- `mean_end_error_ms`

注意这里的 GT 不再是 coarse `typing_2` block 首尾，
而是 `events.csv` 收紧后的 refined password episode 首尾。

---

## Path B：predicted boundary -> onset -> classifier

### 先训练 onset detector（原链路保留）

```bash
python3 onset_detection/onset_preprocessor.py \
  --task onset \
  --project-root . \
  --keyboard-dirs data/raw/single_key data/raw/boost \
  --password-dirs data/raw/password/len_8 \
  --negative-dirs data/raw/onset_negative \
  --output data/processed/onset_dataset.npz

python3 onset_detection/train_onset.py \
  --task onset \
  --project-root . \
  --dataset data/processed/onset_dataset.npz \
  --checkpoint results/onset_detector.pt \
  --scaler results/onset_scaler.npz \
  --report results/onset_training_report.json \
  --device cuda
```

### 再跑 end-to-end Path B

```bash
python3 onset_detection/eval_onset_e2e.py \
  --project-root . \
  --onset-checkpoint results/onset_detector.pt \
  --onset-scaler results/onset_scaler.npz \
  --boundary-checkpoint results/password_boundary_detector.pt \
  --boundary-scaler results/password_boundary_scaler.npz \
  --classifier-checkpoint results/inception_password_final.pt \
  --classifier-scaler results/inception_password_scaler.npz \
  --mixed2-dirs data/raw/onset_mixed2 \
  --report results/onset_e2e_report.json \
  --device cuda
```

Path B 现在走的是：

```text
mixed2 stream
  -> password_boundary detector
  -> predicted refined password episode
  -> onset detector inside predicted episode
  -> per-episode grouping (允许内部短 gap)
  -> existing password classifier
  -> top-k / sequence_topN / CER
```

评估会同时给四组结果：
- `e2e_full`
- `e2e_gt_seg`
- `e2e_gt_aligned`（显式 oracle baseline，允许 GT-assisted grouping）
- `gt_baseline`

其中：
- `e2e_full` 和 `e2e_gt_seg` **不再依赖 GT password group 对齐**
- 如果预测 episode 数量和 GT 不一致，评估时只按时间顺序逐个对应，不做 GT-assisted 重排
- `e2e_gt_aligned` 是故意保留的 oracle 对照线

---

## 设计说明

### 为什么不用 generic mixed activity 多分类
真正影响 password 恢复质量的不是“这是不是 keyboard activity”，而是：
- password episode 是否被完整截住
- `start` 是否足够贴近首个 password 字符真正开始的地方
- `end` 是否足够贴近最后一个 password 字符真正结束的地方
- 内部短暂停顿会不会被误切成 episode 结束
- 会不会把前后的 free typing 或干扰动作混进 password 段

所以当前主任务直接学：
- `start`
- `active`
- `end`
- `non_password`

比继续堆 generic activity label 更贴近最终目标。

### 为什么 password session 不造 start / end 标签
因为 session 边界不是生态真实边界。
如果强行把 session 首尾当成 `password_start / password_end`，会把模型往错误监督上带偏。

所以 password session 只补 `password_active`；
真正的边界监督主要来自：
- `mixed2`
- `events.csv` 收紧后的 refined episode

### 为什么 single_key / boost 当 non-password
因为这些数据很像“会误触发 onset，但不该进入 password classifier 的背景”。
把它们当 hard non-password background，更符合当前主目标。

### 当前真实边界定义
这里明确区分两层：
- **protocol block boundary**：activity log 里粗粒度的 `typing_2` 区间
- **real password activity episode boundary**：更贴近真实恢复目标的 refined 首尾边界

当前 refined 逻辑是：
1. 在 mixed2 的 `typing_2` block 内找到 press 事件
2. `password_start` 锚到 **第一个 password press 前的小段 pre-roll**
3. `password_end` 锚到 **最后一个 password press 后的小段 post-roll**
4. refined 边界附近最模糊的 shell 直接剔除
5. 解码阶段允许 episode 内部存在短 gap；只有持续 non-password 证据才更容易触发结束

所以这个任务学的不是“整个 typing_2 block”，而是更接近：
- 第一个 password 字符真正开始的地方
- 最后一个 password 字符真正结束的地方
- 中间允许短暂停顿的完整 password episode

---

## 当前定位

这套模块现在已经是：
- **可训练**
- **可评估**
- **可接现有 password classifier 推进 Path B**

但也要诚实地说：
- `e2e_gt_aligned` 仍然是 oracle baseline
- `typing_1 / typing_2` 的 style 区分目前主要还是协议内启发式，不是完全 learned classifier

这不影响当前推进 mixed2 / password_boundary 训练与评估。
