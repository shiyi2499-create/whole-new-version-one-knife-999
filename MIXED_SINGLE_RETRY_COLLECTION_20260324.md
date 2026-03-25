# Mixed Single / Retry 补录说明

现有采集代码已经原生支持：

- `mixed_single_training`
- `mixed_retry_training`
- `password_length = 8 / 9 / 10`
- `n_trials = 1`

也就是说，**一次只录一组**这件事，本身就能做到，不需要改采集器逻辑。

真正的采集入口是：

- `onset_detection/onset_collector.py`

我另外加了两个薄包装脚本，专门防止误把多组一次录完：

- `run_collect_mixed_single_once.sh`
- `run_collect_mixed_retry_once.sh`

它们都会强制：

- 只录 `1` 组（`--n-trials 1`）
- 只允许密码长度 `8 / 9 / 10`

## 直接命令

### 录一组 mixed_single

```bash
./run_collect_mixed_single_once.sh p01 8
./run_collect_mixed_single_once.sh p01 9
./run_collect_mixed_single_once.sh p01 10
```

### 录一组 mixed_retry

```bash
./run_collect_mixed_retry_once.sh p01 8
./run_collect_mixed_retry_once.sh p01 9
./run_collect_mixed_retry_once.sh p01 10
```

如果你想固定随机种子，也可以传第三个参数：

```bash
./run_collect_mixed_single_once.sh p01 8 20260324
./run_collect_mixed_retry_once.sh p01 8 20260324
```

## 输出目录

按现有采集器默认规则：

- `mixed_single_training`, `len=8`
  - `data/raw/mixed_single_training/`
- `mixed_single_training`, `len=9`
  - `data/raw/mixed_single_len9/`
- `mixed_single_training`, `len=10`
  - `data/raw/mixed_single_len10/`

- `mixed_retry_training`, `len=8`
  - `data/raw/mixed_retry_training/`
- `mixed_retry_training`, `len=9`
  - `data/raw/mixed_retry_len9/`
- `mixed_retry_training`, `len=10`
  - `data/raw/mixed_retry_len10/`

## 推荐补录计划

你刚才的目标是：

- single 模式：`len8 / len9 / len10` 各补 `3` 组
- retry 模式：`len8 / len9 / len10` 各补 `3` 组

因为我们现在已经强制一次只录一组，所以你就按下面这种顺序跑：

### mixed_single

```bash
./run_collect_mixed_single_once.sh p01 8
./run_collect_mixed_single_once.sh p01 8
./run_collect_mixed_single_once.sh p01 8

./run_collect_mixed_single_once.sh p01 9
./run_collect_mixed_single_once.sh p01 9
./run_collect_mixed_single_once.sh p01 9

./run_collect_mixed_single_once.sh p01 10
./run_collect_mixed_single_once.sh p01 10
./run_collect_mixed_single_once.sh p01 10
```

### mixed_retry

```bash
./run_collect_mixed_retry_once.sh p01 8
./run_collect_mixed_retry_once.sh p01 8
./run_collect_mixed_retry_once.sh p01 8

./run_collect_mixed_retry_once.sh p01 9
./run_collect_mixed_retry_once.sh p01 9
./run_collect_mixed_retry_once.sh p01 9

./run_collect_mixed_retry_once.sh p01 10
./run_collect_mixed_retry_once.sh p01 10
./run_collect_mixed_retry_once.sh p01 10
```

## 说明

这次我没有改 `onset_collector.py` 的核心逻辑，因为它本身已经满足：

- mixed single / retry 模式
- 可选密码长度
- 一次一组

我只加了两个入口脚本，让你不容易误操作。
