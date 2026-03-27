# Still-Password-Still Probe

这是当前协议偏移诊断的最小执行路线：

1. Step 0：录本机 still-password-still probe
2. Step 1：跑 auto / event-window / tight-burst 三种评测

## Step 0: 录制

默认会录 5 条 fair6 里表现较好的密码：

- `b15bp8ws`
- `ijtplv3am8`
- `0xc8pugot`
- `1kfxksa8`
- `kodtpoxk`

命令：

```bash
cd "/Users/shiyi/备份（mac_vs专用）"
python3 clean_password_eval/collect_still_password_probe_batch.py \
  --participant localprobe \
  --dataset-root data/raw/clean_password_probe \
  --idle-sec 3
```

如果想手动指定密码：

```bash
cd "/Users/shiyi/备份（mac_vs专用）"
python3 clean_password_eval/collect_still_password_probe_batch.py \
  --participant localprobe \
  --dataset-root data/raw/clean_password_probe \
  --idle-sec 3 \
  --passwords b15bp8ws ijtplv3am8 0xc8pugot
```

输出：

- `data/raw/clean_password_probe/*_sensor.csv`
- `data/raw/clean_password_probe/*_events.csv`
- `data/raw/clean_password_probe/*_attempts.csv`
- `data/raw/clean_password_probe/*_protocol.json`
- `data/raw/clean_password_probe/batch_manifest.json`

## Step 1: 评测

命令：

```bash
cd "/Users/shiyi/备份（mac_vs专用）"
"/Users/shiyi/apple IMU/.venv/bin/python3" clean_password_eval/eval_still_password_probe.py \
  --dataset-root data/raw/clean_password_probe \
  --output-dir results/still_password_probe_eval \
  --beam-width 500 \
  --tight-threshold 0.7 \
  --tight-margin-sec 0.5
```

输出：

- `results/still_password_probe_eval/report.json`
- `results/still_password_probe_eval/rows.json`
- `results/still_password_probe_eval/rows.csv`

## 结果里会有三种口径

- `auto_fullsample`
  - 全自动：整条样本先过 Stage1，再跑 pipeline / CTC
- `event_window`
  - 用真实按键事件窗口直接切段
  - 具体口径是“第一个非 Enter 按键”到“最后一个非 Enter 按键 + 0.2s 尾巴”
  - `Enter` 只负责结束 trial，不会被当成密码的一部分喂给模型
- `tight_burst`
  - 在 Stage1 best 段内，用高置信 keyness 峰再收紧一次

## 额外信息

每条样本还会输出：

- `inter_key_ms`
- `mean_inter_key_ms`
- `median_inter_key_ms`
- `stage1_segments`
- `tight_burst_debug.anchor_debug`

这样可以直接判断：

1. 这是不是 `password` 这个词的特例
2. keyness 峰在 still 上下文里是否还稳定
3. 仅靠 tight burst 能把 CER 拉回来多少
