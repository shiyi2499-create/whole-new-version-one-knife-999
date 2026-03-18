# Claude Stage 2 Branch

这个目录保留 Claude 路线的 Stage 2 实验，不覆盖主线。

默认前提：
- Stage 1 coarse localization 已经基本可用
- Stage 3 默认接的是当前已验证最充分的 `InceptionTime` password classifier 路线
- 但 Stage 3 也还不是“满血版”；这里的实验重点仍然是 Stage 2 对 mixed2 的贡献，而不是宣称 classifier 已经做到头

## 当前包含
- [password_segment_detector.py](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_claude/password_segment_detector.py)
- [stage2_dp_segmentation.py](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_claude/stage2_dp_segmentation.py)
- [stage2_ctc.py](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_claude/stage2_ctc.py)

## 它尝试过什么
- classifier-in-the-loop constrained DP
- dense CTC sequence decoding

## 当前结论
- 这条线已经被真正接入过 Path B 主流程
- 可以运行、可以比较
- 但 mixed2 上表现很弱，代表性结果约为：
  - `char_top1 ≈ 2.5%`
  - `CER ≈ 97.5%`

## 当前定位
- baseline / side branch
- 保留用于对照，不建议作为当前主线继续重押
