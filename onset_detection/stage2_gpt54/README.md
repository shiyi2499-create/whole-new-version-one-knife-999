# GPT-5.4 Stage 2 Branch

这个目录是当前最接近主线的 Stage 2 探索分支。

## 当前包含
- [password_stage2_model.py](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_gpt54/password_stage2_model.py)
- [password_stage2_preprocessor.py](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_gpt54/password_stage2_preprocessor.py)
- [password_stage2_dataset.py](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_gpt54/password_stage2_dataset.py)
- [stage2_dense_structured.py](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_gpt54/stage2_dense_structured.py)
- [stage2_decoder.py](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_gpt54/stage2_decoder.py)
- [password_segment_detector.py](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_gpt54/password_segment_detector.py)

## 这条线在做什么
- patch-level dense modeling
- heads: `key / boundary / inside`
- structured decode
- top-K hypotheses + global classifier rerank

## 当前 mixed2 结论
- 比 Claude CTC 线更有希望
- 能稳定产生：
  - `Pred groups = 5`
  - `Onsets = 40`
- 但字符恢复仍不够好
- 代表性结果曾到：
  - `char_top1 = 7.5%`
  - `top3 = 15.0%`
  - `top5 = 27.5%`
  - `CER = 90.0%`
- 后续 top-K global rerank 也没有根本救回来

## 当前定位
- 仍然是当前最值得保留的主探索分支
- 但当前证据也在提示：
  - 问题不只在 decoder
  - 更可能在 Stage 2 的训练任务和 mixed2 分布没有真正对齐

## 当前建议
不要把这个目录当成“已经快成功了”的版本。
它更像是：
- 一个已经把问题逼近本质的 dense Stage 2 scaffold
- 下一步应该结合 mixed-style Stage 2 数据和 2A/2B 重建设计继续推进
