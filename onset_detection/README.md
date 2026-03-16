# Onset Detection Briefing Pack

这个文件夹是给外部模型快速理解当前项目状态用的轻量说明包。

目标不是复现实验，而是让对方先理解：
- 我们已经完成了什么
- 当前主线是什么
- password 路线的核心结果是什么
- onset detection 为什么是下一步

## 建议阅读顺序

1. `ROOT_README.md`
2. `PHASE3_STATUS.md`
3. `RESULTS_LEN8.md`
4. `PERMISSION_MODEL.md`
5. `PAPER_OUTLINE.md`

## 当前一句话总结

我们已经证明：
- Apple internal IMU / SPU 在 macOS 上存在 non-root 可读路径
- `single_key + boost` 是强 baseline
- `password-style continuous-string` 路线已经闭环
- 当前最强结果来自 `single_key + password adaptation`
- 下一阶段的关键问题是 continuous-stream onset detection
