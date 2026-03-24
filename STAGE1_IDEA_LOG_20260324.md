# Stage1 Idea Log 2026-03-24

这份记录只保留 2026-03-24 这轮讨论里用户明确提出、并要求后续持续参考的 Stage1 想法与约束。

## 当前共识

- Stage1 的主体仍然是“整段分析”，不是回到旧的“先找单峰再拼段”的主线。
- 第一原则是 **宁可多挖一点，也不能少挖**。
  - 多挖了，后面的按键识别、切窗、字符恢复还有机会纠正。
  - 少挖了，真 password 没进候选池，后面再强也救不回来。
- 评价 Stage1 时，不能只看“有没有大概找到”，而要看：
  - 真 password 段是否进入候选池
  - 是否尽量完整
  - 在复杂连续流里是否会产生过多假段

## 用户明确提出的想法

### 1. 纯净 password 不应该浪费

- `data/raw/password/len_8`
- `data/raw/password/len9`
- `data/raw/password/len10`

这些目录里的数据，虽然不是 mixed 全流，但每次尝试本身就是一条完整 password。
因此可以按完整尝试拆成训练样本，而不是整批闲置。

用户要求：

- 不要只拿 `mixed_training` 训练。
- 要把这批纯净 password 按每次完整输入拆出来。
- 让模型先学习“完整 password 段长什么样”。

### 2. 单密码和复杂流不要全部拿来做测试

用户要求：

- 不要把所有 `mixed_single_*`、`mixed_retry_*` 都留成测试。
- 可以留一小部分硬验证集。
- 其余能进训练的就进训练，避免浪费数据。

这轮已经按这个思路做过一次扩展划分：

- 训练目录：`data/raw/stage1_dense_train_expanded_20260324`
- 小验证集：`data/raw/stage1_dense_eval_dev_20260324`

### 3. 直接把纯净 password 混进当前 dense 主线，不一定对

这轮已经真实验证过：

- 给 `train_eval_stage1_dense_labeling.py` 加了纯净 password 尝试适配
- 真实拆出 300 条完整 password 尝试：
  - 8 键：200
  - 9 键：50
  - 10 键：50
- 和 15 条扩展 mixed 训练记录一起训练，共 315 条训练记录

结果：

- 服务器训练目录：`results/stage1_dense_labeling_v12_passwordaug_dev4_20260324`
- 在小验证集上明显变差
- 说明 **“纯净 password 直接混入当前 dense 主线” 这条接法目前失败**

这条失败结论要保留，避免后面重复走弯路。

### 4. 应该引入纯净负面整段，先做 baseline，再做连续流 adaptation

用户提出的关键新思路：

- 我们之前单字符 / Stage3 的经验是：
  - 先用干净数据训练 baseline
  - 再用更脏、更接近真实场景的数据做 adaptation
- Stage1 也可以照这个逻辑走

具体建议：

- 正样本：
  - `password/len_8`
  - `password/len9`
  - `password/len10`
  - 按完整尝试拆成整段
- 负样本：
  - `data/raw/onset_negative`
  - 尤其包括：
    - `freetyping`
    - `idle`
    - `shake`
    - `trackpad_click`
    - `trackpad_move`

先做一个“纯净整段 password vs 纯净整段负面模式”的 baseline，
再把 mixed 连续流拿来做 adaptation。

这个思路当前尚未完整实现，但已被列为下一条优先主线。

### 5. Stage2 可以作为保底，不要让 Stage1 过度保守

用户明确提出：

- 我们要确保长度覆盖尽量 100%
- 可以多挖一点
- 后面再靠 Stage2 的按键识别、按键数、切窗与分割去纠正

这意味着：

- Stage1 不应该为了“候选少”而过度收紧，导致真段漏掉。
- 在规则和 rerank 设计上，应始终优先保证召回。

## 当前主线状态

### 已落地并保留的主线

- `onset_detection/stage2_segmental/scripts/train_eval_stage1_dense_labeling.py`
  - 当前整段 dense Stage1 主脚本
- `onset_detection/stage2_segmental/scripts/eval_stage1_with_iki_filter.py`
  - 在 dense 候选后追加按键间隔重排

### 当前最好可用结果

- dense 主线 + 段内小峰合并 + 按键间隔重排 + 每条会话只留前 2 个候选
- 结果目录：
  - `results/stage1_with_iki_filter_v8_top2_final_20260324`

当前意义：

- 候选段数明显下降
- 整体质量基本不掉
- `p02` 这类复杂流里，前二已经能变成两段真 password

### 当前剩余问题

- 老 `p01` 第一段 password 仍然不完整
- 纯净 password 直接混 current dense 主线会把模型带偏
- 因此下一条值得认真实现的路线，是：
  - 纯净 password + 纯净 negative 先做整段 baseline
  - 再用 mixed 连续流做 adaptation

## 本文件用途

后续只要继续做 Stage1，默认都应先检查这份记录，避免：

- 又把纯净 password 直接大量混进当前 dense 主线
- 又把所有单密码 / 复杂流全部留成测试不利用
- 又忘了 `onset_negative` 这批纯负面整段数据
- 又把 Stage1 目标从“高召回保住真段”误改成“极度保守只留很少候选”
