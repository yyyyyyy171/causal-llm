# TableShift 外部验证：预注册（写于跑数之前）

日期：2026-10-18。本文件在**任何 TableShift 结果产生之前**写定，之后不改判据。

## 1. 要回答的问题

1. 因果特征选择（IAMB 求标签的马尔可夫毯）选出的特征，从分布内到分布外的
   性能下降，是否小于普通特征选择（互信息 top-k）？
2. 在因果特征选择里加入语言模型先验（候选排序软约束），是否进一步缩小这个下降？

两个问题都按"**gap 更小**"判定，不按"ID 更高"判定；ID 指标同时报告，防止用
牺牲分布内性能换取 gap。

## 2. 数据与划分

- 来源：TableShift（NeurIPS 2023 D&B）。用官方固定划分，不自己重切。
- 任务（3 个，来自不同领域）：
  - `diabetes_readmission`（医疗）
  - `acsfoodstamps`（社会经济）
  - `acsincome`（社会经济）
- 划分：`train` / `validation` / `id_test` / `ood_test`，由
  `scripts/fetch_tableshift.py` 在独立 tableshift 环境里导出为 CSV。
- 编码、缺失填补、连续特征分箱（5 箱）**只在 train 上拟合**；validation 不参与选择与调参。

## 3. 三种方法与预算

| 方法 | 定义 |
|---|---|
| `mi_topk` | 普通特征选择：train 上逐特征互信息，取前 k |
| `iamb_pure` | 纯数据因果特征选择：IAMB（G 检验，α=0.05）求标签的马尔可夫毯 |
| `iamb_soft` | 同一 IAMB，加入 role 先验作为候选排序加成（γ=2.0，与 bnlearn 主实验一致） |

- 预算：`k = min(15, 特征数 − 1)`，三个方法**选同样多的特征**。
- IAMB 的毯子先按互信息排序：超过 k 截断，不足 k 用下一个最高互信息的特征补齐。
  这样做是为了让"因果方法更好"不能归因于"多给了几个特征"。
- 先验只覆盖 (特征, 标签) 对，不枚举特征两两组合；不确定记 0 分。

## 4. 下游模型与指标

- 下游：`HistGradientBoostingClassifier`，固定超参（max_iter=300, lr=0.06,
  min_samples_leaf=40, l2=0.1），只换随机种子。
- 主指标：AUC；次指标：accuracy、F1（阈值 0.5）。
- 核心量：`gap = AUC(id_test) − AUC(ood_test)`。
- 重复：5 个下游种子（101/202/303/404/505），报告均值与标准差；
  方法间比较用**同种子配对差**。

## 5. 判读规则（写死）

- **Q1（因果 vs 普通）**：若 ≥2/3 任务上 `gap(iamb_pure) < gap(mi_topk)`，
  记为"支持因果特征分布外更稳"；否则记为"未支持"。
- **Q2（先验 vs 纯数据）**：逐任务计算 `Δgap = gap(iamb_soft) − gap(iamb_pure)`。
  若 ≥2/3 任务 Δgap < 0，且至少 1 个任务的配对 95% CI 不含 0，记为"有证据"；
  否则记为"未显示优势"。
- 若 soft 的 gap 变小但 ID AUC 明显下降（>0.01），必须同时报告，不得只报 gap。
- 不预设 soft 更好。结果为负就按负结果写。

## 6. 失败处理

- 任务导出失败、OOD 只有单一类别、IAMB 在预算内无法给出非空毯子 → 记录原因并排除该任务，
  **不替换任务**、不改预算。
- 先验文件缺失 → 只跑前两种方法，结果里标记 LLM 臂缺失，不伪造第三条。

## 7. γ=4.0 敏感性（主实验完成后才跑）

- 只比较 `iamb_soft(γ=2.0)` 与 `iamb_soft(γ=4.0)`，同一批种子、同一预算。
- 预注册判据：若 4.0 在 ≥2/3 任务上 gap 更小，**且** ID AUC 不比 2.0 低超过 0.005，
  记为"更强先验有益"；否则主设置保持 γ=2.0。
- 不允许看到结果后再改这条判据；也不允许因为 2.0 不显著就换 4.0 重报。

## 8. 命令

```bash
# 1) 独立 tableshift 环境里导出三个任务
python scripts/fetch_tableshift.py --dry-run
python scripts/fetch_tableshift.py

# 2) 生成 (特征, 标签) 提问
python scripts/tableshift/make_prior_prompts.py --data data/tableshift

# 3) 8B 打角色分（vLLM 环境）
python scripts/tableshift/score_prior_vllm.py --model Qwen/Qwen3-8B-AWQ --samples 5

# 4) 三种方法 + ID/OOD 评测
python scripts/tableshift/run_tableshift.py --data data/tableshift --out results
```

产物：`results/tableshift_metrics.csv`（逐任务逐种子）、`results/tableshift_summary.csv`（均值）。
