# domains/search —— 搜索域蓝本(roadmap:L3 后接入)

> 本目录是**待建域**的契约蓝本。AlgoPilot 的核心主张是:接入新域 = 提供一个目录,
> 引擎一行不改。本文件把"一个搜索域需要什么"说清楚,接手者照单施工即可。

## 域定义

搜索排序算法迭代:给定 query 与候选文档,优化排序质量。

## 四文件契约(与 rec 域同构)

| 文件 | 角色 | 内容建议 |
|---|---|---|
| `prepare.py`(冻结) | 数据装置 | 小型开源数据(如 MS MARCO 子集 / 人工标注的 query-doc 相关性集);确定性 seed;输出 `data/train.npz`、`data/val.npz`、`data/manifest.json`(含 data_hash) |
| `model.py`(唯一可改) | 算法 | 粗排/精排打分模型;KNOBS 旋钮区(如 `_use_semantic_vec`、`_bm25_weight`、`_listwise_loss`、`_lr`、`_patience`);提供 `run_pipeline(data_dir)` |
| `judge.py`(冻结) | 评测器 | 主指标 **NDCG@10**(方向 max);护栏:`val_mrr ≥ 基线-δ`、`train_seconds ≤ 预算`、数据指纹;输出 `###JUDGE###` 协议行 |
| `program.md`(人类) | 组织代码 | 搜索域的迭代策略:相关性信号、语义/词法融合、listwise vs pointwise 等 |

## 指标契约草案

```json
{
  "name": "search",
  "metric": "ndcg_at_10",
  "direction": "max",
  "min_improve": 0.002,
  "budget_seconds": 120,
  "guard_limits": {"train_seconds": ["le", 90], "val_mrr": ["ge", 0.30]},
  "editable_files": ["model.py"],
  "frozen_files": ["prepare.py", "judge.py"]
}
```

## 搜广推三域的差异点(设计时注意)

- rec 是**点态**问题(AUC 排序),search 是**列表态**问题(NDCG 按 query 分组)——
  judge 的评测单元从"全体样本"变成"每个 query 一个排序列表",AUC 换 NDCG,棘轮逻辑不变;
- 搜索有强 **query 意图方差**,失败库价值更高(某类 query 上的提升可能在另一类上回退);
- 多样性/新鲜度等业务护栏比推荐更显性,适合作为 guard_limits 示范。
