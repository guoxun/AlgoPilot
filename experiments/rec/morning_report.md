# AlgoPilot 晨报 · 域[rec]

> 生成时间 2026-11-14 15:24:14 · 主指标 auc(max) · 人工唯一入口 program.md,唯一出口本报告

## 一句话进展
当前最优(gen 2):**auc = 0.636312**,来自 `more-epochs`。共 13 次实验:2 保留 / 5 回退 / 1 失败。

## 指标轨迹(保留与合并事件)

| # | 代 | 想法 | auc | 并入主线 |
|---|---|---|---|---|
| #3 | 0 | feat-cross | 0.626532 | ✓ |
| #11 | 1 | more-epochs | 0.636312 | ✓ |
| #5 | 1 | ⧉ merge: feat-cross | 0.626532 | ✓ |
| #12 | 2 | ⧉ merge: more-epochs | 0.636312 | ✓ |

## 端到端对齐(offline → delivery)

| # | 想法 | 离线 Δ | 交付 Δ | 兑现率 | 结局 |
|---|---|---|---|---|---|
| #3 | feat-cross | +0.0209 | +0.0641 | 3.067 | kept |
| #4 | aggressive-lr | +0.0024 | +0.0103 | 4.319 | rejected_by_delivery |
| #11 | more-epochs | +0.0098 | +0.0312 | 3.195 | kept |

**被交付门禁拦下的离线提升**(组织最贵的教训):
- #4 `aggressive-lr`: 交付护栏 coverage_entropy=0.1636 低于下限 0.55

## 失败教训 Top(避坑库)

- `aggressive-lr` rejected_by_delivery: 交付护栏 coverage_entropy=0.1636 低于下限 0.55
- `cross-finer-buckets` rejected: auc 0.626104 未超过 elite 0.626532+0.001
- `early-stop` rejected: auc 0.626532 未超过 elite 0.626532+0.001
- `pos-weight` failed: 护栏 val_logloss=0.4115 超上限 0.35
- `l2-stronger` rejected: auc 0.626537 未超过 elite 0.626532+0.001

## 已沉淀策略(策略库)

- gen1 `feat-cross`: 开启二阶特征交叉(age×cate、gender×cate、activity×quality 分桶积) → auc=0.626532
- gen2 `more-epochs`: 训练轮数 200→400 → auc=0.636312
