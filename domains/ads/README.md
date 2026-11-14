# domains/ads —— 广告域蓝本(roadmap:L4 后接入)

> 与 search 同构的待建域蓝本。广告域的特殊性:排序目标不是相关性而是**营收与生态的联合**,
> 指标与护栏的博弈空间最大,也最考验"不可博弈指标"的设计功力。

## 域定义

广告竞价/排序算法迭代:给定请求与候选广告,优化出价与排序。

## 四文件契约(与 rec 域同构)

| 文件 | 角色 | 内容建议 |
|---|---|---|
| `prepare.py`(冻结) | 数据装置 | 竞价日志模拟器(pCTR 模型 + 出价分布 + 预算约束,固定 seed);输出曝光/点击/竞价数据 |
| `model.py`(唯一可改) | 算法 | pCVR/pCTR 校准、eCPM 计算、出价策略;KNOBS(如 `_calib_method`、`_bid_shade`、`_ocpx_weight`) |
| `judge.py`(冻结) | 评测器 | 主指标 **模拟 eCPM**(营收代理,方向 max);护栏:`ctr ≥ 基线-δ`(生态)、`win_rate` 区间、`train_seconds`;离线**竞价仿真**作为"在线"的代理 |
| `program.md`(人类) | 组织代码 | 商业化迭代策略:校准优于出价、预算平滑、探索-利用平衡 |

## 指标契约草案

```json
{
  "name": "ads",
  "metric": "ecpm_sim",
  "direction": "max",
  "min_improve": 0.003,
  "budget_seconds": 120,
  "guard_limits": {"train_seconds": ["le", 90],
                    "ctr_ratio": ["ge", 0.98],
                    "win_rate": ["ge", 0.10]},
  "editable_files": ["model.py"],
  "frozen_files": ["prepare.py", "judge.py"]
}
```

## 为什么广告域放在 L4

- 广告的离线代理指标与线上营收的 gap 在三域中最大(竞价博弈是二阶效应);
- 棘轮循环在代理指标上可能找到"钻空子"的解(如压低 win_rate 换高价值曝光)——
  需要先在 rec/search 域验证护栏体系,再开放给博弈性最强的域;
- 这本身就是一次组织成熟度的考验:**L5 的通用能力 = 换域不改引擎,但护栏设计必须逐域重审。**
