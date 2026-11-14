# -*- coding: utf-8 -*-
"""DomainSpec:域契约 —— AlgoPilot 通用迭代能力与具体算法域之间的唯一接口。

引擎(pilot/)不认识任何算法,它只认识这份契约:
  - 哪个文件允许 agent 改(editable_files),哪些是冻结的实验装置(frozen_files)
  - 评测器怎么跑(judge.py,输出 ###JUDGE### JSON 协议)
  - 主指标是什么、方向是什么、护栏指标的上限是什么

换一个算法域 = 提供一个新目录 + domain.json,引擎一行不改。
这是 L5「通用迭代能力」的架构兑现。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# judge.py 输出协议:stdout 的最后一行必须是这个标记 + 空格 + 单行 JSON
JUDGE_MARKER = "###JUDGE###"
# delivery.py(端到端交付仿真)输出协议
DELIVERY_MARKER = "###DELIVERY###"


@dataclass
class DeliverySpec:
    """交付门禁契约:端到端仿真的主指标 + 生态护栏。

    理念:离线棘轮负责走得快,交付对齐负责走得对。棘轮判 kept 后,
    还要在交付仿真里验证「交付指标不回退 + 护栏不破」,才算真正保留。
    """
    file: str = "delivery.py"
    metric: str = "session_ctr"
    direction: str = "max"
    # 允许的最大交付回退(0 = 严格不回退;仿真确定性下 0 是安全的)
    min_keep: float = 0.0
    guard_limits: dict = field(default_factory=dict)

    def better(self, a: float, b: float) -> bool:
        return a > b if self.direction == "max" else a < b

    def gate(self, metrics: dict, baseline: float, slack: float = 0.0) -> tuple[bool, str]:
        """交付门禁:主指标不回退(容忍 slack,来自 warmup 方差)+ 护栏全过。"""
        val = metrics.get(self.metric)
        if val is None:
            return False, f"交付指标 {self.metric} 未上报"
        ok = self.better(val, baseline - slack) if self.direction == "max" \
            else self.better(val, baseline + slack)
        if not ok:
            return False, f"交付 {self.metric} {val:.6f} 回退(基线 {baseline:.6f})"
        for key, (op, limit) in self.guard_limits.items():
            g = metrics.get(key)
            if g is None:
                return False, f"交付护栏 {key} 未上报"
            if op == "le" and g > limit:
                return False, f"交付护栏 {key}={g:.4g} 超上限 {limit}"
            if op == "ge" and g < limit:
                return False, f"交付护栏 {key}={g:.4g} 低于下限 {limit}"
        return True, ""


@dataclass
class DomainSpec:
    name: str
    root: Path
    # 主指标与方向(写死在契约里,循环中不可被 agent 争论)
    metric: str = "auc"
    direction: str = "max"  # "max" 或 "min"
    # 判定一次「有效提升」的最小幅度下限(实际取 max(它, 3×基线方差))
    min_improve: float = 0.001
    # 单实验预算(墙钟秒)。judge 超过 budget_seconds 即判 failed —— 固定预算铁律
    budget_seconds: int = 90
    # 护栏指标上限/下限:{"train_seconds": ("le", 90), ...}
    guard_limits: dict = field(default_factory=dict)
    # 文件权限:agent 唯一可改 / 冻结装置
    editable_files: tuple = ("model.py",)
    frozen_files: tuple = ("prepare.py", "judge.py")
    program_md: str = "program.md"
    # 交付门禁(端到端对齐最终交付);缺省 = 该域暂无交付仿真,棘轮单门禁
    delivery: DeliverySpec | None = None

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, root: str | Path) -> "DomainSpec":
        root = Path(root).resolve()
        meta_path = root / "domain.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"域目录缺少契约文件: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        kwargs = {k: v for k, v in meta.items() if k != "name"}
        # JSON 里数组是 list,契约内部统一 tuple(保证可拼接、不可变)
        if "editable_files" in kwargs:
            kwargs["editable_files"] = tuple(kwargs["editable_files"])
        if "frozen_files" in kwargs:
            kwargs["frozen_files"] = tuple(kwargs["frozen_files"])
        if "delivery" in kwargs:  # 嵌套契约:交付门禁
            kwargs["delivery"] = DeliverySpec(**kwargs["delivery"])
        return cls(name=meta.get("name", root.name), root=root, **kwargs)

    def validate(self) -> list[str]:
        """启动前体检:缺文件/契约矛盾直接 fails loud(继承 2.0 的 fails loud 传统)。"""
        problems = []
        for f in self.editable_files + self.frozen_files + (self.program_md,):
            if not (self.root / f).exists():
                problems.append(f"缺少文件: {self.root / f}")
        overlap = set(self.editable_files) & set(self.frozen_files)
        if overlap:
            problems.append(f"文件同时出现在可改与冻结清单: {overlap}")
        if self.direction not in ("max", "min"):
            problems.append(f"direction 必须是 max/min, 现在是 {self.direction}")
        if self.delivery is not None and (self.root / self.delivery.file).exists() is False:
            problems.append(f"交付仿真缺失: {self.root / self.delivery.file}")
        return problems

    # ------------------------------------------------------------------
    def better(self, a: float, b: float) -> bool:
        """a 是否严格优于 b(按契约方向)。"""
        return a > b if self.direction == "max" else a < b

    def guard_violations(self, guards: dict) -> list[str]:
        """检查护栏:返回违规描述列表,空列表 = 通过。"""
        bad = []
        for key, (op, limit) in self.guard_limits.items():
            val = guards.get(key)
            if val is None:
                bad.append(f"护栏指标 {key} 未上报")
                continue
            if op == "le" and val > limit:
                bad.append(f"护栏 {key}={val:.4g} 超上限 {limit}")
            elif op == "ge" and val < limit:
                bad.append(f"护栏 {key}={val:.4g} 低于下限 {limit}")
        return bad

    def describe(self) -> str:
        """给 agent 看的一句话契约摘要(写进 prompt 与晨报)。"""
        g = ", ".join(f"{k}{op}{v}" for k, (op, v) in self.guard_limits.items())
        d = ""
        if self.delivery is not None:
            dg = ", ".join(f"{k}{op}{v}" for k, (op, v) in self.delivery.guard_limits.items())
            d = f" | 交付门禁 {self.delivery.metric}({self.delivery.direction}, 护栏 {dg or '无'})"
        return (f"域[{self.name}] 主指标 {self.metric}({self.direction}) | "
                f"可改 {list(self.editable_files)} | 冻结 {list(self.frozen_files)} | "
                f"预算 {self.budget_seconds}s | 护栏 {g or '无'}{d}")
