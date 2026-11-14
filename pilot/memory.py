# -*- coding: utf-8 -*-
"""Ledger:实验台账 —— 追加式 JSONL 是唯一事实源(继承 AgenticRec2.0「日志即真相」)。

铁律:
  1. 只追加,不修改。任何一次 kept / rejected / failed 决策都对应台账里的一行。
  2. 每行带 gen(代数)与 parent,从台账可以完整重建「当前最优是从哪条路径迭代来的」。
  3. elite 不是独立状态,而是台账的投影:投影存 elite_state.json,可随时由 rebuild() 重建。
  4. 失败实验不是浪费:failed/rejected 同时沉淀进 failure 库,作为研究员的避坑经验。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

VERDICT_KEPT = "kept"        # 双门禁通过(离线提升 + 交付不回退),保留为新 elite
VERDICT_REJECTED = "rejected"  # 跑完但离线无有效提升(或护栏未过)
VERDICT_FAILED = "failed"    # 没跑完:超时/异常/评审拒绝
VERDICT_REJECTED_BY_DELIVERY = "rejected_by_delivery"  # 离线过了,交付门禁拦下 —— 最贵的教训


class Ledger:
    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.exp_path = self.workdir / "experiments.jsonl"
        self.strategy_path = self.workdir / "strategies.jsonl"
        self.failure_path = self.workdir / "failures.jsonl"
        self.elite_path = self.workdir / "elite_state.json"

    # ------------------------------------------------------------------ 写
    def append(self, record: dict) -> dict:
        record = dict(record)
        record.setdefault("id", self._next_id())
        record.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
        with self.exp_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def add_strategy(self, strategy: dict) -> None:
        strategy = dict(strategy)
        strategy.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
        with self.strategy_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(strategy, ensure_ascii=False) + "\n")

    def add_failure(self, failure: dict) -> None:
        failure = dict(failure)
        failure.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
        with self.failure_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(failure, ensure_ascii=False) + "\n")

    def save_elite(self, state: dict) -> None:
        self.elite_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ 读
    def experiments(self) -> list[dict]:
        return self._read_jsonl(self.exp_path)

    def strategies(self) -> list[dict]:
        return self._read_jsonl(self.strategy_path)

    def failures(self) -> list[dict]:
        return self._read_jsonl(self.failure_path)

    def load_elite(self) -> dict | None:
        if not self.elite_path.exists():
            return None
        return json.loads(self.elite_path.read_text(encoding="utf-8"))

    def tried_ideas(self) -> set[str]:
        """研究员用:已经试过的 idea_id(避免重复劳动)。merge 决策事件不算实验。"""
        return {r["idea_id"] for r in self.experiments()
                if r.get("idea_id") and r.get("verdict") != "merge"}

    def _next_id(self) -> int:
        return len(self.experiments()) + 1

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    # ------------------------------------------------------------------ 投影
    def rebuild_elite(self) -> dict | None:
        """从台账重建 elite 投影(elite 快照目录由 driver 维护,这里只重建元信息)。

        重建优先读 merge 决策事件(L4 并行的"最优者合并"自带 elite 投影);
        无 merge 事件的旧台账回退到"最后一次 kept"逻辑。"""
        exps = self.experiments()
        merges = [r for r in exps if r.get("verdict") == "merge"]
        if merges:
            last = merges[-1]
            state = {
                "elite_id": last.get("winner_id"),
                "idea_id": last.get("idea_id"),
                "metrics": (last.get("elite") or {}).get("metrics"),
                "delivery_metrics": (last.get("elite") or {}).get("delivery_metrics"),
                "gen": last.get("gen", 0),
                "snapshot": str(self.workdir / "elite" / "model.py"),
            }
            self.save_elite(state)
            return state
        kept = [r for r in exps if r.get("verdict") == VERDICT_KEPT]
        if not kept:
            return None
        last_kept = kept[-1]  # 棘轮性质:最后一次 kept 就是当前最优
        state = {
            "elite_id": last_kept["id"],
            "idea_id": last_kept.get("idea_id"),
            "metrics": (last_kept.get("result") or {}).get("metrics"),
            "gen": last_kept.get("gen", 0) + 1,  # 该次保留本身产生新一代
            "snapshot": str(self.workdir / "elite" / "model.py"),
        }
        if last_kept.get("delivery"):  # 交付基线同样从台账重建(日志即真相)
            state["delivery_metrics"] = last_kept["delivery"]
        self.save_elite(state)
        return state

    def summary(self) -> dict:
        exps = self.experiments()
        by = {"kept": 0, "rejected": 0, "failed": 0}
        for r in exps:
            by[r.get("verdict", "?")] = by.get(r.get("verdict", "?"), 0) + 1
        elite = self.load_elite()
        metrics = (elite or {}).get("metrics") or {}
        return {
            "total": len(exps),
            "by_verdict": by,
            "elite_metric": metrics.get("auc"),
            "elite_gen": (elite or {}).get("gen", 0),
            "n_strategies": len(self.strategies()),
            "n_failures": len(self.failures()),
        }
