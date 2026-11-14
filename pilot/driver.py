# -*- coding: utf-8 -*-
"""RatchetDriver:棘轮主循环 —— AlgoPilot 的心脏。

循环(每一步都有台账记录,决策可重建):

    观察(elite+失败库+策略库) → 研究员出假设 → 工程师写 patch
    → 评审员护栏审查 → sandbox 跑 judge(固定预算,超时即死)
    → 棘轮判定:有效提升=保留为新 elite;否则回退 + 记入失败库
    → 重复,直到预算/平台期/研究员举手

三条不可博弈的铁律(继承 autoresearch,见调研报告):
  1. 固定预算:judge 超过 budget_seconds 判 failed,不跟实验争论;
  2. 单一主指标:提升判定只看 spec.metric,护栏超标一票否决;
  3. 棘轮只进不退:elite 永远是历史最好,失败自动回退。
"""
from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .domain import DELIVERY_MARKER, JUDGE_MARKER, DomainSpec
from .memory import (VERDICT_FAILED, VERDICT_KEPT, VERDICT_REJECTED,
                     VERDICT_REJECTED_BY_DELIVERY, Ledger)
from .org import Engineer, Researcher, Reviewer, apply_set_knobs

SANDBOX_TIMEOUT_BUFFER = 30  # 秒,judge 预算之外的启动/导入余量


class RatchetDriver:
    def __init__(self, spec: DomainSpec, llm, workdir: Path,
                 stop_on_plateau: int = 8, elite_recheck_every: int = 8,
                 workers: int = 1):
        self.spec = spec
        self.llm = llm
        self.researcher = Researcher(llm)
        self.engineer = Engineer(llm)
        self.reviewer = Reviewer()
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(self.workdir)
        self.stop_on_plateau = stop_on_plateau
        self.elite_recheck_every = elite_recheck_every
        self.workers = max(1, int(workers))  # L4 多分支并行:批内并行实验,批末串行结算
        self.min_delta = spec.min_improve  # warmup 后按基线方差自适应上调
        self.delivery = spec.delivery      # 交付门禁(可能为 None:域未接交付仿真)
        self.delivery_slack = 0.0          # 交付判定容差(warmup 后按仿真方差上调)

    # ------------------------------------------------------------------ 装置
    def setup(self) -> None:
        """启动体检 + 数据准备 + baseline 方差测量。任何问题 fails loud。"""
        problems = self.spec.validate()
        if problems:
            raise RuntimeError("域契约体检失败:\n  " + "\n  ".join(problems))
        self._prepare_if_needed()
        self._init_elite_from_domain()
        self._baseline_warmup()

    def _prepare_if_needed(self) -> None:
        data_dir = self.spec.root / "data"
        if not (data_dir / "val.npz").exists():
            print(f"[pilot] 首次运行,准备数据(装置冻结,仅此一次)...")
            self._subprocess([sys.executable, "prepare.py"], cwd=self.spec.root,
                             timeout=600)

    def _init_elite_from_domain(self) -> None:
        elite_file = self.workdir / "elite" / "model.py"
        if not elite_file.exists():
            elite_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.spec.root / "model.py", elite_file)
            if not self.ledger.load_elite():
                code = elite_file.read_text(encoding="utf-8")
                self.ledger.append({
                    "verdict": "baseline", "gen": 0, "idea_id": "init",
                    "idea": "域初始基线(人工编写的 model.py)",
                    "patch": {"action": "init"},
                })
                self.ledger.save_elite({
                    "elite_id": 1, "idea_id": "init", "gen": 0,
                    "metrics": None, "snapshot": str(elite_file),
                })

    def _baseline_warmup(self) -> None:
        """同一个 elite 代码跑 2 次 judge + 1 次交付仿真:
        离线方差定「有效提升」阈值,交付基线定「不回退」基准。"""
        if self.ledger.load_elite().get("metrics") is not None:
            self._ensure_delivery_baseline()  # 断点续跑:补齐交付基线(兼容旧台账)
            return
        scores, last_state = [], None
        for i in range(2):
            result = self._run_script(
                "judge.py", JUDGE_MARKER,
                (self.workdir / "elite" / "model.py").read_text(encoding="utf-8"))
            scores.append(result["metrics"][self.spec.metric])
            last_state = result.get("model_state")
        std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        self.min_delta = max(self.spec.min_improve, 3.0 * std)
        delivery_metrics = None
        if self.delivery is not None and last_state:
            d_result = self._run_delivery(
                last_state,
                (self.workdir / "elite" / "model.py").read_text(encoding="utf-8"))
            if "error" not in d_result:
                delivery_metrics = d_result["metrics"]
        self.ledger.append({
            "verdict": "baseline", "gen": 0, "idea_id": "warmup",
            "idea": "baseline 方差测量(warmup x2)" +
                    (" + 交付基线" if delivery_metrics else ""),
            "result": {"scores": scores, "std": std},
            "delivery": delivery_metrics,
            "decision": f"min_delta={self.min_delta:.6g}",
        })
        self.ledger.save_elite({**self.ledger.load_elite(),
                                "metrics": {self.spec.metric: scores[0]},
                                "delivery_metrics": delivery_metrics})
        print(f"[pilot] warmup 完成: {self.spec.metric}={scores[0]:.6f} "
              f"std={std:.2e} → 有效提升阈值 min_delta={self.min_delta:.6g}")
        if delivery_metrics:
            print(f"[pilot] 交付基线: {self.delivery.metric}="
                  f"{delivery_metrics[self.delivery.metric]:.6f} (端到端仿真, seed 固定)")

    def _ensure_delivery_baseline(self) -> None:
        """旧台账续跑时补测交付基线(确定性仿真,一次即准)。"""
        elite = self.ledger.load_elite()
        if self.delivery is None or elite.get("delivery_metrics"):
            return
        result = self._run_script(
            "judge.py", JUDGE_MARKER,
            (self.workdir / "elite" / "model.py").read_text(encoding="utf-8"))
        if "error" in result or not result.get("model_state"):
            return
        d_result = self._run_delivery(
            result["model_state"],
            (self.workdir / "elite" / "model.py").read_text(encoding="utf-8"))
        if "error" not in d_result:
            self.ledger.save_elite({**elite, "delivery_metrics": d_result["metrics"]})
            print(f"[pilot] 补测交付基线: {self.delivery.metric}="
                  f"{d_result['metrics'][self.delivery.metric]:.6f}")

    # ------------------------------------------------------------------ 主循环
    def run(self, max_experiments: int = 20) -> dict:
        """批式循环:批内 workers 路并行实验(各自独立 sandbox),批末按 gen 序串行结算。

        并发正确性来自结算语义:批内结果依次过棘轮+交付门禁,基准是结算时点的
        elite——多个 kept 天然叠加成链,不存在写冲突;台账 append 串行,顺序即事实。
        """
        plateau = 0
        kept_since_recheck = 0
        done = 0
        while done < max_experiments:
            batch = min(self.workers, max_experiments - done)

            # 1) 批内并行提议:研究员出假设 → 工程师补丁 → 评审员护栏(批内 idea 去重)
            fork_state = self.ledger.load_elite()          # 分叉快照:整批的判定基准
            fork_metric = fork_state["metrics"][self.spec.metric]
            fork_delivery = fork_state.get("delivery_metrics")
            obs = self._observe()
            jobs, seen = [], set(obs["tried"])
            for w in range(batch):
                obs["tried"] = sorted(seen)  # 批内互斥:前面同事拿走的方向不再重复
                try:
                    hyp = self.researcher.propose(obs)
                except Exception as e:  # noqa: BLE001
                    self._log(VERDICT_FAILED, None, None, {"stage": "research"},
                              f"研究员异常: {e}")
                    raise
                if hyp is None:
                    break
                seen.add(hyp["idea_id"])
                job = self._prepare_job(hyp, tag=f"w{w}")
                if job is not None:
                    jobs.append(job)
                else:
                    done += 1  # 工程师/评审阶段折损也占预算

            if not jobs:
                if done < max_experiments:
                    print("[pilot] 研究员停止:方向已遍历/主动举手")
                break
            if self.workers > 1 and len(jobs) > 1:
                print(f"[pilot] ▶ 并行批次:{len(jobs)} 路实验 "
                      f"<{', '.join(j['hyp']['idea_id'] for j in jobs)}>")

            # 2) 并行执行 judge(子进程 IO,线程池足够,零新依赖)
            if len(jobs) > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=min(len(jobs), self.workers)) as pool:
                    futures = [pool.submit(self._run_script, "judge.py", JUDGE_MARKER,
                                           j["new_code"], None, j["tag"]) for j in jobs]
                    for j, fut in zip(jobs, futures):
                        j["result"] = fut.result()
            else:
                jobs[0]["result"] = self._run_script("judge.py", JUDGE_MARKER,
                                                     jobs[0]["new_code"], None, jobs[0]["tag"])

            # 3) 串行结算:分支语义 = 「分叉点结算,最优者合并」
            #    每个分支相对分叉时点的 elite 判定(棘轮+交付门禁);kept 中主指标
            #    最高者合并为新 elite,其余 kept 保留在台账(分支经验)但由一条
            #    append-only 的 merge 事件记录"谁胜出"。单 worker 时分叉基线=
            #    结算基线,行为与串行棘轮完全一致。
            kept_jobs = []
            for j in jobs:
                done += 1
                verdict = self._settle(j, fork_metric, fork_delivery)
                if verdict == VERDICT_KEPT:
                    kept_jobs.append(j)
                    plateau = 0
                    kept_since_recheck += 1
                else:
                    plateau += 1

            if kept_jobs:
                winner = max(kept_jobs,
                             key=lambda j: j["result"]["metrics"][self.spec.metric])
                for j in kept_jobs:
                    j["merged"] = j is winner
                self._promote(winner["rec"], winner["new_code"],
                              winner["result"]["metrics"],
                              winner.get("delivery_metrics"))
                # merge 决策事件(append-only):记录胜者与落选分支,elite 投影
                # 由本事件携带 —— 台账重建时优先读它
                new_gen = winner["rec"]["gen"] + 1
                self.ledger.append({
                    "verdict": "merge", "gen": new_gen,
                    "idea_id": winner["hyp"]["idea_id"],
                    "idea": f"批内合并:{winner['hyp']['idea_id']} 胜出(最优者并入主线)",
                    "winner_id": winner["rec"]["id"],
                    "elite": {"metrics": winner["result"]["metrics"],
                              "delivery_metrics": winner.get("delivery_metrics"),
                              "gen": new_gen},
                    "merged_out": [j["hyp"]["idea_id"] for j in kept_jobs if j is not winner],
                })
                if kept_since_recheck >= self.elite_recheck_every:
                    self._recheck_elite()
                    kept_since_recheck = 0

            # 4) 平台期检测
            if plateau >= self.stop_on_plateau:
                print(f"[pilot] 连续 {plateau} 次无有效提升,触发平台期停止")
                break

        self.ledger.rebuild_elite()
        return self.ledger.summary()

    # ------------------------------------------------------------------ 批内步骤
    def _prepare_job(self, hyp: dict, tag: str) -> dict | None:
        """假设 → 补丁 → 护栏审查。通过则返回待执行 job;折损在此入账并返回 None。"""
        try:
            current_code = (self.workdir / "elite" / "model.py").read_text(encoding="utf-8")
            patch = self.engineer.implement(hyp, current_code)
        except Exception as e:  # noqa: BLE001
            self._log(VERDICT_FAILED, hyp, None, {"stage": "engineer"},
                      f"工程师异常: {e}")
            return None
        ok, reasons = self.reviewer.review(patch, current_code)
        if not ok:
            self._log(VERDICT_FAILED, hyp, patch, {"stage": "review"},
                      "评审拒绝: " + "; ".join(reasons))
            return None
        return {"hyp": hyp, "patch": patch, "new_code": self._materialize(patch, current_code),
                "tag": tag, "result": None}

    def _settle(self, job: dict, fork_metric: float,
                fork_delivery: dict | None) -> str:
        """单个分支的结算:相对【分叉点】做棘轮判定(第一道门)→ 交付门禁(第二道门)。

        基准用分叉点而非结算时点:并行分支从同一快照出发,互不背锅——
        批内谁的最优由 merge 事件裁决,不靠结算顺序。"""
        hyp, patch, result = job["hyp"], job["patch"], job["result"]
        verdict, reason = self._ratchet(result, fork_metric)
        delivery_metrics, alignment = None, None

        # 交付门禁(第二道门:端到端对齐最终交付)
        # 离线棘轮负责走得快,交付门禁负责走得对:
        # 会话仿真里交付指标不回退 + 生态护栏不破,才算真正保留
        if verdict == VERDICT_KEPT and self.delivery is not None:
            d_result = self._run_delivery(result.get("model_state"), job["new_code"],
                                          tag=job["tag"])
            if "error" in d_result:
                verdict, reason = VERDICT_FAILED, "交付仿真失败: " + d_result["error"]
            else:
                delivery_metrics = d_result["metrics"]
                baseline = (fork_delivery or delivery_metrics)[self.delivery.metric]
                ok, d_reason = self.delivery.gate(
                    delivery_metrics, baseline, slack=self.delivery_slack)
                if not ok:
                    verdict = VERDICT_REJECTED_BY_DELIVERY
                    reason = d_reason
                else:
                    reason += f" | 交付 {self.delivery.metric}={baseline:.6f}→{delivery_metrics[self.delivery.metric]:.6f}"
                offline_delta = result["metrics"][self.spec.metric] - fork_metric
                d_delta = delivery_metrics[self.delivery.metric] - baseline
                alignment = {
                    "offline_delta": round(offline_delta, 6),
                    "delivery_delta": round(d_delta, 6),
                    "ratio": round(d_delta / offline_delta, 3) if abs(offline_delta) > 1e-12 else None,
                }

        rec = self._log(verdict, hyp, patch, result, reason,
                        delivery=delivery_metrics, alignment=alignment)
        job["rec"] = rec
        job["delivery_metrics"] = delivery_metrics

        if verdict == VERDICT_KEPT:
            print(f"[pilot] #{rec['id']} 候选保留 ▲ {self.spec.metric} "
                  f"{fork_metric:.6f} → {result['metrics'][self.spec.metric]:.6f}"
                  f"  <{hyp['idea_id']}>" +
                  (f" | 交付 {delivery_metrics[self.delivery.metric]:.4f}"
                   if delivery_metrics else ""))
        elif verdict == VERDICT_REJECTED_BY_DELIVERY:
            print(f"[pilot] #{rec['id']} ⚑ 交付拦截  <{hyp['idea_id']}> {reason[:80]}")
        else:
            print(f"[pilot] #{rec['id']} {'✗ 回退' if verdict == VERDICT_REJECTED else '× 失败'}"
                  f"  <{hyp['idea_id']}> {reason[:80]}")
        return verdict

    # ------------------------------------------------------------------ 步骤
    def _observe(self) -> dict:
        elite = self.ledger.load_elite()
        kept = [r for r in self.ledger.experiments() if r.get("verdict") == VERDICT_KEPT]
        return {
            "domain": self.spec.describe(),
            "primary_metric": self.spec.metric,
            "direction": self.spec.direction,
            "min_delta": self.min_delta,
            "elite": {"gen": elite.get("gen"), "metrics": elite.get("metrics"),
                      "delivery_metrics": elite.get("delivery_metrics"),
                      "idea_path": elite.get("idea_id")},
            "tried": sorted(self.ledger.tried_ideas()),
            "recent_failures": self.ledger.failures()[-10:],
            "strategies": [s.get("idea") for s in self.ledger.strategies()],
            # 端到端对齐历史:研究员要以此为鉴——什么样的离线提升能兑现成交付
            "alignment_history": [r.get("alignment") for r in kept
                                  if r.get("alignment")][-8:],
            "delivery_rejected": [f for f in self.ledger.failures()
                                  if f.get("verdict") == VERDICT_REJECTED_BY_DELIVERY],
            "policy_catalog": self._load_policy_catalog(),
        }

    def _load_policy_catalog(self) -> list:
        p = self.spec.root / "mock_policy.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

    def _materialize(self, patch: dict, current_code: str) -> str:
        if patch["action"] == "set_knobs":
            return apply_set_knobs(current_code, patch["knobs"])
        return patch["code"]

    def _run_script(self, filename: str, marker: str, code: str,
                    extra_env: dict | None = None, tag: str = "w0") -> dict:
        """在一次性 sandbox 目录跑装置脚本(judge.py / delivery.py):
        实验永远不污染域目录;协议 = stdout 最后一行 <marker> {json}。

        tag 区分并行的多路实验 —— 每路独立子目录,互不污染。"""
        run_dir = self.workdir / "sandbox" / tag
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        for f in list(self.spec.editable_files) + list(self.spec.frozen_files):
            shutil.copy(self.spec.root / f, run_dir / f)
        (run_dir / self.spec.editable_files[0]).write_text(code, encoding="utf-8")

        env = {**__import__("os").environ,
               "ALGOPILOT_DATA_DIR": str(self.spec.root / "data")}
        if extra_env:
            env.update(extra_env)

        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, filename], cwd=run_dir, env=env,
                capture_output=True,
                timeout=self.spec.budget_seconds + SANDBOX_TIMEOUT_BUFFER)
            wall = time.time() - t0
        except subprocess.TimeoutExpired:
            return {"error": f"超时(预算 {self.spec.budget_seconds}s)", "wall_seconds": time.time() - t0}

        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()[-3:]
            return {"error": f"{filename} 异常: {' | '.join(tail)}", "wall_seconds": wall}

        out = proc.stdout.decode("utf-8", errors="replace")
        for line in reversed(out.splitlines()):
            if line.startswith(marker):
                data = json.loads(line[len(marker):].strip())
                data["wall_seconds"] = round(wall, 2)
                return data
        return {"error": f"{filename} 未输出 {marker} 协议行", "wall_seconds": wall}

    def _run_delivery(self, model_state: dict | None, code: str,
                      tag: str = "w0") -> dict:
        """端到端交付仿真:用 judge 产出的 model_state(与离线评测逐位同一模型)。

        code 必须是产出该 model_state 的那份 model.py —— delivery import 它的
        build_csr,特征空间必须与权重一致。
        """
        if not model_state:
            return {"error": "judge 未产出 model_state,无法交付仿真"}
        state_path = self.workdir / "model_state.json"
        state_path.write_text(json.dumps(model_state), encoding="utf-8")
        return self._run_script(self.delivery.file, DELIVERY_MARKER, code,
                                extra_env={"ALGOPILOT_MODEL_STATE": str(state_path)},
                                tag=tag)

    def _ratchet(self, result: dict, elite_metric: float) -> tuple[str, str]:
        """棘轮:护栏一票否决 → 主指标判定。返回 (verdict, reason)。

        护栏源 = metrics ∪ guards:任何上报的指标都可以在 domain.json 里声明为护栏
        (如 val_logloss 既是指标又是下限护栏),guards 同名键优先。
        """
        if "error" in result:
            return VERDICT_FAILED, result["error"]
        guards = {**result.get("metrics", {}), **result.get("guards", {})}
        bad = self.spec.guard_violations(guards)
        if bad:
            return VERDICT_FAILED, "; ".join(bad)
        metric = result["metrics"][self.spec.metric]
        improved = metric > elite_metric + self.min_delta if self.spec.direction == "max" \
            else metric < elite_metric - self.min_delta
        if improved:
            return VERDICT_KEPT, f"{self.spec.metric} {elite_metric:.6f} → {metric:.6f}"
        return VERDICT_REJECTED, f"{self.spec.metric} {metric:.6f} 未超过 elite {elite_metric:.6f}+{self.min_delta:.2g}"

    def _promote(self, rec: dict, new_code: str, metrics: dict,
                 delivery_metrics: dict | None = None) -> None:
        elite_dir = self.workdir / "elite"
        (elite_dir / "model.py").write_text(new_code, encoding="utf-8")
        gen = rec["gen"] + 1
        state = {**self.ledger.load_elite(),
                 "elite_id": rec["id"], "idea_id": rec["idea_id"],
                 "gen": gen, "metrics": metrics,
                 "snapshot": str(elite_dir / "model.py")}
        if delivery_metrics is not None:
            state["delivery_metrics"] = delivery_metrics
        self.ledger.save_elite(state)
        self.ledger.add_strategy({"idea_id": rec["idea_id"], "idea": rec["idea"],
                                  "gen": gen, "metrics": metrics,
                                  "delivery": delivery_metrics})

    def _recheck_elite(self) -> None:
        """复验 elite:防指标漂移/数据泄漏的微观版本 —— 好实验必须可复现。"""
        code = (self.workdir / "elite" / "model.py").read_text(encoding="utf-8")
        result = self._run_script("judge.py", JUDGE_MARKER, code)
        elite = self.ledger.load_elite()
        recorded = elite["metrics"][self.spec.metric]
        if "error" in result:
            self.ledger.append({"verdict": "recheck", "gen": elite["gen"],
                                "idea_id": "elite_recheck", "idea": "elite 复验",
                                "result": {"error": result["error"]}})
            return
        now = result["metrics"][self.spec.metric]
        drifted = abs(now - recorded) > 2 * self.min_delta
        self.ledger.append({
            "verdict": "recheck", "gen": elite["gen"], "idea_id": "elite_recheck",
            "idea": "elite 复验(可复现性)",
            "result": {"recorded": recorded, "now": now},
            "decision": "漂移告警" if drifted else "通过",
        })
        if drifted:
            print(f"[pilot] ⚠ elite 复验漂移: 记录 {recorded:.6f} vs 复测 {now:.6f}")

    def _log(self, verdict: str, hyp: dict | None, patch: dict | None,
             result: dict, reason: str = "",
             delivery: dict | None = None, alignment: dict | None = None) -> dict:
        elite = self.ledger.load_elite()
        slim = {k: v for k, v in (result or {}).items() if k != "model_state"}
        record = {
            "verdict": verdict,
            "gen": elite.get("gen", 0),
            "idea_id": (hyp or {}).get("idea_id"),
            "idea": (hyp or {}).get("idea"),
            "rationale": (hyp or {}).get("rationale"),
            "patch": (patch or {}).get("action") if patch else None,
            "knobs": (patch or {}).get("knobs") if (patch or {}).get("action") == "set_knobs" else None,
            "result": slim if verdict in (VERDICT_KEPT, VERDICT_REJECTED,
                                          VERDICT_REJECTED_BY_DELIVERY) else
                     {"metrics": slim.get("metrics"),
                      "error": slim.get("error", reason)},
            "delivery": delivery,
            "alignment": alignment,
            "reason": reason,
        }
        rec = self.ledger.append(record)
        if verdict != VERDICT_KEPT:  # 策略库在 _promote 里写
            self.ledger.add_failure({"idea_id": rec["idea_id"], "idea": rec["idea"],
                                     "verdict": verdict, "reason": reason,
                                     "alignment": alignment})
        return rec

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _subprocess(cmd, cwd: Path, timeout: int) -> None:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=timeout)
        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", errors="replace").strip()[-500:]
            raise RuntimeError(f"子进程失败({cmd[-1]}): {tail}")
