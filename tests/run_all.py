# -*- coding: utf-8 -*-
"""AlgoPilot 测试套件(零依赖,直接 python tests/run_all.py)。

延续 AgenticRec2.0 的测试传统:不用 pytest,一个文件跑全部,
覆盖:域契约 / 台账 / 棘轮判定 / 评审护栏 / 补丁应用 / LLM解析 / 装置端到端 / 组织全循环。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pilot.domain import DeliverySpec, DomainSpec, JUDGE_MARKER  # noqa: E402
from pilot.llm import MockLLM, _extract_json  # noqa: E402
from pilot.memory import (Ledger, VERDICT_FAILED, VERDICT_KEPT,
                          VERDICT_REJECTED, VERDICT_REJECTED_BY_DELIVERY)  # noqa: E402
from pilot.org import Reviewer, apply_set_knobs, declared_knobs  # noqa: E402
from pilot.driver import RatchetDriver  # noqa: E402
from pilot.report import render_report  # noqa: E402

REC = ROOT / "domains" / "rec"
RESULTS = []


# ---------------------------------------------------------------- 工具
def ensure_data() -> Path:
    data = REC / "data"
    if not (data / "val.npz").exists():
        subprocess.run([sys.executable, "prepare.py"], cwd=REC, check=True,
                       capture_output=True)
    return data


def expect(name: str, cond: bool, detail: str = ""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  {'✓' if cond else '✗ FAIL'} {name}" + (f"  [{detail}]" if detail and not cond else ""))


# ---------------------------------------------------------------- 用例
def test_domain_spec():
    print("[1] 域契约 DomainSpec")
    spec = DomainSpec.load(REC)
    expect("契约加载", spec.name == "rec" and spec.metric == "auc")
    expect("体检通过", spec.validate() == [])
    expect("方向判定 max", spec.better(0.7, 0.69) and not spec.better(0.69, 0.7))
    bad = spec.guard_violations({"train_seconds": 999})
    expect("护栏超标可检出", len(bad) >= 1 and "train_seconds" in bad[0],
           str(bad))
    expect("护栏缺报可检出", "未上报" in spec.guard_violations({})[0])
    expect("交付门禁契约", spec.delivery is not None
           and spec.delivery.metric == "session_ctr"
           and "delivery.py" in spec.frozen_files)

    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "domain.json").write_text("{}", encoding="utf-8")
        spec2 = DomainSpec.load(td)
        expect("缺文件 fails loud", len(spec2.validate()) > 0)
        expect("无交付域可运行", spec2.delivery is None)


def test_delivery_gate():
    print("[2] 交付门禁 DeliverySpec.gate")
    d = DeliverySpec(metric="session_ctr", direction="max", min_keep=0.0,
                     guard_limits={"coverage_entropy": ["ge", 0.55],
                                   "cold_user_ctr": ["ge", 0.04]})
    ok, why = d.gate({"session_ctr": 0.15, "coverage_entropy": 0.68,
                      "cold_user_ctr": 0.07}, 0.1488)
    expect("双指标健康=放行", ok, why)
    ok, why = d.gate({"session_ctr": 0.14, "coverage_entropy": 0.68,
                      "cold_user_ctr": 0.07}, 0.1488)
    expect("交付回退=拦截", not ok and "回退" in why)
    ok, why = d.gate({"session_ctr": 0.16, "coverage_entropy": 0.50,
                      "cold_user_ctr": 0.07}, 0.1488)
    expect("曝光熵坍缩=拦截", not ok and "coverage_entropy" in why)
    ok, why = d.gate({"session_ctr": 0.16, "coverage_entropy": 0.68,
                      "cold_user_ctr": 0.03}, 0.1488)
    expect("冷启崩塌=拦截", not ok and "cold_user_ctr" in why)
    ok, why = d.gate({"coverage_entropy": 0.68}, 0.1488)
    expect("交付指标缺报=拦截", not ok and "未上报" in why)
    ok, why = d.gate({"session_ctr": 0.1489, "coverage_entropy": 0.68,
                      "cold_user_ctr": 0.07}, 0.1488, slack=0.001)
    expect("容差内放行", ok, why)


def test_ledger():
    print("[2] 台账 Ledger(日志即真相)")
    with tempfile.TemporaryDirectory() as td:
        led = Ledger(Path(td))
        led.append({"verdict": VERDICT_KEPT, "gen": 1, "idea_id": "a", "idea": "A",
                    "metrics": {"auc": 0.62}})
        led.append({"verdict": VERDICT_REJECTED, "gen": 1, "idea_id": "b", "idea": "B"})
        led.append({"verdict": VERDICT_FAILED, "gen": 1, "idea_id": "c", "idea": "C"})
        expect("追加只增", len(led.experiments()) == 3)
        state = led.rebuild_elite()
        expect("elite=最后kept", state["idea_id"] == "a")
        expect("重建幂等", led.rebuild_elite()["idea_id"] == "a")
        expect("tried_ideas", led.tried_ideas() == {"a", "b", "c"})
        s = led.summary()
        expect("摘要三态计数", s["by_verdict"]["kept"] == 1 and s["by_verdict"]["failed"] == 1)


def test_knobs_and_review():
    print("[3] 补丁应用与评审护栏")
    code = (REC / "model.py").read_text(encoding="utf-8")
    patched = apply_set_knobs(code, {"_use_feature_cross": True, "_lr": 0.5})
    d = declared_knobs(patched)
    expect("knob 值已生效", d["_use_feature_cross"] is True and d["_lr"] == 0.5)
    expect("其余 knob 不动", d["_epochs"] == 200)

    rev = Reviewer()
    ok, why = rev.review({"action": "set_knobs", "knobs": {"_not_exist": 1}}, code)
    expect("拒绝未知键", not ok)
    ok, why = rev.review({"action": "set_knobs", "knobs": {"_epochs": "many"}}, code)
    expect("拒绝类型不符", not ok)
    evil = code.replace("import numpy as np",
                        "import numpy as np\nimport subprocess")
    ok, why = rev.review({"action": "rewrite", "code": evil}, code)
    expect("拒绝危险 import", not ok and "subprocess" in "".join(why))
    mutate = code.replace('KNOBS = {', 'KNOBS = {\n    "_evil": 0,')
    ok, why = rev.review({"action": "rewrite", "code": mutate}, code)
    expect("拒绝增删键", not ok and "键集合" in "".join(why))
    ok, _ = rev.review({"action": "rewrite", "code": code}, code)
    expect("等价 rewrite 放行", ok)


def test_ratchet():
    print("[4] 棘轮判定")
    spec = DomainSpec.load(REC)
    with tempfile.TemporaryDirectory() as td:
        d = RatchetDriver(spec, MockLLM(), Path(td))
        d.min_delta = 0.001
        v, r = d._ratchet({"metrics": {"auc": 0.63},
                           "guards": {"train_seconds": 10, "val_logloss": 0.30}}, 0.62)
        expect("有效提升=kept", v == VERDICT_KEPT)
        v, _ = d._ratchet({"metrics": {"auc": 0.6205},
                           "guards": {"train_seconds": 10, "val_logloss": 0.30}}, 0.62)
        expect("小于阈值=rejected", v == VERDICT_REJECTED)
        v, _ = d._ratchet({"metrics": {"auc": 0.63}, "guards": {"val_logloss": 0.9}}, 0.62)
        expect("护栏超标=failed", v == VERDICT_FAILED)
        v, _ = d._ratchet({"error": "judge 超时"}, 0.62)
        expect("异常=failed", v == VERDICT_FAILED)
        v, _ = d._ratchet({"metrics": {"auc": 0.63}}, 0.62)
        expect("护栏未上报=failed", v == VERDICT_FAILED)


def test_llm_parsing():
    print("[5] LLM 接缝")
    m = MockLLM()
    obs = {"tried": ["feat-cross"], "policy_catalog": [
        {"idea_id": "feat-cross", "idea": "x", "knobs": {}},
        {"idea_id": "buckets", "idea": "y", "knobs": {"_n_act_buckets": 8}}]}
    hyp = m.complete("research", obs)
    expect("mock 跳过已试策略", hyp["idea_id"] == "buckets")
    obs2 = {"tried": ["feat-cross", "buckets"], "policy_catalog": obs["policy_catalog"]}
    expect("遍历完毕举手", m.complete("research", obs2).get("stop") is True)
    expect("围栏 JSON 解析", _extract_json('```json\n{"a": 1}\n```') == {"a": 1})
    expect("混排 JSON 解析", _extract_json('前言 {"a": {"b": 2}} 后记') == {"a": {"b": 2}})


def test_device_e2e():
    print("[7] 推荐域装置端到端(judge 协议 + 交付仿真)")
    ensure_data()
    env = {**os.environ, "ALGOPILOT_DATA_DIR": str(REC / "data")}
    proc = subprocess.run([sys.executable, "judge.py"], cwd=REC, env=env,
                          capture_output=True, timeout=180)
    expect("judge 退出码 0", proc.returncode == 0,
           proc.stderr.decode("utf-8", errors="replace")[-200:])
    lines = [l for l in proc.stdout.decode("utf-8", errors="replace").splitlines()
             if l.startswith(JUDGE_MARKER)]
    expect("输出协议行", len(lines) == 1)
    data = json.loads(lines[0][len(JUDGE_MARKER):].strip())
    auc = data["metrics"]["auc"]
    expect(f"基线 AUC 合理({auc:.4f})", 0.55 < auc < 0.72)
    expect("logloss 护栏内", data["metrics"]["val_logloss"] <= 0.35)
    expect("耗时护栏内", data["guards"]["train_seconds"] <= 60)
    expect("数据指纹上报", len(data["guards"]["data_hash"]) == 16)
    expect("model_state 上报", len(data.get("model_state", {}).get("w", [])) > 10)

    # 交付仿真:用 judge 的 model_state,验证协议/确定性/指标区间
    state_file = REC / "model_state_test.json"
    state_file.write_text(json.dumps(data["model_state"]), encoding="utf-8")
    denv = {**env, "ALGOPILOT_MODEL_STATE": str(state_file)}
    outs = []
    for _ in range(2):
        dp = subprocess.run([sys.executable, "delivery.py"], cwd=REC, env=denv,
                            capture_output=True, timeout=180)
        dlines = [l for l in dp.stdout.decode("utf-8", errors="replace").splitlines()
                  if l.startswith("###DELIVERY###")]
        outs.append(json.loads(dlines[-1][len("###DELIVERY### "):]) if dlines else {"error": "no marker"})
    state_file.unlink()
    d = outs[0]
    expect("交付协议行", "error" not in d, str(d))
    if "error" not in d:
        m = d["metrics"]
        expect(f"交付 CTR 高于全局 CTR({m['session_ctr']:.4f})",
               0.11 < m["session_ctr"] < 0.25)  # 排序价值在端到端里兑现
        expect("曝光熵健康", m["coverage_entropy"] > 0.55)
        expect("冷启用户有点击", m["cold_user_ctr"] > 0.02)
        expect("交付仿真确定性", outs[1]["metrics"] == m)


def test_full_loop():
    print("[8] 组织全循环(MockLLM 真跑 4 轮,含 sandbox + 交付门禁)")
    ensure_data()
    spec = DomainSpec.load(REC)
    with tempfile.TemporaryDirectory() as td:
        driver = RatchetDriver(spec, MockLLM(), Path(td) / "exp",
                               stop_on_plateau=8, elite_recheck_every=8)
        driver.setup()
        summary = driver.run(4)
        led = driver.ledger
        exps = led.experiments()
        expect("台账含 baseline+warmup", any(r["idea_id"] == "init" for r in exps)
               and any(r["idea_id"] == "warmup" for r in exps))
        expect("4 次实验入账", summary["total"] == len(exps)
               and summary["by_verdict"]["baseline"] == 2)
        kept = [r for r in exps if r["verdict"] == VERDICT_KEPT]
        expect("至少一次保留(feat-cross 必涨)", len(kept) >= 1)
        expect("每次保留都过交付门禁", all(
            r.get("delivery") is not None and r.get("alignment") is not None
            for r in kept))
        elite = led.load_elite()
        expect("elite 代数前进", elite["gen"] >= 1)
        expect("elite 存交付基线", elite.get("delivery_metrics") is not None)
        expect("elite 快照存在", (Path(td) / "exp" / "elite" / "model.py").exists())
        elite_code = (Path(td) / "exp" / "elite" / "model.py").read_text(encoding="utf-8")
        expect("elite 代码含保留的 knob", "_use_feature_cross\": True" in elite_code
               or '_use_feature_cross": True' in elite_code)
        report = render_report(spec, led)
        expect("晨报含关键节", "指标轨迹" in report and "失败教训" in report)
        # 可复现性:独立重跑 warmup,基线 AUC 应与首次完全一致(确定性装置)
        warm_auc = next(r for r in exps if r["idea_id"] == "warmup")["result"]["scores"][0]
        with tempfile.TemporaryDirectory() as td2:
            d2 = RatchetDriver(spec, MockLLM(), Path(td2) / "exp")
            d2.setup()
            auc_re = d2.ledger.load_elite()["metrics"]["auc"]
            expect(f"装置确定性(warmup {warm_auc:.4f} vs 复跑 {auc_re:.4f})",
                   abs(warm_auc - auc_re) < 1e-9)


def test_parallel_loop():
    print("[9] L4 多分支并行(2 路 worker,分叉点结算、最优者合并)")
    ensure_data()
    spec = DomainSpec.load(REC)
    with tempfile.TemporaryDirectory() as td:
        driver = RatchetDriver(spec, MockLLM(), Path(td) / "exp",
                               stop_on_plateau=8, elite_recheck_every=8, workers=2)
        driver.setup()
        summary = driver.run(8)
        exps = driver.ledger.experiments()
        idea_ids = [r.get("idea_id") for r in exps if r.get("verdict") != "merge"]
        expect("无重复实验 idea(merge 决策事件除外)",
               len(idea_ids) == len(set(idea_ids)), str(idea_ids))
        gens = [r.get("gen", 0) for r in exps]
        expect("gen 序单调不降", gens == sorted(gens), str(gens))
        kept = [r for r in exps if r["verdict"] == VERDICT_KEPT]
        expect("并行多分支出现多个 kept(分叉点结算)", len(kept) >= 2, str(summary))
        blocked = [r for r in exps if r["verdict"] == VERDICT_REJECTED_BY_DELIVERY]
        expect("交付门禁在并行路径自然拦截", len(blocked) >= 1,
               f"verdict 分布 {summary['by_verdict']}")
        merges = [r for r in exps if r.get("verdict") == "merge"]
        expect("merge 决策事件入账", len(merges) >= 1)
        expect("kept 全部带交付与兑现率", all(
            r.get("delivery") and r.get("alignment") for r in kept))
        elite = driver.ledger.load_elite()
        expect("elite = 批内最优(最优者合并)", elite["metrics"]["auc"] == max(
            (r["result"]["metrics"]["auc"] for r in kept), default=0))
        report = render_report(spec, driver.ledger)
        expect("晨报对齐节可渲染", "端到端对齐" in report and "merge" in report)


def test_governance():
    print("[10] program.md 受限自迭代(提案→红线校验→审批生效)")
    from pilot import governance
    program = (REC / "program.md").read_text(encoding="utf-8")
    bounds = governance.immutable_bounds(program)
    expect("IMMUTABLE 区间识别", bounds is not None and bounds[1] > bounds[0])

    with tempfile.TemporaryDirectory() as td:
        led = Ledger(Path(td))
        led.append({"verdict": VERDICT_KEPT, "gen": 1, "idea_id": "weak-dir",
                    "idea": "低兑现率方向",
                    "result": {"metrics": {"auc": 0.62}},
                    "delivery": {"session_ctr": 0.15},
                    "alignment": {"offline_delta": 0.01, "delivery_delta": 0.0005,
                                  "ratio": 0.05}})
        led.add_strategy({"idea_id": "weak-dir", "idea": "低兑现率方向", "gen": 1,
                          "metrics": {"auc": 0.62}})
        led.add_failure({"idea_id": "x", "verdict": "rejected_by_delivery",
                         "reason": "交付护栏 coverage_entropy=0.16 低于下限 0.55"})
        edits = governance.rules_proposal(led, program)
        expect("mock 规则器产出提案(R1 兑现率必出;R2/R3 若已固化进组织代码则幂等跳过)",
               len(edits) >= 1 and any("兑现率提示" in e["line"] for e in edits),
               str([e["line"] for e in edits]))

        ok, rejected = governance.validate(edits, program)
        expect("红线校验放行合法条目", len(ok) >= 1, str(rejected))
        expect("红线校验拒绝越权 anchor", governance.validate(
            [{"anchor": "## 铁律(不可谈判)", "line": "- 改铁律"}], program)[1])

        # 应用:备份 + 小节末尾追加 + diff;应用后再次提案应被幂等拒绝
        target = Path(td) / "program.md"
        target.write_text(program, encoding="utf-8")
        _, diff = governance.apply_edits(ok, target, backup_tag=".gen9")
        new_text = target.read_text(encoding="utf-8")
        expect("备份已生成", (Path(td) / "program.md.gen9").exists())
        expect("追加行进入小节", ok[0]["line"] in new_text)
        expect("铁律区未被触碰", bounds is None or
               program.splitlines()[bounds[0]:bounds[1] + 1] ==
               new_text.splitlines()[bounds[0]:bounds[1] + 1])
        expect("diff 非空且为追加", diff.startswith("---") and "+ " in diff)
        _, rejected2 = governance.validate(edits, new_text)
        expect("已生效条目幂等拒绝(组织不重复道歉)",
               len(rejected2) == len(edits) and all("已存在" in r for r in rejected2),
               str(rejected2))


def test_real_data_adapter():
    print("[11] L3.5 真实数据适配器(data_raw/ CSV → 装置)")
    import csv as _csv
    import shutil as _shutil
    with tempfile.TemporaryDirectory() as td:
        dom = Path(td) / "rec"
        dom.mkdir()
        for f in ("prepare.py", "judge.py", "delivery.py", "model.py", "domain.json"):
            _shutil.copy(REC / f, dom / f)
        raw = dom / "data_raw"
        raw.mkdir()
        for name, n in (("train.csv", 300), ("val.csv", 80)):
            with open(raw / name, "w", encoding="utf-8", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["label", "num_0", "num_1", "num_2", "num_3", "cat_a", "cat_b", "cat_c"])
                rng = np.random.default_rng(1)
                for i in range(n):
                    a, b, c = float(rng.random()), float(rng.random()), float(rng.random())
                    p = 1 / (1 + np.exp(-(3 * a - 2 * b + 1.5 * a * b - 3.0)))
                    w.writerow([int(rng.random() < p), round(a, 3), round(b, 3),
                                round(c, 3), "", f"s{int(rng.random()*7)}",
                                f"m{int(rng.random()*3)}", f"g{int(rng.random()*11)}"])
        env = {**os.environ, "ALGOPILOT_DATA_DIR": str(dom / "data")}
        p = subprocess.run([sys.executable, "prepare.py"], cwd=dom, env=env,
                           capture_output=True, timeout=180)
        expect("prepare 真实路径退出码 0", p.returncode == 0,
               p.stderr.decode("utf-8", errors="replace")[-200:])
        manifest = json.loads((dom / "data" / "manifest.json").read_text(encoding="utf-8"))
        expect("manifest 标记 kind=real", manifest.get("kind") == "real")
        expect("cat_card 来自真实基数", len(manifest.get("cat_card", [])) == 3)

        j = subprocess.run([sys.executable, "judge.py"], cwd=dom, env=env,
                           capture_output=True, timeout=180)
        lines = [l for l in j.stdout.decode("utf-8", errors="replace").splitlines()
                 if l.startswith(JUDGE_MARKER)]
        expect("真实数据 judge 跑通", j.returncode == 0 and len(lines) == 1,
               j.stderr.decode("utf-8", errors="replace")[-200:])
        if lines:
            d = json.loads(lines[0][len(JUDGE_MARKER):].strip())
            expect(f"真实数据 AUC 合理({d['metrics']['auc']:.3f})",
                   0.4 < d["metrics"]["auc"] < 0.95)
            expect("model_state 与自适应特征空间一致",
                   len(d.get("model_state", {}).get("w", [])) == d["guards"]["n_features"])


# ---------------------------------------------------------------- 主入口
def main() -> int:
    print(f"AlgoPilot 测试套件 · {len(RESULTS)} 前置清零")
    tests = [test_domain_spec, test_delivery_gate, test_ledger,
             test_knobs_and_review, test_ratchet, test_llm_parsing,
             test_device_e2e, test_full_loop, test_parallel_loop,
             test_governance, test_real_data_adapter]
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001 测试框架自身不吞错
            expect(f"{t.__name__} 未捕获异常", False, repr(e))
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n共 {len(RESULTS)} 项,失败 {len(failed)} 项")
    for name, _, detail in failed:
        print(f"  ✗ {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
