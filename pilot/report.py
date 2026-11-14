# -*- coding: utf-8 -*-
"""晨报:人只读这一份文档(L2→L3 的授权边界——人审上线,不审过程)。"""
from __future__ import annotations

import time
from pathlib import Path

from .domain import DomainSpec
from .memory import Ledger


def render_report(spec: DomainSpec, ledger: Ledger) -> str:
    exps = ledger.experiments()
    elite = ledger.load_elite() or {}
    strategies = ledger.strategies()
    failures = ledger.failures()
    by = {"kept": 0, "rejected": 0, "failed": 0, "baseline": 0, "recheck": 0}
    for r in exps:
        by[r.get("verdict", "?")] = by.get(r.get("verdict", "?"), 0) + 1

    lines = [
        f"# AlgoPilot 晨报 · 域[{spec.name}]",
        f"",
        f"> 生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')} · "
        f"主指标 {spec.metric}({spec.direction}) · 人工唯一入口 program.md,唯一出口本报告",
        "",
        "## 一句话进展",
    ]
    m = (elite.get("metrics") or {}).get(spec.metric)
    if m is not None:
        lines.append(f"当前最优(gen {elite.get('gen')}):**{spec.metric} = {m:.6f}**"
                     f",来自 `{elite.get('idea_id')}`。共 {len(exps)} 次实验:"
                     f"{by['kept']} 保留 / {by['rejected']} 回退 / {by['failed']} 失败。")
    else:
        lines.append(f"尚无有效 elite。共 {len(exps)} 条台账记录。")

    lines += ["", "## 指标轨迹(保留与合并事件)", ""]
    kepts = [r for r in exps if r.get("verdict") == "kept"]
    merges = [r for r in exps if r.get("verdict") == "merge"]
    merged_out_ids = {i for m in merges for i in m.get("merged_out", [])}
    kept_idea_ids = {r.get("idea_id") for r in kepts}
    # merged_out 记录的是 idea_id,与 kept 记录对齐;胜者的 idea 不在 merged_out 里
    if kepts:
        lines.append("| # | 代 | 想法 | " + spec.metric + " | 并入主线 |")
        lines.append("|---|---|---|---|---|")
        for r in kepts:
            merged_flag = "✓" if r.get("idea_id") not in merged_out_ids else "—(分支经验)"
            lines.append(f"| #{r['id']} | {r.get('gen')} | {r.get('idea_id')} | "
                         f"{(r.get('result', {}).get('metrics') or {}).get(spec.metric, '?')} | {merged_flag} |")
        for m in merges:
            lines.append(f"| #{m['id']} | {m.get('gen')} | ⧉ merge: {m.get('idea_id')} | "
                         f"{(m.get('elite', {}).get('metrics') or {}).get(spec.metric, '?')} | ✓ |")
    else:
        lines.append("(暂无保留事件)")

    lines += ["", "## 端到端对齐(offline → delivery)", ""]
    aligned = [r for r in exps if r.get("alignment")]
    if spec.delivery is None:
        lines.append("(本域未接交付仿真,单门禁运行)")
    elif aligned or any(r.get("verdict") == "rejected_by_delivery" for r in exps):
        lines.append("| # | 想法 | 离线 Δ | 交付 Δ | 兑现率 | 结局 |")
        lines.append("|---|---|---|---|---|---|")
        for r in exps:
            a = r.get("alignment")
            if not a:
                continue
            od, dd, rt = a.get("offline_delta"), a.get("delivery_delta"), a.get("ratio")
            lines.append(f"| #{r['id']} | {r.get('idea_id')} | {od:+.4f} | {dd:+.4f} | "
                         f"{rt if rt is not None else '—'} | {r['verdict']} |")
        blocked = [r for r in exps if r.get("verdict") == "rejected_by_delivery"]
        if blocked:
            lines.append("")
            lines.append("**被交付门禁拦下的离线提升**(组织最贵的教训):")
            for r in blocked:
                lines.append(f"- #{r['id']} `{r.get('idea_id')}`: {r.get('reason')}")
    else:
        lines.append(f"(交付门禁已启用:主指标 {spec.delivery.metric},"
                     f"每次保留都经过端到端会话仿真验证)")

    lines += ["", "## 失败教训 Top(避坑库)", ""]
    if failures:
        seen = set()
        for f in failures:
            key = f.get("idea_id")
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- `{key}` {f.get('verdict')}: {(f.get('reason') or '')[:100]}")
            if len(seen) >= 5:
                break
    else:
        lines.append("(暂无)")

    lines += ["", "## 已沉淀策略(策略库)", ""]
    if strategies:
        for s in strategies[-8:]:
            sm = (s.get("metrics") or {}).get(spec.metric)
            lines.append(f"- gen{s.get('gen')} `{s.get('idea_id')}`: {s.get('idea')}"
                         + (f" → {spec.metric}={sm:.6f}" if isinstance(sm, float) else ""))
    else:
        lines.append("(暂无)")

    return "\n".join(lines) + "\n"


def write_report(spec: DomainSpec, workdir: Path) -> Path:
    ledger = Ledger(workdir)
    text = render_report(spec, ledger)
    out = Path(workdir) / "morning_report.md"
    out.write_text(text, encoding="utf-8")
    return out
