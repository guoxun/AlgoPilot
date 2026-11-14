# -*- coding: utf-8 -*-
"""governance:program.md 受限自迭代 —— L5 的前奏。

理念:组织能力作用于组织自身。autoresearch 里 program.md 是人类唯一入口;
AlgoPilot 走向 L5 的路径是「提案自动化 + 生效人审」:

    台账(教训与兑现率) → 提案器(mock 规则器 / 真 LLM)→ program_proposal.md
    → 红线校验(纯代码,铁面) → 人类 approve → 备份 + 应用 + 台账 program_update 事件

受限的实现是结构性的,不是自觉性的:
  - program.md 的铁律区由 `<!-- IMMUTABLE:RULES --> ... <!-- /IMMUTABLE:RULES -->`
    标记包裹——那是组织的权限结构,任何自动提案不得触碰;
  - 提案形态被限定为「在指定小节标题下追加一行」——可校验、可幂等、不可重写全文;
  - 红线校验是纯代码,永远不交给模型(评审员哲学)。

生效后的 program.md 会塑造下一次 run 的组织行为——**组织代码本身进入台账**,
组织的进化有了自己的历史。
"""
from __future__ import annotations

import difflib
import re
import time
from pathlib import Path

IMMUTABLE_OPEN = "<!-- IMMUTABLE:RULES"
IMMUTABLE_CLOSE = "<!-- /IMMUTABLE:RULES -->"
PROPOSAL_NAME = "program_proposal.md"

# 提案只允许追加到这些小节(白名单;铁律区天然被 IMMUTABLE 排除)
ALLOWED_ANCHORS = ("## 迭代策略建议(研究员必读)", "## 与自动驾驶等级的关系")


def immutable_bounds(text: str) -> tuple[int, int] | None:
    """返回 IMMUTABLE 区间 (start_line, end_line)(0-based,闭区间);无标记返回 None。"""
    lines = text.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        if start is None and line.startswith(IMMUTABLE_OPEN):
            start = i
        elif line.startswith(IMMUTABLE_CLOSE):
            end = i
    if start is not None and end is not None and end > start:
        return (start, end)
    return None


# ---------------------------------------------------------------- 提案器
def rules_proposal(ledger, program_text: str) -> list[dict]:
    """mock 档提案器:确定性规则,从台账统计提炼组织教训。

    规则(全部只追加到非铁律区,幂等——已存在的行不重复):
      R1 低兑现率方向降权:某保留策略兑现率 < 0.3 → 策略建议里标注"兑现率低,谨慎优先"
      R2 高频护栏教训入文:同类失败原因出现 ≥ 2 次 → 策略建议里加避坑行
      R3 交付拦截教训:有 rejected_by_delivery → 加"离线涨交付跌"警示行
    """
    edits: list[dict] = []
    anchor = "## 迭代策略建议(研究员必读)"
    existing = program_text

    # R1:兑现率低的保留策略(alignment 记在台账 kept 记录上)
    for r in ledger.experiments():
        a = r.get("alignment") or {}
        ratio = a.get("ratio")
        if r.get("verdict") == "kept" and isinstance(ratio, (int, float)) and ratio < 0.3:
            line = f"- 兑现率提示:`{r.get('idea_id')}` 离线提升的交付兑现率仅 {ratio}"
            if line not in existing:
                edits.append({"anchor": anchor, "line": line,
                              "why": f"台账显示该方向离线涨交付不涨(ratio={ratio})"})
            existing += "\n" + line  # 幂等检查基准同步推进

    # R2:同类失败 ≥2 次
    reasons: dict[str, int] = {}
    for f in ledger.failures():
        r = (f.get("reason") or "")
        key = r.split(":")[0][:24]
        if key:
            reasons[key] = reasons.get(key, 0) + 1
    for key, n in reasons.items():
        if n >= 2:
            line = f"- 避坑:台账中已有 {n} 次类似失败(「{key}...」),同类想法需先论证差异"
            if line not in existing:
                edits.append({"anchor": anchor, "line": line,
                              "why": f"同类失败 {n} 次,教训固化进组织代码"})
            existing += "\n" + line

    # R3:交付拦截警示
    blocked = [f for f in ledger.failures()
               if f.get("verdict") == "rejected_by_delivery"]
    if blocked and "- 避坑:任何让曝光生态向热门坍缩的改动" not in existing:
        line = ("- 避坑:任何让曝光生态向热门坍缩的改动,即使离线与交付 CTR 双涨,"
                "也会被曝光熵护栏拦下(见台账 rejected_by_delivery)")
        edits.append({"anchor": anchor, "line": line,
                      "why": f"{len(blocked)} 次交付拦截,把生态红线写进策略建议"})
    return edits


def llm_proposal(llm, ledger, program_text: str) -> list[dict]:
    """真 LLM 提案器:读台账摘要产出同构提案(形态受限:仅小节下追加行)。"""
    payload = {
        "ledger_summary": ledger.summary(),
        "strategies": ledger.strategies(),
        "failures": ledger.failures()[-10:],
        "allowed_anchors": list(ALLOWED_ANCHORS),
    }
    out = llm.complete("governance", payload)
    edits = out.get("edits", [])
    clean = []
    for e in edits:
        if e.get("anchor") in ALLOWED_ANCHORS and e.get("line"):
            clean.append({"anchor": e["anchor"], "line": str(e["line"]),
                          "why": str(e.get("why", ""))})
    return clean


# ---------------------------------------------------------------- 校验与应用
def validate(edits: list[dict], program_text: str) -> tuple[list[dict], list[str]]:
    """红线校验(纯代码,铁面):anchor 必须在白名单且不在 IMMUTABLE 区间内;
    行内容不得出现在 IMMUTABLE 区间;幂等跳过已存在的行。"""
    bounds = immutable_bounds(program_text)
    lines = program_text.splitlines()
    ok, rejected = [], []
    for e in edits:
        anchor, line = e.get("anchor", ""), e.get("line", "")
        if anchor not in ALLOWED_ANCHORS:
            rejected.append((e, f"anchor 不在白名单: {anchor}"))
            continue
        if anchor not in program_text:
            rejected.append((e, f"anchor 小节不存在: {anchor}"))
            continue
        anchor_line = next(i for i, l in enumerate(lines) if l.strip() == anchor)
        if bounds and bounds[0] <= anchor_line <= bounds[1]:
            rejected.append((e, "anchor 落在 IMMUTABLE 铁律区(权限结构,禁止自动修改)"))
            continue
        if bounds and any(part.strip().startswith(IMMUTABLE_OPEN)
                          or part.strip().startswith(IMMUTABLE_CLOSE)
                          for part in [line]):
            rejected.append((e, "行内容试图触碰 IMMUTABLE 标记"))
            continue
        if line in program_text:
            rejected.append((e, "幂等:该行已存在,无需重复提案"))
            continue
        ok.append(e)
    return ok, [r[1] for r in rejected]


def apply_edits(edits: list[dict], program_path: Path,
                backup_tag: str = "") -> tuple[str, str]:
    """应用提案:备份 → 在 anchor 小节末尾追加行。返回 (new_text, diff)。"""
    text = program_path.read_text(encoding="utf-8")
    if backup_tag:
        backup = program_path.with_name(program_path.name + backup_tag)
        backup.write_text(text, encoding="utf-8")
    lines = text.splitlines(keepends=True)
    # 从后往前插入,避免行号漂移;每个 anchor 的多行保持顺序插到小节末尾
    by_anchor: dict[str, list[str]] = {}
    for e in edits:
        by_anchor.setdefault(e["anchor"], []).append(e["line"])
    for anchor, new_lines in by_anchor.items():
        anchor_idx = next(i for i, l in enumerate(lines) if l.strip() == anchor)
        insert_at = anchor_idx + 1
        while insert_at < len(lines) and not lines[insert_at].startswith("## "):
            insert_at += 1
        block = "".join(l.rstrip("\n") + "\n" for l in new_lines)
        lines.insert(insert_at, block)
    new_text = "".join(lines)
    diff = "".join(difflib.unified_diff(
        text.splitlines(keepends=True), new_text.splitlines(keepends=True),
        fromfile=f"program.md{backup_tag}", tofile="program.md(new)"))
    program_path.write_text(new_text, encoding="utf-8")
    return new_text, diff


def render_proposal(spec_name: str, edits: list[dict], rejected: list[str],
                    summary: dict) -> str:
    head = [
        f"# program.md 修改提案 · 域[{spec_name}]",
        "",
        f"> 生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')} · "
        f"依据台账:{summary.get('total', 0)} 条记录 / "
        f"{summary.get('n_strategies', 0)} 策略 / {summary.get('n_failures', 0)} 失败",
        "> **生效需要人类 approve。铁律区(IMMUTABLE:RULES)不在提案范围内——"
        "修改权限结构只能由人类手动执行。**",
        "",
    ]
    for i, e in enumerate(edits, 1):
        head += [f"## 修改 {i}(追加到「{e['anchor']}」)",
                 f"- **内容**:{e['line']}",
                 f"- **依据**:{e.get('why', '')}", ""]
    if rejected:
        head += ["## 被红线校验拒绝的条目", ""]
        head += [f"- {r}" for r in rejected] + [""]
    return "\n".join(head)
