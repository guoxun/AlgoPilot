# -*- coding: utf-8 -*-
"""智能组织:研究员 / 工程师 / 评审员。

设计取舍(对应调研报告 §3.1):
  - 研究员、工程师需要"智能",走 LLM 接缝(Mock=策略遍历器 / OpenAI 兼容真模型);
  - 评审员不需要智能,只需要铁面——纯代码实现,不受任何 LLM 花言巧语影响;
  - MVP 阶段角色是同一循环里的三段逻辑,接口先于并行度:L4 要做的只是让
    三个角色各自拥有独立上下文与并发。
"""
from __future__ import annotations

import re

from .llm import LLMError

# rewrite 模式的危险内容黑名单:agent 只改算法,不碰装置与系统
DANGEROUS_PATTERNS = (
    "os.system", "os.popen", "subprocess", "socket", "__import__",
    "eval(", "exec(", "shutil", "rmdir", "unlink",
)


# ---------------------------------------------------------------------------
class Researcher:
    """读台账 → 产出假设。单一变量原则在 prompt 里,在 Mock 策略库里亦然。"""

    def __init__(self, llm):
        self.llm = llm

    def propose(self, observation: dict) -> dict | None:
        out = self.llm.complete("research", observation)
        if out.get("stop"):
            return None
        if not out.get("idea_id") or not out.get("idea"):
            raise LLMError(f"研究员产出不合规: {out}")
        return out


# ---------------------------------------------------------------------------
class Engineer:
    """假设 → 代码 patch。MVP 支持两种动作:set_knobs(可靠)与 rewrite(进阶)。"""

    def __init__(self, llm):
        self.llm = llm

    def implement(self, hypothesis: dict, current_code: str) -> dict:
        out = self.llm.complete(
            "engineer", {"hypothesis": hypothesis, "current_code": current_code})
        action = out.get("action")
        if action not in ("set_knobs", "rewrite"):
            raise LLMError(f"工程师产出不合规: action={action}")
        return out


# ---------------------------------------------------------------------------
def apply_set_knobs(code: str, knobs: dict) -> str:
    """把 knob 变更应用到 model.py 的 KNOBS 字典(标记区内的 `_key": value,` 行)。

    只改值,不增删键——键集合是域契约的一部分,由 Reviewer 复核。
    """
    new_code = code
    for key, value in knobs.items():
        py_value = repr(value)  # JSON 的 true/false/null 已在 loads 时转为 Python 字面量
        pattern = re.compile(
            r'(^(\s*)"%s":\s*)([^,\n]+)(,?\s*(#.*)?)$' % re.escape(key),
            re.M)
        if not pattern.search(new_code):
            raise LLMError(f"KNOBS 中不存在键: {key}(禁止增删键)")
        new_code = pattern.sub(lambda m: f'{m.group(1)}{py_value}{m.group(4)}',
                               new_code)
    return new_code


def declared_knobs(code: str) -> dict:
    """从 model.py 文本解析 KNOBS 键集合与当前值(带 `_` 前缀的行)。"""
    block = re.search(r"KNOBS\s*=\s*\{(.*?)^\}", code, re.S | re.M)
    if not block:
        return {}
    out = {}
    for m in re.finditer(r'"(_\w+)":\s*([^,\n}]+)', block.group(1)):
        raw = m.group(2).strip()
        try:  # 值也解析出来,Reviewer 做类型一致性检查
            val = json_literal(raw)
        except Exception:  # noqa: BLE001 表达式(如 2**14)跳过类型检查
            val = None
        out[m.group(1)] = val
    return out


def json_literal(raw: str):
    import json as _json
    return _json.loads(raw.replace("True", "true").replace("False", "false")
                       .replace("None", "null"))


# ---------------------------------------------------------------------------
class Reviewer:
    """护栏审查:评审员是代码,不是模型 —— 铁面无私,fails loud。"""

    def __init__(self, max_code_lines_ratio: float = 3.0):
        self.max_code_lines_ratio = max_code_lines_ratio

    def review(self, patch: dict, current_code: str) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        action = patch.get("action")

        if action == "set_knobs":
            declared = declared_knobs(current_code)
            for k, v in (patch.get("knobs") or {}).items():
                if k not in declared:
                    reasons.append(f"禁止增删 KNOBS 键: {k}")
                elif declared[k] is not None and type(declared[k]) is not type(v):
                    reasons.append(f"KNOBS {k} 类型不符: 声明 {type(declared[k]).__name__} "
                                   f"改为 {type(v).__name__}")
        elif action == "rewrite":
            code = patch.get("code", "")
            if not code.strip():
                reasons.append("rewrite 的代码为空")
            else:
                # 键集合不变是 rewrite 的硬约束:换了键 = 换了契约
                old_keys = set(declared_knobs(current_code))
                new_keys = set(declared_knobs(code))
                if old_keys and old_keys != new_keys:
                    reasons.append(f"rewrite 改变了 KNOBS 键集合: "
                                   f"增 {new_keys - old_keys} 减 {old_keys - new_keys}")
                for pat in DANGEROUS_PATTERNS:
                    if pat in code:
                        reasons.append(f"代码包含危险内容: {pat}")
                old_lines, new_lines = len(current_code.splitlines()), len(code.splitlines())
                if new_lines > old_lines * self.max_code_lines_ratio:
                    reasons.append(f"改动规模失控: {old_lines} → {new_lines} 行 "
                                   f"(上限 {self.max_code_lines_ratio}x)")
        else:
            reasons.append(f"未知 action: {action}")
        return (len(reasons) == 0), reasons
