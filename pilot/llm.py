# -*- coding: utf-8 -*-
"""LLM 接缝:MockLLM(零依赖)与 OpenAICompatLLM(DeepSeek / 智谱 GLM 等)双档。

与 AgenticRec 的双档传统一致:Mock 档不是假数据,而是「策略遍历器」——
它让自驱动闭环在没有任何 API key 的机器上完整可验证;真收益等接真 LLM。

统一接口:complete(task, payload) -> dict
  task = "research"  payload = 观察上下文      -> 假设(或 {"stop": true})
  task = "engineer"  payload = 假设+当前代码    -> {"action": "set_knobs"|"rewrite", ...}
  task = "report"    payload = 台账摘要        -> 晨报文字(可选)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path


class LLMError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
class MockLLM:
    """策略遍历器:按域提供的 mock_policy.json 有序尝试,已试过/已失败自动跳过。

    它内部不做任何随机——同样的台账状态,永远产出同样的决策。可复现是底线。
    """

    def complete(self, task: str, payload: dict) -> dict:
        if task == "research":
            return self._research(payload)
        if task == "engineer":
            return {"action": "set_knobs",
                    "knobs": dict(payload["hypothesis"].get("knobs", {}))}
        if task == "report":
            return {"text": ""}
        raise LLMError(f"MockLLM 不支持任务: {task}")

    def _research(self, obs: dict) -> dict:
        tried = set(obs.get("tried", []))
        for policy in obs.get("policy_catalog", []):
            pid = policy.get("idea_id")
            if pid and pid not in tried:
                # 策略叠加语义:elite 已有的 knobs 作为基座,策略只写增量
                return {
                    "idea_id": pid,
                    "idea": policy["idea"],
                    "rationale": policy.get("rationale", ""),
                    "expected_effect": policy.get("expected_effect", ""),
                    "risk": policy.get("risk", ""),
                    "knobs": policy.get("knobs", {}),
                }
        return {"stop": True,
                "reason": "策略目录已遍历完毕(mock 档没有真正的创造力)"}


# ---------------------------------------------------------------------------
class OpenAICompatLLM:
    """OpenAI 兼容 chat/completions(标准库 urllib,不引入依赖)。

    环境变量:
      LLM_API_KEY   必填
      LLM_BASE_URL  默认 https://open.deepseek.com  (智谱 oneapi 亦可)
      LLM_MODEL     默认 deepseek-chat
    """

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: int = 120):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL")
                         or "https://api.deepseek.com").rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "deepseek-chat")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        if not self.api_key:
            raise LLMError("缺少 LLM_API_KEY 环境变量(或改用 --engine mock)")
        self.timeout = timeout

    def complete(self, task: str, payload: dict, system: str | None = None,
                 retries: int = 2) -> dict:
        from .prompts import build_prompt  # 延迟导入避免循环
        sys_prompt, user_prompt = build_prompt(task, payload)
        last_err = None
        for _ in range(retries + 1):
            try:
                text = self._chat(sys_prompt, user_prompt)
                return _extract_json(text)
            except Exception as e:  # noqa: BLE001 网络/解析失败都重试
                last_err = e
        raise LLMError(f"LLM 调用失败(task={task}): {last_err}")

    def _chat(self, system: str, user: str) -> str:
        import json as _json
        import urllib.request
        body = _json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.7,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


def _extract_json(text: str) -> dict:
    """容错解析:剥 ```json 围栏,取第一个平衡的 JSON 对象。"""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise LLMError(f"响应中没有 JSON: {text[:200]}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise LLMError("JSON 不平衡")


def make_llm(engine: str):
    if engine == "mock":
        return MockLLM()
    if engine == "llm":
        return OpenAICompatLLM()
    raise LLMError(f"未知引擎: {engine}(可选 mock / llm)")
