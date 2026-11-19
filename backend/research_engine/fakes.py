"""LLM_MODE=fake 的确定性替代实现，见 docs/08 §2。

让完整流程无需网络与密钥即可运行，返回可预测且符合数据结构的节点结果。"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage

from research_engine import prompts


def fake_search(query: str, max_results: int) -> list[dict]:
    return [
        {
            "title": f"Fixture source {i} for {query[:40]}",
            "url": f"https://example.com/{re.sub(r'[^a-z0-9]+', '-', query.lower())[:30]}/{i}",
            "snippet": f"Fixture snippet {i}: authoritative-sounding fact about {query[:40]}.",
        }
        for i in range(1, max_results + 1)
    ]


def fake_read_webpage(url: str) -> dict:
    return {
        "url": url,
        "title": f"Fixture page at {url}",
        "text": f"Fixture body for {url}. Contains a citable fact and figures like 42%.",
        "error": None,
    }


class ScriptedScenario(StrEnum):
    """脚本调用所模拟的具体场景。

    场景不等于角色：critic 同时承载评审、矛盾检测、引用核验；synthesizer 承载草稿与引用修复。
    必须区分场景，否则同一角色会得到不符合当前契约的脚本回复。"""

    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"
    CONTRADICTION_DETECTOR = "contradiction_detector"
    CITATION_VERIFY = "citation_verify"
    CLAIM_DECOMPOSE = "claim_decompose"
    CLAIM_VERIFY = "claim_verify"
    CITATION_RELEVANCE = "citation_relevance"
    INFERENCE_VERIFY = "inference_verify"
    SELECTION_REPAIR = "selection_repair"
    SYNTHESIZER = "synthesizer"
    SYNTHESIZER_REPAIR = "synthesizer_repair"
    CHAT = "chat"


#: 按完整提示词常量匹配场景，不依赖手写短语，避免提示词改写后脚本静默失配。
SCENARIO_BY_PROMPT: dict[str, ScriptedScenario] = {
    prompts.PLANNER_PROMPT_V2: ScriptedScenario.PLANNER,
    prompts.EXECUTOR_PROMPT: ScriptedScenario.EXECUTOR,
    prompts.CRITIC_PROMPT_V2: ScriptedScenario.CRITIC,
    prompts.CONTRADICTION_DETECTOR_PROMPT: ScriptedScenario.CONTRADICTION_DETECTOR,
    prompts.CITATION_VERIFY_PROMPT: ScriptedScenario.CITATION_VERIFY,
    prompts.CLAIM_DECOMPOSE_PROMPT: ScriptedScenario.CLAIM_DECOMPOSE,
    prompts.CLAIM_VERIFY_PROMPT: ScriptedScenario.CLAIM_VERIFY,
    prompts.CITATION_RELEVANCE_PROMPT: ScriptedScenario.CITATION_RELEVANCE,
    prompts.INFERENCE_VERIFY_PROMPT: ScriptedScenario.INFERENCE_VERIFY,
    prompts.SELECTION_REPAIR_PROMPT: ScriptedScenario.SELECTION_REPAIR,
    # 与评审提示词一同发送，因此采用评审脚本。
    prompts.SELECTION_CRITIC_ADDENDUM: ScriptedScenario.CRITIC,
    prompts.SYNTHESIZER_PROMPT_V2: ScriptedScenario.SYNTHESIZER,
    prompts.SYNTHESIZER_REPAIR_PROMPT: ScriptedScenario.SYNTHESIZER_REPAIR,
    prompts.CHAT_PROMPT: ScriptedScenario.CHAT,
    prompts.PROJECT_CHAT_PROMPT: ScriptedScenario.CHAT,
}

#: 自定义提示词无法匹配预设常量时，以调用方明确提供的角色作为回退。
SCENARIO_BY_ROLE: dict[str, ScriptedScenario] = {
    "planner": ScriptedScenario.PLANNER,
    "executor": ScriptedScenario.EXECUTOR,
    "critic": ScriptedScenario.CRITIC,
    "synthesizer": ScriptedScenario.SYNTHESIZER,
    "chat": ScriptedScenario.CHAT,
}


class _ScriptedModel(FakeMessagesListChatModel):
    """按明确传入的角色和可识别场景返回脚本回复。

    不再从提示词中的固定短语猜角色，避免改写提示词后返回空 JSON。
    填入模拟用量供费用累加流程使用；结构化调用返回可解析 JSON。"""

    #: 必须显式提供角色，不能用默认值掩盖调用方遗漏。
    role: str

    def __init__(self, role: str) -> None:
        super().__init__(responses=[AIMessage(content="")], role=role)

    def scenario_for(self, system: str) -> ScriptedScenario:
        """优先根据已知提示词选择场景，否则使用角色。

        无法选择时立即报错，不回退空 JSON，避免到后续节点才出现不相关错误。"""
        for prompt, scenario in SCENARIO_BY_PROMPT.items():
            if prompt in system:
                return scenario
        try:
            return SCENARIO_BY_ROLE[self.role]
        except KeyError:
            raise ValueError(
                f"no scripted behaviour for role {self.role!r}: add it to SCENARIO_BY_ROLE "
                "or route the call through a role that has one"
            ) from None

    def bind_tools(self, tools, **kwargs):
        # 脚本执行器通过 submit_evidence 工具调用表示完成，与真实执行器契约一致。
        return self

    def _reply(self, messages: list[BaseMessage]) -> AIMessage:
        system = ""
        human = ""
        for m in messages:
            kind = getattr(m, "type", "")
            if kind == "system":
                system += str(m.content) + "\n"
            elif kind == "human":
                human += str(m.content) + "\n"

        scenario = self.scenario_for(system)

        if scenario is ScriptedScenario.PLANNER:
            content = json.dumps(
                {
                    "tasks": [
                        {"id": 1, "query": "background and definitions", "rationale": "context"},
                        {"id": 2, "query": "current state and data", "rationale": "evidence"},
                    ]
                }
            )
        elif scenario is ScriptedScenario.EXECUTOR:
            # 证据只能由 submit_evidence 提交，脚本也遵守同一契约，见 docs/12 M7。
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_evidence",
                        "args": {
                            "evidence": [
                                {
                                    "task_id": 1,
                                    "source_url": "https://example.com/fixture/1",
                                    "source_title": "Fixture Source 1",
                                    "snippet": "Fixture snippet supporting the claim.",
                                    "key_fact": "A citable fact.",
                                    "retrieved_at": "2026-01-01T00:00:00Z",
                                },
                                {
                                    "task_id": 1,
                                    "source_url": "https://example.com/fixture/2",
                                    "source_title": "Fixture Source 2",
                                    "snippet": "A second independent snippet.",
                                    "key_fact": "A corroborating fact.",
                                    "retrieved_at": "2026-01-01T00:00:00Z",
                                },
                            ]
                        },
                        "id": "fake-submit-evidence-1",
                    }
                ],
                usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            )
        elif scenario is ScriptedScenario.CRITIC:
            content = json.dumps(
                {"passed": True, "confidence": 0.9, "reasons": ["two independent sources"]}
            )
        elif scenario is ScriptedScenario.CONTRADICTION_DETECTOR:
            content = self._contradiction_reply(human)
        elif scenario is ScriptedScenario.SYNTHESIZER_REPAIR:
            # 每条断言都带 [n]，让引用修复一次收敛；标题短于 Claim 长度阈值。
            content = (
                "# Fixture Report\n\n## Summary\nDeterministic summary [1].\n\n"
                "## Findings\nA citable fact [1]. A corroborating fact [2].\n\n"
                "## Analysis\nAnalysis grounded in evidence [1][2].\n\n"
                "## Limitations\nFixture data only.\n\n"
                "## Sources\n[1] https://example.com/fixture/1\n[2] https://example.com/fixture/2\n"
            )
        elif scenario is ScriptedScenario.CLAIM_DECOMPOSE:
            # 每句生成一个覆盖完整句子的 Claim，并保留全部引用编号，让 fake 选择研究走完整校验流程。
            # 脚本判断仅验证流程，质量对抗测试需另外提供 Judge。
            items = []
            for number, sentence in re.findall(r"Sentence (\d+): (.*)", human):
                cites = sorted(
                    {
                        int(p)
                        for m in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", sentence)
                        for p in m.split(",")
                    }
                )
                items.append(
                    {
                        "sentence": int(number),
                        "claims": [
                            {
                                "text": sentence,
                                "span": [0, len(sentence)],
                                "cites": cites,
                                "calculation": None,
                            }
                        ],
                    }
                )
            content = json.dumps(items)
        elif scenario in (ScriptedScenario.CLAIM_VERIFY, ScriptedScenario.INFERENCE_VERIFY):
            n_claims = len(re.findall(r"Claim \d+:", human)) or 1
            content = "\n".join(f"Claim {i}: SUPPORTED" for i in range(1, n_claims + 1))
        elif scenario is ScriptedScenario.SELECTION_REPAIR:
            # 仅复述已核验的 Claim，不添加内容。
            block = human.split("Verified claims in this passage:\n", 1)[-1].split("\n\n", 1)[0]
            kept = [line[2:] for line in block.splitlines() if line.startswith("- ")]
            content = " ".join(kept) or "REMOVE"
        elif scenario is ScriptedScenario.CITATION_RELEVANCE:
            n_pairs = len(re.findall(r"Pair \d+:", human)) or 1
            content = "\n".join(f"Pair {i}: YES" for i in range(1, n_pairs + 1))
        elif scenario is ScriptedScenario.CITATION_VERIFY:
            # 引用核验器逐 Claim 要求 YES/NO；固定数据均有支持，按请求数量返回 YES，保持演示草稿稳定。
            n_claims = len(re.findall(r"Claim \d+:", human)) or 1
            content = "\n".join(f"Claim {i}: YES" for i in range(1, n_claims + 1))
        elif scenario is ScriptedScenario.SYNTHESIZER:
            content = (
                "# Fixture Report\n\n## Executive Summary\nDeterministic summary [1].\n\n"
                "## Key Findings\n- A citable fact [1]\n- A corroborating fact [2]\n\n"
                "## Detailed Analysis\nAnalysis grounded in evidence [1][2].\n\n"
                "## Limitations\nFixture data only.\n\n"
                "## Sources\n[1] https://example.com/fixture/1\n[2] https://example.com/fixture/2\n"
            )
        else:  # ScriptedScenario.CHAT
            content = f"Based on the report, here is a grounded answer to: {human.strip()[:80]}"

        return AIMessage(
            content=content,
            usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )

    @staticmethod
    def _contradiction_reply(human: str) -> str:
        """脚本矛盾检测器，见 docs/12 M11。

        默认固定证据无冲突；测试插入 CONTRADICTION-FIXTURE 时，
        使用实际提供的前两个来源和片段构造冲突，以便通过真实 URL 校验并渲染展示。"""
        if "CONTRADICTION-FIXTURE" not in human:
            return json.dumps({"pairs": []})
        # 每来源一块，与 contradictions.build_detector_input 的输入格式一致。
        blocks = re.findall(
            r"Source: (\S+)\n<untrusted_web_content>\n(.*?)\n</untrusted_web_content>",
            human,
            re.S,
        )
        seen: dict[str, str] = {}
        for url, body in blocks:
            if url not in seen:
                quoted = re.findall(r'- "([^"]*)"', body)
                seen[url] = quoted[0] if quoted else ""
        distinct = [u for u in seen if seen[u]]
        if len(distinct) < 2:
            return json.dumps({"pairs": []})
        a, b = distinct[0], distinct[1]
        return json.dumps(
            {
                "pairs": [
                    {
                        "claim_a": "the measured output was 42 units",
                        "snippet_a": seen[a],
                        "source_a": a,
                        "claim_b": "the measured output was 17 units",
                        "snippet_b": seen[b],
                        "source_b": b,
                        "nature": "The two sources report incompatible values for the same measurement.",
                    }
                ]
            }
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any):
        from langchain_core.outputs import ChatGeneration, ChatResult

        return ChatResult(generations=[ChatGeneration(message=self._reply(messages))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs: Any):
        from langchain_core.outputs import ChatGeneration, ChatResult

        message = await self._reply_async(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _reply_async(self, messages: list[BaseMessage]) -> AIMessage:
        """优先异步回复，只有资料库模式的执行器需要 await。"""
        from research_engine.runconfig import get_run_config

        system = "\n".join(str(m.content) for m in messages if getattr(m, "type", "") == "system")
        if "Research Executor" in system and get_run_config().corpus_mode:
            return await self._corpus_evidence(messages)
        return self._reply(messages)

    async def _corpus_evidence(self, messages: list[BaseMessage]) -> AIMessage:
        """资料库脚本执行器先检索，再逐字提交实际返回的证据。

        引用携带真实 corpus:// 定位，才能验证导入到引用的完整链路；使用固定网页 URL 不能证明这一点。"""
        from research_engine.retrievers import search

        task_id, query = 1, ""
        for m in messages:
            if getattr(m, "type", "") == "human":
                match = re.match(r"Task (\d+): (.+)", str(m.content).strip())
                if match:
                    task_id, query = int(match.group(1)), match.group(2)
        try:
            results = await search(query or "corpus", max_results=2)
        except Exception:  # noqa: BLE001 — 失败时返回空证据，不编造 URL
            results = []
        evidence = [
            {
                "task_id": task_id,
                "source_url": r["url"],
                "source_title": r["title"],
                # EvidenceChunk 的片段上限为 500 字符，见 schemas.py。
                "snippet": r["snippet"][:480],
                "key_fact": r["snippet"][:80],
                "retrieved_at": "2026-01-01T00:00:00Z",
            }
            for r in results
        ]
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "submit_evidence",
                    "args": {"evidence": evidence},
                    "id": "fake-submit-evidence-corpus",
                }
            ],
            usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )


def fake_model(role: str) -> _ScriptedModel:
    return _ScriptedModel(role)
