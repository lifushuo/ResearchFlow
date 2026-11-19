"""首次启动演示会话的数据，见 docs/17 §6.1。

这里不同于测试占位数据。来源真实存在，片段逐字采自对应 arXiv 摘要，采集日期为 2026-08-15。
修改片段必须重新抓取核对，不能改写后冒充原文。
演示报告的每条 Claim 都由对应片段支持，不添加片段中没有的数字，由测试持续校验。"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from research_engine.fakes import ScriptedScenario, _ScriptedModel

DEMO_QUERY = "What is retrieval-augmented generation, and when does it beat fine-tuning?"

# 对应 URL 的逐字摘要，保持在 EvidenceChunk 的 500 字符上限内，避免执行器修改。
DEMO_SOURCES: list[dict[str, Any]] = [
    {
        "index": 1,
        "url": "https://arxiv.org/abs/2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "snippet": (
            "Large pre-trained language models have been shown to store factual knowledge "
            "in their parameters, and achieve state-of-the-art results when fine-tuned on "
            "downstream NLP tasks. However, their ability to access and precisely "
            "manipulate knowledge is still limited, and hence on knowledge-intensive "
            "tasks, their performance lags behind task-specific architectures. "
            "Additionally, providing provenance for their decisions and updating their "
            "world knowledge remain open research problems."
        ),
        "key_fact": (
            "Parametric knowledge is hard to access precisely, and provenance and updates "
            "remain open problems."
        ),
    },
    {
        "index": 2,
        "url": "https://arxiv.org/abs/2312.10997",
        "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
        "snippet": (
            "Retrieval-Augmented Generation (RAG) has emerged as a promising solution by "
            "incorporating knowledge from external databases. This enhances the accuracy "
            "and credibility of the generation, particularly for knowledge-intensive "
            "tasks, and allows for continuous knowledge updates and integration of "
            "domain-specific information."
        ),
        "key_fact": (
            "External-database knowledge improves accuracy and credibility and allows "
            "continuous updates."
        ),
    },
]

# Claim 不带数字，因为 _numbers_grounded 要求数字逐字出现在证据中；日期放在来源标题。
DEMO_REPORT = """# Retrieval-augmented generation versus fine-tuning

## Summary
Retrieval-augmented generation and fine-tuning address different problems: fine-tuning \
adapts a model's parameters, while retrieval supplies knowledge the model can cite at \
answer time [1][2].

## Findings
Large pre-trained language models store factual knowledge in their parameters and \
achieve state-of-the-art results when fine-tuned on downstream NLP tasks [1].
Their ability to access and precisely manipulate that knowledge is still limited, and on \
knowledge-intensive tasks their performance lags behind task-specific architectures [1].
Providing provenance for a model's decisions and updating its world knowledge remain \
open research problems [1].
Incorporating knowledge from external databases enhances the accuracy and credibility of \
the generation, particularly for knowledge-intensive tasks [2].
Because the knowledge sits outside the parameters, retrieval allows for continuous \
knowledge updates and integration of domain-specific information [2].

## Analysis
Retrieval is the stronger choice where knowledge changes or must be attributable, since \
it updates continuously and carries provenance that parametric memory does not [1][2].
Fine-tuning remains the mechanism for adapting a model to a downstream task rather than \
for teaching it facts it must later be able to cite [1].

## Limitations
This is a demonstration run on scripted models over two sources captured in advance. A \
real run searches the live web, may surface contradicting sources, and will not \
reproduce this report.

## Sources
[1] https://arxiv.org/abs/2005.11401
[2] https://arxiv.org/abs/2312.10997
"""


def demo_search(query: str, max_results: int) -> list[dict]:
    """演示搜索结果只使用真实来源，不编造来源。"""
    return [
        {"title": s["title"], "url": s["url"], "snippet": s["snippet"]}
        for s in DEMO_SOURCES[:max_results]
    ]


def demo_read_webpage(url: str) -> dict:
    """演示网页读取只返回已采集来源的原文；其他 URL 明确拒绝，不能临时编造页面。"""
    for source in DEMO_SOURCES:
        if source["url"] == url:
            return {
                "url": url,
                "title": source["title"],
                "text": source["snippet"],
                "error": None,
            }
    return {"url": url, "title": "", "text": "", "error": "not part of the demo corpus"}


class _DemoModel(_ScriptedModel):
    """用真实演示内容替代测试占位文本的脚本模型。

    继承 _ScriptedModel 的评审、矛盾检测和引用核验契约，仅覆盖用户实际阅读的角色输出。"""

    def _reply(self, messages: list[BaseMessage]) -> AIMessage:
        system = "\n".join(str(m.content) for m in messages if getattr(m, "type", "") == "system")
        scenario = self.scenario_for(system)

        if scenario is ScriptedScenario.PLANNER:
            return self._usage(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "id": 1,
                                "query": "what retrieval-augmented generation is",
                                "rationale": "define the mechanism before comparing it",
                            },
                            {
                                "id": 2,
                                "query": "when retrieval beats fine-tuning",
                                "rationale": "the comparison the question actually asks for",
                            },
                        ]
                    }
                )
            )

        if scenario is ScriptedScenario.EXECUTOR:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_evidence",
                        "args": {
                            "evidence": [
                                {
                                    "task_id": 1,
                                    "source_url": s["url"],
                                    "source_title": s["title"],
                                    "snippet": s["snippet"],
                                    "key_fact": s["key_fact"],
                                    "retrieved_at": "2026-08-15T00:00:00Z",
                                }
                                for s in DEMO_SOURCES
                            ]
                        },
                        "id": "demo-submit-evidence-1",
                    }
                ],
                usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            )

        if scenario in (ScriptedScenario.SYNTHESIZER, ScriptedScenario.SYNTHESIZER_REPAIR):
            return self._usage(DEMO_REPORT)

        return super()._reply(messages)

    @staticmethod
    def _usage(content: str) -> AIMessage:
        return AIMessage(
            content=content,
            usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )


def demo_model(role: str) -> _DemoModel:
    return _DemoModel(role)
