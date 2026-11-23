"""真实规划与独立评分，沿用 ResearchFlow 的模型路由和运行上下文。"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import nullcontext
from dataclasses import replace

from langchain_core.messages import HumanMessage, SystemMessage

from evals.harness import _pipeline_scope, validate_judge
from evals.judge_config import judge_client_config
from evals.paid_call_budget import (
    BudgetError,
    BudgetExhausted,
    RequestOutcomeUnknown,
    UnknownCallBlocked,
)
from research_engine import graph, llm_factory
from research_engine.prompt_composition import recording_provenance
from research_engine.prompts import PLANNER_PROMPT_V2
from research_engine.runconfig import RunConfig, reset_run_config, set_run_config
from research_engine.runner import initial_state
from research_engine.token_usage import PaidRequestUncertain
from research_engine.trace_content import content_snapshot

from .experiment import Case, Coverage, Observation, covered_indices
from .judge_prompt import PLANNER_JUDGE_PROMPT, PLANNER_JUDGE_PROMPT_V2
from .metrics_v2 import CoverageV2


class JudgeDeadlineExceeded(TimeoutError):
    """Our local Judge deadline elapsed, as opposed to a provider timeout."""


class LiveRuntime:
    """受控次序运行真实 Planner 节点。GEPA 两臂共享同一套 Planner 配置和独立 Judge。"""

    def __init__(
        self,
        config: RunConfig,
        judge: str,
        reflector: str,
        timeout: int,
        secrets: tuple[str, ...],
        metric_version: str = "v1",
        budget=None,
        judge_timeout: int | None = None,
    ) -> None:
        self.config = config
        self.judge = judge
        self.reflector = reflector
        self.timeout = timeout
        self.judge_timeout = timeout if judge_timeout is None else judge_timeout
        self.secrets = secrets
        # Optional paid-call budget gate. None preserves legacy behavior; when a
        # PaidCallBudget is supplied, every paid role call is reserved first.
        self.budget = budget
        # v3 复用 v2 的 Judge 轴与契约：改变的只是聚合方式，不是评分对象。
        if metric_version not in ("v1", "v2", "v3"):
            raise ValueError(f"unknown metric_version: {metric_version}")
        self.metric_version = metric_version

    async def _judge_wait(self, call):
        deadline = asyncio.timeout(self.judge_timeout)
        try:
            async with deadline:
                return await call()
        except TimeoutError as exc:
            if deadline.expired():
                raise JudgeDeadlineExceeded(
                    f"configured Judge deadline {self.judge_timeout}s elapsed; outcome remains unknown"
                ) from exc
            raise

    async def _paid(self, role: str, purpose: str, call, input_data=None):
        """Run one paid call through the budget gate when configured."""
        if self.budget is None:
            return await call()
        try:
            return await self.budget.paid_call_async(role, purpose, call, input_data=input_data)
        except BudgetError:
            raise
        except Exception as exc:
            unknown = self.budget.state()["unknown"]
            slot = next((s for s in unknown if s.startswith(role + "-")), None)
            if slot:
                raise RequestOutcomeUnknown(slot, role, type(exc).__name__) from exc
            raise

    async def _planner_call(self, call):
        """Planner goes through token_usage.invoke, which reserves per REQUEST.

        The Planner graph may retry once and fall back to JSON mode, so a single
        observation can issue up to four real requests. Charging at the runtime
        boundary as well would double-count; only the per-request hook applies.
        """
        if self.budget is None:
            return await call()
        from research_engine.token_usage import gate_scope

        with gate_scope(self.budget):
            try:
                result = await call()
            except PaidRequestUncertain as exc:
                raise RequestOutcomeUnknown(exc.slot_id, exc.role, exc.cause_type) from exc
        state = self.budget.state()
        if state["unknown"]:
            raise UnknownCallBlocked("Planner request has unresolved outcome")
        if result.get("error") and not state["remaining_by_role"]["planner"]:
            raise BudgetExhausted("Planner request quota exhausted")
        return result

    def checked_text(self, text: str, limit: int = 3000) -> str:
        snapshot = content_snapshot(text, secrets=self.secrets, max_chars=limit)
        if snapshot["value"] != text:
            raise ValueError("text is out of bounds or contains sensitive material")
        return text

    async def observe(self, case: Case, prompt: str | None) -> Observation:
        started = time.monotonic()
        fields = {
            "case_id": case.case_id,
            "split": case.split,
            "required_count": len(case.required_aspects),
        }
        stage = "planner"
        retry_scope = llm_factory.no_sdk_retries() if self.budget else nullcontext()
        retry_scope.__enter__()
        token = set_run_config(self.config)
        try:
            overrides = {"planner": prompt} if prompt else None
            with (
                _pipeline_scope(overrides, base_config=self.config),
                recording_provenance() as used,
            ):
                state = initial_state(
                    session_id=f"planner-evolution-{case.case_id}",
                    user_id="planner-evaluation",
                    query=case.query,
                    depth=case.depth,
                )
                plan = await self._planner_call(
                    lambda: asyncio.wait_for(graph.planner_node(state), self.timeout),
                )
            fields["planner_tokens"] = (plan.get("tokens_input", 0), plan.get("tokens_output", 0))
            if used.get("planner.main", {}).get("effective_prompt") != (
                prompt or PLANNER_PROMPT_V2
            ):
                raise ValueError("executed Planner prompt does not match experiment")
            if not plan.get("tasks") or plan.get("error"):
                raise ValueError(plan.get("error") or "Planner produced no tasks")
            queries = [self.checked_text(task["query"], limit=2000) for task in plan["tasks"]]
            fields["task_queries"] = queries
            stage = "judge"
            with judge_client_config(self.judge):
                llm = llm_factory.get_llm("critic")
                method = (
                    {"method": "function_calling"}
                    if self.judge.startswith(("custom:", "ollama:", "openrouter:"))
                    else {}
                )
                schema, prompt_text = (
                    (CoverageV2, PLANNER_JUDGE_PROMPT_V2)
                    if self.metric_version in ("v2", "v3")
                    else (Coverage, PLANNER_JUDGE_PROMPT)
                )
                scorer = llm.with_structured_output(schema, include_raw=True, **method)
                payload = {
                    "question": case.query,
                    "required_aspects": dict(enumerate(case.required_aspects, 1)),
                    "tasks": dict(enumerate(queries, 1)),
                }
                answer = await self._paid(
                    "judge",
                    f"judge:{case.case_id}:{case.split}",
                    lambda: self._judge_wait(
                        lambda: scorer.ainvoke(
                            [
                                SystemMessage(content=prompt_text),
                                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                            ]
                        ),
                    ),
                    input_data=payload,
                )
            raw = answer.get("raw")
            fields["judge_tokens"] = llm_factory.token_counts(raw)
            served = llm_factory.served_model_id(raw)
            if served:
                fields["served_judge"] = self.checked_text(served, limit=200)
                validate_judge(
                    f"{self.judge.partition(':')[0]}:{served}",
                    {**self.config.models, "reflection": self.reflector},
                )
            if answer.get("parsing_error") or answer.get("parsed") is None:
                raise ValueError("Judge returned no valid structured coverage verdict")
            if self.metric_version in ("v2", "v3"):
                verdict = CoverageV2.model_validate(answer["parsed"])
                spec_by_id = {t.task_id: t.specificity for t in verdict.task_specificity}
                if set(spec_by_id) != set(range(1, len(queries) + 1)):
                    raise ValueError(
                        "judge must return a specificity score for every task exactly once"
                    )
                fields["task_specificity"] = [spec_by_id[i + 1] for i in range(len(queries))]
                plan_by_index = {m.aspect_index: m.plan_quality for m in verdict.matches}
                fields["plan_quality"] = [
                    plan_by_index.get(i + 1, 0) for i in range(len(case.required_aspects))
                ]
                # CoverageV2 鸭子类型兼容 v1 的 covered_indices 契约（matches 子集同形）。
                fields["covered_indices"] = covered_indices(
                    verdict, len(case.required_aspects), len(queries)
                )
                fields["task_matches"] = verdict.matches
            else:
                verdict = Coverage.model_validate(answer["parsed"])
                fields["covered_indices"] = covered_indices(
                    verdict, len(case.required_aspects), len(queries)
                )
                fields["task_matches"] = verdict.matches
        except BudgetError:
            raise
        except Exception as exc:
            # 远端异常和缺失评分保持未测量，不能当成 0 分或通过。
            detail = content_snapshot(str(exc), secrets=self.secrets, max_chars=320)["value"]
            fields["error"] = f"{stage}: {type(exc).__name__}: {detail}"
        finally:
            reset_run_config(token)
            retry_scope.__exit__(None, None, None)
        return Observation(**fields, latency_seconds=time.monotonic() - started)

    async def reflect(self, prompt: str) -> str:
        retry_scope = llm_factory.no_sdk_retries() if self.budget else nullcontext()
        retry_scope.__enter__()
        token = set_run_config(
            replace(self.config, models={**self.config.models, "planner": self.reflector})
        )
        try:
            with llm_factory.output_limit("planner", 4096):
                model = llm_factory.get_llm("planner")
            answer = await self._paid(
                "reflection",
                "reflection",
                lambda: asyncio.wait_for(
                    model.ainvoke([HumanMessage(content=prompt)]), self.timeout
                ),
                input_data={
                    "prompt_sha": __import__("hashlib").sha256(prompt.encode()).hexdigest()
                },
            )
            return self.checked_text(llm_factory.text_of(answer), limit=4000)
        except BudgetError:
            raise
        except Exception as exc:
            detail = content_snapshot(str(exc), secrets=self.secrets, max_chars=320)["value"]
            raise ValueError(f"reflection: {type(exc).__name__}: {detail}") from None
        finally:
            reset_run_config(token)
            retry_scope.__exit__(None, None, None)
