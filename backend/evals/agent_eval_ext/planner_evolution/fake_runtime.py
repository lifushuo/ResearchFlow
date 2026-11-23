"""Three deterministic offline model ports for the SAME GEPA v3 CLI and ledger."""

import asyncio
import os

from research_engine.prompts import PLANNER_PROMPT_V2

from .experiment import Observation, TaskMatch

MARKER = "Measure named benchmarks and concrete evidence for every required aspect."


class FakePlanner:
    async def invoke(self, case, prompt):
        scenario = os.environ.get("RF_GEPA_FAKE_SCENARIO", "")
        if scenario == "slow":
            await asyncio.sleep(30)
        if scenario == "model_error":
            raise RuntimeError("offline simulated model error")
        improved = MARKER in (prompt or "") or (
            getattr(self, "candidate_prompt", None) is not None and prompt == self.candidate_prompt
        )
        return {
            "queries": [
                f"{aspect} evidence target {i} dataset comparison"
                for i, aspect in enumerate(case.required_aspects)
            ],
            "improved": improved,
        }


class FakeJudge:
    async def invoke(self, case, plan):
        scenario = os.environ.get("RF_GEPA_FAKE_SCENARIO", "")
        improved = plan["improved"] and scenario != "no_improvement"
        covered = list(range(1, len(case.required_aspects) + 1))
        if improved and scenario == "coverage_regression":
            covered = covered[:-1]
        return {
            "specificity": [2 if improved else 0] * len(plan["queries"]),
            "quality": [
                (2 if improved else 1) if i in covered else 0
                for i in range(1, len(case.required_aspects) + 1)
            ],
            "covered": covered,
        }


class FakeReflection:
    async def invoke(self, prompt):
        return "```\n" + PLANNER_PROMPT_V2 + "\n" + MARKER + "\n```"


class FakeRuntime:
    def __init__(self, budget):
        self.budget = budget
        self.planner, self.judge, self.reflector = FakePlanner(), FakeJudge(), FakeReflection()

    def checked_text(self, text):
        return text

    async def observe(self, case, prompt):
        plan = await self.budget.paid_call_async(
            "planner", f"fake-planner:{case.case_id}", lambda: self.planner.invoke(case, prompt)
        )
        verdict = await self.budget.paid_call_async(
            "judge", f"fake-judge:{case.case_id}", lambda: self.judge.invoke(case, plan)
        )
        return Observation(
            case_id=case.case_id,
            split=case.split,
            required_count=len(case.required_aspects),
            task_queries=plan["queries"],
            covered_indices=verdict["covered"],
            task_matches=[
                TaskMatch(aspect_index=i, task_ids=[i] if i in verdict["covered"] else [])
                for i in range(1, len(case.required_aspects) + 1)
            ],
            task_specificity=verdict["specificity"],
            plan_quality=verdict["quality"],
            served_judge="fake-judge",
            planner_tokens=(32, 24),
            judge_tokens=(40, 24),
            latency_seconds=0.0,
        )

    async def reflect(self, prompt):
        return await self.budget.paid_call_async(
            "reflection", "fake-reflection", lambda: self.reflector.invoke(prompt)
        )
