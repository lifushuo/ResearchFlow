"""单次 Planner 节点真实调用：为 Judge 校准表产出任务列表证据。

只执行 ``graph.planner_node``——不运行 Executor/Critic/Synthesizer，
不写研究报告，不触碰任何数据集或评测结果。``llm_mode=real`` 时每次
调用恰好消耗一次 Planner 模型调用；fake 模式仅供离线测试接线。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_engine.runconfig import RunConfig

SCHEMA_VERSION = 1
MAX_TASKS = 30


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def fingerprint(payload: dict) -> str:
    """Content hash over everything except the fingerprint key itself."""
    unsigned = {k: v for k, v in payload.items() if k != "probe_evidence_sha256"}
    return hashlib.sha256(_canonical(unsigned).encode("utf-8")).hexdigest()


def build_probe_payload(
    *,
    case_id: str,
    query: str,
    depth: str,
    llm_mode: str,
    prompt_sha256: str | None,
    planner_tokens: tuple[int, int] | None,
    tasks: list[dict],
) -> dict:
    """Normalize a planner_node result into the probe evidence contract.

    Task rules mirror the whole pipeline: 1–30 tasks, each query 5–500 chars.
    A malformed plan stays malformed — never trimmed into something plausible.
    """
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= MAX_TASKS:
        raise ValueError("planner produced no usable task list; nothing was recorded")
    extracted = []
    for index, task in enumerate(tasks, 1):
        if not isinstance(task, dict) or not isinstance(task.get("query"), str):
            raise ValueError("planner returned a malformed task entry")
        value = task["query"].strip()
        if not 5 <= len(value) <= 500:
            raise ValueError("planner task query length out of the calibrated range")
        extracted.append({"id": str(task.get("id", index))[:40], "query": value})
    payload = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "query": query,
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "depth": depth,
        "llm_mode": llm_mode,
        "prompt_sha256": prompt_sha256,
        "planner_tokens": list(planner_tokens) if planner_tokens else None,
        "recorded_at": datetime.now(UTC).isoformat(),
        "tasks": extracted,
    }
    payload["probe_evidence_sha256"] = fingerprint(payload)
    return payload


async def run_probe(case_id: str, query: str, depth: str, timeout: int, config: RunConfig) -> dict:
    """The same planner-stage wiring LiveRuntime.observe uses — planner node only."""
    from evals.harness import _pipeline_scope
    from research_engine import graph
    from research_engine.prompt_composition import recording_provenance
    from research_engine.runconfig import reset_run_config, set_run_config
    from research_engine.runner import initial_state

    token = set_run_config(config)
    try:
        with _pipeline_scope(None, base_config=config), recording_provenance() as used:
            state = initial_state(
                session_id=f"planner-probe-{case_id}",
                user_id="planner-evaluation",
                query=query,
                depth=depth,
            )
            plan = await asyncio.wait_for(graph.planner_node(state), timeout)
        recorded = used.get("planner.main", {}).get("effective_prompt")
        prompt_sha256 = (
            hashlib.sha256(recorded.encode("utf-8")).hexdigest()
            if isinstance(recorded, str)
            else None
        )
        if plan.get("error"):
            raise ValueError(f"planner error: {plan.get('error')}")
        return build_probe_payload(
            case_id=case_id,
            query=query,
            depth=depth,
            llm_mode=config.llm_mode,
            prompt_sha256=prompt_sha256,
            planner_tokens=(plan.get("tokens_input", 0), plan.get("tokens_output", 0)),
            tasks=plan.get("tasks") or [],
        )
    finally:
        reset_run_config(token)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--depth", choices=("fast", "balanced", "comprehensive"), default="balanced"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--allow-fake",
        action="store_true",
        help="offline wiring test only; fake planner output is refused by the calibration fill step",
    )
    args = parser.parse_args(argv)

    from research_engine.local import load_env_file, run_config_from_env

    load_env_file(Path(__file__).resolve().parents[2].parent / ".env")
    # 默认强制真实调用（缺密钥时 SystemExit 带供应商指引）；
    # --allow-fake 只用于离线接线测试，其产物会被回填步骤拒绝。
    config = run_config_from_env(fake=args.allow_fake)
    try:
        payload = asyncio.run(run_probe(args.case_id, args.query, args.depth, args.timeout, config))
    except (TimeoutError, ValueError) as exc:
        print(f"probe failed: {exc}; nothing was recorded", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        f"recorded {len(payload['tasks'])} real planner tasks for {args.case_id} "
        f"(llm_mode={payload['llm_mode']}, fingerprint={payload['probe_evidence_sha256'][:12]}…): {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
