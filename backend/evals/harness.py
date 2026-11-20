"""
Eval harness (docs/08 §5). Runs the fixed query set through the compiled graph up to
the review gate and records per-report quality metrics to a dated JSON file, so report
quality is diffable over time.

Per-commit CI uses fake models; evals measure real-model quality. Run:

    make eval                       # fake mode (deterministic, no keys) — smoke/baseline
    LLM_MODE=real make eval EVAL_ARGS="--judge anthropic:claude-sonnet-4-6"

The committed baseline in results/ is a fake-mode run: it exercises the metric plumbing
and pins structural numbers. Real-model runs additionally compute an LLM-judged citation
support rate and are what the release criteria gate on — judged by the model `--judge`
names, which must be independent of every route under evaluation (RG-5).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from evals import metrics
from evals.agent_eval_ext import long_run
from evals.agent_eval_ext.execution_observer import environment_secrets
from evals.agent_eval_ext.judge_runtime import (
    judge_trace_artifacts,
    token_usage,
    trace_judge_batch,
    trace_judge_case,
    trace_judge_restored_batch,
    trace_judge_stage,
    trace_judge_wait,
)
from evals.judge_config import judge_client_config, judge_provider_keys
from evals.query_selection import select_queries
from research_engine.eval_trace import eval_trace_run, trace_span, traced_ainvoke
from research_engine.graph import build_graph
from research_engine.local import load_env_file, run_config_from_env
from research_engine.prompts import PLANNER_PROMPT_V2
from research_engine.runconfig import (
    RunConfig,
    reset_run_config,
    set_process_default,
    set_run_config,
)
from research_engine.runner import initial_state

# The harness is a host, so it installs the engine's config (docs/13 §2) — built straight
# from the environment rather than through `app.config`. That is why an eval run now needs
# no DATABASE_URL and no JWT_SECRET_KEY: the `os.environ.setdefault` block that used to sit
# here was the clearest evidence the engine was wrongly coupled to the server, and deleting
# it is M6's acceptance test (docs/12).
# Mode is read from the *real* environment before `.env` is loaded, and `.env` supplies
# keys only. This ordering is deliberate: a developer `.env` commonly carries
# `LLM_MODE=real` for the app, and letting that reach the harness would silently turn
# `make eval` — documented and relied on as the free, deterministic default — into a run
# that spends money on every invocation. Spending is opt-in, per invocation:
#
#     make eval                    # always fake, whatever .env says
#     LLM_MODE=real make eval      # explicit, and picks up keys from .env
LLM_MODE = os.environ.get("LLM_MODE", "fake")
load_env_file()
RUN_CONFIG = run_config_from_env(fake=LLM_MODE != "real")
set_process_default(RUN_CONFIG)

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"

# Release criteria (docs/08 §5).
MIN_CITATION_SUPPORT = 0.95
MIN_COMPLETION_RATE = 0.90

_ROLES = ("planner", "executor", "critic", "synthesizer", "chat")

#: The judge's own prompt is part of the run's configuration fingerprint: a resumed run
#: must not mix verdicts produced under a different instruction. Keep the text here, not
#: inline in `judge_citation_support`, so there is one string to hash and one to edit.
JUDGE_SYSTEM_PROMPT = (
    "You judge whether claims are supported by their cited evidence. "
    "For each numbered claim, respond with exactly one line: "
    '"Claim N: YES" if the evidence supports the claim, or '
    '"Claim N: NO" if it does not. Answer for every claim.'
)
JUDGE_HUMAN_PREFIX = (
    "For each claim below, determine if it is supported by its cited evidence.\n"
    'Answer with one line per claim in format: "Claim N: YES" or "Claim N: NO"\n\n'
)
JUDGE_PROMPT = JUDGE_SYSTEM_PROMPT + JUDGE_HUMAN_PREFIX


# ── The judge (RG-5) ──────────────────────────────────────────────────────────────
#
# The citation judge used to be `get_llm("critic")` — the pipeline's own critic model grading
# the pipeline's own citations. That is a self-judged number, and RG-5 requires "a disclosed
# independent judge that actually answered" (internal/rfcs/V3.0-agentspec-scope.md §20). So
# the judge is named per invocation, checked against the routes under evaluation before a
# single query runs, and recorded beside them — never inferred, never defaulted.


class JudgeConfigError(ValueError):
    """A judge that cannot serve as the independent one, or no judge at all."""


#: The route `judge_citation_support` builds its judge from, set by `configure_judge`. None
#: until then — and judging then refuses rather than falling back to the pipeline's critic,
#: which is the self-judged measurement this exists to end.
JUDGE_ROUTE: str | None = None


def _is_router_alias(model: str) -> bool:
    """`auto`, `auto/*` and `*/auto` resolve per call (AGENTS.md: not pinned models)."""
    model = model.lower()
    return model == "auto" or model.startswith("auto/") or model.endswith("/auto")


def _model_identity(model: str) -> str:
    """The model a route names, with the provider's spelling removed.

    Catches one model reached two ways — `openrouter:google/gemini-2.5-flash` and
    `google:gemini-2.5-flash`, `ollama:qwen2.5` and `ollama:qwen2.5:latest` — which a plain
    string comparison of routes would call independent. It does not decide whether two
    *different* models are too closely related; the result names both, so a reader can.
    """
    name = model.lower().rsplit("/", 1)[-1]
    return name.removesuffix(":latest")


def validate_judge(route: str, pipeline: dict[str, str]) -> str:
    """`route` if it can judge `pipeline` independently; otherwise say exactly why not."""
    provider, _, model = route.partition(":")
    if not provider or not model:
        raise JudgeConfigError(
            f"a judge must be 'provider:model' (got {route!r}), e.g. anthropic:claude-sonnet-4-6"
        )
    if _is_router_alias(model):
        raise JudgeConfigError(
            f"{route} is a router alias, not a pinned model: it can resolve to a different "
            "model on every call, including one the pipeline uses, so it cannot be disclosed "
            "as the judge"
        )
    for role, used in sorted(pipeline.items()):
        used_model = used.partition(":")[2]
        if _is_router_alias(used_model):
            raise JudgeConfigError(
                f"the pipeline's {role} routes to the alias {used}; what it resolves to is "
                "unknown, so no judge can be shown to be independent of it. Pin the model."
            )
        if used == route or _model_identity(used_model) == _model_identity(model):
            raise JudgeConfigError(
                f"{route} is the model the pipeline's {role} runs on ({used}); the system "
                "under evaluation cannot also be its judge"
            )
    return route


def configure_judge(route: str) -> str:
    """Install the judge for this process, checked against every route the harness runs."""
    global JUDGE_ROUTE
    try:
        judge_provider_keys(route)
        from evals.judge_config import judge_max_output_tokens

        judge_max_output_tokens()
    except ValueError as exc:
        raise JudgeConfigError(str(exc)) from exc
    JUDGE_ROUTE = validate_judge(route, {role: RUN_CONFIG.models[role] for role in _ROLES})
    return JUDGE_ROUTE


def _judge_llm():
    """Build the configured judge through the engine's factory — every provider the product
    supports, no second client. `critic` is borrowed only for its zero-temperature settings;
    the route is the judge's, installed for the build and removed straight after, as
    `benchmark.py::_build_judge` does."""
    if JUDGE_ROUTE is None:
        raise JudgeConfigError(
            "no independent judge is configured (--judge provider:model). Judging never falls "
            "back to the pipeline's own critic: a self-judged rate is not evidence for RG-5."
        )
    from research_engine import llm_factory
    from research_engine.response_diagnostics import diagnostic_models

    with judge_client_config(JUDGE_ROUTE), diagnostic_models():
        llm = llm_factory.get_llm("critic")
    # A gateway reports in response headers which connection and model served a call, and the
    # OpenAI-compatible client drops them unless asked (`gateway_routing` reads them). Set on
    # this judge's own instance only — the pipeline's clients are untouched — and only where
    # the client declares the option; a native SDK client or a test double is left as built.
    if "include_response_headers" in getattr(type(llm), "model_fields", {}):
        llm.include_response_headers = True
    return llm


#: The headers OmniRoute documents for "what actually served this call". Read by name, and
#: nothing else from a response's headers is kept — so a credential, token or cookie a
#: gateway happened to send back can never reach a result file.
_ROUTING_HEADERS = {
    "provider": "x-omniroute-provider",
    "model": "x-omniroute-model",
    "decision": "x-omniroute-decision",
}


def gateway_routing(response) -> dict[str, str | None]:
    """The connection, model and routing decision a gateway reports for one judge call.

    Observational only: a header the response did not carry is None, never filled in from
    the configured route. A direct provider client sends none of them, so every field is
    None there — which says "no gateway reported this", not "served by the route".
    """
    meta = getattr(response, "response_metadata", None) or {}
    headers = {str(k).lower(): v for k, v in (meta.get("headers") or {}).items()}
    return {
        field: (str(headers[name]) if headers.get(name) else None)
        for field, name in _ROUTING_HEADERS.items()
    }


def _decision_strategy(decision: str) -> str | None:
    """`single`, a combo strategy, … — the `strategy=` field of OmniRoute's documented
    `strategy=<name>; provider=<alias>; latency_ms=<n>` decision header, or None."""
    for part in decision.split(";"):
        key, _, value = part.strip().partition("=")
        if key == "strategy" and value:
            return value
    return None


def judge_tally(rows: list[dict]) -> dict:
    """What the judge actually did across a run's reports: how many claims it ruled on, how
    many it never reached, and which model the provider says answered.

    `served_models` is read from each response (`llm_factory.served_model_id`), not copied
    from the route, so a judge that silently answered as something else shows it. Empty when
    the provider discloses nothing — never filled in from the route.
    """
    verdicts = [c for r in rows for c in r.get("claim_verdicts") or []]
    judged = [c for c in verdicts if c.get("judged")]
    routed = [c.get("judge_routing") or {} for c in judged]
    return {
        "claims_judged": len(judged),
        "claims_unjudged": len(verdicts) - len(judged),
        "served_models": sorted({c["judged_by"] for c in judged if c.get("judged_by")}),
        # What the gateway said served each ruling. `strategy` single means the pinned
        # connection answered directly — no combo, no fallback to another provider.
        "routing": {
            "providers": sorted({r["provider"] for r in routed if r.get("provider")}),
            "models": sorted({r["model"] for r in routed if r.get("model")}),
            "strategies": sorted(
                {
                    s
                    for r in routed
                    if r.get("decision")
                    for s in [_decision_strategy(r["decision"])]
                    if s
                }
            ),
            "rulings_without_routing": sum(1 for r in routed if not r.get("provider")),
        },
    }


def judge_section(route: str, rows: list[dict], candidate_rows: list[dict] | None) -> dict:
    """The result file's account of its judge, beside — never inside — `models`.

    `models` is the system under evaluation; this is what graded it. Keeping them apart is
    what lets a reader check the claim "independent" without trusting the harness's word.
    """
    section = {
        "route": route,
        "independent_of": {role: RUN_CONFIG.models[role] for role in _ROLES},
        "baseline": judge_tally(rows),
    }
    if candidate_rows is not None:
        section["candidate"] = judge_tally(candidate_rows)
    tallies = [section["baseline"], *([section["candidate"]] if candidate_rows is not None else [])]
    section["answered"] = any(t["claims_judged"] for t in tallies)
    return section


def _routing_slug() -> str:
    """Filename-safe tag for the routing a run used: the planner role's provider.

    Routing is `provider:model` split on the **first** colon (AGENTS.md), so
    `ollama:qwen2.5:7b` yields `ollama`. Used to give each result file a run identity —
    see `_result_path`.
    """
    provider = RUN_CONFIG.models["planner"].split(":", 1)[0]
    cleaned = "".join(c if c.isalnum() else "-" for c in provider).strip("-").lower()
    return cleaned or "unknown"


def _result_path(run_date: str, *, custom_spec: bool = False) -> Path:
    """Where this run's result goes — never onto an existing file.

    A date is not a run identity. The old default was `eval-<date>.json`, so two runs on
    one day collided, and that is exactly how a real 10/10 ollama measurement was
    destroyed: `cbde168` overwrote `eval-2026-08-13.json` with a failed Gemini run and
    nothing warned. Results are write-once (AGENTS.md; CI job `eval-artifacts`), so carry
    the routing and take the next free run number rather than clobbering.

    A custom-spec run (RFC §14.2) takes a `-custom` stem as well. It holds a baseline *and* a
    candidate, so it is a different kind of evidence; sharing the numbering with plain runs
    would let the two be mistaken for each other, and a distinct stem means neither can ever
    resolve to the other's file.
    """
    routing = "fake" if RUN_CONFIG.llm_mode == "fake" else _routing_slug()
    stem = f"eval-{run_date}-{routing}" + ("-custom" if custom_spec else "")
    path = RESULTS_DIR / f"{stem}.json"
    n = 2
    while path.exists():
        path = RESULTS_DIR / f"{stem}-run{n}.json"
        n += 1
    return path


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


def _check_output_path(out: Path) -> None:
    """先实际测试目录写入，避免模型调用完成后才发现容器挂载目录只读。"""
    if out.exists():
        raise FileExistsError(
            f"{out} already exists and eval results are write-once. "
            "Pass a new --out, or omit --out to get an auto-numbered filename."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=out.parent, prefix=".eval-write-check-", mode="w", encoding="utf-8"
    ) as probe:
        probe.write("output check\n")
        probe.flush()


@contextmanager
def _pipeline_scope(
    prompt_overrides: dict[str, str] | None, *, base_config: RunConfig | None = None
) -> Iterator[None]:
    """The config the pipeline runs under — baseline or candidate — for this block only.

    **Each experiment uses one base config for both arms.** Report evaluation uses
    `RUN_CONFIG`. The Planner optimizer supplies its explicitly configured local run
    config so it can reuse this boundary without changing the report harness defaults.

    **Scoped, never process-wide.** `set_process_default` would outlive the query and still
    be live when the judge runs. A context-local config exists only inside the `with`, so the
    sequence is fixed: install, run the pipeline, reset, *then* judge (RFC §14.1: a user's
    prompt may change what the pipeline produces, never what the harness measures).
    Synchronous by design — entered with `with`, not `async with`, around an awaited call.
    """
    token = set_run_config(
        replace(base_config or RUN_CONFIG, prompt_overrides=prompt_overrides or {})
    )
    try:
        yield
    finally:
        reset_run_config(token)


class CandidateSpecError(ValueError):
    """A `--candidate` file that cannot be evaluated as written."""


def load_candidate(path: Path) -> dict[str, str]:
    """Read a candidate spec — `{role: prompt}` — and validate it as production would.

    **From an explicit file, never from a user's stored preferences** (RFC §14.1): the eval
    measures a spec someone hands it, and must not reach into a database for one.

    Validated through `usable_overrides`, the same all-or-nothing rule a run's snapshot goes
    through, rather than a second copy of it. An unusable spec is refused outright instead of
    being evaluated as the shipped prompts: in production that fallback is the honest
    outcome, but here it would publish the baseline's numbers under the candidate's name.
    """
    from app.services.run_config import OVERRIDES_APPLIED, usable_overrides

    try:
        raw = json.loads(Path(path).read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise CandidateSpecError(f"cannot read candidate spec {path}: {e}") from e
    overrides, status = usable_overrides(raw)
    if status != OVERRIDES_APPLIED:
        reason = (
            "it replaces no prompt, so it would measure the shipped prompts twice"
            if not raw
            else "it names an unknown role or carries a prompt that is not non-empty text"
        )
        raise CandidateSpecError(f"candidate spec {path} cannot be evaluated: {reason}")
    return overrides


def candidate_section(overrides: dict[str, str], rows: list[dict]) -> dict:
    """The candidate's half of a custom-spec result, judged against the same thresholds.

    Beside the baseline rather than instead of it (RFC §14.2), and never gating the run: a
    spec below `MIN_CITATION_SUPPORT` is reported, because the user chose it and the eval's
    job is to measure honestly, not to refuse. The digest identifies exactly which text was
    measured without anyone having to diff prompts by eye.
    """
    import hashlib

    agg = aggregate(rows)
    return {
        "prompt_overrides": dict(overrides),
        "prompt_sha256": {
            role: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for role, text in sorted(overrides.items())
        },
        "aggregate": agg,
        "release_criteria": check_release_criteria(agg),
        "results": rows,
    }


def _trace_artifacts(writer, *, requested: bool, artifact_root: Path | None = None) -> dict:
    if not requested:
        return {}
    if writer is None:
        return {"runtime_trace_status": "unavailable"}

    def shown(path: Path) -> str:
        try:
            return str(path.relative_to(artifact_root or EVALS_DIR))
        except ValueError:
            return str(path)

    result = {
        "runtime_trace": shown(writer.path),
        "runtime_trace_status": "partial" if writer.write_failed else "recorded",
    }
    if writer.bad_case_written:
        result["runtime_bad_case"] = shown(writer.bad_case_path)
    return result


def _jsonable(value: object, depth: int = 0) -> object:
    """Bounded, JSON-safe projection of graph state for a local progress artifact."""
    import math

    if depth > 6:
        return "[depth-limit]"
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k)[:120]: _jsonable(v, depth + 1) for k, v in list(value.items())[:200]}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, depth + 1) for v in list(value)[:200]]
    return str(value)[:2000]


def _interrupt_payload(result: object) -> dict:
    """Which human gate fired — read off the interrupt, never assumed (AGENTS.md)."""
    if not isinstance(result, dict):
        return {}
    interrupts = result.get("__interrupt__") or ()
    if not interrupts:
        return {}
    first = interrupts[0]
    value = getattr(first, "value", None)
    if not isinstance(value, dict):
        value = first.get("value") if isinstance(first, dict) else None
    if not isinstance(value, dict):
        return {}
    return {k: _jsonable(value.get(k)) for k in ("type", "word_count", "source_count")}


def _save(ctx, store, action: str, *args, **kwargs):
    """A failed artifact write becomes an explicit state, never a silent loss.

    `action` is a *method name*, not a bound method: resolving it here means a
    `None` store is a cheap no-op instead of an AttributeError raised while
    Python is still evaluating the call's arguments.
    """
    if store is None:
        return None
    try:
        return long_run.persist(getattr(store, action), *args, **kwargs)
    except long_run.ArtifactSaveFailed as exc:
        store.save_failed = True
        if ctx is not None:
            ctx.exit_code = long_run.EXIT_ARTIFACT_SAVE_FAILED
            ctx.note(f"artifact_save_failed:{exc}")
        return None


def _row_failure(query: dict, started: float, error: object, runtime_trace, requested, root):
    return {
        "id": query["id"],
        "domain": query.get("domain"),
        "completed": False,
        "error": str(error)[:300],
        "latency_s": round(time.time() - started, 2),
        **_trace_artifacts(runtime_trace, requested=requested, artifact_root=root),
    }


async def run_one(
    query: dict,
    *,
    prompt_overrides: dict[str, str] | None = None,
    evaluation_arm: str | None = None,
    trace_dir: Path | None = None,
    trace_run_id: str | None = None,
    trace_artifact_root: Path | None = None,
    trace_content: bool = False,
) -> dict:
    """Run one query to the gate; return its report metrics + timing.

    `prompt_overrides` is a candidate spec (RFC §14.2), already validated by the caller. It
    reaches the pipeline and is gone before the judge runs — see `_pipeline_scope`.

    With progress enabled the draft is written the moment the graph stops at the human
    review gate — before a single judge call — so a killed process keeps its evidence.
    A validated resume reuses that draft and never re-runs the research graph; the
    `interrupt()` is still left unapproved: the harness does not resume to the finalizer.
    """
    ctx = long_run.current()
    if ctx is not None and ctx.stop.requested:
        # A cooperative stop must never start another paid research graph. The
        # caller's result and the already-flushed progress stay authoritative.
        return _row_failure(
            query,
            time.time(),
            f"cooperative_stop:{ctx.stop.reason}",
            None,
            False,
            trace_artifact_root,
        )
    store = None
    if ctx is not None:
        ctx.budget.start_case(ctx.remaining_cases)
        ctx.remaining_cases = max(0, ctx.remaining_cases - 1)
        config = long_run.build_config(
            run_config=RUN_CONFIG,
            judge_route=JUDGE_ROUTE,
            judge_prompt=JUDGE_PROMPT,
            case_id=query["id"],
            query=query["query"],
            depth=query.get("depth", "balanced"),
            prompt_overrides=prompt_overrides,
            evaluation_arm=evaluation_arm,
        )
        store = ctx.store_for(config)
        try:
            long_run.persist(store.preflight)
        except long_run.ArtifactSaveFailed as exc:
            store.save_failed = True
            ctx.exit_code = long_run.EXIT_ARTIFACT_SAVE_FAILED
            ctx.note(f"preflight:{exc}")
            ctx.stop.request("artifact_save_failed")
            row = _row_failure(query, time.time(), exc, None, False, trace_artifact_root)
            row["research_state"] = long_run.ARTIFACT_SAVE_FAILED
            return row

    started = time.time()
    runtime_trace = None
    trace_requested = trace_dir is not None or os.environ.get("RF_EVAL_TRACE") == "1"
    trace_directory = trace_dir or Path(
        os.environ.get("RF_EVAL_TRACE_DIR") or RESULTS_DIR / "traces"
    )

    reused = None
    if store is not None and store.source_dir is not None:
        # A fingerprint mismatch refuses the whole run: neither a paid research graph nor
        # a paid judge call may silently reinterpret saved evidence.
        resume = (
            store.load_for_rejudge(case_id=query["id"])
            if ctx.rejudge
            else store.load(case_id=query["id"])
        )
        ctx.batches_restored += len(resume.batches)
        ctx.research_reused += 1
        _save(ctx, store, "adopt_resume")
        _save(
            ctx,
            store,
            "save_progress",
            status=long_run.RESEARCH_DRAFT_READY,
            reason="resumed_from_saved_draft",
        )
        reused = resume.draft
        state = {
            "draft_report": reused["draft_report"],
            "sources": reused["sources"],
            "error": reused.get("error"),
            "cost_usd": reused.get("cost_usd", 0.0),
            "tokens_input": reused.get("tokens_input", 0),
            "tokens_output": reused.get("tokens_output", 0),
            "evidence": reused.get("evidence") or [],
        }
        interrupt = reused.get("interrupt") or {}
    else:
        graph = build_graph(MemorySaver())
        thread_id = f"eval-{query['id']}"
        config = {"configurable": {"thread_id": thread_id}}
        initial = initial_state(
            session_id=thread_id,
            user_id="eval",
            query=query["query"],
            depth=query.get("depth", "balanced"),
        )
        invoke_result = None
        try:
            # The candidate is live for the pipeline alone. Reading the state back composes
            # no prompt, so it runs after the reset; so does everything below, the judge.
            with _pipeline_scope(prompt_overrides):
                with eval_trace_run(
                    case_id=query["id"],
                    run_id=trace_run_id or uuid.uuid4().hex,
                    thread_id=thread_id,
                    directory=trace_directory,
                    enabled=trace_requested,
                    capture_content=trace_content or os.environ.get("RF_EVAL_CONTENT_TRACE") == "1",
                    redaction_secrets=environment_secrets(),
                ) as runtime_trace:
                    if ctx is not None:
                        invoke_result = await long_run.with_deadline(
                            graph.ainvoke(initial, config), budget=ctx.budget, stage="research"
                        )
                    else:
                        invoke_result = await graph.ainvoke(initial, config)
            state = (await graph.aget_state(config)).values
        except (long_run.StageDeadlineExceeded, long_run.CooperativeStop) as exc:
            _save(ctx, store, "save_progress", status=long_run.RESEARCH_FAILED, reason=str(exc))
            row = _row_failure(
                query, started, exc, runtime_trace, trace_requested, trace_artifact_root
            )
            if ctx is not None:
                row["research_state"] = long_run.RESEARCH_FAILED
                ctx.exit_code = long_run.EXIT_JUDGE_PARTIAL
            return row
        except Exception as e:  # noqa: BLE001
            _save(
                ctx,
                store,
                "save_progress",
                status=long_run.RESEARCH_FAILED,
                reason=type(e).__name__,
            )
            row = _row_failure(
                query, started, e, runtime_trace, trace_requested, trace_artifact_root
            )
            if ctx is not None:
                row["research_state"] = long_run.RESEARCH_FAILED
            return row
        interrupt = _interrupt_payload(invoke_result)

    latency = (
        round(time.time() - started, 2) if reused is None else float(reused.get("latency_s", 0.0))
    )
    error = str(state["error"])[:300] if state.get("error") else None
    report = state.get("draft_report") or ""
    sources = state.get("sources") or []
    completed = bool(report) and error is None
    claim_lines = [c for c in metrics.claim_lines(report) if metrics.CITE_RE.search(c)]
    citations = [{"claim": c, "cites": metrics.extract_citations(c)} for c in claim_lines]
    research_state = long_run.RESEARCH_DRAFT_READY if completed else long_run.RESEARCH_FAILED

    if store is not None and reused is None:
        draft_doc = {
            "state": research_state,
            "draft_report": report,
            "sources": _jsonable(sources),
            "evidence": _jsonable(state.get("evidence") or []),
            "citations": _jsonable(citations),
            "claim_lines": claim_lines,
            "report_sha256": long_run.sha256_text(report),
            "sources_sha256": long_run.canonical_hash(sources),
            "claims_sha256": long_run.canonical_hash(claim_lines),
            "tokens_input": state.get("tokens_input", 0),
            "tokens_output": state.get("tokens_output", 0),
            "cost_usd": round(state.get("cost_usd", 0.0), 6),
            "latency_s": latency,
            "error": error,
            # The human gate is deliberately left suspended. Nothing here approves the
            # draft or continues the graph to `finalizer_node`.
            "interrupt": {
                **interrupt,
                "gate": "hitl_gate" if completed else None,
                "approved": False,
                "resume_action": "not_taken",
                "word_count": len(report.split()),
                "source_count": len(sources),
            },
        }
        _save(ctx, store, "save_draft", query["id"], draft_doc)
        _save(
            ctx,
            store,
            "save_progress",
            status=research_state,
            reason=None if completed else (error or "no_draft_report"),
            extra={
                "claim_count": len(claim_lines),
                "tokens_input": state.get("tokens_input", 0),
                "tokens_output": state.get("tokens_output", 0),
            },
        )

    result = {
        "id": query["id"],
        "domain": query.get("domain"),
        "completed": completed,
        "error": error,
        "latency_s": latency,
        "cost_usd": round(state.get("cost_usd", 0.0), 6),
        "tokens": state.get("tokens_input", 0) + state.get("tokens_output", 0),
        **metrics.report_metrics(report, sources),
        **_trace_artifacts(
            runtime_trace, requested=trace_requested, artifact_root=trace_artifact_root
        ),
    }
    if ctx is not None:
        result["research_state"] = research_state
        if reused is not None:
            result["research_reused"] = True

    if completed and RUN_CONFIG.llm_mode == "real" and not (store and store.save_failed):
        # Judge observation is outside both candidate prompts and the graph trace (docs/08 §5).
        judge_trace_dir = os.environ.get("RF_JUDGE_BATCH_TRACE_DIR")
        phase = evaluation_arm or ("candidate" if prompt_overrides is not None else "baseline")
        if store is not None and store.resume is not None:
            phase += "-resume"
        if ctx is not None:
            ctx.store = store
        with trace_judge_case(
            case_id=query["id"],
            phase=phase,
            trace_dir=judge_trace_dir,
            judge_route=JUDGE_ROUTE,
            run_id=trace_run_id,
            capture_content=trace_content or os.environ.get("RF_EVAL_CONTENT_TRACE") == "1",
            redaction_secrets=environment_secrets(),
            total_claims=len([c for c in metrics.claim_lines(report) if metrics.CITE_RE.search(c)]),
            batch_size=long_run.JUDGE_BATCH_SIZE,
        ) as judge_trace:
            rate, claim_rows = await judge_citation_support(report, sources)
        result.update(
            judge_trace_artifacts(
                judge_trace, requested=bool(judge_trace_dir), artifact_root=trace_artifact_root
            )
        )
        result["citation_support_rate"] = rate
        # The per-claim ruling — not just the aggregate — is what makes a miss
        # publishable and debuggable (docs/12 M5: "including the misses"). It is also
        # the only way to diff the judge against the graph's own citation-fidelity pass.
        result["claim_verdicts"] = claim_rows
        result["report"] = report
        if ctx is not None:
            result["judge_status"] = ctx.last_judge_status or long_run.JUDGE_FAILED
    return result


async def judge_citation_support(
    report: str, sources: list[dict]
) -> tuple[float | None, list[dict]]:
    """Real-mode only: ask a model whether each cited claim is actually supported by the
    snippet it cites. Batches up to 4 claims per LLM call.

    Returns `(supported / *judged* claims, per-claim rows)`. The denominator counts only
    claims the judge actually ruled on: a provider error, a partial reply, or an
    unparseable answer leaves a claim unjudged, and unjudged is **excluded** rather than
    scored as unsupported. The rate is None when nothing could be judged — "we could not
    measure this" has to stay distinct from "it scored zero" (docs/16 §6). The identical
    rule lives in `benchmark.py::calc_support_rate`; change both (AGENTS.md).

    With a reliability context, every batch is atomically persisted as soon as it ends,
    waits happen only before a real next request, and a stop/deadline yields a partial
    result that can be resumed. Without one (the CLI default and every existing test)
    the behaviour is the same as before, minus the unconditional 15 s sleeps.
    """

    from langchain_core.messages import HumanMessage, SystemMessage

    from evals.agent_eval_ext.batch_ledger import BatchBusy
    from research_engine.llm_factory import served_model_id

    ctx = long_run.current()
    store = ctx.store if ctx is not None else None
    by_index = {s.get("index"): s for s in sources if isinstance(s, dict)}
    claims = [c for c in metrics.claim_lines(report) if metrics.CITE_RE.search(c)]
    if not claims:
        if ctx is not None:
            ctx.last_judge_status = long_run.JUDGE_COMPLETED
        return None, []

    # The configured independent judge, or a refusal — never the pipeline's critic (RG-5).
    llm = _judge_llm()

    # Build per-claim evidence blocks once, then batch.
    claim_evidence: list[tuple[str, str]] = []
    for claim in claims:
        cited = [by_index.get(n) for n in metrics.extract_citations(claim)]
        # Show every snippet extracted from each cited source, not just the first
        # (docs/12 M5, D3). A source backs ~8 claims per report; judging each against a
        # single stored snippet measured a snippet-retention bug rather than the model's
        # citation quality.
        snippets = "\n".join(
            f"- {text}"
            for s in cited
            if s
            for text in (s.get("snippets") or ([s["snippet"]] if s.get("snippet") else []))
        )
        claim_evidence.append((claim, snippets))

    BATCH_SIZE = long_run.JUDGE_BATCH_SIZE
    report_hash = long_run.sha256_text(report)
    sources_hash = long_run.canonical_hash(sources)
    resumed = store.resume.batches if store is not None and store.resume is not None else {}
    total_batches = (len(claim_evidence) + BATCH_SIZE - 1) // BATCH_SIZE
    # Only claims the judge actually ruled on enter this map, and every number below is
    # derived from it — so an unjudged claim cannot reach the rate as a miss.
    verdict_by_claim: dict[int, bool] = {}
    served_by_claim: dict[int, str | None] = {}
    routing_by_claim: dict[int, dict[str, str | None]] = {}
    batch_statuses: list[str] = []
    attempted = 0
    stop_reason: str | None = None
    previous_batch_no = 0

    with ExitStack() as ownership:
        for batch_start in range(0, len(claim_evidence), BATCH_SIZE):
            batch_no = batch_start // BATCH_SIZE + 1
            batch = claim_evidence[batch_start : batch_start + BATCH_SIZE]
            batch_claims = [c for c, _ in batch]

            lease = None
            if store is not None and batch_no not in resumed:
                try:
                    lease = ownership.enter_context(store.batch_lease(batch_no))
                except BatchBusy:
                    stop_reason = f"batch_in_progress:{batch_no}"
                    break
                except (OSError, ValueError) as exc:
                    store.save_failed = True
                    stop_reason = f"batch_ownership_failed:{type(exc).__name__}"
                    break
                store.lease = lease
                if lease.uncertain_previous_request:
                    ctx.note(f"batch_{batch_no}_previous_request_may_have_been_billed")
                if lease.committed is not None:
                    _save(ctx, store, "restore_commit", batch_no, lease.committed)
                    if store.save_failed:
                        stop_reason = "artifact_save_failed"
                        break
                    resumed[batch_no] = lease.committed

            if batch_no in resumed:
                saved = resumed[batch_no]
                if saved.get("claim_sha256") != long_run.canonical_hash(batch_claims) or saved.get(
                    "expected_count"
                ) != len(batch):
                    raise long_run.ResumeRejected(f"batch_claims_changed:{batch_no}")
                for key, value in (saved.get("verdicts") or {}).items():
                    idx = int(key)
                    verdict_by_claim[idx] = bool(value["supported"])
                    served_by_claim[idx] = value.get("served_model")
                    routing_by_claim[idx] = value.get("judge_routing")
                status = saved.get("status", "unjudged")
                batch_statuses.append(status)
                attempted += 1
                trace_judge_restored_batch(
                    batch_no=batch_no,
                    batch_start=batch_start,
                    expected_count=int(saved.get("expected_count", len(batch))),
                    judged_count=int(saved.get("judged_count", 0)),
                    unjudged_count=int(saved.get("unjudged_count", 0)),
                    yes_count=int(saved.get("yes_count", 0)),
                    no_count=int(saved.get("no_count", 0)),
                    status=status,
                    restored_from=f"batch-{batch_no:04d}.json",
                    reason_hint=saved.get("reason_hint"),
                    served_model=saved.get("served_model"),
                    response_diagnostic=saved.get("response_diagnostic"),
                )
                previous_batch_no = batch_no
                if lease is not None:
                    lease.release()
                    store.lease = None
                continue

            if ctx is not None:
                if ctx.throttle.blocked_reason:
                    stop_reason = ctx.throttle.blocked_reason
                    break
                if ctx.stop.requested:
                    stop_reason = ctx.stop.reason or "cooperative_stop"
                    break
                try:
                    ctx.budget.check("judge")
                except long_run.StageDeadlineExceeded as exc:
                    stop_reason = f"judge_deadline:{exc}"
                    break
                # Only the *next* request waits, and only for as long as it must. A 429's
                # Retry-After is honoured here; the failed batch itself is never re-sent.
                seconds, reason, retry_after = ctx.throttle.required()
                seconds = min(seconds, ctx.budget.remaining("judge"))
                planned = (seconds, reason, retry_after)
                traced_wait = seconds > 0 and previous_batch_no > 0
                try:
                    if traced_wait:
                        with trace_judge_wait(
                            batch_no=previous_batch_no,
                            seconds=seconds,
                            reason=reason,
                            provider=ctx.throttle.provider,
                            retry_after=retry_after,
                        ) as observed:
                            record = await ctx.throttle.wait_before_next(
                                batch_no=previous_batch_no, budget=ctx.budget, planned=planned
                            )
                            observed["elapsed_s"] = record.actual_s
                    else:
                        record = await ctx.throttle.wait_before_next(
                            batch_no=previous_batch_no, budget=ctx.budget, planned=planned
                        )
                except long_run.CooperativeStop as exc:
                    stop_reason = exc.reason
                    break
                except asyncio.CancelledError:
                    # A cancel *between* batches still owns the batches already measured.
                    # Flush them before the exception closes the trace scope.
                    ctx.last_judge_status = long_run.JUDGE_PARTIAL
                    ctx.exit_code = long_run.EXIT_JUDGE_PARTIAL
                    if store is not None:
                        _save(
                            ctx,
                            store,
                            "save_progress",
                            status=long_run.JUDGE_PARTIAL,
                            reason="cancelled_during_throttle",
                        )
                        _save(
                            ctx,
                            store,
                            "save_final",
                            status=long_run.JUDGE_PARTIAL,
                            reason="cancelled_during_throttle",
                            extra={"attempted_batches": attempted, "total_batches": total_batches},
                        )
                    raise
                if store is not None:
                    store.note_wait(record)
                ctx.budget.spend("judge", record.actual_s)
                if store is not None:
                    _save(
                        ctx,
                        store,
                        "save_progress",
                        status=long_run.JUDGE_PARTIAL,
                        reason="judging",
                        extra={"wait": record.as_dict(), "next_batch_no": batch_no},
                    )
                    if store.save_failed:
                        stop_reason = "artifact_save_failed"
                        break

            # Build a single prompt covering all claims in this batch.
            claim_blocks = [
                f"Claim {i}: {claim}\nEvidence {i}:\n{snippets}"
                for i, (claim, snippets) in enumerate(batch, start=1)
            ]
            messages = [
                # Literal on purpose: `test_every_judge_system_message_is_literal_text` forbids
                # any Name/Call/Attribute in a judge system message, so an override or prompt
                # registry can never reach it. `JUDGE_SYSTEM_PROMPT` is the identical text used
                # for the config fingerprint; a drift test pins the two together.
                SystemMessage(
                    content="You judge whether claims are supported by their cited evidence. "
                    "For each numbered claim, respond with exactly one line: "
                    '"Claim N: YES" if the evidence supports the claim, or '
                    '"Claim N: NO" if it does not. Answer for every claim.'
                ),
                HumanMessage(content=JUDGE_HUMAN_PREFIX + "\n\n".join(claim_blocks)),
            ]

            if store is not None:
                _save(ctx, store, "request_started")
                if store.save_failed:
                    stop_reason = "artifact_save_failed"
                    break
            request_started_at = datetime.now(UTC).isoformat(timespec="milliseconds")
            trace_judge_stage(
                "batch_started",
                batch_no=batch_no,
                batch_start_zero_based=batch_start,
                expected_count=len(batch),
                details={"input": messages},
            )
            observed_started = time.perf_counter()
            observed_response = None
            observed_text = ""
            observed_error: BaseException | None = None
            observed_received = False
            observed_served = None
            from research_engine.response_diagnostics import (
                ResponseObservation,
                response_observation,
            )

            response_diag = ResponseObservation(
                lease.batch_id
                if lease is not None
                else long_run.canonical_hash(
                    {
                        "report": report_hash,
                        "sources": sources_hash,
                        "batch_start": batch_start,
                        "claims": batch_claims,
                        "route": JUDGE_ROUTE,
                        "prompt": JUDGE_PROMPT,
                    }
                )
            )
            preexisting_indices = set(verdict_by_claim)
            if ctx is not None:
                ctx.throttle.note_request_started()
            try:
                with response_observation(response_diag):
                    invoke = traced_ainvoke(
                        llm, messages, role="judge", configured_route=JUDGE_ROUTE
                    )
                    resp = (
                        await long_run.run_request(invoke, budget=ctx.budget)
                        if ctx is not None
                        else await invoke
                    )
                    from research_engine.response_diagnostics import observe

                    observe("after_conversion", resp)
                observed_response = resp
                observed_received = True
                _save(ctx, store, "response_received")
                text = resp.content if isinstance(resp.content, str) else ""
                observed_text = text
                served = served_model_id(resp)
                observed_served = served
                routing = gateway_routing(resp)
                # Batch-local labels only; `evals/judge_labels.py` is the one parser.
                from evals.judge_labels import claim_labels

                for local_idx, supported in claim_labels(text, len(batch)).items():
                    idx = batch_start + local_idx
                    verdict_by_claim[idx] = supported
                    served_by_claim[idx] = served
                    routing_by_claim[idx] = routing
            except asyncio.CancelledError as e:
                observed_error = e
                if ctx is None:
                    raise
                stop_reason = "cancelled"
            except long_run.CooperativeStop as e:
                observed_error = e
                stop_reason = e.reason
            except long_run.StageDeadlineExceeded as e:
                # The throttle wait consumed the last of the judge budget before this request.
                observed_error = e
                stop_reason = f"judge_deadline:{e}"
            except Exception as e:  # noqa: BLE001 — one failed batch must not sink the run
                observed_error = e
                # The batch's claims stay UNJUDGED and are excluded from the denominator.
                # A 429 paces the next request; it never triggers a duplicate paid call.
                if ctx is not None:
                    retry = ctx.throttle.note_rate_limit(e, status=long_run.http_status(e))
                    if retry is not None:
                        ctx.note(f"batch_{batch_no}_429_retry_after_s={retry}")
                print(f"Citation judging error (batch {batch_no}): {type(e).__name__}")
            finally:
                observed_ms = (time.perf_counter() - observed_started) * 1000
                trace_judge_batch(
                    batch_no=batch_no,
                    batch_start=batch_start,
                    expected_count=len(batch),
                    verdict_by_claim=verdict_by_claim,
                    response=observed_response,
                    text=observed_text,
                    error=observed_error,
                    elapsed_ms=observed_ms,
                    response_received=observed_received,
                    preexisting_indices=preexisting_indices,
                    served_model=observed_served,
                    response_diagnostic=response_diag.snapshot(
                        harness_text_chars=len(observed_text)
                        if observed_received
                        else "unavailable"
                    ),
                )

            local = {
                str(i): {
                    "supported": bool(verdict_by_claim[i]),
                    "served_model": served_by_claim.get(i),
                    "judge_routing": routing_by_claim.get(i),
                }
                for i in range(batch_start, batch_start + len(batch))
                if i in verdict_by_claim and i not in preexisting_indices
            }
            judged = len(local)
            unjudged = len(batch) - judged
            status = "complete" if unjudged == 0 else "unjudged" if judged == 0 else "partial"
            batch_statuses.append(status)
            attempted += 1
            if isinstance(observed_error, asyncio.CancelledError):
                reason_hint = "evaluation_cancelled"
            elif observed_error is not None:
                reason_hint = (
                    "response_processing_exception" if observed_received else "invoke_exception"
                )
            elif status == "complete":
                reason_hint = None
            elif not observed_text.strip():
                reason_hint = "empty_or_nontext_response"
            elif judged == 0:
                reason_hint = "unparseable_response"
            else:
                reason_hint = "partial_response"

            if store is not None:
                _save(
                    ctx,
                    store,
                    "save_batch",
                    batch_no,
                    {
                        "report_sha256": report_hash,
                        "sources_sha256": sources_hash,
                        "claim_sha256": long_run.canonical_hash(batch_claims),
                        "expected_count": len(batch),
                        "judged_count": judged,
                        "unjudged_count": unjudged,
                        "yes_count": sum(1 for v in local.values() if v["supported"]),
                        "no_count": sum(1 for v in local.values() if not v["supported"]),
                        "status": status,
                        "reason_hint": reason_hint,
                        "verdicts": local,
                        "served_model": observed_served,
                        "application_level_calls": 1,
                        "elapsed_ms": round(observed_ms, 3),
                        "error_type": type(observed_error).__name__ if observed_error else None,
                        "http_status": long_run.http_status(observed_error),
                        "started_at": request_started_at,
                        "response_received": observed_received,
                        "response_diagnostic": response_diag.snapshot(
                            harness_text_chars=len(observed_text)
                            if observed_received
                            else "unavailable"
                        ),
                        "tokens_input": token_usage(observed_response, "input_tokens"),
                        "tokens_output": token_usage(observed_response, "output_tokens"),
                        "cost_usd": None,
                        "finished_at": datetime.now(UTC).isoformat(timespec="milliseconds"),
                    },
                )
                if ctx is not None and not store.save_failed:
                    ctx.batches_persisted += 1
                    if status == "complete":
                        ctx.batches_completed += 1
            if lease is not None:
                lease.release()
                store.lease = None
            previous_batch_no = batch_no
            if stop_reason or (store is not None and store.save_failed):
                stop_reason = stop_reason or "artifact_save_failed"
                break

    if store is not None:
        store.lease = None
    rows = [
        {
            "claim": claim,
            # None means the judge never ruled on this claim. Distinct from False.
            "supported": verdict_by_claim.get(i),
            "judged": i in verdict_by_claim,
            "judged_by": served_by_claim.get(i),
            "judge_routing": routing_by_claim.get(i),
            "cites": metrics.extract_citations(claim),
        }
        for i, claim in enumerate(claims)
    ]

    if ctx is not None:
        if store is not None and store.save_failed:
            status = long_run.ARTIFACT_SAVE_FAILED
        elif not verdict_by_claim and not stop_reason:
            status = long_run.JUDGE_FAILED
        elif (
            stop_reason or attempted < total_batches or any(s != "complete" for s in batch_statuses)
        ):
            status = long_run.JUDGE_PARTIAL
        else:
            status = long_run.JUDGE_COMPLETED
        ctx.last_judge_status = status
        if status == long_run.JUDGE_PARTIAL:
            ctx.exit_code = long_run.EXIT_JUDGE_PARTIAL
        elif status == long_run.JUDGE_FAILED:
            ctx.exit_code = long_run.EXIT_JUDGE_FAILED
        elif status == long_run.ARTIFACT_SAVE_FAILED:
            ctx.exit_code = long_run.EXIT_ARTIFACT_SAVE_FAILED
        if store is not None:
            _save(
                ctx,
                store,
                "save_progress",
                status=status,
                reason=stop_reason,
                extra={"attempted_batches": attempted, "total_batches": total_batches},
            )
            _save(
                ctx,
                store,
                "save_final",
                status=status,
                reason=stop_reason,
                extra={"attempted_batches": attempted, "total_batches": total_batches},
            )

    if not verdict_by_claim:
        # Nothing was measured. Returning 0.0 here would publish an unmeasured run as a
        # total quality failure — the unmeasured-vs-zero conflation AGENTS.md calls a P0.
        return None, rows
    supported = sum(1 for ok in verdict_by_claim.values() if ok)
    return round(supported / len(verdict_by_claim), 4), rows


async def run_one_memory(query: dict) -> dict:
    import re

    from langchain_core.messages import HumanMessage, SystemMessage

    from research_engine.llm_factory import get_llm
    from research_engine.prompt_composition import system_prompt

    started = time.time()
    sources_json = json.dumps(query["excerpts"], indent=2)

    system = (
        f"{system_prompt('chat.project')}\n\n"
        f"--- EXCERPTS ---\n<untrusted_web_content>\n{sources_json}\n</untrusted_web_content>"
    )

    messages = [SystemMessage(content=system), HumanMessage(content=query["query"])]

    llm = get_llm("chat")
    error = None
    response_text = ""
    try:
        resp = await llm.ainvoke(messages)
        response_text = resp.content if isinstance(resp.content, str) else ""
    except Exception as e:
        error = str(e)[:300]

    latency = round(time.time() - started, 2)

    cite_re = re.compile(r"\[R\d+\]")
    has_citations = bool(cite_re.search(response_text))

    is_refusal = not has_citations and (
        "not cover" in response_text.lower()
        or "doesn't cover" in response_text.lower()
        or "does not cover" in response_text.lower()
        or "not answer" in response_text.lower()
        or "does not mention" in response_text.lower()
        or "no excerpts" in response_text.lower()
        or "not found" in response_text.lower()
        or "cannot answer" in response_text.lower()
        or "not explicitly mentioned" in response_text.lower()
        or "do not contain" in response_text.lower()
    )

    if query["type"] == "supported":
        pass_test = has_citations
    else:
        pass_test = is_refusal

    return {
        "id": query["id"],
        "type": query["type"],
        "completed": error is None,
        "error": error,
        "latency_s": latency,
        "response": response_text,
        "has_citations": has_citations,
        "is_refusal": is_refusal,
        "pass_test": pass_test,
    }


def aggregate(rows: list[dict]) -> dict:
    """The run's summary. Every rate over nothing is `None`, never `0.0` (RFC §14.3, AC-13).

    A completion rate over zero queries is not a rate of zero: `0.0` reads as "every query
    failed", and against `MIN_COMPLETION_RATE` it reports a failure that never happened.
    `citation_support_rate` and the averages already returned `None` on an empty input; the
    completion rate was the one that did not.
    """
    n = len(rows)
    done = [r for r in rows if r.get("completed")]
    completion_rate = round(len(done) / n, 4) if n else None

    def mean(key: str, source=done) -> float | None:
        vals = [r[key] for r in source if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    support_vals = [
        r["citation_support_rate"] for r in done if r.get("citation_support_rate") is not None
    ]
    citation_support = round(sum(support_vals) / len(support_vals), 4) if support_vals else None
    rulings = [c for r in done for c in r.get("claim_verdicts", [])]
    full_input_known = bool(done) and all(isinstance(r.get("claim_verdicts"), list) for r in done)
    claim_total = len(rulings) if full_input_known else None
    claim_judged = sum(c.get("judged") is True for c in rulings) if full_input_known else None

    return {
        "queries": n,
        "completion_rate": completion_rate,
        "avg_source_count": mean("source_count"),
        "avg_uncited_claims": mean("uncited_claim_count"),
        "avg_resolution_rate": mean("resolution_rate"),
        "avg_cost_usd": mean("cost_usd"),
        "avg_latency_s": mean("latency_s"),
        "citation_support_rate": citation_support,
        "claims_total": claim_total,
        "claims_judged": claim_judged,
        "full_judge_coverage": claim_judged / claim_total if claim_total else None,
    }


#: How an unmeasured value is shown to a person — the same words `benchmark.py` and
#: `retrieval.py` already use. The result file stores `null`; this is only the console's
#: rendering, so a reader of the terminal never sees a bare `null` and wonders whether it
#: meant zero (RFC §14.3: an unmeasured result is reported as `n/a (unmeasured)`).
UNMEASURED = "n/a (unmeasured)"


def _for_display(summary: dict) -> dict:
    """`summary` with every unmeasured value spelled out, for the console only."""
    return {k: (UNMEASURED if v is None else v) for k, v in summary.items()}


#: The memory eval's pass threshold. Named rather than left inline in `main()`, where it
#: was untestable; the value is unchanged.
MIN_MEMORY_PASS_RATE = 0.90


def aggregate_memory(rows: list[dict]) -> dict:
    """The memory eval's summary, with the same rule as `aggregate`: nothing measured is
    `None`. This was inline in `main()` and published `0.0` for both fields on an empty run."""
    n = len(rows)
    passed = sum(1 for r in rows if r.get("pass_test"))
    return {
        "queries": n,
        "pass_rate": round(passed / n, 4) if n else None,
        "avg_latency_s": round(sum(r.get("latency_s", 0) for r in rows) / n, 4) if n else None,
    }


def check_memory_criteria(agg: dict) -> dict:
    return {
        "pass_rate_ok": _meets(agg["pass_rate"], MIN_MEMORY_PASS_RATE),
        "thresholds": {"min_pass_rate": MIN_MEMORY_PASS_RATE},
    }


def _retriever_in_use() -> str:
    """Which search backend this run actually had available.

    Recorded because it materially changes the evidence a report is built from, and the
    keyless DuckDuckGo fallback is rate-limited enough to depress quality on its own — a
    published number without this is not reproducible.
    """
    if RUN_CONFIG.llm_mode == "fake":
        return "fixtures (no network)"
    if RUN_CONFIG.tavily_api_key:
        return "tavily (→ brave → duckduckgo fallback)"
    if RUN_CONFIG.brave_api_key:
        return "brave (→ duckduckgo fallback)"
    return "duckduckgo (keyless fallback — rate-limited)"


def _meets(value: float | None, threshold: float) -> bool | None:
    """`True`/`False` against an inclusive threshold, or `None` when nothing was measured.

    `None` is a third answer, not a failure: "could not tell" must never be reported as
    "fell short", or an empty run becomes indistinguishable from a regression.
    """
    return None if value is None else value >= threshold


def check_release_criteria(agg: dict) -> dict:
    result = {
        "completion_rate_ok": _meets(agg["completion_rate"], MIN_COMPLETION_RATE),
        "citation_support_ok": _meets(agg.get("citation_support_rate"), MIN_CITATION_SUPPORT),
        "full_judge_coverage_ok": _meets(agg.get("full_judge_coverage"), 1.0),
        "gate_version": "report-full-input/1",
        "thresholds": {
            "min_completion_rate": MIN_COMPLETION_RATE,
            "min_citation_support": MIN_CITATION_SUPPORT,
        },
    }
    checks = [
        result[k] for k in ("completion_rate_ok", "citation_support_ok", "full_judge_coverage_ok")
    ]
    result["gate_passed"] = False if False in checks else None if None in checks else True
    return result


async def _pace_between_queries(ctx) -> None:
    """Optional, recorded pacing — only when a next query exists."""
    if ctx is None:
        return
    seconds = min(long_run.query_interval(), ctx.budget.remaining("research"))
    if seconds <= 0:
        return
    record = await ctx.throttle.wait_fixed(seconds, reason="inter_query_throttle")
    ctx.budget.spend("research", record.actual_s)


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the report-quality eval suite.")
    parser.add_argument(
        "--mode",
        choices=["report", "memory"],
        default="report",
        help="Which eval to run (report or memory)",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=None, help="run only the first N queries")
    selection.add_argument(
        "--case-id", help="run exactly one original report case by ID; incompatible with --limit"
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="record runtime metadata (also RF_EVAL_TRACE=1; report mode only)",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="trace directory; default beside --out; also RF_EVAL_TRACE_DIR",
    )
    parser.add_argument(
        "--trace-content",
        action="store_true",
        help="enable trace with bounded redacted prompts, responses, tool inputs/outputs and errors",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="output path (default results/eval-<date>.json)"
    )
    parser.add_argument("--date", type=str, default=None, help="override the run date (YYYY-MM-DD)")
    parser.add_argument(
        "--candidate",
        type=str,
        default=None,
        help="JSON file of {role: prompt}. Runs the query set twice — shipped prompts, then "
        "this spec — and reports both against the same thresholds (report mode only)",
    )
    parser.add_argument(
        "--baseline",
        help="Optional approved {role: prompt} spec to compare with --candidate in report mode",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=None,
        help="provider:model of the citation judge (default EVAL_JUDGE_MODEL). Required for a real-mode report eval, and "
        "refused if it is a router alias or any model the pipeline runs on (RG-5)",
    )
    parser.add_argument(
        "--progress-dir",
        type=Path,
        default=None,
        help="atomic reliability artifacts (default RF_EVAL_RELIABILITY_DIR); off when unset",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="a previous --progress-dir to resume; validates report and judge config first",
    )
    parser.add_argument(
        "--rejudge-from",
        type=Path,
        default=None,
        help="reuse a validated draft with a new output budget; rescore ALL judge batches",
    )
    args = parser.parse_args()
    if args.case_id is not None and args.mode != "report":
        parser.error("--case-id supports --mode report only; nothing was run")
    if args.rejudge_from is not None and args.mode != "report":
        parser.error("--rejudge-from supports --mode report only")
    queries_file = EVALS_DIR / ("memory_queries.json" if args.mode == "memory" else "queries.json")
    try:
        queries = select_queries(
            json.loads(queries_file.read_text(encoding="utf-8-sig"))["queries"],
            case_id=args.case_id,
            limit=args.limit,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"{exc}; no queries were run")
    tracing = args.trace or args.trace_content or os.environ.get("RF_EVAL_TRACE") == "1"
    if tracing and args.mode != "report":
        parser.error("runtime tracing currently supports --mode report only")

    # Before any query runs: a run that would be self-judged must cost nothing.
    judging = args.mode == "report" and RUN_CONFIG.llm_mode == "real"
    if judging:
        args.judge = args.judge or os.environ.get("EVAL_JUDGE_MODEL")
        if not args.judge:
            parser.error(
                "a real-mode report eval needs --judge provider:model or EVAL_JUDGE_MODEL, a pinned model "
                "independent of the pipeline's routes (RG-5); nothing was run"
            )
        try:
            configure_judge(args.judge)
        except JudgeConfigError as e:
            parser.error(f"{e}; nothing was run")
    elif args.judge:
        parser.error("--judge applies to a real-mode report eval; this run judges nothing")

    candidate = None
    baseline = None
    if args.baseline and (not args.candidate or args.mode != "report"):
        parser.error("--baseline requires --candidate in report mode")
    if args.candidate:
        if args.mode != "report":
            parser.error("--candidate applies to the report eval (RFC §14.2), not --mode memory")
        try:
            candidate = load_candidate(Path(args.candidate))
            if args.baseline:
                baseline = load_candidate(Path(args.baseline))
        except CandidateSpecError as e:
            parser.error(str(e))

    run_date = args.date or datetime.now(UTC).strftime("%Y-%m-%d")
    out = Path(args.out) if args.out else _result_path(run_date, custom_spec=candidate is not None)
    try:
        _check_output_path(out)
    except FileExistsError as exc:
        raise SystemExit(f"{exc} No queries were run.") from exc
    except OSError as exc:
        parser.error(
            f"cannot write evaluation result to {out}: {exc}. "
            "Pass --out in a writable directory; no queries were run. "
            "In Docker, use --out /tmp/eval.json and keep the container for docker cp."
        )

    progress_root = args.progress_dir or _env_path("RF_EVAL_RELIABILITY_DIR")
    resume_root = args.resume_from or _env_path("RF_EVAL_RESUME_FROM")
    if args.rejudge_from is not None:
        if resume_root is not None:
            parser.error("--rejudge-from cannot be combined with --resume-from")
        resume_root = args.rejudge_from
    if resume_root is not None and progress_root is None:
        raise SystemExit("resume requires a separate --progress-dir; no queries were run")
    ctx = None
    if progress_root is not None:
        provider = (JUDGE_ROUTE or RUN_CONFIG.models["critic"]).partition(":")[0] or "unknown"
        try:
            # A second real write, before any model call: a mounted directory that exists
            # but refuses writes must fail here, not after a 180 s research run (AGENTS.md).
            ctx = long_run.build_context(
                root=progress_root,
                judge_provider=provider,
                resume_root=resume_root,
                case_count=len(queries) * (2 if candidate is not None else 1),
            )
            long_run.preflight_root(progress_root)
            ctx.rejudge = args.rejudge_from is not None
        except (OSError, long_run.ReliabilityError) as exc:
            raise SystemExit(f"reliability preflight failed: {exc}; no queries were run") from exc
        long_run.activate(ctx)
        if resume_root is not None:
            print(f"Resuming from {resume_root} (report + judge config must match)…")

    trace_kwargs = {}
    if tracing:
        trace_kwargs = {
            "trace_dir": args.trace_dir
            or Path(os.environ.get("RF_EVAL_TRACE_DIR") or out.parent / "traces"),
            "trace_run_id": os.environ.get("RF_EVAL_TRACE_RUN_ID") or uuid.uuid4().hex,
            "trace_artifact_root": out.parent,
            **({"trace_content": True} if args.trace_content else {}),
        }
    elif judging and os.environ.get("RF_JUDGE_BATCH_TRACE_DIR"):
        trace_kwargs = {"trace_artifact_root": out.parent}

    print(f"Running {len(queries)} queries in LLM_MODE={RUN_CONFIG.llm_mode} mode={args.mode}…")
    rows = []
    for index, q in enumerate(queries):
        if ctx is not None and ctx.stop.requested:
            print(f"  cooperative stop before {q['id']}: {ctx.stop.reason}")
            break
        if args.mode == "memory":
            row = await run_one_memory(q)
            rows.append(row)
            flag = "✓" if row.get("pass_test") else "✗"
            print(f"  {flag} {q['id']:24s} type={q['type']} latency={row.get('latency_s')}s")
        else:
            baseline_kwargs = (
                {"prompt_overrides": baseline, "evaluation_arm": "baseline"}
                if baseline is not None
                else {}
            )
            row = await run_one(q, **baseline_kwargs, **trace_kwargs)
            rows.append(row)
            flag = "✓" if row.get("completed") else "✗"
            print(
                f"  {flag} {q['id']:24s} sources={row.get('source_count')} "
                f"uncited={row.get('uncited_claim_count')} cost=${row.get('cost_usd')}"
            )

        # No unconditional 60 s sleep after every query (the last one included). Pacing is
        # configurable, recorded, and applied only when another query actually follows.
        if index < len(queries) - 1:
            await _pace_between_queries(ctx)

    candidate_rows: list[dict] = []
    if candidate is not None:
        roles = ", ".join(sorted(candidate))
        print(f"\nCandidate spec ({roles}) — the same {len(queries)} queries, the same thresholds…")
        for index, q in enumerate(queries):
            if ctx is not None and ctx.stop.requested:
                print(f"  cooperative stop before candidate {q['id']}: {ctx.stop.reason}")
                break
            row = await run_one(q, prompt_overrides=candidate, **trace_kwargs)
            candidate_rows.append(row)
            flag = "✓" if row.get("completed") else "✗"
            print(
                f"  {flag} {q['id']:24s} sources={row.get('source_count')} "
                f"uncited={row.get('uncited_claim_count')} cost=${row.get('cost_usd')}"
            )
            if index < len(queries) - 1:
                await _pace_between_queries(ctx)

    if args.mode == "memory":
        agg = aggregate_memory(rows)
        release_criteria = check_memory_criteria(agg)
    else:
        agg = aggregate(rows)
        release_criteria = check_release_criteria(agg)

    payload = {
        "date": run_date,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "llm_mode": RUN_CONFIG.llm_mode,
        "mode": args.mode,
        # In fake mode `llm_factory` returns `fake_model(role)` and no provider is contacted,
        # so writing five real-looking ids here would record models we never called —
        # the exact rule in AGENTS.md ("never record a model id you did not actually
        # call"). eval-2026-07-23.json is the artifact that did it.
        "models": (
            {"_note": "fake mode — no provider was called"}
            if RUN_CONFIG.llm_mode == "fake"
            else {role: RUN_CONFIG.models[role] for role in _ROLES}
        ),
        "method": {
            "metrics_version": metrics.METRICS_VERSION,
            "retriever": _retriever_in_use() if args.mode == "report" else "none (static memory)",
            "query_set": f"evals/{'memory_' if args.mode == 'memory' else ''}queries.json",
            # How the pipeline was driven — read from the environment (research_engine/local.py),
            # so a result that omitted them could not be reproduced from its own file. The
            # release run sets `max_parallel_tasks` to 1 so a provider sees sequential calls,
            # not bursts; that changes wall-clock and nothing the harness measures.
            "limits": {
                "max_parallel_tasks": RUN_CONFIG.max_parallel_tasks,
                "max_critic_loops": RUN_CONFIG.max_critic_loops,
                "max_cost_per_session_usd": RUN_CONFIG.max_cost_per_session_usd,
                "max_wallclock_seconds": RUN_CONFIG.max_wallclock_seconds,
                "max_input_tokens": RUN_CONFIG.max_input_tokens,
            },
        },
        "aggregate": agg,
        "release_criteria": release_criteria,
        "results": rows,
    }
    if candidate is not None:
        # The top-level `aggregate`/`release_criteria`/`results` stay the baseline,
        # exactly as in a plain run, so a reader of the existing schema reads it unchanged.
        payload["candidate"] = candidate_section(candidate, candidate_rows)
        payload["method"]["case_set_sha256"] = long_run.canonical_hash(queries)
        payload["method"]["baseline_prompt_sha256"] = {
            role: long_run.sha256_text(text)
            for role, text in {"planner": PLANNER_PROMPT_V2, **(baseline or {})}.items()
        }
    if args.case_id is not None:
        payload["method"]["case_id"] = args.case_id
    if judging:
        # Only where a judge ran. A scripted run calls no judge, and naming one it never
        # called is the "model id you did not call" rule the `models` note above follows.
        payload["judge"] = judge_section(
            JUDGE_ROUTE, rows, candidate_rows if candidate is not None else None
        )

    if ctx is not None:
        judged_rows = rows + candidate_rows
        judge_states = [
            r.get("judge_status") for r in judged_rows if r.get("completed") and judging
        ]
        if ctx.save_failed:
            reliability_status = long_run.ARTIFACT_SAVE_FAILED
        elif any(r.get("research_state") == long_run.RESEARCH_FAILED for r in judged_rows):
            reliability_status = long_run.RESEARCH_FAILED
        elif not judging:
            reliability_status = (
                long_run.RESEARCH_DRAFT_READY
                if all(r.get("completed") for r in judged_rows)
                else long_run.RESEARCH_FAILED
            )
        elif not judge_states or any(s == long_run.JUDGE_FAILED for s in judge_states):
            reliability_status = long_run.JUDGE_FAILED
        elif any(s != long_run.JUDGE_COMPLETED for s in judge_states):
            reliability_status = long_run.JUDGE_PARTIAL
        else:
            reliability_status = long_run.JUDGE_COMPLETED
        reliability_reason = ctx.stop.reason or (
            "incomplete_judge"
            if reliability_status
            in (long_run.JUDGE_PARTIAL, long_run.JUDGE_FAILED, long_run.ARTIFACT_SAVE_FAILED)
            else None
        )
        try:
            long_run.write_root_status(
                progress_root,
                status=reliability_status,
                reason=reliability_reason,
                extra={"exit_code": ctx.exit_code, "judge_route": JUDGE_ROUTE},
            )
        except OSError as exc:
            reliability_status = long_run.ARTIFACT_SAVE_FAILED
            ctx.exit_code = long_run.EXIT_ARTIFACT_SAVE_FAILED
            ctx.note(f"root_status_failed:{type(exc).__name__}")
        payload["reliability"] = long_run.reliability_section(
            ctx, status=reliability_status, reason=reliability_reason
        )
        # A partial measurement must not clear the 0.95 gate on the few claims that
        # happened to answer. Legacy `aggregate`/`judge` fields stay exactly as they were.
        if reliability_status in (
            long_run.JUDGE_PARTIAL,
            long_run.JUDGE_FAILED,
            long_run.ARTIFACT_SAVE_FAILED,
        ):
            release_criteria["citation_support_ok"] = None
            if candidate is not None:
                payload["candidate"]["release_criteria"]["citation_support_ok"] = None

    # 检查后仍可能有另一个进程占用同名路径；独占创建保护既有评测证据。
    with trace_span("result.save", kind="artifact_write", inputs={"case_count": len(rows)}):
        with out.open("x", encoding="utf-8") as result_file:
            result_file.write(json.dumps(payload, indent=2) + "\n")
    print(f"\nAggregate: {json.dumps(_for_display(agg))}")
    if judging:
        judge = payload["judge"]
        served = sorted(
            {
                m
                for part in ("baseline", "candidate")
                for m in judge.get(part, {}).get("served_models", [])
            }
        )
        print(f"Judge: {judge['route']} (answered as: {', '.join(served) or 'undisclosed'})")
    print(f"Release criteria: {json.dumps(_for_display(payload['release_criteria']))}")
    if candidate is not None:
        section = payload["candidate"]
        print(f"Candidate aggregate: {json.dumps(_for_display(section['aggregate']))}")
        print(f"Candidate criteria: {json.dumps(_for_display(section['release_criteria']))}")
    # Show a repo-relative path when the output is inside the repo, and the plain path
    # when it isn't — `relative_to` raises on an outside path, which used to crash the
    # run *after* the results file had already been written.
    try:
        shown = out.relative_to(EVALS_DIR.parent)
    except ValueError:
        shown = out
    print(f"Wrote {shown}")
    if ctx is not None and ctx.exit_code:
        # A named partial/failed state, not a silent success. The result and all progress
        # artifacts are already on disk before this non-zero exit.
        raise SystemExit(ctx.exit_code)


async def main() -> None:
    root = os.environ.get("RF_EVAL_EXECUTION_ROOT")
    try:
        if root and os.environ.get("RF_EVAL_CONTENT_TRACE") == "1":
            from evals.agent_eval_ext.execution_observer import execution_scope

            with execution_scope(
                Path(root),
                os.environ.get("RF_EVAL_TRACE_RUN_ID") or uuid.uuid4().hex,
                "harness.run",
            ):
                await _main()
        else:
            await _main()
    except long_run.ResumeRejected as exc:
        ctx = long_run.current()
        if ctx is not None:
            try:
                long_run.write_root_status(
                    ctx.root,
                    status=long_run.JUDGE_FAILED,
                    reason=f"resume_rejected:{exc}",
                    extra={"exit_code": long_run.EXIT_JUDGE_FAILED},
                )
            except OSError:
                pass
        print(f"Resume refused: {exc}; no model calls were made")
        raise SystemExit(long_run.EXIT_JUDGE_FAILED) from exc
    finally:
        long_run.deactivate()


if __name__ == "__main__":
    asyncio.run(main())
