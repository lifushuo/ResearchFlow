# Testing and evaluation

Two different questions, deliberately kept apart:

- **Tests** ask *does it work* — deterministic, free, and on every commit.
- **Evaluations** ask *is the output any good* — real models, real money, run on purpose.

## The pyramid

| Layer | Tooling | Scope | Speed |
|---|---|---|---|
| Unit (backend) | pytest | Schemas, guards, cost maths, citation and prompt validators, chunking, normalisation | ms |
| Integration (backend) | pytest + httpx ASGI client + real Postgres/Redis | API endpoints, auth flows, migrations, project isolation | s |
| Pipeline (backend) | pytest with scripted models | The compiled graph end to end, no network and no keys | s |
| Unit (frontend) | Vitest + Testing Library | SSE parser, citation renderer, pipeline derivations, formatters | ms |
| Golden end-to-end | Playwright against the real stack in fake mode | Whole user journeys through a real browser | min |
| Evaluations | Custom harness | Report quality against a fixed query set with real models | min–$ |

`LLM_MODE=fake` is what makes the middle layers deterministic and free: the factory returns
scripted models and the retriever chain returns fixture results, so no network call and no
API key is involved.

## The golden journeys

Five journeys run in CI against the packaged stack, and they encode the product's promises.

| Journey | What it proves |
|---|---|
| **0 — the design gate** | Editing the plan before any search runs changes the run, then the draft is reached |
| **1 — research reaches the gate** | Register → submit → live pipeline events over SSE → a cited draft renders |
| **2 — approval completes the session** | Approving finalises **without re-running research**, and Markdown export downloads |
| **3 — rework loops and chat works** | Rejecting with feedback re-runs synthesis, the gate is re-reached, approval works, and chat answers from the report |
| **6 — corpus documents preview in place** | A document is uploaded and read without leaving the page |

Journey 2 is the load-bearing one: it asserts that resume enters the graph **at the gate**,
not at the planner.

```bash
cd frontend && npm run e2e
```

**Running it locally.** The suite drives the real stack, so bring one up first — `./start.sh
--fake` is enough, and fake mode is what CI uses. Then point Playwright at it:

```bash
E2E_NO_SERVER=1 E2E_BASE_URL=http://localhost:3031 npm run e2e
```

Omit `E2E_NO_SERVER` and Playwright builds and starts the frontend itself; set it when one
is already running, which is faster and what CI does.

**Repeated local runs are safe, and that is not an accident.** Registration is capped at 5
per IP per hour and the cap is *not* configurable — it is brute-force protection, and one a
test suite can switch off is not a cap. The suite registers one account per journey and has
more journeys than that, so it would exhaust the cap on the first run and fail on the
second. `e2e/fixtures.ts` therefore deletes this suite's own `rl:*` counters before each
journey; it never issues `FLUSHDB`, because the same Redis carries the SSE channels and the
search cache and blowing those away would change what the run under test actually does.

If you do hit the cap — running one spec against a stack whose Redis the fixture cannot
reach, say — the failure now names itself rather than surfacing as a navigation timeout.
Clear the counters by hand with:

```bash
redis-cli -n 1 --scan --pattern 'rl:*' | xargs -r redis-cli -n 1 del
```

The cap itself stays armed: `backend/tests/security/test_auth_rate_limit.py` proves a sixth
registration from one address is refused, and pins the published limit so widening it has
to be deliberate.

CI starts the API and the worker in fake mode, then Playwright. It waits for an explicit
worker preload signal rather than for a ping — a Celery worker answers a ping while LangGraph
and the provider clients are still importing, which measurably cost tens of seconds on the
first task and made these journeys look flaky when the real cause was a worker that had not
finished booting.

## Testing an absence

A recurring trap worth its own section. When you write a test for an *absence* — no network
egress, no writes, no spend — check what the fixtures replaced.

A test once asserted zero network calls in corpus mode while injecting a fake embedder, and
the query embedding was the only call that egressed. The suite was green precisely because it
had replaced the defect. **If the mechanism under test is the thing you mocked, the test is
decorative.**

## What CI enforces

Run what CI runs, not what is convenient:

```bash
cd backend && ruff check app/ research_engine/ tests/ evals/ \
  && ruff format --check app/ research_engine/ tests/ evals/ \
  && python -m pytest
```

```bash
cd frontend && npm run lint && npm run typecheck && npm test && npm run build
```

Plus, in the frontend job, [four bespoke greps](32-development.md#the-four-greps-that-fail-the-build)
that `npm run lint` does not run.

Jobs: `backend`, `frontend`, `golden-e2e`, and `eval-artifacts`. Dependency audits
(`pip-audit`, `npm audit`) run non-blocking.

### Evaluation results are write-once

The `eval-artifacts` job fails the build if a committed file under `backend/evals/results/`
is *modified*. Adding a new result is always fine; changing one is not.

This exists because a frontend commit once silently overwrote a real 10/10 measurement with a
failed run, leaving the README citing numbers whose proof no longer existed. Nothing failed
and nothing warned. Record a new run under a new name — `eval-<date>-<routing>-run<N>.json`,
because a date alone is not a run identity.

## Evaluations

The harness runs a versioned query set through the compiled graph up to the review gate and
records per-report metrics to a dated JSON file, so report quality is diffable over time.

```bash
make eval                                   # fake mode: deterministic, free, no keys
LLM_MODE=real make eval EVAL_ARGS="--judge anthropic:claude-sonnet-4-6"   # real models
```

Mode is read from the *real* environment before `.env` is loaded, and `.env` supplies keys
only. That ordering is deliberate: a developer `.env` commonly carries `LLM_MODE=real` for
the app, and letting that reach the harness would silently turn `make eval` — documented and
relied on as the free default — into a run that spends money on every invocation. **Spending
is opt-in, per invocation.**

### Metrics

| Metric | Definition |
|---|---|
| Completion rate | Fraction of queries that reached a report |
| Citation resolution rate | In-text `[n]` markers that resolve to a real source |
| Citation support rate | Cited claims an independent judge rules are actually supported by their snippet |
| Uncited claims | Assertive sentences carrying no citation marker |
| Contradictions surfaced | Distinct conflicting pairs shown to the reader |
| Source count, cost, latency | Per report |

Every run records its own method block and a `metrics_version`, bumped whenever a definition
changes, so two runs are never silently compared across incompatible metrics.

### Retrieval quality

Report quality is bounded by retrieval quality — [`docs/01`](../getting-started/01-overview.md),
[`docs/10`](../project/10-roadmap.md), [`docs/21`](../getting-started/21-configuration.md) and
[`docs/33`](33-contributing.md) all say so — and for a long time nothing measured it. A second
harness does now, separate from the report harness above because it answers a narrower
question: given a query, does the retriever return the right documents?

```bash
python -m evals.retrieval             # measure and print
python -m evals.retrieval --write     # also record a write-once result file
```

It measures; it does not rank. Queries go through `CorpusStore.search` — the same call the
executor makes — so what is recorded is production behaviour rather than a reimplementation
of it. The dataset is a frozen 16-document corpus with a fingerprint, so a changed corpus
cannot be mistaken for a changed retriever.

| Metric | Definition |
|---|---|
| recall@k | Of the documents that should be found, the fraction retrieved in the top *k* |
| precision@k | Of the top *k* retrieved, the fraction that should have been found |
| nDCG@10 | Rank-aware relevance — being right at position 1 counts for more than at 10 |
| dedup rate | Near-duplicate documents collapsed rather than returned twice |

**A real embedder is required, and its absence is reported rather than filled in.**
`FakeEmbeddings` hashes text with blake2b — deterministic, and carrying no semantic signal —
so recall over those vectors would be a number with no meaning attached. With no embedding
endpoint reachable the harness refuses and records `"status": "unmeasured"` with the reason.
That is the same rule as [Unmeasured is not zero](#unmeasured-is-not-zero), one level up.

The recorded baseline
(`backend/evals/results/retrieval-2026-09-09-corpus-local.json`, `ollama:nomic-embed-text`,
12 queries):

| recall@1 | precision@1 | nDCG@10 | dedup rate |
|---|---|---|---|
| 0.792 | 0.917 | 0.944 | 0.000 |

This baseline is the gate a retrieval change has to clear: no change merges without a
measured, committed result either side of it. Hybrid retrieval was built and measured against
exactly this and did not beat it, so it was rejected — see
[the roadmap](../project/10-roadmap.md).

#### Retrieval over selected files

```bash
python -m evals.selection_retrieval           # measure and print
python -m evals.selection_retrieval --write   # also record a write-once result file
```

A run over selected files searches only those files (`ScopedCorpus.search`). This eval
checks two things about that path, on the same embedder:

- **Leakage.** The share of retrieved chunks from a file outside the selection. **The target
  is exactly 0**, and the command exits non-zero when any query leaks. The dataset
  (`evals/datasets/selection-retrieval/`) is built so a leak has something to find: one
  fixture of every shipped format plus the corpus above, where unselected files often
  answer the query better, and two selections leave out an identical copy of the chosen
  file.
- **The per-file hit cap.** A selection takes `max(1, ceil(k / files))` hits from each file,
  where every search used to take one. It is measured against one hit per file. Document
  rankings are identical by construction, so the measure is **passage recall**: whether
  every passage a question needs comes back, judged by verbatim anchors.

The recorded result (`backend/evals/results/retrieval-2026-10-28-selection-local.json`,
`ollama:nomic-embed-text`, 16 queries, 35 documents):

| | leakage (max) | passages per query | passage recall | nDCG@5 |
|---|---|---|---|---|
| per-file cap | **0.000** (0 of 16 queries) | 2.81 | **1.000** | 0.963 |
| one hit per file | — | 1.94 | 0.458 | 0.963 |

### Unmeasured is not zero

The rule the whole harness is built around, and the reason it exists at all.

A provider error, a partial model reply, and an unparseable answer must not each silently
contribute a *miss* to a quality score — that makes an exhausted quota indistinguishable from
a quality collapse. So:

- unjudged claims are **excluded from the denominator**, not counted as failures;
- a metric returns `None` when nothing could be judged, and `None` renders as
  `n/a (unmeasured)`, never `0.0`;
- a **rate over nothing is `None` too** — a completion or pass rate over zero queries, and an
  average over none. `0.0` completion would read as "every query failed" and report a
  failure against the threshold that never happened;
- a threshold check has **three answers**: met, missed, or `None` when there was nothing to
  measure. `None` is never reported as a miss;
- a trace records the model that actually answered, never the one that was requested.

The result file stores an unmeasured value as `null`; the console prints it as
`n/a (unmeasured)`, the same words the harness, the benchmark and the retrieval eval all use.

**The rule has two homes** — the harness and the benchmark — and they must change together.

### The judge

裁判可单独使用阿里云 Kimi K3，研究流程仍保持原来的 `MODEL_*` 路由。
在仓库根目录 `.env` 设置以下变量（真实密钥只写入本地 `.env`）：

```ini
ALIYUN_API_KEY=填写百炼密钥
EVAL_JUDGE_MODEL=custom:kimi-k3
EVAL_JUDGE_BASE_URL=https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
EVAL_JUDGE_API_KEY=${ALIYUN_API_KEY}
```

Base URL 应以百炼控制台为准；上例来自 [Kimi 官方文档](https://help.aliyun.com/zh/model-studio/kimi-api)，
需要替换业务空间 ID。`ALIYUN_API_KEY` 通过 dotenv 插值提供给裁判，不能仅添加它而不配置
`EVAL_JUDGE_API_KEY`。裁判的地址和密钥必须成对提供，构建客户端后立即恢复研究模型配置。

`evals.harness` 在显式启用 `LLM_MODE=real` 的报告评测中读取 `EVAL_JUDGE_MODEL`，
命令行 `--judge` 优先；默认 fake 运行仍不调用裁判。
`evals.benchmark` 的路由优先级为 `BENCHMARK_JUDGE_MODEL`、`EVAL_JUDGE_MODEL`、原有默认值。
这两个入口均使用上述独立端点；已有结果及 grounding 数据集的冻结标签不会被修改。
更换裁判后，应另存新的结果文件，并记录裁判路由，避免把不同裁判的分数直接当作质量变化。

Windows 的 Docker 挂载目录可能拒绝容器写入，即使容器内使用 root。
报告评测会在调用模型前试写所选输出目录，已存在的结果文件也会提前拒绝；
指定 `--out` 后不再要求默认 `evals/results` 目录可写。
建议将源码只读挂载，结果写入容器内 `/tmp/eval.json`，并保留容器直到导出完成。
以下 PowerShell 命令执行一道真实研究题，再由 Kimi K3 评分，会产生 API 费用：

```powershell
Set-Location D:\work\ResearchFlow
$runId = Get-Date -Format 'yyyyMMddHHmmssfff'
$container = "researchflow-judge-$runId"
$out = ".\backend\evals\results\eval-$(Get-Date -Format 'yyyy-MM-dd')-deepseek-kimi-k3-run$runId.json"
docker run --name $container --env LLM_MODE=real --env MAX_PARALLEL_TASKS=1 `
  --mount "type=bind,source=D:\work\ResearchFlow,target=/workspace,readonly" `
  --workdir /workspace/backend researchflow-worker `
  python -m evals.harness --mode report --limit 1 --judge custom:kimi-k3 --out /tmp/eval.json
if ($LASTEXITCODE -ne 0) { throw "评测失败；容器 $container 已保留，可用 docker logs 检查" }
if (Test-Path -LiteralPath $out) { throw "目标结果已存在，请另选文件名；容器已保留" }
docker cp "${container}:/tmp/eval.json" $out
if ($LASTEXITCODE -ne 0) { throw "复制失败；容器 $container 已保留，可重新执行 docker cp" }
Write-Output "结果已导出到 $out；容器 $container 保留至确认后再删除"
```

`cost_usd=0.0` 对 `custom:` 路由表示缺少可用的价格目录，不能据此认定本次调用免费。

Citation support is graded by the model `--judge` or `EVAL_JUDGE_MODEL` names, and a real-mode report eval refuses
to start without one. It is refused, before a single query runs, if it is:

- a route the pipeline itself runs, or the same model reached another way
  (`openrouter:google/gemini-2.5-flash` is `google:gemini-2.5-flash`);
- a router alias (`auto/*`), which can resolve to a different model on every call;
- set while any pipeline role is on a router alias, since then nothing can be shown to be
  independent of it.

Judging never falls back to the pipeline's own critic. That was the harness's behaviour until
V3, which is why the runs recorded before it are self-judged.

The result file names the judge in a `judge` block beside `models`, never inside it:
`models` is the system under evaluation, and `judge` is what graded it. The block holds:

- the judge's route;
- the routes it was checked against;
- for the baseline and any candidate, how many claims it ruled on, how many it never
  reached, and which model the provider reported answering;
- whether it answered at all.

Each claim verdict also records the model that served it. That name is read from the
provider's response, never copied from the route: a judge that answered as something else,
or did not disclose what answered, shows it. A scripted run calls no judge and writes no
`judge` block.

### The release run

RG-5 is this harness on the fixed query set with a named judge and the committed candidate,
nothing more. The pipeline runs one research task at a time: a hosted route that throttles
bursts (Antigravity via OmniRoute returned empty responses after a burst of parallel calls in
the preflight) would otherwise fail queries for reasons that have nothing to do with quality.

```bash
cd backend
LLM_MODE=real MAX_PARALLEL_TASKS=1 \
  MODEL_PLANNER=custom:antigravity/gemini-2.5-flash MODEL_EXECUTOR=custom:antigravity/gemini-2.5-flash \
  MODEL_CRITIC=custom:antigravity/gemini-2.5-flash MODEL_SYNTHESIZER=custom:antigravity/gemini-2.5-flash \
  MODEL_CHAT=custom:antigravity/gemini-2.5-flash \
  python -m evals.harness --judge custom:kiro/claude-sonnet-4.5 \
    --candidate evals/candidates/v3-release-planner.json
```

Parallelism changes wall-clock, not what is measured, and the result records it: `method.limits`
holds every execution limit the run used, so a result can be reproduced from its own file.

### Release criteria

Citation support ≥ 0.95 and completion ≥ 0.90 on the fixed set. Both are inclusive — a run at
exactly the threshold meets it — and both are fixed: nothing a user configures changes them. The V3.0.0
release run clears both under an independent judge; see
[Citation-fidelity benchmark](../research/16-citation-fidelity-benchmark.md) for what has
actually been measured, and under what caveats.

### Custom-spec evaluation

A user can replace a role's system prompt, so the harness can measure a replacement against
the shipped prompts:

```bash
cd backend && python -m evals.harness --candidate spec.json   # {"planner": "…", "critic": "…"}
```

It runs the fixed ten-query set twice — once with the shipped prompts, once with the
candidate — and judges both against **the same thresholds**. The result file keeps the
shipped run where a plain run puts it (`aggregate`, `release_criteria`, `results`) and adds
the candidate beside it under `candidate`, with each replaced prompt and its SHA-256. It is
written to its own write-once file, `eval-<date>-<routing>-custom.json`, so it can never land
on a plain run's result.

**A candidate below threshold is reported, not blocked.** The user chose it; the harness's
job is to measure honestly, not to refuse. A threshold miss never stops the process — the
only exit is the write-once guard.

**The candidate is refused, not quietly substituted, if it cannot be used.** It is validated
by the same all-or-nothing rule a run's own override snapshot goes through. In production an
unusable snapshot falls back to the shipped prompts and records that it did; here the same
fallback would publish the baseline's numbers under the candidate's name. It comes from the
file you pass, never from a user's stored preferences, and applies to the report eval only.

#### What the candidate can and cannot reach

A user's prompt may change **what the pipeline produces**, never **what the harness
measures**. Two independent protections keep the judge out of reach:

- **Scope.** The candidate is installed around the pipeline run alone and removed before the
  judge is called. It is never installed process-wide, which would outlive the query and
  still be live when the judge ran. Both runs are built from the same base configuration, so
  they differ in their prompts and nothing else — the same models, the same limits.
- **Construction.** Both judges carry their system prompt as literal text and never compose
  one through the prompt machinery. A replaced prompt has no path into a judge's messages
  even if it were live at the moment the judge ran.

The project-chat prompt the memory eval composes is a protected purpose, so no override
reaches it either.

#### What scripted mode proves, and what it does not

`make eval` runs scripted models, and so does CI. That proves the **mechanism** — the
candidate reaches the pipeline, both runs are recorded, the same thresholds apply, the judge
never sees the candidate. It does **not** show that one prompt is better than another: the
judge runs only against a real provider, so citation support is unmeasured for both runs, and
the scripted models answer by role, so a candidate produces the baseline's output. Comparing
the quality of two specs needs a real-model run.

### Experimental offline evaluation audit

The additive `evals.agent_eval_ext` CLI reads existing report results without calling models:

```bash
cd backend
python -m evals.agent_eval_ext.cli audit --input evals/results/YOUR_RUN.json
python -m evals.agent_eval_ext.cli judge-batches --input evals/results/YOUR_RUN.json --batch-size 4
python -m evals.agent_eval_ext.cli gate --input evals/results/YOUR_RUN.json
python -m evals.agent_eval_ext.cli compare --baseline evals/results/BASELINE.json --candidate evals/results/YOUR_RUN.json
python -m evals.agent_eval_ext.cli calibration --input evals/agent_eval_ext/fixtures/judge_calibration.jsonl
```

The audit separates support among judged claims from judge coverage. Unjudged claims remain
unmeasured, never failures. A report missing its claim records leaves overall coverage
unmeasured. Audit support is claim-weighted; the official aggregate may average per-report
rates, so the audit preserves the original aggregate separately. Experimental artifacts use
`extension_schema_version=1`, without changing the official metric definitions.

The separate experimental gate adds coverage ≥0.98 to completion ≥0.90 and support ≥0.95.
This is a proposed reliability threshold, not a change to the official release criteria.
Gate and tool-trajectory commands exit 1 on failure; `--out NEW_FILE.json` saves the analysis
before that exit and always refuses an existing file. No output option means stdout only.

`judge-batches` infers groups from saved verdict order and an assumed historical batch
size (default 4). A measured claim requires both `judged=true` and a boolean `supported`;
NO counts as judged. It reports complete, partial and wholly unjudged groups with 1-based
claim positions, including short final groups. Missing claim records in completed reports
leave overall coverage unmeasured. Its input SHA256 and grouping assumptions are recorded.
These groups are not observed API requests: historical JSON cannot establish raw replies,
HTTP/parse failures, retries or root causes. The command adds no runtime logging or retries
and makes no model calls. The standalone `evals.agent_eval_ext.judge_batch_audit` entry point
uses the same input validation and write-once artifact format.

Comparison aligns case IDs and reports configuration differences. A supplied dataset file
only fingerprints that file today; without a dataset hash recorded in both original runs
and prompt snapshot hashes, comparison cannot claim matched controls. Live web drift still
prevents attributing score differences to a code change alone. Nested custom-spec candidate
results are not selected automatically: this adapter audits the top-level shipped baseline.

The 24 synthetic calibration controls contain labels but no model predictions. Their
offline accuracy is therefore null. The separate `evals.agent_eval_ext.calibrate_live`
entry point reuses the original independent judge, requires both `--allow-paid` and explicit
process `LLM_MODE=real`, and trial-writes its output directory before any model call. It
preserves returned claim verdicts rather than attributing answers to the requested model.

`trajectory` compares supplied tool-call fixtures in strict, unordered, subset or superset
mode. Existing backend CI discovers the extension tests with no new external service.

#### Opt-in judge batch observation

`RF_JUDGE_BATCH_TRACE_DIR` enables separate JSONL files around actual report-harness judge
calls. It is off by default, does not cause a fake or failed research run to call a judge,
and runs outside candidate prompt scope and graph traces. Baseline/candidate files are
distinct. `judge_batch_trace` and `judge_batch_trace_status` (`recorded`, `partial`,
`unavailable`) are added only when observation is requested, with paths relative to the
report directory where possible. Trace I/O failures do not change scoring.

Each existing application invocation records coverage, YES/NO counts, elapsed time,
response content kind and label counts, duplicates/out-of-range labels, exception class,
structured HTTP status when exposed, and cancellation. Exceptions are observed in `finally`
without adding retries or changing the parser, prompts, rate-limit sleeps or denominator.
JSONL and observer warnings omit prompts, claim/evidence text, raw replies, headers and
exception messages. `preexisting_judged_count` exposes entries already in the legacy verdict
map before the batch; map coverage is distinct from labels in the current response.

```bash
python -m evals.agent_eval_ext.cli judge-runtime --input PATH_TO_JUDGE_TRACE.jsonl
```

The inspector validates identity, sequence, event order and counts. Truncated traces leave
overall coverage and total calls unmeasured and expose observed subtotals. One recorded
`application_level_calls=1` is not one HTTP request: `sdk_retry_attempts` stays null. Hard-killed
in-flight calls, provider internals, tokens and billing are not measured. A reason hint is an
observed symptom, not proof of a provider root cause. Historical missing responses remain
unrecoverable. This automatic scope covers `harness.run_one`, not benchmark or live-calibration
entry points; direct callers can wrap `judge_citation_support` in `trace_judge_case` explicitly.

#### Opt-in research runtime trace

Report evaluation accepts `--trace` (or `RF_EVAL_TRACE=1`). The default is off. Nine actual
graph nodes, direct model calls in `graph.py`, and Executor tool executions write metadata
spans through `research_engine.eval_trace`, which does not depend on the eval host. The
independent judge executes outside that scope. The manual v0.1 recorder remains available.

```bash
LLM_MODE=fake python -m evals.harness --limit 1 --trace --out /tmp/eval-output/report.json
python -m evals.agent_eval_ext.cli trace --input /tmp/eval-output/traces/TRACE_FILE.jsonl
```

Trace files are exclusively created under `--trace-dir` / `RF_EVAL_TRACE_DIR`, defaulting
beside `--out` in `traces/`. The optional `runtime_trace` and `runtime_bad_case` paths are
relative to the report directory where possible, so export the directory as a whole. Trace
I/O failures do not change the research outcome: `runtime_trace_status` records `recorded`,
`partial`, or `unavailable`. Each file carries case/run/thread identity and ordered spans;
ContextVars isolate parallel graph tasks. Application planner attempts and parsing/recovery
steps are observed without adding retries or recording payloads. SDK-internal attempts are
not separately observed.

Only allowlisted metadata is written: role, configured route in real mode, disclosed model,
provider-supplied token counts, tool name/call ID, schema status, attempt and strategy.
Prompts, tool arguments, source content, request headers and exception messages are excluded.
Missing usage stays unmeasured, including in the trace inspector. Graph interrupts are
control flow; cancellation, exceptions and returned business errors produce symptom-only
Bad Case files with an unknown root cause. A recovered failure can leave such a file even
when the final outcome succeeds. Historical failures without traces cannot be reconstructed.

Models called internally by other modules, including grounding, are not all instrumented;
trace totals are not a complete bill. Historical response replay, full HTTP fault injection and
always-on server/desktop tracing remain unimplemented. Dedicated runtime unit and compiled
graph integration tests run with fake models and no paid requests.

#### Controlled planner fault replay

`evals.agent_eval_ext.replay_cli` runs versioned cases against the actual compiled graph
with an in-memory checkpointer and explicit fake configuration. Fixtures cover healthy
report approval, invalid Planner schema, and simulated application-level 429/522 exceptions.
The latter do not exercise HTTP clients or SDK retries; this is controlled re-execution,
not reconstruction of missing historical responses.

```bash
python -m evals.agent_eval_ext.replay_cli --case evals/agent_eval_ext/fixtures/replay_planner_invalid.json --trace-dir /tmp/replay/traces --out /tmp/replay/result.json
```

The CLI trial-writes its output directory before graph execution and creates results
exclusively. It records raw fixture and normalized case hashes, measured Planner calls,
state/count/error assertions, and relative Trace/Bad Case paths where possible. Unrequested
assertions remain null. Trace availability/completeness and the corresponding terminal node
are required for a passing replay. An expected failed research state can therefore pass
regression checks and exit 0; unmet assertions save a new result and exit 1. Unhandled graph
exceptions and unrecognized interrupts cannot masquerade as the expected terminal state.

Temporary model replacement is process-wide. Use a dedicated CLI/test process: overlapping
replays are rejected, but unrelated business work in that interpreter cannot be isolated.
Configuration, user keys, cache/corpus ports and the event sink are scoped and restored on
success, failure and cancellation. Existing backend pytest CI collects the new regressions;
no product routes, production data stores or paid models are needed. General tool fault
injection, HTTP SDK behavior and automatic code/prompt revision checkout remain separate work.

#### Process-isolated replay suites

`evals.agent_eval_ext.replay_suite` adds `validate`, `run` and `compare` over the existing
v0.5 replay CLI. `fixtures/replay_suite_v1.json` defines the four fixed cases. All cases
are validated before execution, then their original bytes and the manifest are frozen
under a newly created output directory. Each child reads a snapshot and executes through
the current Python interpreter with `shell=False`, a fixed backend working directory,
fake mode and a default 180-second per-process timeout. It runs sequentially and continues
after individual failures. The timeout bounds the test process, not Agent or SDK retries.

```bash
python -m evals.agent_eval_ext.replay_suite run --manifest evals/agent_eval_ext/fixtures/replay_suite_v1.json --out-dir /tmp/replay-suite --revision supplied-label
```

The runner verifies child exit status, result/fixture identity, assertions, linked Trace
integrity and Planner counts, terminal nodes and Bad Case identity. Missing or inconsistent
artifacts cannot PASS, and a timeout cannot be masked by a previously written passing
result. Inputs and verified artifact byte hashes accompany `suite.json`. API keys and
business/observation settings are removed from child environments and common telemetry
exporters are disabled; Docker network isolation is still recommended. Network attempts
are not measured.

Comparison requires valid case fingerprints and matching nonempty case sets; changed
contracts are not scored as regressions. It reports recoveries/regressions and blocks
unless all candidate cases pass. The dictionary comparison API checks declared metadata;
CLI comparison additionally re-verifies saved inputs and linked artifact bytes, so the
entire suite directory must be retained. Integrity failures block the gate and leave change
counts unmeasured. `--revision` is an unverified human label, not proof of a Git or prompt
revision. This layer neither changes v0.5 injection/scoring nor adds a CI workflow; existing
backend pytest discovery covers its tests.

#### Unified evaluation adapters and runner

`evals.agent_eval_ext.unified_cli` provides `validate`, `collect`, `run` and `compare`.
`fixtures/unified_dataset_v1.json` versions a wrapper around the existing four-case replay
manifest; its wrapper, manifest and original case byte fingerprints are recorded separately.
These fingerprints describe the files read now, not an inferred historical input revision.

```bash
python -m evals.agent_eval_ext.unified_cli validate --dataset evals/agent_eval_ext/fixtures/unified_dataset_v1.json
python -m evals.agent_eval_ext.unified_cli run --dataset evals/agent_eval_ext/fixtures/unified_dataset_v1.json --out-dir /tmp/unified --revision supplied-label
python -m evals.agent_eval_ext.unified_cli collect --dataset evals/agent_eval_ext/fixtures/unified_dataset_v1.json --replay-suite /tmp/unified/replay/suite.json --report EXISTING_REPORT.json --retrieval EXISTING_RETRIEVAL.json --grounding EXISTING_GROUNDING.json --out /tmp/collected.json
```

`run` only executes the controlled fake replay suite. Optional report, retrieval and
grounding paths attach existing historical artifacts; they do not run paid evaluations.
Inputs are validated before execution, the new output directory is trial-written, and
dataset/case/optional source bytes are frozen before child processes start. Final outputs
are exclusive. `collect` executes no graph and verifies supplied replay suites using the
existing v0.6 result/Trace/Bad Case checks; retain the complete suite directory.

Four separate dimensions are emitted: outcome, reliability, retrieval and grounding.
Metrics carry status, value, unit and a metric-specific denominator (`n`). Unmeasured
values remain null, invalid grounding stays invalid, and unverified pricing remains
unmeasured. Report citation support is claim-weighted among judged claims, with coverage
separate and the official aggregate retained. Completed-report P95 is unmeasured if any
successful latency is missing. Both retrieval layouts preserve their recorded aggregates
and count available observations separately per metric. Grounding retains conditional
denominators, reference-label provenance and configured routing/alias declarations.
Neither legacy report completion nor a passing fault contract proves human-approved
research success: replay `test_result=pass` may coexist with `agent_status=failed`.

Unified `compare` requires identical versioned datasets and nonempty case ID/byte hashes,
with internally consistent counts and rates. Its gate covers deterministic replay only;
other dimensions are neither paired nor gated and no combined score is produced. The
comparison validates normalized records without re-opening source chains
(`artifact_verification_repeated=false`). Re-collect inputs or use the v0.6 suite CLI
comparison when fresh source-artifact verification is required. A supplied revision label
does not verify Git/Prompt revisions or establish a controlled code improvement.

#### Controlled tool contracts and the offline Actions gate

`evals.agent_eval_ext.tool_safety_cli score` validates `tool_gold_v1.json` against controlled
positive/negative observations. Only `controlled_fixture` provenance is accepted; all
results set `actual_agent_executed=false`, even when a fixture declares the executed stage.
Arguments are compared in memory and omitted from output. This layer does not capture real
Agent arguments or establish real tool accuracy, safety enforcement or tool execution success.

```bash
python -m evals.agent_eval_ext.tool_safety_cli score --gold evals/agent_eval_ext/fixtures/tool_gold_v1.json --observed evals/agent_eval_ext/fixtures/tool_observed_pass_v1.json --out /tmp/tools-pass.json
```

Strict mode checks the complete ordered name sequence and required positional arguments;
its selection and trajectory flags both incorporate name order. Unordered mode requires
an equal number of one-to-one matching calls, while subset mode permits additional allowed
calls. Unknown/forbidden tools always fail. Required argument keys form a top-level subset;
nested values preserve boolean/number distinctions. This does not validate the full tool
schema. Missing observations remain unmeasured and cannot pass a gate. Comparisons require
matching Gold fingerprints/dataset identity and internally consistent cases, metrics and
counts; measured observation stages must match. Fingerprints in the dictionary API are
caller-supplied; file-based CLI reads hash the exact bytes decoded. Comparison does not
re-open the original input chain. Outputs are trial-written then exclusively created.

`.github/workflows/agent-evaluation.yml` adds the unique `Agent evaluation offline gate`
job on every PR or manual dispatch, alongside existing CI. It builds the existing Worker,
runs Ruff and the dedicated evaluator tests, checks both tool controls, then executes the
four compiled-graph Replay cases. The negative tool control must exit exactly 1; it is a
scorer control, not an Agent failing in production. Evaluators run fake without credentials,
with network disabled, read-only root/source mounts, dropped capabilities and no-new-privileges.
Image builds and GitHub action downloads still require network access.

Persistent outputs use new Docker volumes mounted at `/app/data`, whose ownership is
initialized from the image's appuser-owned directory. `/tmp` is ephemeral tmpfs and must
not hold exportable results. After each container exits, `docker cp` must succeed before
cleanup; export failure blocks a passing run. Run, export and effective gate exit codes
are recorded separately. Logs and artifacts are uploaded with `always()` as a best effort,
including after test/gate failures; runner loss or cancellation can still prevent upload.
The workflow runs no paid quality evaluation and produces no seven-dimension release score.
Its local shell/Docker validation is not a GitHub-hosted Actions run or an upload validation.
To enforce merges, configure the job as a [required status check in branch protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches#require-status-checks-before-merging);
adding the workflow alone does not configure that repository setting.

#### Offline closeout reports and compiled Executor contracts

`evals.agent_eval_ext.interview_closeout` renders a self-contained HTML report, a JSON
summary and a conservative triage draft from normalized inputs. Optional `--tool`,
`--executor` and `--baseline` attach separate controlled evidence; no combined model score
is computed. Missing and invalid measurements remain distinct. Inputs are escaped and a
CSP forbids scripts and network resources. Outputs use unique paths and exclusive creation.
Triage records symptoms with unknown root causes and never automatically promotes them
into Gold. Expected failed/cancelled Agent states with passing contracts are not defects;
synthetic tool controls are not real Agent bad cases.

```bash
python -m evals.agent_eval_ext.executor_regression --out-dir /tmp/executor-suite
python -m evals.agent_eval_ext.interview_closeout --unified UNIFIED.json --executor /tmp/executor-suite/suite.json --html /tmp/report.html --summary /tmp/summary.json --triage /tmp/triage.json
```

Seven fixed Executor cases execute the actual compiled graph to the Executor checkpoint:
search/read/submit, malformed URL, SSRF rejection, timeout recovery, duplicate reads,
duplicate submissions and cancellation. Tool schemas, HTTP body parsing, SSRF and verbatim
snippet verification remain real. All models are scripted and DNS/HTTP boundaries are
controlled; `RunConfig.llm_mode=real` deliberately exercises guards skipped by fake mode.
Use dedicated isolated processes and Docker network isolation; no provider client is
created. This measures fixed integration contracts, not live model tool accuracy, actual
wall-clock timeouts, SDK retries or universal submission idempotency. Result/Trace/Bad Case
links and fingerprints are verified before their suite can be shown as passing.

**Judge metric version 5** fixes batch-local label bounds in both report and comparative
benchmark scoring. `evals/judge_labels.py` only accepts `1..len(batch)`, including a short
tail. `Claim 0` cannot index the last claim, and overflow cannot contaminate the next batch
or denominator. The duplicate policy remains last valid label wins; prompts, routes,
waits/retries and the exclusion of unjudged claims remain unchanged. New benchmark traces
and summaries disclose version 5. Saved version 4 artifacts are not rescored, and the
synthetic before/after repair is not evidence for the cause of historical unjudged claims.
Existing offline Actions tests also collect the closeout, bounds and Executor regressions;
this does not imply an online Actions run or a change to branch protection.

## Quality gates

| Gate | Enforced by |
|---|---|
| No merge with failing golden journeys | Branch protection on `main` |
| No schema change without a migration round-trip | The backend CI job |
| Committed evaluation results are never modified | The `eval-artifacts` CI job |
| No banned frontend pattern | The four greps |
| Prompt or model changes carry an evaluation run | Review |
| Documentation updated in the same pull request as behaviour | Review |

## 评测结果、比较与质量门禁

服务端评测中心复用现有登录会话、共享评测卷和两条隔离 Worker lane。
统一 Pipeline 在运行前固定样本及源文件指纹，运行后只读取本轮逐题产物。
结果保存在评测卷的 `closure-v1.sqlite3`，不连接业务 ORM；已有旧审核表不删除，但不再提供审核操作。

工作台包含总览、评测运行、逐题、轨迹、坏例排查、版本对比和数据集。
旧单任务队列、历史目录工作台、人工审核/回归晋升和付费模型辅助归因已移除。
自动坏例与执行异常按账号隔离、去重并只读展示，不改分、不自动确认根因。
研究业务的计划审批、报告审批与 Checkpoint 不受影响。

版本比较要求非空、完整、同口径的案例与输入集合，在线模型身份未验证时不作严格配对声明。
质量门禁返回 passed / failed / blocked；页面要求完整测量且全部通过，缺失与未判定不能放行。
检索没有明确逐题合格阈值时保持未判定，各维指标不合成总准确率。
门禁检查保留策略指纹和审计，不请求模型。

`python -m app.services.evaluation_closure.cli --db EXISTING_DB --owner OWNER --run-id RUN_ID`
检查已有结果并追加门禁审计；退出码为 0（通过）、2（失败）、3（阻塞或错误）。

### 按原始 Case ID 运行单题报告评测

`evals.harness --case-id quantum-error-correction --out NEW_FILE.json` 只选择
`evals/queries.json` 中该 ID 的原始输入，不改数据或题目顺序。`--case-id` 与 `--limit` 互斥，
且只用于 report 模式；未知 ID、空 ID 或源数据中的重复 ID 在模型调用前报错，不回退全量。
不传 `--case-id` 时仍按原来的 `--limit` / 全套模式执行。附加 `--candidate` 时，
基线与候选仍分别执行同一个指定案例，不能把它理解成只有一次模型请求。

统一 Pipeline 的可选 `case_id` 只允许与 `datasets=["report-benchmark"]`、`max_cases=1`
一起使用；原费用确认仍必需。API 校验、持久化任务和幂等比较包含此字段，在线 Worker 将它
传入原 harness，使用 `--case-id` 而非 `--limit`。Worker 还核对实际结果只有一个匹配的案例。
失败回流冻结的输入、案例集合和源指纹采用同一选择规则；旧任务不传此字段时行为不变。
浏览器的报告单案例选项来自现有只读数据集接口，选择后明确显示只运行该案例。

离线检查（容器内、显式 fake）可运行：

```bash
LLM_MODE=fake python -m evals.harness --case-id quantum-error-correction --out /tmp/report-single-case.json
```

此命令只验证脚本模型执行与选择逻辑，不是历史案例的真实模型复测成绩。
真实评测必须显式 `LLM_MODE=real`，并配置独立 Judge。单题仍会多次调用研究模型、
独立 Judge 和检索服务；原有 token/轮次/耗时策略不变，案例数量限制不是费用硬上限。
`custom:` 等缺少本地计价的路由显示零估价也不代表免费，不应据此给出美元费用承诺。

### 运行诊断与执行异常

顶部 Pipeline / 数据集选择统一驱动总览、逐题、Trace 和候选比较，切换时清除旧证据并丢弃迟到响应。
八套适配器提供真实运行状态和可取得的逐题记录；Report、Planner Replay 另提供图节点/模型/工具事件。
在线 Report 同时传入 `--trace --trace-dir`，Judge 批次另存任务目录。未采集的事件或用量保持缺失。
只读异常池按账号、数据集、案例选择、输入指纹及原因去重；批次超时不推断每题均发生质量失败。

诊断结合研究图、Judge 批次与运行器起止记录计算阶段耗时。早期已恢复工具错误只作为症状，
不能代替终止原因；并发和父子 span 的累计时间不能直接相加作为墙钟。
无明确等待记录时仅能推断，SDK 内部重试仍未测量。

Trace 默认仅元数据；专用评测 Worker 显式启用的正文经过脱敏、限额并披露省略。
浏览器支持筛选、展开和导出已加载轨迹，不再将事件外发给辅助归因模型。
Judge 响应观察比较 HTTP JSON、SDK 已解析对象与消息转换边界的安全统计，缺失字段为 `unavailable`。
不为补采结束原因保存响应头、隐藏推理或原始响应正文。

### 长任务可靠性（P0）

在线 Worker 的单题 420 秒硬上限保持不变，默认在 405 秒开始协作停止，15 秒宽限位于上限之内。
多题沿用原 Worker 的题量上限，研究、评分及整体默认预算随实际题数调整，不把整个多题任务压成单题预算。
`evals/agent_eval_ext/long_run.py` 在其内部给出分阶段预算，
只有 `report-benchmark` 通过 `RF_EVAL_RELIABILITY_DIR` 启用；其余数据集、离线通道和旧成绩不受影响。
`MemorySaver` 只是进程内检查点，不能用于跨进程恢复：可恢复的证据一律来自磁盘上的原子产物。

| 环境变量 | 默认 | 含义 |
| --- | --- | --- |
| `RF_EVAL_RESEARCH_DEADLINE_S` | 240 | 研究图阶段预算 |
| `RF_EVAL_JUDGE_DEADLINE_S` | 未设置 | 默认共享剩余总预算；显式设置时作为严格 Judge 阶段上限 |
| `RF_EVAL_TOTAL_DEADLINE_S` | 390 | 整体预算，必须与停止宽限一起小于 420 |
| `RF_EVAL_JUDGE_REQUEST_TIMEOUT_S` | 45 | 单次 Judge 请求上限 |
| `RF_EVAL_STOP_GRACE_S` | 15 | SIGTERM 后等待协作退出的宽限 |
| `RF_EVAL_JUDGE_MIN_INTERVAL_S` | 0 | 全局 Judge 最小请求间隔，0 表示不等待 |
| `RF_EVAL_JUDGE_THROTTLE_<PROVIDER>_S` | 同全局 | 按路由 Provider 覆盖，例如 `..._GOOGLE_S`、`..._CUSTOM_S` |
| `RF_EVAL_QUERY_MIN_INTERVAL_S` | 0 | 两个真实研究查询之间的可选间隔 |
| `RF_EVAL_JUDGE_RETRY_AFTER_MAX_S` | 60 | 可等待的最大 `Retry-After`；超过则保存未完成状态并停止，不能缩短服务端冷却后继续请求 |
| `RF_EVAL_RESUME_FROM` | 无 | 上一轮的 `--progress-dir`，续评前校验指纹 |
| `RF_EVAL_BATCH_LEDGER_ROOT` | 进度目录旁的 `.judge-ledger` | 共享批次提交与跨进程锁目录；在线 Worker 自动设为评测历史卷根下的 `judge-batch-ledger` |

默认使用 `shared-deadline/1` 调度：研究提前完成的时间可用于评分，每次请求仍取单请求、阶段及总剩余时间的最小值。
多题/候选臂按剩余题数分配当前题的执行份额，未花完的份额向后滚动；首题不能耗尽后续题预算。
总预算的落盘和停止预留不因共享而扩大，显式阶段上限也不会被忽略。运行策略版本独立于评分定义，
更改额度不会让同一评分定义下的有效历史批次失效。

启动前会实际写入 `write-check.json`；运行目录不可写时直接拒绝，不做任何付费调用。研究图一到达人工审核
`interrupt()`，草稿（报告、Evidence、Citation、Token、成本、配置指纹）立即落盘，且 `approved=false`、
`resume_action=not_taken`，绝不擅自继续到 `finalizer_node`。每个 Judge 批次用独占创建原子写入：
`batch-0001.json` 仅表示「全部主张已判定」的完成批次，永不覆盖；`429`、空回复、取消、超时的未完批次写成
`batch-0001.attempt-*.json` 诊断记录，续评时重新评分而不是重放。

状态写入 `progress.json` / `final-status.json` 与结果文件顶层的 `reliability` 段：
`research_draft_ready`、`research_failed`、`judge_partial`、`judge_completed`、`judge_failed`、
`artifact_save_failed`。旧字段 `models` / `aggregate` / `release_criteria` / `results` / `judge` / `method`
保持原样；部分评分时 `release_criteria.citation_support_ok` 置 `None`，未测量不能变相通过 0.95 门禁。
`judge_partial` / `judge_failed` / `artifact_save_failed` 分别以退出码 3 / 4 / 5 明确上报，
Worker 在硬超时仍保留历史 `runner_timeout` 归因，并把可靠性状态追加到 `process.json`。

超时或取消先向子进程组发 SIGTERM：Suite 保持存活，harness 接收同组信号，唤醒事件循环并取消在途请求或等待，
等待子任务清理、保存已有进度后退出；只有超过停止宽限仍未退出才发 SIGKILL。
续评使用 `--resume-from <上一轮 progress-dir> --progress-dir <新的独立目录>`，禁止写回来源目录，会校验
`judge_config_version`、报告/来源/主张哈希与原始运行配置指纹；已完成的批次以 `batch_restored`
事件回到 Trace 且 `application_level_calls=0`，研究图不会重跑。任何校验失败都拒绝续评且不做模型调用。
恢复后的新目录同时持有草稿及已完成批次，支持再次中断后继续恢复。判定字段只接受真正的布尔值，批次数值必须
与逐条判定相符。基线和候选 Prompt 使用不同进度目录与指纹，不互相覆盖或复用评分。
配置指纹版本为 `rf-judge-reliability/2`；旧版本进度不能静默当作新版兼容数据继续评分。

### 全量口径、批次提交与恢复一致性

新 Judge Trace 在开始时记录全量 Claim 数与批次大小。诊断也能从同次、同案例、同实验臂的完整结果补充旧 Trace
缺少的分母；没有全量输入时，整体覆盖率为 `null`。分别展示成功批次/总批次、尝试批次/总批次、
全量 Claim 覆盖率和已尝试范围覆盖率，不能把轨迹正常闭合或请求成功返回当作评分完整。
`report-full-input/1` 门禁要求全量 Claim 覆盖率为 1，再检查原有质量及研究完成率判据；
未判定仍排除支持率分母，绝不改成 NO。历史结果只读，不回写新增字段或重判成绩。

研究草稿保存、评分完成、评分截止原因、已提交批次、恢复校验、最终完整性和质量门禁分别表达。
恢复只读校验通过表示保存产物内部一致；实际启动续评时仍需与当前模型/Prompt 配置匹配。

批次稳定 ID 绑定运行血缘、报告、来源、Evidence、Claim 分片、Judge 配置、`claim-labels/v5` 和 `fixed-four/v1`。
新草稿额外保存 Evidence 哈希；历史 v2 草稿缺少该哈希时，只能校验当时已保存的报告/来源/Claim 哈希，
不能宣称补回了历史缺失的校验字段。独立新运行有不同血缘，不会因文本相同而复用另一场实验评分。

并发续评必须共享同一 `RF_EVAL_BATCH_LEDGER_ROOT`。跨进程文件锁持有至本批请求和提交结束；占用时保存
可恢复状态，不重复请求。OS 锁随进程退出释放，不使用定时租约过期来重发仍在途的请求。
此机制要求支持进程文件锁的本地/容器共享文件系统，不宣称适用于不支持这些语义的网络文件系统。
完成评分先写共享 `committed.json`，再写本轮局部副本；副本写入前崩溃时，续评仍可验证并复用提交记录。
仅有 started/returned/partial/cancelled 记录不算有效提交，不能跳过；批次划分或评分规则不匹配直接拒绝。

这保证有效评分不会重复提交，不保证绝对零重复计费：供应商返回后、提交前崩溃仍可能需要重发，
恢复记录明确标注之前的请求可能已计费。没有应用层自动无限重试；Retry-After 仍必须完整遵守。
逐批保存开始/结束时间、延迟、返回状态、已披露 Token 及等待原因，缺失用量与未知成本保留 `null`。
不依此自动扩大批次、并发度或输出预算；先用固定输入、故障注入检查完整性和吞吐，再另行授权真实评测。

### Judge 空回复的响应观测

独立 Judge 不再借用研究 Critic 的输出额度，默认使用 4096 个输出 Token（包含供应商计入的推理 Token）。
可用 `EVAL_JUDGE_MAX_OUTPUT_TOKENS=1..32768` 明确设置；它不改变研究角色的输出参数，
也不扩大运行器硬截止、增加应用层自动重试或改写空回复判定。4096 是有界默认值，不保证任何模型都能完成评分。
显式设为 `1000` 时兼容旧版恢复指纹；其他值写入 `judge_generation` 并改变配置指纹。
同配置中断使用 `--resume-from`，只跳过校验通过的持久化批次。
改变输出预算后应使用 `--rejudge-from <原 reliability 根目录> --progress-dir <新目录> --out <新结果>`：
它先校验来源报告、Evidence、Claim、Prompt、路由及规则，只复用研究草稿，全部 Judge 批次重新评分，
不混入旧配置的成功评分，不覆盖旧目录，也不运行或批准研究图。除输出预算外的配置变化会被拒绝。
该命令仍可能收费，必须先取得调用授权；65 条 Claim、每批 4 条时预计 17 次应用请求，研究请求为 0。

以下响应观测本身不改变生成参数；独立输出预算由上述配置单独控制。

Report harness 的独立 Judge 使用本地、无正文的 `response_diagnostic`。它只观测，不改变 Prompt、
请求参数、评分解析、重试、预算或门禁。`research_engine.observed_openai.ObservedChatOpenAI` 保留固定版本
SDK 的传输与解析实现，仅在评测宿主创建的 OpenAI 兼容 Judge 实例上启用；普通研究客户端不受影响。

在 SDK 已缓存 HTTP 响应时，只从其 JSON 提取首个 Choice 的内容类型/正文长度、结束原因、响应 ID、
输出 Token 与供应商提供的推理 Token；只读取响应头中的 `x-request-id` 作为请求关联 ID，不保存头集合。
随后记录 OpenAI SDK 已解析对象，以及 LangChain 转换后消息的同类统计。实际输出参数取 SDK 生成的
请求 payload，因此能区分 `max_tokens` 与 `max_completion_tokens`，不以模型对象上的默认值冒充实际请求。
正文、隐藏推理、工具参数、密钥和 HTTP 原始字节均不落盘；即使显式启用正文 Trace，Judge 回复仍只保存统计。
只检查已缓存且不超过 4 MiB 的 HTTP 正文，不读取流、不额外请求；非 JSON、不可读取的边界、原生 SDK、
Responses API 等未覆盖路径明确记录 `unavailable`，不得凭 Token 等于上限补造结束原因。

| 类别 | 必要证据与边界 |
|---|---|
| `output_truncated` | 返回元数据明确报告 `length` / `max_tokens`；表示观测到输出限制，不证明空正文的唯一根因 |
| `provider_empty` | 可观测供应商边界正文长度为 0、SDK 后长度为 0，并正常 `stop` / `end_turn`；不能继续归咎于远端网关内部 |
| `sdk_conversion_empty` | 转换前确有正文，转换后变空或转换异常；关联具体转换边界，不修补返回文本 |
| `sdk_conversion_error` | 观测到 SDK 解析/消息转换异常，但不足以证明曾有正文 |
| `nontext_response` | harness 无普通文本，但可观测非文本内容、工具回复或列表型消息；不擅自解释成 YES/NO |
| `unknown` | 缺少足够证据，包含仅 Token 达到上限、调用取消及未覆盖的边界 |
| `normal` | harness 收到非空文本；此类别不代表标签有效、质量通过或评分完整 |

类别确认只描述响应现象，`root_cause_confirmed` 始终为 false。字段缺失使用字面值 `unavailable`，
供应商报告的 0 则保留为 0。读入产物时再次按白名单投影、重新计算类别，不信任文件中的分类声明。

同一 `observation_id` / `batch_id` 关联 LLM Trace、Judge 批次 Trace 与新批次落盘记录；未完成或异常响应
另存 `judge-*.badcase.json`，包含 Trace 文件/序号，不冒充已确认根因。诊断 API 与模型证据包保留这份安全投影，
外发仍须既有人工确认。历史没有采集的字段不回填；恢复的旧批次不会新增模型调用，有观测时原样关联。
观测字段不进入恢复指纹，不改变 `rf-judge-reliability/2`、`METRICS_VERSION=5` 或既有成绩。
