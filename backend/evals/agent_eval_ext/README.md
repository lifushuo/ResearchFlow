# Agent evaluation extension

这是接入现有评测 JSON 的实验性离线分析层。原有 report、memory、retrieval、grounding、
contradiction 评测入口和 `metrics_version` 均保持原定义。扩展输出单独使用
`extension_schema_version=1`，不重新判定历史报告。

## 已接入的能力

| 模块 | 作用 |
|---|---|
| `audit.py` | 校验 report JSON、统计 YES/NO/未判定和覆盖率、列出失败案例与保守的错误症状分类 |
| `judge_batch_audit.py` | 按保存的 Claim 顺序和假定批大小推算批次覆盖情况，保留未知根因 |
| `judge_runtime.py` | 可选的实际 report Judge 批次观察，记录覆盖、标签异常、取消和结构化错误元数据 |
| `judge_runtime_inspect.py` | 检查 Judge JSONL 身份、序号和完整性，汇总实际应用层调用与症状 |
| `replay.py` / `replay_cli.py` | 显式 fake 的真实编译图故障重执行与自动断言；四个固定 Planner 场景 |
| `replay_suite.py` | 每案例独立解释器的顺序批量回放、产物核对与套件契约回归比较 |
| `compare.py` | 按题目 ID 对齐版本，披露模型、裁判、指标、限额、题集和 prompt 指纹的差异或缺失 |
| `calibration.py` | 对人工标签和已有预测计算混淆矩阵、正负召回率、覆盖率；没有预测时准确率为 null |
| `calibrate_live.py` | 显式付费入口，复用原 harness 的独立裁判和解析逻辑，并保存返回的逐条裁判记录 |
| `trajectory.py` | 对提供的工具调用记录做 strict、unordered、subset、superset 比较，检查参数与禁用工具 |
| `trace.py` | 手动 JSONL span 记录器；新文件独占创建、并发父子关联、写盘错误可观察且不掩盖业务错误 |
| `research_engine/eval_trace.py` | 可选的真实图节点、直接 LLM 请求与工具执行 Trace；独立于 evals，不改变流程和重试 |
| `trace_inspect.py` | 读取运行时 JSONL，区分节点返回、模型/工具调用、审批中断、错误症状和用量缺失 |
| `gates.py` | 独立实验门禁：完成率 ≥90%、已判定支持率 ≥95%、裁判覆盖率 ≥98% |
| `artifacts.py` | 调用模型前实际试写输出目录，最终独占创建，拒绝覆盖历史文件 |
| `cli.py` | 离线命令入口；不导入 harness、不读取 `.env`、不调用模型 |

`fixtures/judge_calibration.jsonl` 含 24 条人工设计的合成对照样本：12 正、12 负，
其中 dev 16 条、holdout 8 条。标签是校准参考，不是已经获得的模型成绩。

## 离线命令

从 `backend/` 运行，使用项目的 Python 3.14 环境：

```powershell
python -m evals.agent_eval_ext.cli audit --input evals/results/YOUR_RUN.json
python -m evals.agent_eval_ext.cli judge-batches --input evals/results/YOUR_RUN.json --batch-size 4
python -m evals.agent_eval_ext.cli judge-runtime --input PATH_TO_JUDGE_TRACE.jsonl
python -m evals.agent_eval_ext.cli compare --baseline evals/results/BASELINE.json --candidate evals/results/YOUR_RUN.json
python -m evals.agent_eval_ext.cli gate --input evals/results/YOUR_RUN.json
python -m evals.agent_eval_ext.cli calibration --input evals/agent_eval_ext/fixtures/judge_calibration.jsonl
python -m evals.agent_eval_ext.cli trajectory --input evals/agent_eval_ext/fixtures/trajectory_example.json
python -m evals.agent_eval_ext.cli trace --input PATH_TO_RUNTIME_TRACE.jsonl
python -m pytest -q tests/task/test_agent_eval_ext.py
```

任何离线子命令都可加 `--out NEW_FILE.json`。不指定时只打印结果，指定后只创建新文件。
`gate` 和 `trajectory` 未通过（包括未测量的门禁项）时退出码为 1，通过为 0。
门禁先保存分析结果再退出，因此退出码 1 不表示结果文件未写入。
`compare` 的阻断项仅阻止受控比较的声明，命令仍以 0 退出并输出描述性比较。
这些新测试由现有 backend CI 的 `pytest tests/` 和 `ruff ... evals/` 自动覆盖，
不需要另一套 Python 版本或专用 workflow。

没有本地 Python 环境时，可在仓库根目录用现有镜像运行只读审计：

```powershell
docker run --rm --network none `
  --mount "type=bind,source=D:\work\ResearchFlow,target=/workspace,readonly" `
  -w /workspace/backend researchflow-worker `
  python -m evals.agent_eval_ext.cli audit --input evals/results/YOUR_RUN.json
```

容器中需要保存新产物时，使用容器内可写目录并保留容器供 `docker cp` 导出。
不要重新写入挂载的历史结果文件。

## 结果怎样解释

- 支持率只以已判定 Claim 为分母，未判定既不是 YES 也不是 NO。
- 审计的 `support_among_judged` 按 Claim 加权；官方 aggregate 可能平均各报告支持率。
  两种权重可能得出不同数字，`aggregate_reported` 保留原值，不静默替换它。
- 覆盖率为已判定数 / 全部记录数。有成功报告缺少整个 `claim_verdicts` 时，总覆盖率为 null。
- `supported_among_all_cited_claims` 是全部记录中已确认支持的比例，不是官方支持率，
  也不把未判定 Claim 宣判为错误。
- 98% 是可讨论的实验性覆盖率门槛，不替代原官方发布条件。
- `--dataset` 或比较命令的 `--baseline-dataset` / `--candidate-dataset` 只给事后文件计算指纹。
  历史 JSON 未保存 `method.query_set_sha256` 时仍不得声称题集已受控。
  prompt 指纹缺失时也不声称严格受控；现有历史成绩出现这类 blocker 是预期行为。
- 输入含 `candidate` 时只审计顶层 shipped baseline，并明确警告；不偷偷改成候选成绩。
- 错误分类是症状标签，例如 `valid task list` 对应结构化输出/校验现象，不能证明根因。
- 无法从 report JSON 恢复未保存的裁判原始响应、证据片段或运行轨迹。
- custom 路由的 `$0.0` 可能缺少定价，不能作为免费调用的证据。

## 离线 Judge 批次审计（v0.3）

`judge-batches` 按每份报告的 `claim_verdicts` 顺序推算分组，`--batch-size` 默认 4，
是对历史 harness 的批大小假设。Claim 序号从 1 开始，批次号在每份报告中重新计数。
只有 `judged=true` 且 `supported` 为布尔值才计为已判定，因此 NO 同样计入 `complete`。
`partial` 表示部分有有效判定，`unjudged` 表示该组全部未判定；短尾组也完整计数，
`is_full_batch` 表示该组是否含指定数量的 Claim。

成功报告若缺失整份 Claim 记录，整体覆盖率为 null，而非用其余报告冒充完整测量。
输出保存缺失案例、非法判定位置、原输入 SHA256 和显式的分组假设。
嵌套 candidate 不自动选取，CLI 与独立入口均复用现有只读加载和独占输出逻辑：

```powershell
python -m evals.agent_eval_ext.judge_batch_audit --input evals/results/YOUR_RUN.json --out NEW_BATCH_AUDIT.json
```

这只是历史记录的离线覆盖率重建。它不观察真实请求，不恢复原始回复、HTTP 状态、
解析错误或重试次数，问题组的根因保持 unknown。即使 4 条连续未判定，也不能据此断言
发生了一次 API 异常。本版本没有增加运行时 Judge 日志、自动重试或付费请求。

## 可选 Judge 运行时观察（v0.4）

设置 `RF_JUDGE_BATCH_TRACE_DIR` 后，report harness 的 `run_one()` 在实际 Judge 执行前
创建独立 JSONL。默认关闭；研究失败或 fake 模式不自动调用 Judge，也不会为了记录而调用。
每个案例和 baseline/candidate 使用不同文件；作用域在候选 Prompt 和研究图 Trace 之外。
独立 `benchmark.py` 和 `calibrate_live.py` 没有自动建此作用域，直接调用者可显式使用
`trace_judge_case(...)` 包裹 `harness.judge_citation_support(...)`。

每次 `llm.ainvoke()` 及原有响应处理在 `finally` 中产生 `batch_finished`，记录实际批号、
预期/已判定/未判定数、YES/NO 数、耗时、标签总数、重复/越界标签数、异常类型和可取得的
结构化 HTTP 状态码。空文本、非文本响应（记录内容类型）、不可解析、漏答、调用异常、
响应处理异常及取消可区分；这些是应用层症状，不是供应商内部根因。

JSONL 和观察层告警不保存 Prompt、Claim、Evidence、响应正文、请求头或异常消息。
模型标识只来自实际响应，路由别名单列为 `disclosed_alias`，缺失信息不猜测。
`application_level_calls=1` 只代表一次应用层调用，`sdk_retry_attempts=null`，不能当作
一次 HTTP 请求。原评分、正则解析、15 秒等待和重试策略均未调整。

`judged_count` 反映原判定映射在本组范围内的有效布尔条目，
`parsed_unique_in_range_labels` 反映当前回复的有效编号数。原解析器若因前一批越界标签
提前写入后一批，可通过 `out_of_range_labels` / `preexisting_judged_count` 观察；本观察版本
不修正该评分行为。即使覆盖完整，标签异常或响应处理异常仍会出现在检查器的问题列表。

写盘错误不改变 Judge 结果；启用时报告新增 `judge_batch_trace` 与
`judge_batch_trace_status=recorded|partial|unavailable`，路径尽可能相对报告目录。
独占文件、ContextVar 隔离、锁内序号与开始/结束事件用于检查身份和完整性。
关闭或打开失败的内层作用域不会借用外层记录器。检查器拒绝混合身份和非法计数，
截断文件的整体覆盖率与总调用数保留 null，同时列出已记录部分。

未来真实评测可在原 Docker 命令中加入：

```powershell
-e RF_JUDGE_BATCH_TRACE_DIR=/tmp/eval-output/judge-batches
```

并使用 `--out /tmp/eval-output/report.json`，保留容器后通过 `docker cp` 导出整个
`/tmp/eval-output` 目录。真实调用仍遵循原入口要求；本次未运行付费评测。
导出后可离线查看：

```powershell
python -m evals.agent_eval_ext.cli judge-runtime --input PATH_TO_JUDGE_TRACE.jsonl
python -m evals.agent_eval_ext.judge_runtime_inspect PATH_TO_JUDGE_TRACE.jsonl
```

检查器只统计已经落盘的完成钩子；进程被强杀时的在途调用、SDK HTTP 重试、网关内部行为、
token 和费用未测量。历史的 27 条未判定仍不能由新模块事后定位。本版本没有新增自动重试。

## 可选的真实裁判校准

以下命令会调用收费 API。只有同时传入 `--allow-paid` 并在进程环境设置
`LLM_MODE=real` 才会运行；输出目录在导入并调用裁判前实际试写，已有文件提前拒绝。
该命令保留原独立裁判检查以及 `.env` 中的独立端点配置。

```powershell
$env:LLM_MODE = "real"
python -m evals.agent_eval_ext.calibrate_live `
  --input evals/agent_eval_ext/fixtures/judge_calibration.jsonl `
  --judge custom:kimi-k3 --split dev --limit 4 --allow-paid `
  --out NEW_CALIBRATION_RUN.json
```

不能把人工标签直接复制成预测来宣称校准成功。开发集用于调整，holdout 留作独立检查。

## 可选运行时 Trace（v0.2）

报告评测加 `--trace` 或进程环境 `RF_EVAL_TRACE=1`，即可记录真实编译的 LangGraph。
默认关闭；开启后原来的九个节点都被观察，包含人工审批、拒绝后的重写和恢复。
`--trace-dir` / `RF_EVAL_TRACE_DIR` 可指定位置，默认 `--out` 同级的 `traces/`。
Trace 创建/写入失败只记录日志，研究继续；报告中的 `runtime_trace_status` 明确区分
`recorded`、`partial`、`unavailable`，不会假装有完整轨迹。

```powershell
# 从仓库根目录，真实编译图 + scripted 模型，不调用付费 API。
docker run --name rf-trace-demo-run1 --network none `
  -e LLM_MODE=fake `
  --mount "type=bind,source=D:\work\ResearchFlow,target=/workspace,readonly" `
  -w /workspace/backend researchflow-worker `
  python -m evals.harness --limit 1 --trace --out /tmp/eval-output/report.json

# 等命令完成后导出整个目录，新目录名避免覆盖既有证据。
docker cp rf-trace-demo-run1:/tmp/eval-output .\trace-demo-run1
```

报告的 `runtime_trace` / `runtime_bad_case` 默认相对报告输出目录，目录整体导出后仍可定位。
自定义 Trace 放在输出目录之外时，会记录该位置，需要另行导出。
真实研究仍需显式 `LLM_MODE=real` 与独立裁判；此开关不改变评分定义或模型模式。
Judge 在研究 Trace 作用域之外，Trace 中不包括裁判调用。

事件含 `trace_id`、`run_id`、`case_id`、`thread_id`、span/parent、单调耗时、异常类型和
白名单元数据。只保存供应商实际返回的模型 ID/用量；未知用量为缺失值，检查器汇总为 null。
fake 模式不把配置中的真实模型路由当作已调用模型记录。
响应中若仅披露 `auto/*` 等路由别名，保留为 `disclosed_alias`，不当作实际回答模型 ID。
不保存 prompt、网页正文、工具参数、请求头、密钥或原始异常消息。

规划器 Trace 保留实际两次应用层尝试、结构化输出与 JSON 恢复步骤；HTTP 异常若提供
`status_code` 则记录。SDK 内部重试没有单独计数，不将一次 `ainvoke` 冒充一次 HTTP 请求。
正常审批中断是 `interrupt`，不是失败；异常、业务错误或取消产生独占的 Bad Case 文件，
其中根因保持 `unknown`。恢复前出现过的失败也可能留下症状文件，文件存在不代表最终运行失败。

## 当前边界

运行时追踪覆盖 graph.py 中直接调用的 LLM 边界及 Executor 实际工具调用；
grounding 等子模块内部另行发出的模型调用尚未逐一追踪，不得把此 Trace 的调用数/用量当作全系统账单。
`TraceWriter` 仍是兼容 v0.1 的独立手动记录器；本版没有接入全局服务端/桌面常驻采集。
历史请求/响应重放、完整 HTTP 故障注入和前端面板尚未实现；四类专项结果的统一适配见 v0.7。

## 受控 Planner 故障回放（v0.5）

`replay_cli` 读取版本化 JSON 案例，通过临时模型工厂注入执行真实
`build_graph(MemorySaver())`。显式设置 fake、跳过计划审批、子任务并发为 1，不读取 `.env`。
使用空用户密钥、NullCache、NoCorpus 和丢弃事件的 sink，避免继承调用者的业务端口；
正常结束、异常与取消均恢复上下文和模型工厂。没有改动 graph/harness/Judge 的行为。

四个 fixture 为 `replay_healthy.json`、`replay_planner_invalid.json`、
`replay_planner_429.json`、`replay_planner_522.json`。它们固定输入与应用层故障，
不是从旧报告恢复历史响应。429/522 为模拟 RuntimeError，不经过 HTTP 客户端；
`sdk_retry_attempts=null`，不得解释成 SDK 或 HTTP 重试验证。

```powershell
python -m evals.agent_eval_ext.replay_cli `
  --case evals/agent_eval_ext/fixtures/replay_planner_invalid.json `
  --trace-dir /tmp/replay/traces --out /tmp/replay/result.json
```

CLI 在执行前实际试写输出目录，最终独占保存；已存在文件绝不覆盖。
案例原始字节哈希和规范化配置哈希分别记录。Trace/Bad Case 路径尽可能相对输出目录，
导出整个目录即可保持关联。Tracer 不可用、Trace 不完整或对应终态节点缺失时，
即使状态/次数符合预期也不判通过。未要求的断言保留 null，Planner 次数包含健康路径的实际计数。
`passed=true` 表示本次处理符合案例预期；故障案例可以同时 `status=failed`、CLI 退出 0。
断言未通过时先保存新结果，再以 1 退出。未知中断、未处理图崩溃不能冒充正常审批或预期失败。

模型工厂 patch 是进程级的。本模块拒绝同进程重叠 Replay，但不能隔离该进程内无关的业务研究，
因此只在独立 CLI/测试进程中使用，不接入常驻 API/Worker。多个案例顺序执行，或各用独立进程；
这不是对任意业务任务都安全的全局注入器。

新增测试已由现有 backend pytest CI 自动收集，不另加工作流。Tool 故障、HTTP SDK 回放、
取消幂等性与自动检出不同代码/Prompt 版本的受控实验尚未统一接入。

## 回放回归套件（v0.6）

新增独立 `replay_suite` 入口，复用 v0.5 CLI，不改变注入、图、Judge 或评分逻辑。
`fixtures/replay_suite_v1.json` 固定列出 Healthy、非法 Schema、模拟 429、模拟 522 四题。
在 `backend/` 执行：

```powershell
python -m evals.agent_eval_ext.replay_suite validate --manifest evals/agent_eval_ext/fixtures/replay_suite_v1.json
python -m evals.agent_eval_ext.replay_suite run --manifest evals/agent_eval_ext/fixtures/replay_suite_v1.json --out-dir /tmp/suite-baseline --revision baseline-label
python -m evals.agent_eval_ext.replay_suite compare --baseline /tmp/suite-baseline/suite.json --candidate /tmp/suite-candidate/suite.json --out /tmp/comparison.json
```

`validate` 只校验清单和全部案例，不执行图、不创建输出。`run` 先校验所有案例、拒绝重复 ID
和越界/链接路径，再新建输出目录并实际试写；现有目录一律拒绝复用。将原始清单与案例字节
保存在 `inputs/`，子进程只读取这些快照，避免运行中原文件变化。每题独立启动当前 Python，
`shell=False`，工作目录固定为 backend，默认每进程超时 180 秒（可设置 1～3600）。
顺序执行所有案例，超时、启动失败、缺失/损坏结果均记录为套件失败并继续后续题目。

子进程强制 fake，移除 API Key、业务数据库/Redis 和 Judge 观察配置，并关闭常见 tracing
导出开关；仍建议用 Docker `--network none` 强制隔离。没有测量真实网络尝试或 HTTP SDK
重试，不能用套件通过证明零网络调用。v0.5 的宿主端口隔离逻辑保持原样。

每题不仅检查退出码和 `passed`，还核对案例原始/规范化哈希、断言一致性、实际 Trace
身份/完整性、Planner span 次数、终态节点和故障 Bad Case 身份。产物路径必须落在案例目录，
通过核对的文件保存字节哈希。进程超时即使已写出 passing result 也不通过。
结果在 `<case_id>/result.json`，关联文件在同一案例目录，汇总为 `suite.json`。
这是“符合预期故障处理”的门禁，不是研究报告质量分数。

比较按非空案例集的 ID 与有效 SHA256 对齐；缺失指纹或自相矛盾的汇总会被拒绝，
案例集/字节不同则 `comparable=false`、门禁失败，退化/恢复数保持 null。
可比较时计算 recovered / regressed / unchanged，候选集必须全通过才放行；
即使没有新退化，只要候选仍有失败，门禁仍不通过。

`compare_suites(dict, dict)` 仅比较并校验声明的契约元数据。
CLI `compare` 还会重新检查保存的输入、结果、Trace/Bad Case 字节哈希；
必须保留整个套件目录，仅拷贝 `suite.json` 不足以通过完整性门禁。
发现变更或缺失时不宣称恢复/退化数量已测量，也不放行。
未通过验证的子结果不能变为 PASS，原先的超时/启动失败不会被后续坏文件覆盖。

`--revision` 仅写入 `revision_supplied`，`revision_verified=false`。
模块不检出 Git Commit、不验证 Prompt 版本，也不将两次相同工作区执行说成代码改进。
`run` 与 `compare` 门禁失败返回 1，保存采用独占新文件；既有案例、实现和历史结果不改写。
该层没有新增 Tool 故障或 SDK 级注入，也没有新增 CI workflow，测试由现有 backend CI 收集。
这里也不接入 LangSmith、Langfuse、Braintrust、Ragas 或 Promptfoo 的服务/SDK。

## 统一结果与执行入口（v0.7）

独立的 `unified_cli` 提供 `validate`、`collect`、`run`、`compare`；不修改原评测入口、
Judge 或 v0.6 套件。`fixtures/unified_dataset_v1.json` 包装原四题清单并记录数据集版本。
在 `backend/` 执行：

```powershell
python -m evals.agent_eval_ext.unified_cli validate --dataset evals/agent_eval_ext/fixtures/unified_dataset_v1.json
python -m evals.agent_eval_ext.unified_cli run --dataset evals/agent_eval_ext/fixtures/unified_dataset_v1.json --out-dir /tmp/unified --revision supplied-label
python -m evals.agent_eval_ext.unified_cli collect --dataset evals/agent_eval_ext/fixtures/unified_dataset_v1.json --replay-suite /tmp/unified/replay/suite.json --report EXISTING_REPORT.json --retrieval EXISTING_RETRIEVAL.json --grounding EXISTING_GROUNDING.json --out /tmp/collected.json
python -m evals.agent_eval_ext.unified_cli compare --baseline BASELINE_UNIFIED.json --candidate CANDIDATE_UNIFIED.json --out /tmp/unified-comparison.json
```

`run` **只新执行 fake Replay**，每题仍是独立进程执行真实编译图。可选的 `--report`、
`--retrieval`、`--grounding` 都读取已有成绩，不启动这三类评测，也不调用收费模型。
先校验所有输入、实际试写新输出目录，再冻结 dataset/清单/案例和可选历史输入的原始字节；
子进程与后续适配使用快照。已有目录拒绝复用，结果独占保存。`collect` 不执行图；
Replay 必须提供完整套件目录，采集时复用 v0.6 的 Result/Trace/Bad Case 哈希及身份核对。
`validate` 只读校验，显示此时读取到的案例哈希；这不是历史运行时指纹。

统一 JSON 保留四个独立维度：`outcome`、`reliability`、`retrieval`、`grounding`。
每个指标包含 `status/value/unit/n/note`；未测量为 null，原始 invalid 保持 invalid。
Report 的支持率是已判定 Claim 加权值，覆盖率另列；原报告 aggregate 原样保留，不把二者混为
官方逐报告平均。成本未验证计价时保持未测量，完整报告审批不由 legacy completed 标志证明。
P95 只取全部成功报告的研究耗时，缺任一成功样本耗时则不报 P95。
Retrieval 支持 corpus overall 和 selection after-cap 两种布局，各指标 n 仅统计自身有效观测，
没有逐题记录时 n 为 null，不一律填总题数。Grounding 错误率使用各自条件分母，并保留
reference kind 和路由别名声明；不冒充报告引用支持率或人工标准答案。

`test_result=pass` 表示故障处理符合预期，可以同时 `agent_status=failed`。
没有输入的维度保持未测量，四个维度不平均成总分，历史资料汇总不构成同一次配对实验。
`compare` 只对齐同版本 dataset、清单与非空案例 ID/SHA，并校验统一记录中通过率和次数一致；
候选 Replay 必须全部通过才退出 0，退化/残留失败/不可比较退出 1，非法输入或输出错误退出 2。
其 `quality_gate_passed` 仅适用于 Replay，另外三维不比较、不放行；不是全产品发布门禁。
统一 JSON 的比较不重新打开原产物链，明确 `artifact_verification_repeated=false`；
需要重新验证源文件时重新 `collect` 或使用 v0.6 套件 CLI `compare` 并保留整个源目录。
`--revision` 仍是未验证标签，不检出 Git，不证明代码或 Prompt 的受控 A/B。

## 工具契约评分与 CI（v0.8）

`tool_safety_cli` 只接受 `controlled_fixture`，按 Gold 评分合成的工具名称、参数、顺序和禁止调用。
没有接入真实 Agent 参数采集，结果始终标记 `actual_agent_executed=false`；fixture 中的
`stage=executed` 也不是实际运行证明。参数只在内存中比较，输出保留工具名、计数、阶段、
匹配模式和布尔判定，不保存原参数。阶段变化不能直接作为同口径的版本比较。

从 `backend/` 执行：

```powershell
python -m evals.agent_eval_ext.tool_safety_cli score --gold evals/agent_eval_ext/fixtures/tool_gold_v1.json --observed evals/agent_eval_ext/fixtures/tool_observed_pass_v1.json --out /tmp/tools-pass.json
python -m evals.agent_eval_ext.tool_safety_cli score --gold evals/agent_eval_ext/fixtures/tool_gold_v1.json --observed evals/agent_eval_ext/fixtures/tool_observed_fail_v1.json --out /tmp/tools-negative.json
python -m evals.agent_eval_ext.tool_safety_cli compare --baseline /tmp/tools-pass.json --candidate /tmp/tools-negative.json --out /tmp/tools-comparison.json
```

正例预期退出 0；反例及其退化比较预期退出 1，这是评分器能拒绝错误的负控，不能作为正式门禁的 passing 结果。
非法输入或输出错误退出 2。CLI 实际试写目录，最终独占保存，不覆盖既有产物；序列化失败不留下半个文件。
原始 Gold 字节哈希必须与 observed 声明相等，读取与指纹来自同一份字节；比较还校验
案例状态、指标、分母和汇总的一致性，缺失观测为 null，候选覆盖不全不能放行。
字典 API 的文件指纹由调用者提供；CLI 才直接核对读入字节。结果比较不是重新打开原始输入链。

三种模式为：strict 按位置检查完整名称序列与必需参数；unordered 做一对一无序匹配，数量必须相等；
subset 允许额外的允许工具，但仍拒绝禁止或未列入 allowed_tools 的工具。重复工具不能替代另一条调用。
Gold 参数是顶层必需键的子集，额外参数允许，嵌套值比较保留布尔/数字区别；这不验证工具完整 Schema。
strict 的 selection_correct 本身包含名称顺序，trajectory_correct 同该顺序判定；
非 strict 的 trajectory_correct 为 null，不能把它称为通用执行成功率。

`.github/workflows/agent-evaluation.yml` 增加独立的 `Agent evaluation offline gate` 检查。
所有 PR 或手动运行都会构建现有 Worker，执行 Ruff 与专项离线测试、工具正反控和四题真实图 Replay。
评测容器禁网、fake、只读根文件系统与源码挂载，不传密钥；镜像构建和 Actions 下载仍需网络。
`/tmp` 是临时 tmpfs，仅放临时文件。评分与 Replay 输出写入继承 appuser 所有权的 `/app/data` 新数据卷，
容器退出后检查 `docker cp`；只有导出成功才清理对应容器/卷。导出失败同样阻止通过，原模型/图运行错误码
不会被导出错误覆盖：分别记录 run_exit_code、export_exit_code、gate_exit_code。
上传步骤使用 always() 尝试保存日志、Result、Trace、Bad Case；不能保证强制取消或 runner 丢失时上传成功。
已有主 CI 不替换，Report/Retrieval/Grounding 不在此处重跑，也不变成七维门禁。
工作流文件不等于 GitHub 已运行或分支已受保护；尚需线上运行验证，并把该检查纳入 Required 才约束合并。

## 离线收尾报告与 Executor 案例（v0.9）

`interview_closeout` 将 v0.7 的统一结果、可选 v0.8 工具结果及可选 Executor 套件生成独立 HTML、
摘要 JSON 和 Bad Case 待排查草稿；不运行模型、不重算历史 Judge，不产生统一总分。
输入中的覆盖率与条件支持率分开显示，未测量保持 null，invalid 可保留缺失值。
指标及案例/汇总一致性先校验，工具复用 v0.8 校验器，Executor 还核对完整 Result/Trace/Bad Case 链。
输出独占创建，拒绝与输入同路径；HTML 转义输入，CSP 禁止脚本和网络资源，无外部字体或脚本。

```powershell
python -m evals.agent_eval_ext.executor_regression --out-dir /tmp/executor-suite
python -m evals.agent_eval_ext.interview_closeout --unified UNIFIED.json --executor /tmp/executor-suite/suite.json --tool TOOLS.json --baseline BASELINE.json --html /tmp/report.html --summary /tmp/summary.json --triage /tmp/triage.json
```

`executor_regression` 固定七个案例：搜索→读取→提交、畸形 URL、SSRF 拒绝、超时恢复、重复读取、
重复提交和取消。每题独立 Python 进程，真实编译图执行到 Executor 检查点，在 Critic 前暂停。
为了覆盖真实的片段核验与 HTTP/SSRF 分支，RunConfig 使用 real；模型工厂全部替换为脚本模型，
HTTP 使用 MockTransport，DNS 受控，密钥/业务端口隔离，必须在专用离线进程运行。
这不是付费 real-model 评测，不测真实网络超时或 SDK 重试，也不证明通用提交幂等性。
重复提交案例仅验证首个有效提交使本轮结束；取消案例要求实际任务传播取消并清理子任务。
输出不落盘模型工具参数；不是自由模型的 Tool Calling 准确率。

待排查池只根据未完成报告或失败的 Replay/Executor 契约生成症状草稿，根因 unknown，
不会自动修改 Gold 或晋升回归集；预期 failed/cancelled 的测试通过不算待修复缺陷。
合成工具反例不冒充真实 Agent Bad Case。

Judge 编号缺陷已在 v0.9 单独修复：两个评分入口共用 `evals/judge_labels.py`，仅接受批次内编号，
有效冲突标签仍最后一次为准。新口径 metrics_version=5；旧 v4 文件和成绩未改写，
不能直接用 v5 数值宣称模型提升，也不能据此解释历史 27 条未判定。
演示与修复证据在 `internal/agent-evaluation/phase9-validation-2026-11-12-run1/`。
