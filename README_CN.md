# ResearchFlow 评测中心

评测中心位于 Web 服务的 `/evaluation`，使用现有登录账号与独立评测卷。
统一工作台包含七个分区：总览、评测运行、逐题用例、执行轨迹、坏例排查、版本对比、数据集。

## 保留的能力

- 八套固定数据集适配：Report、Corpus、Selection、Grounding、Judge Calibration、Planner Replay、Executor Regression、Tool Contract。
- 单一 Pipeline 队列，离线与在线 Worker 分离。离线受控回归不调用真实模型；在线研究、评分和向量化可能收费，需显式确认。
- 按当前 Pipeline / 数据集查看逐题结果、阶段耗时、Trace 和去重异常池，缺失指标明确显示未测量。
- 基线比较先校验案例、输入与评分口径；质量门禁不把部分评分或未测量当成通过。
- 报告草稿和有效 Judge 批次持久化，CLI 支持校验配置后续评；共享批次账本避免重复提交。
- Judge 响应诊断保留 HTTP JSON、SDK 对象、消息转换边界的安全元数据，不保存隐藏推理或密钥。

## 运行

沿用仓库原 Compose 配置：`docker-compose.full.yml` 与 `docker-compose.evaluation.override.yml`。
Windows 的 `start-evaluation.ps1 -CheckOnly` 只预检；部署使用 `start-evaluation.ps1 -Build`。
当前精简不自动部署、不重启现有容器、不触发真实模型。

新运行通过「评测运行」或「数据集」显式创建。原始结果只能写入新目录。
续评沿用 `python -m evals.harness --mode report --case-id <ID> --resume-from <原 reliability 根目录> --progress-dir <新目录> --out <新结果>`；
Judge Trace 另设 `RF_JUDGE_BATCH_TRACE_DIR`，并共用 `RF_EVAL_BATCH_LEDGER_ROOT`。
续评仍可能收费，没有新增前端续评按钮，也不承诺崩溃窗口内绝对零重复计费。

独立 Judge 默认输出预算为 4096 Token，可通过 `EVAL_JUDGE_MAX_OUTPUT_TOKENS` 配置，研究 Critic 保持原参数。
旧版 1000 Token 成绩需显式保留该值才能按原指纹续评；改变预算时使用 `--rejudge-from` 复用草稿并全批重评，
不会重跑研究或覆盖原成绩。真实调用需另行授权，完整评分仍要求全量 Claim 均已判定。

## 已移除的外围功能

旧 v1.2 单任务队列、历史目录工作台、人工审核/回归晋升流程、付费模型辅助归因、旧版本源码自动安装器。
比较和质量门禁保留在「版本对比」。旧成绩和研究业务人工审批均保留。
历史验收说明只描述当时状态；截图预览和重复安装说明已清理，删除项可从本地回滚包恢复。

## 简历表述

构建多 Agent 可信评测中心，统一接入 8 套固定评测集与隔离的离线/在线执行通道，支持逐题结果、Trace/Bad Case、基线比较及全量 Claim 覆盖率门禁；实现 Judge 批次持久化、配置指纹校验续评和多层响应诊断，避免部分评分与未测量结果被误报为通过。

八套接入表示可执行适配与数据契约，不代表全部真实模型评测已通过；不宣称未经对照验证的准确率或 P95 提升。

当前代码和验收入口见 [精简记录](internal/agent-evaluation/cleanup-2026-11-16-run1/README.md)。

## Planner Prompt 自进化

使用 GEPA 从规划漏项反馈生成候选 Prompt，在独立留出集上比较必答维度覆盖率，再用现有 Report Harness 复评报告完成和引用支持。通过两层门禁后可生成 Agent 设置的发布与回滚配置。

代码包内含无密钥演示。数据准备、PowerShell 命令与复评口径见 [Planner Prompt 自进化使用说明](docs/developers/08-testing-and-evaluation.md#planner-prompt-自进化)。

这是 CLI 驱动的受控实验能力：坏例维度由人工标注，候选经门禁后导出配置并人工应用。
前端「自进化」页面描述闭环流程、提供坏例回流入口并展示实验证据；付费运行与发布仍只在 CLI。
复用现有运行级 Prompt 快照，支持回滚配置；离线演示通过不代表已经证明真实模型质量提升。
