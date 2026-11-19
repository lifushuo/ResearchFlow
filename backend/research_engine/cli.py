"""research-engine：本地宿主的参考实现，见 docs/12 M6。

无需 Docker、Postgres、Redis、登录或服务器，状态保存在单个 SQLite 文件。
研究暂停后进程可以退出；之后用 --approve 从检查点继续，不重复已付费的研究。
拒绝时可用 --reject <session-id> -f 提供反馈；演示运行可使用 --fake。
桌面端通过相同引擎、端口和本地适配器包装此流程。"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import uuid
from pathlib import Path

from research_engine.local import (
    InProcessEventSink,
    SqliteCache,
    default_data_dir,
    load_env_file,
    run_config_from_env,
)
from research_engine.runner import RunOutcome, resume, run

# ── 终端输出 ──

_AGENT_MARK = {
    "planner": "◆",
    "executor": "▸",
    "critic": "✓",
    "synthesizer": "■",
}


def _print_event(event: dict) -> None:
    agent = event.get("agent") or "-"
    message = event.get("message") or ""
    print(f"  {_AGENT_MARK.get(agent, '·')} {agent:<12} {message}", file=sys.stderr)


_SOURCES_HEADING_RE = re.compile(
    r"^#{1,6}\s*(sources|references|citations|bibliography)\b", re.I | re.M
)


def _print_outcome(outcome: RunOutcome, session_id: str) -> None:
    report = outcome.report or "(no report produced)"
    print()
    print(report)

    # 综合器通常已输出来源列表，仅在缺失时补面板，避免重复显示 Sources。
    if outcome.sources and not _SOURCES_HEADING_RE.search(report):
        print()
        print("## Sources")
        for source in outcome.sources:
            print(f"[{source.get('index')}] {source.get('title') or '(untitled)'}")
            print(f"    {source.get('url')}")

    print()
    print(
        f"— {outcome.status} · {len(outcome.sources)} sources · "
        f"${outcome.cost_usd:.4f} · {outcome.tokens_input + outcome.tokens_output} tokens"
        + (f" · {outcome.elapsed_seconds}s" if outcome.elapsed_seconds is not None else "")
    )

    if outcome.status == "awaiting_approval":
        print()
        print("Paused at the review gate. Nothing is final until you approve it:")
        print(f"  research-engine --approve {session_id}")
        print(f'  research-engine --reject {session_id} -f "what to fix"')
    elif outcome.status == "awaiting_plan":
        # 此 CLI 的配置不会启用设计审批关卡，因为没有对应交互入口。
        # 仍显式处理该状态，避免意外配置启用后只打印状态却不告诉用户如何继续。
        print()
        print(
            "Paused at the research design gate, which this CLI cannot review — "
            "use the app, or leave RunConfig.skip_plan_gate at its default."
        )
    elif outcome.status == "failed":
        print(f"\nFailed: {outcome.error}", file=sys.stderr)


# ── 运行与恢复 ──


async def _drive(args: argparse.Namespace) -> tuple[RunOutcome, str]:
    """打开本地检查点和缓存，再启动或恢复研究。

    同时返回结果和会话 ID；新运行的 ID 随机生成，必须展示给用户以便恢复草稿。"""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    load_env_file()

    # 本地只有一套密钥，已由配置携带。单独 provider_keys 端口留给服务端用户密钥覆盖部署密钥的场景。
    ports = {
        "event_sink": InProcessEventSink(on_event=None if args.quiet else _print_event),
        "cache": SqliteCache(data_dir / "cache.sqlite"),
        "run_config": run_config_from_env(fake=args.fake),
    }

    async with AsyncSqliteSaver.from_conn_string(str(data_dir / "checkpoints.sqlite")) as saver:
        await saver.setup()

        if args.approve or args.reject:
            session_id = args.approve or args.reject
            outcome = await resume(
                checkpointer=saver,
                session_id=session_id,
                approved=bool(args.approve),
                feedback=args.feedback,
                **ports,
            )
            return outcome, session_id

        session_id = args.session_id or f"local-{uuid.uuid4().hex[:12]}"
        outcome = await run(
            checkpointer=saver,
            session_id=session_id,
            user_id="local",
            query=args.query,
            depth=args.depth,
            **ports,
        )
        # 同一进程内批准也从检查点恢复，重新进入关卡，不会再次规划。
        if args.yes and outcome.status == "awaiting_approval":
            outcome = await resume(
                checkpointer=saver, session_id=session_id, approved=True, **ports
            )
        return outcome, session_id


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="research-engine",
        description="Run the research pipeline locally — no server, no Docker, no login.",
    )
    parser.add_argument("query", nargs="?", help="the research question")
    parser.add_argument(
        "--depth", default="balanced", choices=["fast", "balanced", "comprehensive"]
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="scripted models and fixture sources — no API key, no network",
    )
    parser.add_argument("--approve", metavar="SESSION_ID", help="finalize a paused draft")
    parser.add_argument("--reject", metavar="SESSION_ID", help="send a paused draft back")
    parser.add_argument("-f", "--feedback", help="feedback to accompany --reject")
    parser.add_argument(
        "--yes", action="store_true", help="approve immediately (skips the gate; for demos)"
    )
    parser.add_argument("--session-id", help="use a specific session id instead of a random one")
    parser.add_argument(
        "--data-dir", help=f"where to keep local state (default: {default_data_dir()})"
    )
    parser.add_argument("--json", action="store_true", help="emit the outcome as JSON")
    parser.add_argument("--quiet", action="store_true", help="don't stream progress to stderr")

    args = parser.parse_args(argv)
    if not (args.query or args.approve or args.reject):
        parser.error("give a query, or --approve/--reject a session id")
    if args.reject and not args.feedback:
        parser.error("--reject needs -f/--feedback saying what to fix")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        outcome, session_id = asyncio.run(_drive(args))
    except KeyboardInterrupt:  # pragma: no cover — 需要交互输入
        print("\ninterrupted", file=sys.stderr)
        return 130
    except ValueError as e:
        # 模型工厂和配置错误已有可操作提示，不再用堆栈淹没这些信息。
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.json:
        payload = {
            "status": outcome.status,
            "report": outcome.report,
            "sources": outcome.sources,
            "cost_usd": outcome.cost_usd,
            "tokens_input": outcome.tokens_input,
            "tokens_output": outcome.tokens_output,
            "rework_count": outcome.rework_count,
            "elapsed_seconds": outcome.elapsed_seconds,
            "error": outcome.error,
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_outcome(outcome, session_id)

    return 1 if outcome.status == "failed" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
