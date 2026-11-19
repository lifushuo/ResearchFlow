"""独立研究证据包核验器，见 docs/12 M12。

无需 AI、网络或应用安装，读取 .bundle.json，输出可读或 JSON 报告，退出 0 表示通过、1 表示失败。
运行方式：python -m research_engine.verify_bundle path/to/bundle.json，可加 --format json。
除 Pydantic 证据包结构外仅需标准库，Python 3.14+ 可运行。

检查数据结构、证据包哈希、报告哈希、片段哈希、引用解析、Claim 与证据联系、审批链；
轨迹状态仅作信息披露。兼容 v1、v2，v2 另含受哈希保护的实际提示词及摘要。"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from research_engine.bundle import (
    BundleManifest,
    compute_bundle_hash,
    content_hash,
)

# 有意保留 claims 引用模式的本地副本，维持独立核验器的最小依赖。
# 一致性测试保证副本不偏离唯一标准，不能各自修改。
_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SOURCES_HEADING_RE = re.compile(r"^#{1,6}\s*(sources|references|citations|bibliography)\b", re.I)

#: 唯一能够授权报告及其产物的批准动作。
REPORT_APPROVAL_ACTION = "approved"
#: 计划关卡动作仅记入审批链，不授予报告发布权限。
PLAN_GATE_ACTIONS = frozenset({"plan_approved", "plan_rework_requested", "plan_rejected"})


# ── 核验结果类型 ──


#: 核验器支持的格式版本。v2 只增加提示词来源，其余检查对两个版本相同。
#:
#: 已发布的旧核验器不会自动接受新格式，因此先发布支持新版本的核验器，再让生成端输出新版包。
SUPPORTED_BUNDLE_VERSIONS: frozenset[int] = frozenset({1, 2})


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class VerifyResult:
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # 演示状态与 passed 并列披露。脚本证据包也能完整通过哈希核验，不能只显示 PASS 让人误以为是真研究。
    demo: bool = False

    # 提示词来源与通过状态并列，避免完整性通过掩盖角色指令已被替换；v1 无此信息。
    prompt_provenance: list = field(default_factory=list)

    # 直接复制 NONE、APPLIED、UNUSABLE；v1 为 None。
    # 快照不可用后采用默认提示词，不等于从未配置自定义提示词，因此状态单独保留。
    prompt_overrides_status: str | None = None


# ── 各项核验 ──


def _check_bundle_integrity(bundle: BundleManifest) -> CheckResult:
    """核对证据包自身哈希及内部各摘要一致性。

    外层哈希能发现组装后篡改，但攻击者若修改提示词再重算包哈希，仍可能留下旧提示词摘要。
    因此这里同时复算来源摘要。v1 无提示词来源，不改变其原有行为。"""
    expected = compute_bundle_hash(bundle)
    if bundle.bundle_hash != expected:
        return CheckResult(
            "bundle_integrity",
            False,
            f"bundle_hash mismatch: recorded {bundle.bundle_hash[:16]}… "
            f"but computed {expected[:16]}… — the bundle was modified after assembly",
        )

    # 不能携带声明版本之后才引入的字段。否则 v1 中的新版来源字段不受哈希保护，
    # 新读取器却可能误当作已核验内容，旧核验器也无法发现。
    if bundle.bundle_version < 2 and (bundle.prompt_provenance or bundle.prompt_overrides_status):
        return CheckResult(
            "bundle_integrity",
            False,
            f"bundle declares version {bundle.bundle_version} but carries v2 prompt "
            "provenance, which that version's hash does not cover",
        )

    for record in bundle.prompt_provenance:
        recomputed = content_hash(record.effective_prompt)
        if record.effective_prompt_sha256 != recomputed:
            return CheckResult(
                "bundle_integrity",
                False,
                f"prompt provenance for {record.purpose} is inconsistent: recorded "
                f"{record.effective_prompt_sha256[:16]}… but the prompt text hashes to "
                f"{recomputed[:16]}… — the recorded prompt and its hash disagree",
            )

    return CheckResult("bundle_integrity", True)


def _check_report_integrity(bundle: BundleManifest) -> CheckResult:
    expected = content_hash(bundle.report)
    if bundle.report_hash == expected:
        return CheckResult("report_integrity", True)
    return CheckResult(
        "report_integrity",
        False,
        f"report_hash mismatch: recorded {bundle.report_hash[:16]}… "
        f"but computed {expected[:16]}… — the report text was modified",
    )


def _check_evidence_integrity(bundle: BundleManifest) -> CheckResult:
    bad: list[str] = []
    for i, e in enumerate(bundle.evidence):
        expected = content_hash(e.snippet)
        if e.content_hash != expected:
            bad.append(f"  snippet {i} from {e.source_url}: hash mismatch")
    if not bad:
        return CheckResult("evidence_integrity", True)
    return CheckResult(
        "evidence_integrity",
        False,
        f"{len(bad)} snippet(s) tampered:\n" + "\n".join(bad),
    )


def _body_before_sources(text: str) -> str:
    for i, line in enumerate(text.splitlines()):
        if _SOURCES_HEADING_RE.match(line.strip()):
            return "\n".join(text.splitlines()[:i])
    return text


def _check_citation_resolution(bundle: BundleManifest) -> CheckResult:
    body = _body_before_sources(bundle.report)
    valid_indices = {s.get("index") for s in bundle.sources if isinstance(s.get("index"), int)}
    cited: list[int] = []
    for m in _CITE_RE.finditer(body):
        cited.extend(int(p.strip()) for p in m.group(1).split(","))
    unresolved = sorted({n for n in cited if n not in valid_indices})
    if not unresolved:
        return CheckResult("citation_resolution", True)
    return CheckResult(
        "citation_resolution",
        False,
        f"Unresolved citation markers: {unresolved} — no matching source entry",
    )


def _check_claim_evidence_linkage(bundle: BundleManifest) -> CheckResult:
    source_indices = {s.get("index") for s in bundle.sources if isinstance(s.get("index"), int)}
    urls_by_index: dict[int, str] = {}
    for s in bundle.sources:
        idx = s.get("index")
        if isinstance(idx, int):
            urls_by_index[idx] = s.get("url", "")

    evidence_urls = {e.source_url for e in bundle.evidence}
    gaps: list[str] = []
    for claim in bundle.claims:
        for idx in claim.citation_indices:
            if idx not in source_indices:
                gaps.append(
                    f'  claim cites [{idx}] which has no source entry: "{claim.sentence[:60]}…"'
                )
            elif urls_by_index.get(idx, "") not in evidence_urls:
                gaps.append(
                    f"  claim cites [{idx}] ({urls_by_index.get(idx, '?')}) "
                    f"which has no evidence snippet"
                )
    if not gaps:
        return CheckResult("claim_evidence_linkage", True)
    # 多条 Claim 引用同一个缺证据来源时，去重展示。
    unique = sorted(set(gaps))
    return CheckResult(
        "claim_evidence_linkage",
        False,
        f"{len(unique)} linkage gap(s):\n" + "\n".join(unique),
    )


def _check_approval_chain(bundle: BundleManifest) -> CheckResult:
    if not bundle.approval_chain:
        return CheckResult(
            "approval_chain",
            False,
            "No approval records — this report was never human-reviewed",
        )

    # 只有 REPORT 批准可以授权报告。plan_approved 发生时草稿尚不存在，其 hash 不能满足报告审批检查。
    #
    # 显式排除计划批准，防止未来重命名后不小心获得报告授权。
    approved_entries = [
        a
        for a in bundle.approval_chain
        if a.action == REPORT_APPROVAL_ACTION and a.action not in PLAN_GATE_ACTIONS
    ]
    if not approved_entries:
        return CheckResult(
            "approval_chain",
            False,
            "No 'approved' entry in the chain — the report was never approved",
        )

    empty_hashes = [a for a in bundle.approval_chain if not a.draft_hash]
    if empty_hashes:
        return CheckResult(
            "approval_chain",
            False,
            f"{len(empty_hashes)} approval record(s) have an empty draft_hash",
        )

    # 关键检查：至少有一条批准记录的 draft_hash 与当前 report_hash 一致，证明批准针对这份报告。
    linked = any(a.draft_hash == bundle.report_hash for a in approved_entries)
    if not linked:
        return CheckResult(
            "approval_chain",
            False,
            "No 'approved' entry's draft_hash matches report_hash — "
            "the approval record does not apply to this report",
        )

    return CheckResult("approval_chain", True)


#: 指定文件运行的证据包首条轨迹记录类型，见 app/run_bundle.py。
CORPUS_SELECTION_EVENT = "corpus_selection"


def _check_corpus_selection(bundle: BundleManifest) -> CheckResult | None:
    """检查指定文件运行的每个来源都属于冻结选择中的文档版本。

    仅在证据包记录选择时执行；外层哈希保护选择和来源，
    此检查进一步发现内容自洽却越过授权文件范围的包，见 RFC V3.1 §8.6。"""
    entries = [
        t for t in bundle.trace if isinstance(t, dict) and t.get("event") == CORPUS_SELECTION_EVENT
    ]
    if not entries:
        return None
    if len(entries) > 1:
        return CheckResult("corpus_selection", False, "more than one selection is recorded")
    resolved = entries[0].get("resolved")
    if not isinstance(resolved, list) or not resolved:
        return CheckResult("corpus_selection", False, "the recorded selection is empty")
    allowed = {str(item.get("document_id")) for item in resolved if isinstance(item, dict)}

    outside: list[str] = []
    for source in bundle.sources:
        url = str(source.get("url") or "") if isinstance(source, dict) else ""
        if not url.startswith("corpus://"):
            outside.append(url or "(no url)")
            continue
        doc = url[len("corpus://") :].split("#", 1)[0]
        if doc not in allowed:
            outside.append(url)
    if outside:
        return CheckResult(
            "corpus_selection",
            False,
            f"{len(outside)} source(s) are outside the run's selected files: {outside[:3]}",
        )
    return CheckResult(
        "corpus_selection", True, f"all {len(bundle.sources)} source(s) are selected files"
    )


# ── 顶层核验入口 ──


def verify(bundle: BundleManifest) -> VerifyResult:
    """执行全部检查，返回含逐项详情的 VerifyResult。"""
    checks = [
        CheckResult("schema_validity", True),  # 此前已经解析过数据结构。
        _check_bundle_integrity(bundle),
        _check_report_integrity(bundle),
        _check_evidence_integrity(bundle),
        _check_citation_resolution(bundle),
        _check_claim_evidence_linkage(bundle),
        _check_approval_chain(bundle),
    ]
    selection = _check_corpus_selection(bundle)
    if selection is not None:
        checks.append(selection)

    notes: list[str] = []
    if not bundle.trace_available:
        notes.append(
            "Trace unavailable: this bundle was produced by a host without durable event logging."
        )
    elif not bundle.trace:
        notes.append("Trace is empty (no agent events recorded for this session).")

    passed = all(c.passed for c in checks)
    return VerifyResult(
        passed=passed,
        checks=checks,
        notes=notes,
        demo=bundle.demo,
        prompt_provenance=list(bundle.prompt_provenance),
        prompt_overrides_status=bundle.prompt_overrides_status,
    )


def verify_file(path: str | Path) -> VerifyResult:
    """加载并核验 .bundle.json，结构解析失败也算核验失败。"""
    p = Path(path)
    try:
        data = json.loads(p.read_text("utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return VerifyResult(
            passed=False,
            checks=[CheckResult("schema_validity", False, f"Cannot read bundle: {e}")],
        )

    try:
        bundle = BundleManifest.model_validate(data)
    except Exception as e:  # noqa: BLE001
        return VerifyResult(
            passed=False,
            checks=[CheckResult("schema_validity", False, f"Schema validation failed: {e}")],
        )

    if bundle.bundle_version not in SUPPORTED_BUNDLE_VERSIONS:
        supported = ", ".join(str(v) for v in sorted(SUPPORTED_BUNDLE_VERSIONS))
        return VerifyResult(
            passed=False,
            checks=[
                CheckResult(
                    "schema_validity",
                    False,
                    f"Unsupported bundle_version {bundle.bundle_version} "
                    f"(this verifier supports {supported})",
                )
            ],
        )

    return verify(bundle)


# ── 人工可读输出 ──


#: 固定非 ASCII 图符及其在不支持终端上的替代文字。
#:
#: 第三方可能在任意终端离线核验。Windows 某些编码无法打印勾号，
#: 曾导致全部检查已通过却在输出时抛 UnicodeEncodeError，用户看不到结论。
#:
#: 不统一替换成问号，否则用户无法区分通过与失败。
#: 固定图符用明确替代，任意详情文字另按编码转换。
_ASCII_FALLBACK = {"✓": "[PASS]", "✗": "[FAIL]", "ℹ": "[note]", "—": "--", "…": "..."}


def _encodable(stream) -> bool:
    """检查输出流能否无异常地显示这些图符。"""
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        "".join(_ASCII_FALLBACK).encode(encoding)
    except UnicodeEncodeError, LookupError:
        return False
    return True


def _degrade(text: str) -> str:
    """生成任意终端都能打印的同一份报告。"""
    for glyph, plain in _ASCII_FALLBACK.items():
        text = text.replace(glyph, plain)
    return text.encode("ascii", "replace").decode("ascii")


def format_text(result: VerifyResult, stream=None) -> str:
    rich = _encodable(stream if stream is not None else sys.stdout)
    lines: list[str] = []
    for c in result.checks:
        mark = "✓" if c.passed else "✗"
        lines.append(f"  {mark} {c.name}")
        if c.detail:
            for d in c.detail.splitlines():
                lines.append(f"    {d}")
    for note in result.notes:
        lines.append(f"  ℹ {note}")
    if result.prompt_provenance:
        # 这里只输出提示词摘要；完整正文保留在证据包中，避免长提示词淹没核验结论。
        customised = [r for r in result.prompt_provenance if getattr(r, "overridden", False)]
        lines.append(
            f"  ℹ AgentSpec provenance: {len(result.prompt_provenance)} purposes recorded, "
            f"{len(customised)} running a replaced prompt"
        )
        for record in customised:
            lines.append(f"    replaced: {record.purpose} ({record.effective_prompt_sha256[:16]}…)")
        if result.prompt_overrides_status == "UNUSABLE":
            lines.append(
                "    note: the run's stored overrides could not be used; "
                "every purpose ran on its shipped prompt"
            )

    verdict = "PASS" if result.passed else "FAIL"
    lines.insert(0, f"Bundle verification: {verdict}")
    if any(getattr(r, "overridden", False) for r in result.prompt_provenance):
        # 在 PASS 前展示配置改动。哈希通过只证明产物一致，读者仍需先知道代理指令被自定义。
        lines.insert(
            0,
            "!! CUSTOMISED AGENTS — one or more prompts were replaced by the run's owner.\n"
            "!! The checks below confirm what those prompts were and that they are\n"
            "!! unmodified since assembly, not that they were sound.\n",
        )
    if result.demo:
        # 在结论前显示演示标记。脚本产物也能通过完整性检查，不能让单独 PASS 误导成真实研究。
        lines.insert(
            0,
            "!! DEMO BUNDLE — scripted models and fixture sources. NOT REAL RESEARCH.\n"
            "!! The checks below confirm this file is internally consistent,\n"
            "!! not that its findings are true.\n",
        )
    text = "\n".join(lines)
    return text if rich else _degrade(text)


def format_json(result: VerifyResult) -> str:
    return json.dumps(
        {
            "passed": result.passed,
            "demo": result.demo,
            "prompt_overrides_status": result.prompt_overrides_status,
            "prompt_provenance": [
                {
                    "purpose": r.purpose,
                    "role": r.role,
                    "policy": r.policy,
                    "overridden": r.overridden,
                    "effective_prompt_sha256": r.effective_prompt_sha256,
                }
                for r in result.prompt_provenance
            ],
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail or None}
                for c in result.checks
            ],
            "notes": result.notes,
        },
        indent=2,
    )


# ── 命令行入口 ──


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("Usage: python -m research_engine.verify_bundle <path.bundle.json> [--format json]")
        return 0

    path = args[0]
    fmt = "json" if "--format" in args and "json" in args else "text"

    result = verify_file(path)
    output = format_json(result) if fmt == "json" else format_text(result)
    print(output)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
