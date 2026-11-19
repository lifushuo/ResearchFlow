"""防止指定文件报告展示无依据结论，见 RFC V3.1 §9.4。

grounding.validate 测量，本模块决定展示内容。含不支持 Claim 的单元由受保护提示词
根据已验证内容重写，再从头核验，避免简单删句留下悬空引用或缺失前提的结论。
达到 MAX_ATTEMPTS 仍失败则整单元移除并提示；重新引入已拒绝 Claim 不再调用模型放行。
只有 complete 可以到审批关卡；unmeasured 或 no_evidence 均明确失败。"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field

from research_engine import checker_accuracy, grounding
from research_engine.grounding import (
    CAPTION,
    CONCLUSION,
    HEADING,
    LIMITATION,
    LIST_ITEM,
    PARAGRAPH,
    SUPPORTED,
    TABLE_ROW,
    UNMEASURED,
    Budget,
    BudgetExhausted,
    Validation,
)
from research_engine.runconfig import get_run_config

MAX_ATTEMPTS = 2

REMOVED_NOTICE = grounding.REMOVED_NOTICE

#: 这些原因表示检查器无法运行，改写不能解决，应停止循环以免继续浪费预算。
_TERMINAL_CAUSES = {"budget_exhausted", "provider_outage", "rate_limited", "timeout"}

_SOURCES_SPLIT_RE = re.compile(r"^#{1,6}\s*(sources|references|bibliography)\b", re.I | re.M)
_REWRITABLE = {PARAGRAPH, LIST_ITEM, LIMITATION, CONCLUSION}


@dataclass
class GroundingResult:
    text: str
    report: dict
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    failure: str | None = None


@dataclass
class _Ledger:
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    calls: int = 0
    retries: int = 0
    estimated_usd: float = 0.0
    units_repaired: int = 0
    units_dropped: int = 0
    rejected: set[str] = field(default_factory=set)
    #: 记录核验开始时间，让 G-5 能在同一运行中同时评价核验延迟与费用，见 RFC §14.1。
    started: float = field(default_factory=time.monotonic)

    def absorb(self, v: Validation) -> None:
        self.cost_usd += v.cost_usd
        self.tokens_in += v.tokens_in
        self.tokens_out += v.tokens_out
        self.calls += v.calls
        self.retries += v.retries


def _norm(text: str) -> str:
    text = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", " ", text.lower())
    return " ".join(re.findall(r"[a-z0-9.%]+", text))


def _split(draft: str) -> tuple[str, str]:
    m = _SOURCES_SPLIT_RE.search(draft or "")
    return (draft[: m.start()], draft[m.start() :]) if m else (draft or "", "")


def _preflight(body: str, budget: Budget) -> tuple[float, str | None]:
    """估算 body 的核验费用，预算容纳不下时返回拒绝原因，见 §9.9。"""
    units, _ = grounding.parse_units(body)
    sentences = sum(len(grounding.sentences_of(u)) for u in units)
    estimate = grounding._estimate(sentences)  # noqa: SLF001 — 复用唯一估算实现
    tokens = (sentences + 2 * max(1, -(-sentences // grounding.BATCH))) * 900
    if budget.cost_cap and estimate and budget.spent_usd + estimate > budget.cost_cap:
        return estimate, (
            f"budget_insufficient: checking this report is estimated at ${estimate:.4f}; "
            f"${budget.spent_usd:.4f} of the ${budget.cost_cap:.2f} limit is already spent"
        )
    if budget.token_cap and budget.tokens_in + tokens > budget.token_cap:
        return estimate, (
            f"budget_insufficient: checking this report needs about {tokens:,} input tokens; "
            f"{budget.tokens_in:,} of the {budget.token_cap:,} limit are already used"
        )
    return estimate, None


async def ground(
    draft: str,
    sources: list[dict],
    *,
    question: str,
    admissible,
    budget: Budget,
) -> GroundingResult:
    """核验草稿并修复不合格内容，返回允许展示的正文。"""
    body, _sources_tail = _split(draft)
    ledger = _Ledger()
    estimate, refused = _preflight(body, budget)
    ledger.estimated_usd = estimate
    if refused:
        return _finish(body, sources, None, ledger, status=UNMEASURED, failure=refused)

    validation: Validation | None = None
    # 最多重写 MAX_ATTEMPTS 次，每次后完整重验。最后一次核验后不再改写，
    # 确保用户看到的每个新词都经过检查；仍失败的单元整块移除，不加入新断言。
    for round_ in range(MAX_ATTEMPTS + 1):
        validation = await grounding.validate(
            body, sources, question=question, admissible=admissible, budget=budget
        )
        ledger.absorb(validation)
        for claim in validation.claims:
            if claim.state == SUPPORTED and _norm(claim.text) in ledger.rejected:
                claim.state, claim.cause = grounding.NOT_SUPPORTED, "reintroduced"
        if any(c.cause in _TERMINAL_CAUSES for c in validation.claims) or validation.unparsed_lines:
            break
        failing = sorted({c.unit for c in validation.claims if c.state != SUPPORTED})
        if not failing or round_ == MAX_ATTEMPTS:
            break
        for claim in validation.claims:
            if claim.state != SUPPORTED:
                ledger.rejected.add(_norm(claim.text))
        lines = body.splitlines()
        for index in failing:
            unit = validation.units[index]
            if unit.kind not in _REWRITABLE:
                lines[unit.line - 1] = (
                    "" if unit.kind in (TABLE_ROW, HEADING, CAPTION) else REMOVED_NOTICE
                )
                ledger.units_dropped += 1
                continue
            try:
                rewritten = await _rewrite(unit, validation, budget, ledger)
            except BudgetExhausted as exc:
                # 与其他未测量结局保持同一结构；修复中超限也必须说明具体原因，见 §9.9、AC-43。
                return _finish(
                    body,
                    sources,
                    validation,
                    ledger,
                    status=UNMEASURED,
                    failure=f"grounding_unmeasured: budget_exhausted ({exc.reason})",
                )
            except Exception as exc:  # noqa: BLE001 — 重写失败记为未测量原因
                return _finish(
                    body,
                    sources,
                    validation,
                    ledger,
                    status=UNMEASURED,
                    failure=f"grounding_unmeasured: the repair call failed ({grounding.unmeasured_cause(exc)})",
                )
            if rewritten is None:
                lines[unit.line - 1] = REMOVED_NOTICE
                ledger.units_dropped += 1
            else:
                prefix = re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", lines[unit.line - 1])
                lines[unit.line - 1] = (prefix.group(0) if prefix else "") + rewritten
                ledger.units_repaired += 1
        body = "\n".join(lines) + ("\n" if body.endswith("\n") else "")

    assert validation is not None
    if validation.unparsed_lines or any(c.state == UNMEASURED for c in validation.claims):
        causes = sorted({c.cause for c in validation.claims if c.state == UNMEASURED and c.cause})
        if "budget_exhausted" in causes and validation.budget_breach:
            causes[causes.index("budget_exhausted")] += f" ({validation.budget_breach})"
        if validation.unparsed_lines:
            causes.append("unparsed_content")
        return _finish(
            body,
            sources,
            validation,
            ledger,
            status=UNMEASURED,
            failure=f"grounding_unmeasured: {', '.join(causes)}",
        )
    if any(c.state != SUPPORTED for c in validation.claims):
        # 最后一轮仍有失败单元且没有剩余重写次数，整块移除。
        lines = body.splitlines()
        for index in sorted({c.unit for c in validation.claims if c.state != SUPPORTED}):
            unit = validation.units[index]
            lines[unit.line - 1] = (
                "" if unit.kind in (TABLE_ROW, HEADING, CAPTION) else REMOVED_NOTICE
            )
            ledger.units_dropped += 1
        body = "\n".join(lines)
    if not any(c.state == SUPPORTED for c in validation.claims):
        return _finish(
            body,
            sources,
            validation,
            ledger,
            status="no_evidence",
            failure="selection_no_evidence",
        )
    body = _drop_non_supporting(body, validation)
    return _finish(body, sources, validation, ledger, status="complete", failure=None)


async def _rewrite(unit, validation: Validation, budget: Budget, ledger: _Ledger) -> str | None:
    """根据已支持 Claim 重写一个单元；没有可写内容时返回 None。"""
    own = [c for c in validation.claims if c.unit == unit.index]
    verified = [c for c in own if c.state == SUPPORTED]
    rejected = [c for c in own if c.state != SUPPORTED]
    if not verified:
        return None
    findings = "\n".join(f"- {c.text}" for c in validation.claims if c.state == SUPPORTED)
    human = (
        f"Passage:\n{unit.text}\n\nVerified claims in this passage:\n"
        + "\n".join(f"- {c.text}" for c in verified)
        + "\n\nRejected claims (must not appear):\n"
        + "\n".join(f"- {c.text}" for c in rejected)
    )
    if unit.kind == CONCLUSION:
        human += f"\n\nThe only findings a conclusion may rest on:\n{findings}"
    scratch = Validation(units=[], claims=[])
    text = await grounding.call_model(
        "synthesizer.selection_repair", human, budget=budget, result=scratch
    )
    ledger.absorb(scratch)
    text = (text or "").strip()
    if not text or text.upper() == "REMOVE":
        return None
    return " ".join(text.splitlines()).strip()


def _drop_non_supporting(body: str, validation: Validation) -> str:
    """删除逐引用检查确认对该 Claim 毫无支持的标记。"""
    lines = body.splitlines()
    for claim in validation.claims:
        if claim.state != SUPPORTED or not claim.non_supporting_cites:
            continue
        unit = validation.units[claim.unit]
        drop = frozenset(claim.non_supporting_cites)
        lines[unit.line - 1] = re.sub(
            r"\[(\d+(?:\s*,\s*\d+)*)\]",
            lambda m, drop=drop: _without(m.group(1), drop),
            lines[unit.line - 1],
        )
    return "\n".join(lines)


def _without(group: str, drop: frozenset[int]) -> str:
    nums = [n.strip() for n in group.split(",") if int(n) not in drop]
    return f"[{', '.join(nums)}]" if nums else ""


def _finish(body, sources, validation, ledger: _Ledger, *, status: str, failure: str | None):
    used = sorted({n for line in body.splitlines() for n in grounding.markers_in(line)})
    by_index = {s.get("index"): s for s in sources or [] if isinstance(s, dict)}
    tail = ""
    if used:
        tail = (
            "\n## Sources\n"
            + "\n".join(f"[{n}] {by_index[n].get('url', '')}" for n in used if n in by_index)
            + "\n"
        )
    text = body.rstrip() + "\n" + tail
    counts = validation.counts() if validation is not None else {}
    report = {
        "status": status,
        "failure": failure,
        **counts,
        "units_repaired": ledger.units_repaired,
        "units_dropped": ledger.units_dropped,
        "validation_cost": {
            "estimated_usd": round(ledger.estimated_usd, 6),
            "actual_usd": round(ledger.cost_usd, 6),
            "calls": ledger.calls,
            "tokens_in": ledger.tokens_in,
            "tokens_out": ledger.tokens_out,
            "retries": ledger.retries,
            "repairs": ledger.units_repaired,
            "seconds": round(time.monotonic() - ledger.started, 3),
        },
        "claims": [c.as_dict() for c in validation.claims] if validation is not None else [],
        "checker_accuracy": checker_accuracy.for_route(get_run_config().model_for("critic")),
        "validated_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()
        if status == "complete"
        else None,
    }
    return GroundingResult(
        text=text,
        report=report,
        cost_usd=ledger.cost_usd,
        tokens_in=ledger.tokens_in,
        tokens_out=ledger.tokens_out,
        failure=failure,
    )
