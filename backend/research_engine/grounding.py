"""指定文件研究的逐 Claim 依据核验，见 RFC V3.1 §9。

模型拆解句子并判断含义，代码核对拆解是否覆盖原句、引用是否来自冻结选择及数字是否有据。
相关提示词受保护，用户不能改写自己的评分规则。
最终只有 supported、not_supported、contradicted、unmeasured 四种状态。
未测量必须带原因，不能当作支持或反对；仍有未测量项时不允许作为已核验报告交付。
本模块负责测量，grounding_repair 负责改写，由综合节点驱动。"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from research_engine import claims, quantities
from research_engine.llm_factory import estimate_cost, get_llm, text_of, token_counts
from research_engine.prompt_composition import system_prompt
from research_engine.runconfig import get_run_config

# ── 状态、原因与文本单元 ──

SUPPORTED, NOT_SUPPORTED, CONTRADICTED, UNMEASURED = (
    "supported",
    "not_supported",
    "contradicted",
    "unmeasured",
)

#: 拒绝原因：uncited 表示无有效引用，number_not_in_snippet 表示数字无据，其余来自模型判定。
REJECTION_CAUSES = (
    "uncited",
    "number_not_in_snippet",
    "not_supported",
    "contradicted",
    "reintroduced",
    "inference_rejected",
)

#: 无法测量的原因按异常类型和状态码分类，不依赖供应商可能随时改写的错误文字。
UNMEASURED_CAUSES = (
    "provider_outage",
    "rate_limited",
    "timeout",
    "budget_exhausted",
    "malformed_response",
    "decomposition_invalid",
    "unparsed_content",
)

#: 待核验的报告单元；标签若有引用、数字或超过 12 个词，也视为 Claim，见 §9.1。
PARAGRAPH, LIST_ITEM, TABLE_ROW, CAPTION, BLOCKQUOTE, HEADING, LIMITATION, CONCLUSION = (
    "paragraph",
    "list_item",
    "table_row",
    "caption",
    "blockquote",
    "heading",
    "limitation",
    "conclusion",
)

#: 修复循环移除内容后写入的说明，本身不陈述外部事实，不再作为 Claim 校验。
REMOVED_NOTICE = (
    "*A passage was removed here: its claims are not supported by your selected files.*"
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SOURCES_RE = re.compile(r"^#{1,6}\s*(sources|references|bibliography)\b", re.IGNORECASE)
_LIMITATIONS_RE = re.compile(
    r"^#{1,6}\s*(limitations|what your files don.t cover)\b", re.IGNORECASE
)
_CONCLUSION_RE = re.compile(
    r"^#{1,6}\s*(conclusion|conclusions|summary|executive summary|overall|in short|key findings)\b",
    re.IGNORECASE,
)
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_TABLE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")
_CAPTION_RE = re.compile(r"^(figure|table|source)\s*[\d:.]", re.IGNORECASE)
_ITALIC_RE = re.compile(r"^(\*[^*].*\*|_[^_].*_)$")
_UNPARSEABLE_RE = re.compile(r"^(```|~~~|<[a-zA-Z/!])")
_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SELECTION_PHRASE_RE = re.compile(
    r"\b(the selected (files|documents)|these (files|documents)|your (selected )?files|"
    r"none of the \d+ (selected )?(files|documents)|the files you selected)\b",
    re.IGNORECASE,
)


@dataclass
class Unit:
    """按文档顺序排列的一个待核验草稿单元。"""

    index: int
    kind: str
    text: str
    line: int
    section: str = ""


@dataclass
class Claim:
    """一条原子 Claim 及其全部判定信息。"""

    unit: int
    sentence: str
    text: str
    span: tuple[int, int]
    cites: list[int]
    calculation: dict | None = None
    #: Claim 出现但片段没有的数字，允许核验器提出计算，由代码重新算一遍确认；无缺失时为空。
    missing: list[str] = field(default_factory=list)
    valid_cites: list[int] = field(default_factory=list)
    invalid_cites: list[int] = field(default_factory=list)
    non_supporting_cites: list[int] = field(default_factory=list)
    state: str = UNMEASURED
    cause: str | None = None

    def as_dict(self) -> dict:
        return {
            "unit": self.unit,
            "text": self.text,
            "cites": self.cites,
            "valid_cites": self.valid_cites,
            "invalid_cites": self.invalid_cites,
            "non_supporting_cites": self.non_supporting_cites,
            "state": self.state,
            "cause": self.cause,
        }


@dataclass
class Validation:
    """一次核验结果，units 与 claims 均保持文档顺序。"""

    units: list[Unit]
    claims: list[Claim]
    unparsed_lines: list[int] = field(default_factory=list)
    limitation_violations: list[int] = field(default_factory=list)
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    calls: int = 0
    retries: int = 0
    estimated_usd: float = 0.0
    failure: str | None = None
    #: 具体触发的运行限制及超出量；不能只在 Claim 中留下笼统的 budget_exhausted。
    budget_breach: str | None = None

    def counts(self) -> dict:
        rejected = {cause: 0 for cause in REJECTION_CAUSES}
        unmeasured = {cause: 0 for cause in UNMEASURED_CAUSES}
        supported = 0
        for c in self.claims:
            if c.state == SUPPORTED:
                supported += 1
            elif c.state == UNMEASURED:
                unmeasured[c.cause or "malformed_response"] += 1
            else:
                rejected[c.cause or c.state] += 1
        if self.unparsed_lines:
            unmeasured["unparsed_content"] += len(self.unparsed_lines)
        return {
            "claims_total": len(self.claims),
            "claims_supported": supported,
            "claims_rejected": sum(rejected.values()),
            "rejected_by_cause": rejected,
            "claims_unmeasured": sum(unmeasured.values()),
            "unmeasured_by_cause": unmeasured,
            "invalid_citations": sum(len(c.invalid_cites) for c in self.claims),
            "non_supporting_citations": sum(len(c.non_supporting_cites) for c in self.claims),
        }


# ── 将草稿解析为文本单元，见 §9.1 ──


def parse_units(draft: str) -> tuple[list[Unit], list[int]]:
    """每行必须恰好归入一个单元，否则列为未解析。

    到 Sources 标题停止；代码围栏、原始 HTML 等无法识别的行单独返回，
    不能让内容因格式特殊而逃过核验，见 AC-38。"""
    units: list[Unit] = []
    unparsed: list[int] = []
    section, in_table, last_was_table = "", False, False
    for number, raw in enumerate((draft or "").splitlines(), start=1):
        line = raw.strip()
        if not line:
            in_table = False
            continue
        if _SOURCES_RE.match(line):
            break
        if line == REMOVED_NOTICE:
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            section = line
            units.append(Unit(len(units), HEADING, heading.group(2).strip(), number, section))
            in_table = last_was_table = False
            continue
        if _UNPARSEABLE_RE.match(line):
            unparsed.append(number)
            continue
        kind_in_section = (
            LIMITATION
            if _LIMITATIONS_RE.match(section)
            else CONCLUSION
            if _CONCLUSION_RE.match(section)
            else None
        )
        if line.startswith("|"):
            if _TABLE_SEPARATOR_RE.match(line):
                continue
            if not in_table:
                in_table = last_was_table = True
                continue  # 表头行属于标签。
            units.append(Unit(len(units), TABLE_ROW, line.strip("|").strip(), number, section))
            continue
        if line.startswith(">"):
            units.append(Unit(len(units), BLOCKQUOTE, line.lstrip("> ").strip(), number, section))
            last_was_table = False
            continue
        if _CAPTION_RE.match(line) or (last_was_table and _ITALIC_RE.match(line)):
            units.append(Unit(len(units), CAPTION, line.strip("*_ "), number, section))
            last_was_table = False
            continue
        last_was_table = False
        if _LIST_RE.match(line):
            text = _LIST_RE.sub("", line)
            units.append(Unit(len(units), kind_in_section or LIST_ITEM, text, number, section))
            continue
        units.append(Unit(len(units), kind_in_section or PARAGRAPH, line, number, section))
    return units, unparsed


def heading_is_claim(text: str) -> bool:
    """标题通常是主题标签；包含引用、数字或完整句子时仍需核验。"""
    return bool(_MARKER_RE.search(text)) or bool(re.search(r"\d", text)) or len(text.split()) > 12


def limitation_is_about_the_selection(sentence: str, question: str) -> bool:
    """关于文件未覆盖内容的说明必须描述选择范围，而非断言外部事实。

    确定性检查其提到文件选择、无引用且没有问题之外的新数字；不符合时作为普通正文 Claim 核验。"""
    if _MARKER_RE.search(sentence) or not _SELECTION_PHRASE_RE.search(sentence):
        return False
    return not [q for q in quantities.extract(sentence) if q not in quantities.extract(question)]


def sentences_of(unit: Unit) -> list[str]:
    if unit.kind in (TABLE_ROW, HEADING, CAPTION):
        return [unit.text]
    return [s.strip() for s in claims.split_sentences(unit.text) if s.strip()]


def markers_in(sentence: str) -> list[int]:
    return sorted({int(p) for m in _MARKER_RE.finditer(sentence) for p in m.group(1).split(",")})


# ── Claim 拆解，见 §9.2 ──

_FACTUAL_TOKEN_RE = re.compile(r"\d[\d,.]*|\"[^\"]+\"|“[^”]+”|\b[A-Z][a-zA-Z]{2,}\b")


def factual_token_positions(sentence: str) -> list[int]:
    """返回拆解必须覆盖的事实标记起点：数字、引号文本和大写词，排除句首词与引用标记。"""
    stripped = _MARKER_RE.sub(lambda m: " " * len(m.group(0)), sentence)
    first = re.match(r"\s*\S+", stripped)
    first_end = first.end() if first else 0
    return [m.start() for m in _FACTUAL_TOKEN_RE.finditer(stripped) if m.start() >= first_end]


#: 拆解可以略过的连接或框定词；不包含否定、因果、先后词，因为它们会改变断言含义。
_FUNCTION_WORDS = frozenset(
    """a an the and or but nor so yet while whereas both either also then too as of in on at
    to for by with from into onto over this that these those which who whom whose it its
    they their them there is are was were be been being has have had do does did
    however meanwhile notably overall additionally moreover furthermore similarly likewise
    instead respectively""".split()
)
_WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*")


def content_word_positions(sentence: str) -> list[int]:
    """返回除事实标记外仍须覆盖的实义词位置。

    没有数字或大写词的附带说明也可能是断言，不能因留在引文之外而漏检。"""
    stripped = _MARKER_RE.sub(lambda m: " " * len(m.group(0)), sentence)
    return [
        m.start() for m in _WORD_RE.finditer(stripped) if m.group(0).lower() not in _FUNCTION_WORDS
    ]


def locate_quote(sentence: str, quote: str, after: int = 0) -> tuple[int, int] | None:
    """定位 Claim 引文在原句中的字符区间，无法匹配时返回 None。

    由模型引用原词、代码计算偏移，避免模型数错字符。仅允许空白差异；
    从 after 之后取首次出现，保证重复短语与前一个 Claim 不重叠。"""
    words = quote.split()
    if not words:
        return None
    pattern = re.compile(r"\s+".join(re.escape(w) for w in words))
    found = pattern.search(sentence, after) or pattern.search(sentence)
    return found.span() if found else None


def check_decomposition(sentence: str, decomposed: list[dict]) -> list[Claim] | None:
    """校验模型拆解结构；失败时返回 None。

    通过 quote 定位 span，测试脚本可直接给 span；区间必须在句内、互不重叠、
    覆盖事实标记和实义词，引用编号必须原句已有。失败则整句未测量，不能默认支持。"""
    if not isinstance(decomposed, list):
        return None
    allowed = set(markers_in(sentence))
    out: list[Claim] = []
    spans: list[tuple[int, int]] = []
    for item in decomposed:
        if not isinstance(item, dict):
            return None
        quote = item.get("quote")
        if quote is not None:
            if not isinstance(quote, str):
                return None
            located = locate_quote(sentence, quote, after=max((e for _, e in spans), default=0))
            if located is None:
                return None
            span = list(located)
        else:
            span = item.get("span")
        text = item.get("text")
        cites = item.get("cites") or []
        if (
            not isinstance(span, list | tuple)
            or len(span) != 2
            or not all(isinstance(x, int) for x in span)
            or not isinstance(text, str)
            or not text.strip()
            or not isinstance(cites, list)
            or not all(isinstance(n, int) for n in cites)
        ):
            return None
        start, end = span
        if not (0 <= start < end <= len(sentence)):
            return None
        if any(start < e and s < end for s, e in spans):
            return None
        if not set(cites) <= allowed:
            return None
        calculation = item.get("calculation")
        if calculation is not None and not isinstance(calculation, dict):
            return None
        spans.append((start, end))
        out.append(
            Claim(
                unit=-1,
                sentence=sentence,
                text=text.strip(),
                span=(start, end),
                cites=sorted(set(cites)),
                calculation=calculation,
            )
        )
    for position in (*factual_token_positions(sentence), *content_word_positions(sentence)):
        if not any(s <= position < e for s, e in spans):
            return None
    return out


# ── 模型调用、预算与失败分类 ──

#: 单次调用重试的退避间隔，放在模块级便于测试设为零。
RETRY_BACKOFF_SECONDS = 0.5
MAX_RETRIES = 2


class BudgetExhausted(Exception):
    """核验期间超出运行限制；reason 说明具体限制及超出量。"""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def unmeasured_cause(exc: BaseException) -> str:
    """根据异常类型和状态码确定未测量原因。"""
    if isinstance(exc, BudgetExhausted):
        return "budget_exhausted"
    if isinstance(exc, asyncio.TimeoutError | TimeoutError) or "Timeout" in type(exc).__name__:
        return "timeout"
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    if status == 429 or type(exc).__name__ == "RateLimitError":
        return "rate_limited"
    return "provider_outage"


class Budget:
    """将运行已有预算同时用于核验调用，见 §9.9。

    从本轮已消耗量起算，每次核验调用继续累加；在调用边界检查，不截断正在返回的响应。"""

    def __init__(self, *, spent_usd: float, tokens_in: int, started_at: float | None) -> None:
        cfg = get_run_config()
        self.cost_cap = cfg.max_cost_per_session_usd
        self.token_cap = cfg.max_input_tokens
        self.time_cap = cfg.max_wallclock_seconds
        self.spent_usd = spent_usd
        self.tokens_in = tokens_in
        self.started_at = started_at if started_at is not None else time.time()

    def breach(self) -> str | None:
        if self.cost_cap and self.spent_usd >= self.cost_cap:
            return f"cost ceiling reached during validation: ${self.spent_usd:.4f} of ${self.cost_cap:.2f}"
        if self.token_cap and self.tokens_in >= self.token_cap:
            return (
                f"input-token ceiling reached during validation: {self.tokens_in:,} of "
                f"{self.token_cap:,}"
            )
        elapsed = time.time() - self.started_at
        if self.time_cap and elapsed >= self.time_cap:
            return f"time limit reached during validation: {elapsed:.0f}s of {self.time_cap}s"
        return None


async def call_model(purpose: str, human: str, *, budget: Budget, result: Validation) -> str:
    """执行受保护模型调用，暂时故障可有限重试，并计入预算。

    调用前已超限则抛 BudgetExhausted；达到 MAX_RETRIES 后重抛最后错误，由调用方记录未测量原因。"""
    breach = budget.breach()
    if breach:
        result.budget_breach = breach
        raise BudgetExhausted(breach)
    role = purpose.split(".", 1)[0]
    model = get_llm(role)
    messages = [SystemMessage(content=system_prompt(purpose)), HumanMessage(content=human)]
    attempt = 0
    while True:
        try:
            from research_engine.eval_trace import traced_ainvoke

            response = await traced_ainvoke(model, messages, role=role)
            break
        except Exception as exc:  # noqa: BLE001 — 下方会分类、重试或重新抛出
            cause = unmeasured_cause(exc)
            if (
                cause not in ("rate_limited", "timeout", "provider_outage")
                or attempt >= MAX_RETRIES
            ):
                raise
            attempt += 1
            result.retries += 1
            await asyncio.sleep(RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    cost = estimate_cost(response, role)
    i, o = token_counts(response)
    result.cost_usd += cost
    result.tokens_in += i
    result.tokens_out += o
    result.calls += 1
    budget.spent_usd += cost
    budget.tokens_in += i
    return text_of(response)


_VERDICT_RE = re.compile(
    r"^\s*Claim\s+(\d+)\s*:\s*(SUPPORTED|NOT_SUPPORTED|CONTRADICTED)\s*$", re.I
)
_PAIR_RE = re.compile(r"^\s*Pair\s+(\d+)\s*:\s*(YES|NO)\s*$", re.I)
_CALC_RE = re.compile(
    r"^\s*Calc\s+(\d+)\s*:\s*(add|subtract|multiply|divide|percent_change)\s+(\S+)\s+(\S+)"
    r"\s*=\s*(\S+)\s*$",
    re.I,
)


def parse_calculations(text: str, n: int) -> list[tuple[str, list[str], str] | None]:
    """提取每条 Claim 提出的计算：运算、操作数、结果；未提供时为 None。"""
    out: list[tuple[str, list[str], str] | None] = [None] * n
    for line in (text or "").splitlines():
        m = _CALC_RE.match(line)
        if m and 1 <= int(m.group(1)) <= n:
            out[int(m.group(1)) - 1] = (m.group(2).lower(), [m.group(3), m.group(4)], m.group(5))
    return out


def parse_verdicts(text: str, n: int) -> list[str | None]:
    """解析每条 Claim 的三分类判定；缺失或格式错误为 None，表示未测量。"""
    out: list[str | None] = [None] * n
    for line in (text or "").splitlines():
        m = _VERDICT_RE.match(line)
        if m and 1 <= int(m.group(1)) <= n:
            out[int(m.group(1)) - 1] = m.group(2).lower()
    return out


def parse_pairs(text: str, n: int) -> list[bool | None]:
    out: list[bool | None] = [None] * n
    for line in (text or "").splitlines():
        m = _PAIR_RE.match(line)
        if m and 1 <= int(m.group(1)) <= n:
            out[int(m.group(1)) - 1] = m.group(2).upper() == "YES"
    return out


def _json_array(text: str):
    """提取模型回复中的 JSON 数组，兼容外层代码围栏。"""
    body = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if fence:
        body = fence.group(1).strip()
    start = body.find("[")
    if start < 0:
        raise ValueError("no JSON array in the reply")
    return json.loads(body[start : body.rfind("]") + 1])


# ── 依据核验 ──

BATCH = 4


@dataclass
class Evidence:
    """引用编号对应的片段及是否允许引用。

    来源不在冻结选择内或没有实际检索证据时 admissible 为 False，该标记不构成有效引用。"""

    snippets: list[str]
    admissible: bool


def evidence_by_index(
    sources: list[dict], admissible: Callable[[str], bool]
) -> dict[int, Evidence]:
    out: dict[int, Evidence] = {}
    for source in sources or []:
        if not isinstance(source, dict) or not isinstance(source.get("index"), int):
            continue
        snippets = [s for s in (source.get("snippets") or [source.get("snippet") or ""]) if s]
        out[source["index"]] = Evidence(
            snippets=snippets, admissible=bool(snippets) and admissible(source.get("url") or "")
        )
    return out


async def validate(
    draft: str,
    sources: list[dict],
    *,
    question: str,
    admissible: Callable[[str], bool],
    budget: Budget,
) -> Validation:
    """解析、拆解并核验草稿每条 Claim。模型故障转成对应 Claim 的未测量原因，不直接抛出。"""
    units, unparsed = parse_units(draft)
    result = Validation(units=units, claims=[], unparsed_lines=unparsed)
    evidence = evidence_by_index(sources, admissible)

    # 确定哪些句子需要拆解，以及各自适用的规则。
    todo: list[tuple[Unit, str]] = []
    for unit in units:
        if unit.kind == HEADING and not heading_is_claim(unit.text):
            continue
        for sentence in sentences_of(unit):
            if not claims.is_claim_sentence(sentence) and unit.kind != HEADING:
                continue
            if unit.kind == LIMITATION and limitation_is_about_the_selection(sentence, question):
                continue
            if unit.kind == LIMITATION:
                result.limitation_violations.append(unit.index)
            todo.append((unit, sentence))

    result.estimated_usd = _estimate(len(todo))

    await _decompose(todo, result, budget)
    decided = [c for c in result.claims if c.state == UNMEASURED and c.cause is None]
    _check_citations(decided, evidence)
    _check_quantities(decided, evidence)
    pending = [c for c in decided if c.state == UNMEASURED and c.cause is None]
    conclusion_units = {u.index for u in units if u.kind == CONCLUSION}
    body = [c for c in pending if c.unit not in conclusion_units]
    conclusions = [c for c in pending if c.unit in conclusion_units]
    for c in conclusions:
        if c.missing:
            c.state, c.cause = NOT_SUPPORTED, "number_not_in_snippet"
    conclusions = [c for c in conclusions if not c.missing]
    await _verify(body, evidence, result, budget, purpose="critic.claim_verify")
    await _relevance(body, evidence, result, budget)
    supported = [c for c in result.claims if c.state == SUPPORTED]
    await _verify_inference(conclusions, supported, evidence, result, budget)
    return result


def _estimate(sentences: int) -> float:
    """调用前估算核验费用：调用数乘观察到的 Token 数和单价。

    每句一次拆解，每四条 Claim 一次核验和一次相关性检查，粗略按每句一条 Claim 估计。
    路由无价格时返回 0.0，但调用方必须显示未知，不能显示免费。"""
    from research_engine import catalog
    from research_engine.llm_factory import model_name_for

    spec = catalog.get(model_name_for("critic"))
    if spec is None or not spec.priced:
        return 0.0
    calls = sentences + 2 * max(1, -(-sentences // BATCH))
    tokens_in, tokens_out = 900, 200
    return calls * (tokens_in * spec.input_per_mtok + tokens_out * spec.output_per_mtok) / 1e6


async def _decompose(todo: list[tuple[Unit, str]], result: Validation, budget: Budget) -> None:
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        human = "\n\n".join(f"Sentence {i}: {s}" for i, (_, s) in enumerate(batch, start=1))
        cause = None
        parsed = None
        try:
            parsed = _json_array(
                await call_model("critic.claim_decompose", human, budget=budget, result=result)
            )
        except ValueError, json.JSONDecodeError:
            cause = "malformed_response"
        except Exception as exc:  # noqa: BLE001 — 每类故障转成明确原因
            cause = unmeasured_cause(exc)
        by_sentence: dict[int, list] = {}
        if parsed is not None:
            for item in parsed if isinstance(parsed, list) else []:
                if isinstance(item, dict) and isinstance(item.get("sentence"), int):
                    by_sentence[item["sentence"]] = item.get("claims")
        for i, (unit, sentence) in enumerate(batch, start=1):
            if cause is not None:
                result.claims.append(_undecided(unit, sentence, cause))
                continue
            checked = check_decomposition(sentence, by_sentence.get(i))
            if checked is None:
                result.claims.append(_undecided(unit, sentence, "decomposition_invalid"))
                continue
            for claim in checked:
                claim.unit = unit.index
                result.claims.append(claim)


def _undecided(unit: Unit, sentence: str, cause: str) -> Claim:
    return Claim(
        unit=unit.index,
        sentence=sentence,
        text=sentence,
        span=(0, len(sentence)),
        cites=markers_in(sentence),
        state=UNMEASURED,
        cause=cause,
    )


def _check_citations(pending: list[Claim], evidence: dict[int, Evidence]) -> None:
    """在模型调用前完成引用结构检查，见 §9.3。"""
    for claim in pending:
        for n in claim.cites:
            ev = evidence.get(n)
            (claim.valid_cites if ev is not None and ev.admissible else claim.invalid_cites).append(
                n
            )
        if not claim.valid_cites:
            claim.state, claim.cause = NOT_SUPPORTED, "uncited"


def _snippets(claim: Claim, evidence: dict[int, Evidence], cites: list[int] | None = None) -> str:
    return "\n".join(
        f"- {s}"
        for n in (cites if cites is not None else claim.valid_cites)
        for s in evidence[n].snippets
    )


def _check_quantities(pending: list[Claim], evidence: dict[int, Evidence]) -> None:
    for claim in pending:
        if claim.state != UNMEASURED or claim.cause is not None:
            continue
        snippets = _snippets(claim, evidence)
        # 引用标记不是断言数字，[1] 不代表 Claim 陈述数值 1。
        missing = quantities.unsupported(_MARKER_RE.sub(" ", claim.text), snippets)
        calc = claim.calculation
        if (
            missing
            and calc
            and quantities.calculation_holds(
                str(calc.get("op")),
                [str(v) for v in calc.get("operands") or []],
                str(calc.get("result")),
                snippets,
            )
        ):
            # 计算操作数已在片段中核对，只有代码复算得到的结果允许不直接出现在片段中。
            result_values = {str(v) for v in [calc.get("result")]}
            missing = [m for m in missing if not any(m in r or r in m for r in result_values)]
        # 暂不拒绝片段中未逐字出现的数字；核验器可提出基于片段数值的计算，
        # 但只有 _verify 重新计算确认后才接受。
        claim.missing = missing


def _as_written(claim: Claim) -> str:
    """取 Claim 来源的报告原文，并移除引用标记。"""
    words = claim.sentence[claim.span[0] : claim.span[1]]
    return " ".join(_MARKER_RE.sub(" ", words).split())


def _verify_human(i: int, claim: Claim, evidence: dict[int, Evidence]) -> str:
    lines = [f"Claim {i}: {claim.text}", f'As written in the report: "{_as_written(claim)}"']
    if claim.missing:
        lines.append(
            f"Figures not found in the evidence as written: {', '.join(claim.missing)}. "
            f"If they are computed from figures the evidence states, add the line "
            f'"Calc {i}: <add|subtract|multiply|divide|percent_change> <a> <b> = <result>".'
        )
    lines.append(f"Evidence {i}:\n{_snippets(claim, evidence)}")
    return "\n".join(lines)


def _account_for_figures(claim: Claim, calc, evidence: dict[int, Evidence]) -> bool:
    """检查计算能否解释片段缺失的每个数值：操作数必须有据，代码复算正确，结果与 Claim 一致。"""
    if calc is None:
        return False
    op, operands, outcome = calc
    snippets = _snippets(claim, evidence)
    if not quantities.calculation_holds(op, operands, outcome, snippets):
        return False
    text = _MARKER_RE.sub(" ", claim.text)
    return not quantities.unsupported(text, f"{snippets}\n{outcome}")


async def _verify(
    pending: list[Claim],
    evidence: dict[int, Evidence],
    result: Validation,
    budget: Budget,
    *,
    purpose: str,
) -> None:
    for start in range(0, len(pending), BATCH):
        batch = pending[start : start + BATCH]
        human = "\n\n".join(_verify_human(i, c, evidence) for i, c in enumerate(batch, start=1))
        try:
            reply = await call_model(purpose, human, budget=budget, result=result)
        except Exception as exc:  # noqa: BLE001
            cause = unmeasured_cause(exc)
            for c in batch:
                c.state, c.cause = UNMEASURED, cause
            continue
        verdicts = parse_verdicts(reply, len(batch))
        calcs = parse_calculations(reply, len(batch))
        for c, verdict, calc in zip(batch, verdicts, calcs, strict=True):
            if verdict is None:
                c.state, c.cause = UNMEASURED, "malformed_response"
            elif (
                verdict == "supported" and c.missing and not _account_for_figures(c, calc, evidence)
            ):
                c.state, c.cause = NOT_SUPPORTED, "number_not_in_snippet"
            elif verdict == "supported":
                c.state, c.cause = SUPPORTED, None
            elif verdict == "contradicted":
                c.state, c.cause = CONTRADICTED, "contradicted"
            else:
                cause = "number_not_in_snippet" if c.missing else "not_supported"
                c.state, c.cause = NOT_SUPPORTED, cause


async def _relevance(
    claims_: list[Claim], evidence: dict[int, Evidence], result: Validation, budget: Budget
) -> None:
    """单独核验已支持 Claim 的每个引用是否至少支持部分内容。

    无关引用必须移除并计数，即使其他引用足够支持整条 Claim，见 §9.3。"""
    pairs = [(c, n) for c in claims_ if c.state == SUPPORTED for n in c.valid_cites]
    for start in range(0, len(pairs), BATCH):
        batch = pairs[start : start + BATCH]
        human = "\n\n".join(
            f"Pair {i}: Claim: {c.text}\nSnippet:\n{_snippets(c, evidence, [n])}"
            for i, (c, n) in enumerate(batch, start=1)
        )
        try:
            answers = parse_pairs(
                await call_model("critic.citation_relevance", human, budget=budget, result=result),
                len(batch),
            )
        except Exception as exc:  # noqa: BLE001
            cause = unmeasured_cause(exc)
            for c, _ in batch:
                c.state, c.cause = UNMEASURED, cause
            continue
        for (c, n), answer in zip(batch, answers, strict=True):
            if answer is None:
                c.state, c.cause = UNMEASURED, "malformed_response"
            elif answer is False and n not in c.non_supporting_cites:
                c.non_supporting_cites.append(n)
    for c in claims_:
        if c.state == SUPPORTED and set(c.non_supporting_cites) >= set(c.valid_cites):
            # 全部引用单独看都无关，不能保留此前组合起来的支持判定。
            c.state, c.cause = NOT_SUPPORTED, "not_supported"


async def _verify_inference(
    conclusions: list[Claim],
    supported: list[Claim],
    evidence: dict[int, Evidence],
    result: Validation,
    budget: Budget,
) -> None:
    """结论 Claim 同时对照已支持的研究发现核验，不仅对照片段。"""
    if not conclusions:
        return
    findings = "\n".join(f"- {c.text}" for c in supported) or "- (no supported findings)"
    for start in range(0, len(conclusions), BATCH):
        batch = conclusions[start : start + BATCH]
        human = "\n\n".join(
            f"Claim {i}: {c.text}\nFindings:\n{findings}\nSnippets:\n{_snippets(c, evidence)}"
            for i, c in enumerate(batch, start=1)
        )
        try:
            verdicts = parse_verdicts(
                await call_model("critic.inference_verify", human, budget=budget, result=result),
                len(batch),
            )
        except Exception as exc:  # noqa: BLE001
            cause = unmeasured_cause(exc)
            for c in batch:
                c.state, c.cause = UNMEASURED, cause
            continue
        for c, verdict in zip(batch, verdicts, strict=True):
            if verdict is None:
                c.state, c.cause = UNMEASURED, "malformed_response"
            elif verdict == "supported":
                c.state, c.cause = SUPPORTED, None
            else:
                c.state, c.cause = NOT_SUPPORTED, "inference_rejected"
