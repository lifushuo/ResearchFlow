"""按明确且保守的规则比较 Claim 和片段中的数量，见 RFC V3.1 §9.3。

数字、日期、百分比需在证据中出现；数值相同只是必要条件，主体是否正确仍由模型判断。
歧义表示按原文比较，不能宽泛归一化后把编造数字误当有据。

支持明确千位分隔、k/thousand、M/million/mn、bn/billion，采用短级差制。
12% 与 12 percent 等价，但 0.12 不自动等价于 12%。
单位仅在同量纲内换算；月份名和 ISO 日期可归一化，纯数字歧义日期保持原样。
允许按 Claim 精度四舍五入，不允许增加精度。计算必须明确提出、操作数有据并由代码复算。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

_MONTHS = {
    name: i
    for i, names in enumerate(
        [
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ],
        start=1,
    )
    for name in names
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))

#: 不把裸 m、b 当作百万、十亿，避免把 5 m（米）误当 500 万。
_SCALE = {
    "k": 3,
    "thousand": 3,
    "mn": 6,
    "million": 6,
    "bn": 9,
    "billion": 9,
    "trillion": 12,
}

#: 量纲与到基本单位的倍数；基本单位为 m、g、s、l、byte、Wh。
_UNITS = {
    "mm": ("length", Decimal("0.001")),
    "cm": ("length", Decimal("0.01")),
    "m": ("length", Decimal("1")),
    "km": ("length", Decimal("1000")),
    "mg": ("mass", Decimal("0.001")),
    "g": ("mass", Decimal("1")),
    "kg": ("mass", Decimal("1000")),
    "t": ("mass", Decimal("1000000")),
    "ms": ("time", Decimal("0.001")),
    "s": ("time", Decimal("1")),
    "sec": ("time", Decimal("1")),
    "min": ("time", Decimal("60")),
    "h": ("time", Decimal("3600")),
    "hr": ("time", Decimal("3600")),
    "second": ("time", Decimal("1")),
    "seconds": ("time", Decimal("1")),
    "secs": ("time", Decimal("1")),
    "minute": ("time", Decimal("60")),
    "minutes": ("time", Decimal("60")),
    "mins": ("time", Decimal("60")),
    "hour": ("time", Decimal("3600")),
    "hours": ("time", Decimal("3600")),
    "hrs": ("time", Decimal("3600")),
    "day": ("time", Decimal("86400")),
    "days": ("time", Decimal("86400")),
    "metre": ("length", Decimal("1")),
    "metres": ("length", Decimal("1")),
    "meter": ("length", Decimal("1")),
    "meters": ("length", Decimal("1")),
    "kilometres": ("length", Decimal("1000")),
    "kilometers": ("length", Decimal("1000")),
    "gram": ("mass", Decimal("1")),
    "grams": ("mass", Decimal("1")),
    "kilograms": ("mass", Decimal("1000")),
    "litre": ("volume", Decimal("1")),
    "litres": ("volume", Decimal("1")),
    "liter": ("volume", Decimal("1")),
    "liters": ("volume", Decimal("1")),
    "ml": ("volume", Decimal("0.001")),
    "l": ("volume", Decimal("1")),
    "kb": ("data", Decimal("1000")),
    "mb": ("data", Decimal("1000000")),
    "gb": ("data", Decimal("1000000000")),
    "wh": ("energy", Decimal("1")),
    "kwh": ("energy", Decimal("1000")),
    "mwh": ("energy", Decimal("1000000")),
    "gwh": ("energy", Decimal("1000000000")),
    "twh": ("energy", Decimal("1000000000000")),
    "kilowatt hours": ("energy", Decimal("1000")),
    "megawatt hours": ("energy", Decimal("1000000")),
    "gigawatt hours": ("energy", Decimal("1000000000")),
    "terawatt hours": ("energy", Decimal("1000000000000")),
    "°c": ("celsius", Decimal("1")),
    "°f": ("fahrenheit", Decimal("1")),
    "usd": ("usd", Decimal("1")),
    "$": ("usd", Decimal("1")),
    "eur": ("eur", Decimal("1")),
    "€": ("eur", Decimal("1")),
    "gbp": ("gbp", Decimal("1")),
    "£": ("gbp", Decimal("1")),
}
_UNIT_RE = "|".join(
    re.escape(u)
    for u in sorted((u for u in _UNITS if u not in {"$", "€", "£"}), key=len, reverse=True)
)


@dataclass(frozen=True)
class Quantity:
    """原文数量的十进制值、类型和精度。

    kind 区分数值、百分比、日期与单位；exponent 表示最后有效位的位置。
    literal 保存日期或可能为年份的四位数文本；step 保存单位换算后的精度步长，
    例如整小时精度为 3600 秒，不能用十的幂表示。"""

    value: Decimal | None
    kind: str
    exponent: int = 0
    literal: str = ""
    step: Decimal | None = None


_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{1,3}(?: \d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_NUMERIC_DATE = re.compile(r"\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b")
_DMY = re.compile(rf"\b(\d{{1,2}})\s+({_MONTH_RE})\.?,?\s+(\d{{4}})\b", re.IGNORECASE)
_MDY = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.IGNORECASE)
_MY = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{4}})\b", re.IGNORECASE)
_VALUE = re.compile(
    rf"(?P<cur>[$€£])?\s?(?P<num>{_NUMBER})"
    rf"(?:\s?(?P<pct>%|percent\b|per cent\b))?"
    rf"(?:\s?(?P<scale>thousand|million|billion|trillion|mn|bn|k)\b)?"
    rf"(?:\s?(?P<unit>{_UNIT_RE})\b)?",
    re.IGNORECASE,
)


def _decimal(text: str) -> Decimal:
    return Decimal(text.replace(",", "").replace(" ", ""))


def _exponent(text: str) -> int:
    """返回最后有效数字的位数；整数末尾零不增加精度，2300000 与 2.3 million 等价。"""
    digits = text.replace(",", "").replace(" ", "")
    if "." in digits:
        return -len(digits.split(".", 1)[1])
    return len(digits) - len(digits.rstrip("0")) if digits.strip("0") else 0


def extract(text: str) -> list[Quantity]:
    """提取文本全部数量，先处理日期，避免把日期数字再次当作普通数字。"""
    found: list[Quantity] = []
    spans: list[tuple[int, int]] = []

    def taken(start: int, end: int) -> bool:
        return any(s < end and start < e for s, e in spans)

    for m in _NUMERIC_DATE.finditer(text):
        spans.append(m.span())
        found.append(Quantity(None, "date", literal=m.group(0)))
    for m in _ISO_DATE.finditer(text):
        if taken(*m.span()):
            continue
        spans.append(m.span())
        found.append(Quantity(None, "date", literal=f"{m.group(1)}-{m.group(2)}-{m.group(3)}"))
    for regex, order in ((_DMY, "dmy"), (_MDY, "mdy")):
        for m in regex.finditer(text):
            if taken(*m.span()):
                continue
            spans.append(m.span())
            if order == "dmy":
                day, month, year = m.group(1), m.group(2), m.group(3)
            else:
                month, day, year = m.group(1), m.group(2), m.group(3)
            iso = f"{int(year):04d}-{_MONTHS[month.lower().rstrip('.')]:02d}-{int(day):02d}"
            found.append(Quantity(None, "date", literal=iso))
    for m in _MY.finditer(text):
        if taken(*m.span()):
            continue
        spans.append(m.span())
        iso = f"{int(m.group(2)):04d}-{_MONTHS[m.group(1).lower().rstrip('.')]:02d}"
        found.append(Quantity(None, "date", literal=iso))

    for m in _VALUE.finditer(text):
        if taken(*m.span("num")):
            continue
        num = m.group("num")
        try:
            value = _decimal(num)
        except InvalidOperation:  # pragma: no cover — 正则已保证这里只包含数字
            continue
        exponent = _exponent(num)
        scale = (m.group("scale") or "").lower()
        if scale:
            power = _SCALE[scale]
            value = value.scaleb(power)
            exponent += power
        if m.group("pct"):
            found.append(Quantity(value, "percent", exponent))
            continue
        unit = (m.group("unit") or "").lower() or (m.group("cur") or "")
        if unit:
            dimension, factor = _UNITS[unit]
            # 精度随单位换算：5 km 的精度仍是千米，而非米。
            # 用 step 保留此粗粒度，尤其小时等非十进制换算不能靠十的幂舍入。
            found.append(
                Quantity(
                    value * factor,
                    f"unit:{dimension}",
                    exponent + factor.adjusted(),
                    step=factor.scaleb(exponent),
                )
            )
            continue
        # 年份范围的四位整数既保留数值，也保留日期文本身份，使完整日期可支持其中的年份断言。
        maybe_year = num if num.isdigit() and len(num) == 4 and 1500 <= int(num) <= 2199 else ""
        found.append(Quantity(value, "number", exponent, literal=maybe_year))
    return found


def _rounds_to(claim: Quantity, snippet: Quantity) -> bool:
    """检查证据值按 Claim 精度四舍五入后是否与其相同。"""
    if claim.value is None or snippet.value is None:
        return False
    if claim.step is not None and snippet.step is not None:
        if claim.step < snippet.step:
            return False  # Claim 比证据更精确。
        rounded = (snippet.value / claim.step).to_integral_value(ROUND_HALF_UP) * claim.step
        return rounded == claim.value
    if claim.exponent < snippet.exponent:
        return False  # Claim 增加了证据没有提供的有效位。
    quantum = Decimal(1).scaleb(claim.exponent)
    try:
        rounded = snippet.value.quantize(quantum, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        rounded = (snippet.value / quantum).to_integral_value(ROUND_HALF_UP) * quantum
    return rounded == claim.value


def matches(claim: Quantity, snippet: Quantity) -> bool:
    """按上述规则判断一个证据数量是否支持一个 Claim 数量。"""
    if claim.kind == "date" or snippet.kind == "date":
        if claim.kind == "date" and snippet.kind == "date":
            return claim.literal == snippet.literal or (
                # 某年某月的断言可由该月任何完整日期支持。
                len(claim.literal) == 7 and snippet.literal.startswith(claim.literal + "-")
            )
        if claim.kind == "number" and claim.literal and snippet.kind == "date":
            return snippet.literal.startswith(claim.literal + "-")
        return False
    if claim.kind != snippet.kind:
        return False
    return _rounds_to(claim, snippet)


def unsupported(claim_text: str, snippets: str) -> list[str]:
    """返回 Claim 中无法被任何证据数量支持的数值。

    空列表表示数字全部有据；不含数字的 Claim 不在这里拒绝，其含义仍需模型判断。"""
    available = extract(snippets)
    missing: list[str] = []
    for quantity in extract(claim_text):
        if not any(matches(quantity, s) for s in available):
            missing.append(quantity.literal or str(quantity.value))
    return missing


_OPS = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b if b else None,
    "percent_change": lambda a, b: (b - a) / a * 100 if a else None,
}


def _as_stated(operand: Quantity, available: list[Quantity]) -> Quantity | None:
    """将计算操作数对应到片段中实际陈述的值，缺失时返回 None。

    仅在片段明确有相同比例且无同值裸数字时，允许用 31 表示 31%；
    不猜测单位或尺度。"""
    if any(matches(operand, s) for s in available):
        return operand
    if operand.kind == "number":
        lifted = Quantity(operand.value, "percent", operand.exponent)
        if any(s.kind == "percent" and s.value == operand.value for s in available):
            return lifted
    return None


def calculation_holds(op: str, operands: list[str], result: str, snippets: str) -> bool:
    """核对明确提出的计算：每个操作数必须在片段中，结果按声明精度重新计算，不采信模型口算。"""
    func = _OPS.get(op)
    if func is None or len(operands) != 2:
        return False
    parsed = [extract(o) for o in operands]
    outcome = extract(result)
    if any(len(p) != 1 for p in parsed) or len(outcome) != 1:
        return False
    available = extract(snippets)
    a, b = (_as_stated(p[0], available) for p in parsed)
    if a is None or b is None:
        return False
    if op in ("add", "subtract") and a.kind != b.kind:
        return False  # 31% 减去 19 人不是可解释的同类数值计算。
    if a.value is None or b.value is None or outcome[0].value is None:
        return False
    computed = func(a.value, b.value)
    if computed is None:
        return False
    exact = Quantity(computed, outcome[0].kind, -28, step=Decimal(1).scaleb(-28))
    return _rounds_to(outcome[0], exact)
