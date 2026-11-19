"""披露引用检查器在指定路由上的已测准确性，见 RFC V3.1 §17、D-24。

低于门槛只标注，不拒绝；没有测过则明确显示未测量。
MEASURED 来自最新有效 G-6 成绩，由测试与 evals/results 核对；
引擎运行时不能读取评测目录，因为桌面包不包含 evals。"""

from __future__ import annotations

MEETS = "meets_threshold"
BELOW = "below_threshold"
NOT_MEASURED = "not_measured"

#: 路由对应的状态与成绩来源；未测量 Claim 超过 5% 的运行无效，不作为测量依据。
MEASURED: dict[str, tuple[str, str]] = {
    "custom:auto/best-fast": (MEETS, "eval-2026-11-02-custom-auto-best-fast-grounding-run1.json"),
    "custom:agy/gemini-3-flash": (
        BELOW,
        "eval-2026-11-02-custom-agy-gemini-3-flash-grounding-run1.json",
    ),
}

_WORDS = {
    MEETS: "meets the measured threshold",
    BELOW: "below the measured threshold",
    NOT_MEASURED: "not measured",
}


def for_route(route: str | None) -> dict:
    """返回指定检查路由的已记录结论和展示文字。

    custom 路由名称由网关维护者定义，同名不保证同一模型；标签说明测量来源，不替其他网关背书。"""
    status, result = MEASURED.get(route or "", (NOT_MEASURED, None))
    label = f"checker accuracy on this model: {_WORDS[status]}"
    if status != NOT_MEASURED and (route or "").startswith("custom:"):
        label += " (measured on the maintainers' gateway; a custom endpoint chooses its model)"
    return {"routing": route, "status": status, "result": result, "label": label}
