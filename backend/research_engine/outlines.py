"""报告大纲模板，见 internal/07 Phase 4。

模板集中存为数据，界面从 API 获取，只提交模板 ID。
规划器展开后交用户编辑，综合器最终使用批准后的列表，避免界面承诺与实际提示词不一致。
custom 空模板表示用户自己编写；None 表示无约束默认结构，含义不同。"""

from __future__ import annotations

#: 模板 ID 对应有序章节；标题成为 ## 标题，description 仅作生成指导。
TEMPLATES: dict[str, dict] = {
    "literature_review": {
        "label": "Literature Review",
        "summary": "Background, how the field studies it, what is known, what is missing.",
        "sections": [
            {"title": "Background", "description": "The problem and why it is studied."},
            {"title": "Methods", "description": "How the literature investigates it."},
            {"title": "Findings", "description": "What the sources establish, with weight."},
            {"title": "Gaps", "description": "Open questions and thin evidence."},
        ],
    },
    "systematic_comparison": {
        "label": "Systematic Comparison",
        "summary": "Compare named alternatives against shared criteria, then conclude.",
        "sections": [
            {"title": "Criteria", "description": "The dimensions being compared."},
            {"title": "Alternatives", "description": "Each option described on its own terms."},
            {"title": "Comparison", "description": "Alternatives against criteria, side by side."},
            {"title": "Trade-offs", "description": "What each choice costs."},
        ],
    },
    "methods_survey": {
        "label": "Methods Survey",
        "summary": "Survey the techniques in an area and how they are evaluated.",
        "sections": [
            {"title": "Scope", "description": "Which techniques are in and out of scope."},
            {"title": "Techniques", "description": "Each method and its mechanism."},
            {"title": "Evaluation", "description": "How the literature measures them."},
            {"title": "Limitations", "description": "Where the methods break down."},
        ],
    },
    "custom": {
        "label": "Custom",
        "summary": "Start from an empty outline and write your own sections at the gate.",
        "sections": [],
    },
}


def sections_for(template_id: str | None) -> list[dict]:
    """返回模板章节；未知或缺失时返回空列表。

    不因请求拼写错误中断运行，回退无约束大纲，用户在研究花费前仍可看到并修改。"""
    template = TEMPLATES.get(template_id or "")
    return [dict(s) for s in template["sections"]] if template else []


def catalog() -> list[dict]:
    """返回选择器数据：ID、标签、单行摘要和章节列表。"""
    return [
        {
            "id": template_id,
            "label": entry["label"],
            "summary": entry["summary"],
            "sections": [dict(s) for s in entry["sections"]],
        }
        for template_id, entry in TEMPLATES.items()
    ]
