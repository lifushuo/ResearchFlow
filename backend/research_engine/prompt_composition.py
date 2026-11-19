"""统一决定每次调用实际使用的系统提示词，见范围冻结 RFC §17。

五个角色对应多个用途，同为 critic 的研究评审可替换，引用核验等用途却必须保护。
因此可覆盖性按用途判断，不能只按角色判断。此模块可由引擎、服务端、桌面和评测共同导入。

默认或受保护用途返回原常量对象，不复制、不重新包装，保证脚本模型匹配保持一致。
仅用户自定义正文需要前后补不可信内容说明；默认提示词本来已有说明，不能重复添加。"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import NamedTuple

from research_engine import prompts
from research_engine.runconfig import get_run_config

#: 按 <角色>.<用途> 索引默认提示词，角色直接由键推导，避免两处声明漂移。
#:
#: 多个用途共用五个角色，不能只用角色索引来决定替换权限。
PURPOSE_CONSTANTS: Mapping[str, str] = {
    "planner.main": prompts.PLANNER_PROMPT_V2,
    "executor.main": prompts.EXECUTOR_PROMPT,
    "critic.research": prompts.CRITIC_PROMPT_V2,
    "critic.citation_verify": prompts.CITATION_VERIFY_PROMPT,
    "critic.contradiction_detector": prompts.CONTRADICTION_DETECTOR_PROMPT,
    "synthesizer.main": prompts.SYNTHESIZER_PROMPT_V2,
    "synthesizer.repair": prompts.SYNTHESIZER_REPAIR_PROMPT,
    "critic.claim_decompose": prompts.CLAIM_DECOMPOSE_PROMPT,
    "critic.claim_verify": prompts.CLAIM_VERIFY_PROMPT,
    "critic.citation_relevance": prompts.CITATION_RELEVANCE_PROMPT,
    "critic.inference_verify": prompts.INFERENCE_VERIFY_PROMPT,
    "critic.selection_addendum": prompts.SELECTION_CRITIC_ADDENDUM,
    "synthesizer.selection_repair": prompts.SELECTION_REPAIR_PROMPT,
    "chat.general": prompts.CHAT_PROMPT,
    "chat.project": prompts.PROJECT_CHAT_PROMPT,
}

#: 从用途键推导模型角色，新角色出现时不会被第二份手写列表漏掉。
ROLES: tuple[str, ...] = tuple(dict.fromkeys(p.split(".", 1)[0] for p in PURPOSE_CONSTANTS))

#: 可替换用途的唯一显式允许清单，只有精确匹配才能覆盖。
#:
#: 默认保护：PROTECTED_PURPOSES 取允许清单的补集，新用途忘记登记时仍受保护。
OVERRIDABLE_PURPOSES: frozenset[str] = frozenset(
    {
        "planner.main",
        "executor.main",
        "critic.research",
        "synthesizer.main",
        "chat.general",
    }
)

#: 其余用途全部保护，由补集计算，不再手工枚举。
#:
#: 下列用途不允许用户改写的原因：
#:
#: 引用核验决定证据支持情况，允许改写等于自己评分；矛盾检测可被诱导忽略冲突；
#: 引用修复可能被改成伪造引用；项目聊天的拒绝规则缺失会让无据回答混入。
PROTECTED_PURPOSES: frozenset[str] = frozenset(PURPOSE_CONSTANTS) - OVERRIDABLE_PURPOSES

#: 默认提示词含不可信内容说明的用途，替换正文时需重新补上保护框定。
#:
#: 从提示词实际内容推导，不手写清单，避免新增保护说明后在自定义替换时漏掉。
NOTE_CARRYING_PURPOSES: frozenset[str] = frozenset(
    p for p, constant in PURPOSE_CONSTANTS.items() if prompts.UNTRUSTED_CONTENT_NOTE in constant
)

#: 区分研究运行会执行和不会执行的用途。
#:
#: 这里只决定产物应记录哪些提示词来源，不决定覆盖权限。
#: 未执行的聊天用途不能出现在研究运行来源中，正如不能记录未调用的模型。
#:
#: 必须显式分类，不能对新用途默认包含或默认排除，否则都会静默错报来源；未分类让测试失败。
RUN_PURPOSES: frozenset[str] = frozenset(
    {
        "planner.main",
        "executor.main",
        "critic.research",
        "critic.citation_verify",
        "critic.contradiction_detector",
        "synthesizer.main",
        "synthesizer.repair",
        # 仅指定文件研究使用，见 RFC V3.1 §9.7。
        "critic.claim_decompose",
        "critic.claim_verify",
        "critic.citation_relevance",
        "critic.inference_verify",
        "critic.selection_addendum",
        "synthesizer.selection_repair",
    }
)

#: 仅指定文件研究会执行的逐 Claim 核验用途；其他运行及其证据包不记录这些用途。
SELECTION_ONLY_PURPOSES: frozenset[str] = frozenset(
    {
        "critic.claim_decompose",
        "critic.claim_verify",
        "critic.citation_relevance",
        "critic.inference_verify",
        "critic.selection_addendum",
        "synthesizer.selection_repair",
    }
)

#: 显式列出互补用途以供双向核对。聊天针对已完成报告，不属于研究执行。
NON_RUN_PURPOSES: frozenset[str] = frozenset({"chat.general", "chat.project"})


def role_of(purpose: str) -> str:
    """返回用途所属的模型角色，与 get_llm 使用同一角色字符串。"""
    return _checked(purpose).split(".", 1)[0]


def is_overridable(purpose: str) -> bool:
    """判断此用途是否允许用户替换提示词。

    只能按用途决定；同一 critic 角色的不同用途权限不同。
    未知用途抛错，不能静默返回 False 掩盖拼写错误。"""
    return _checked(purpose) in OVERRIDABLE_PURPOSES


class EditableDefault(NamedTuple):
    """提供某个可编辑角色的默认提示词内容。"""

    role: str
    text: str
    #: 编辑器可据此说明系统会自动添加不可信内容保护，而不让用户编辑保护说明本身。
    framed: bool


def editable_defaults() -> tuple[EditableDefault, ...]:
    """按 ROLES 顺序返回各角色编辑器的初始文本，见范围冻结 §12。

    仅展示允许替换用途，运行时权限仍由 system_prompt 独立强制检查。
    移除默认正文中系统管理的不可信内容说明，保存后由系统包裹，避免重复三次。
    此函数不读取运行配置或用户状态。"""
    by_role: dict[str, str] = {}
    for purpose in sorted(OVERRIDABLE_PURPOSES):
        role = role_of(purpose)
        if role in by_role:
            raise RuntimeError(
                f"role {role!r} has two overridable purposes ({by_role[role]}, {purpose}); "
                "a role-keyed editor cannot say which one it replaces"
            )
        by_role[role] = purpose

    note = prompts.UNTRUSTED_CONTENT_NOTE
    defaults = []
    for role in ROLES:
        purpose = by_role.get(role)
        if purpose is None:
            raise RuntimeError(f"role {role!r} has no overridable purpose to edit")
        framed = purpose in NOTE_CARRYING_PURPOSES
        text = PURPOSE_CONSTANTS[purpose]
        if framed:
            text = text.replace(f"{note}\n", "").replace(note, "")
        defaults.append(EditableDefault(role=role, text=text.strip(), framed=framed))
    return tuple(defaults)


def system_prompt(purpose: str) -> str:
    """返回指定用途的实际系统提示词，并在允许时应用覆盖。

    唯一运行时保护点：受保护用途在读取覆盖配置之前就返回默认值。
    即使配置手工构造或绕过 API 校验，此顺序也不能放宽。"""
    p = _checked(purpose)
    shipped = PURPOSE_CONSTANTS[p]

    if p not in OVERRIDABLE_PURPOSES:
        # 受保护用途完全不读取覆盖配置。_observed 在决定之后记录，且无法访问配置，不会削弱检查。
        return _observed(p, shipped, overridden=False)

    # 防御绕过宿主构造的坏配置：只有整体可用文本才应用，否则回退全部默认提示词。
    # 节点内不因坏覆盖中断半份报告；宿主在运行记录上披露 UNUSABLE。
    stored = get_run_config().prompt_overrides
    body = stored.get(role_of(p)) if isinstance(stored, Mapping) else None
    if not isinstance(body, str) or not body.strip():
        return _observed(p, shipped, overridden=False)

    if p in NOTE_CARRYING_PURPOSES:
        note = prompts.UNTRUSTED_CONTENT_NOTE
        body = f"{note}\n\n{body}\n\n{note}"
    return _observed(p, body, overridden=True)


class PromptProvenanceConflict(RuntimeError):
    """同一运行的同一用途产生了两份不同提示词。

    冻结配置下不应发生，必须抛错以暴露冻结失效，不能任选一份伪装成唯一来源。"""


#: 有观测器时，system_prompt 在这里报告刚刚实际组装的提示词。
#:
#: 只观察，不另行组装。运行结束后按今日配置重建会得到今日默认值，不能冒充历史实际指令。
#:
#: ContextVar 持有可变字典，子任务继承同一字典引用，确保各节点写入同一运行记录。
_provenance: ContextVar[dict[str, dict] | None] = ContextVar("_prompt_provenance", default=None)


def _observed(purpose: str, text: str, *, overridden: bool) -> str:
    """记录实际组装结果，再原样返回同一个字符串对象。

    脚本场景依赖提示词对象身份；overridden 记录实际走过的分支，而不是比较内容是否不同。
    同用途重复调用必须完全一致，否则抛错，避免只保留第一份而隐藏模型实际见过两套指令。"""
    records = _provenance.get()
    if records is None:
        return text

    record = {
        "purpose": purpose,
        "role": role_of(purpose),
        "policy": "OVERRIDABLE" if purpose in OVERRIDABLE_PURPOSES else "PROTECTED",
        "overridden": overridden,
        "effective_prompt": text,
    }
    existing = records.get(purpose)
    if existing is not None and existing != record:
        raise PromptProvenanceConflict(
            f"{purpose} composed twice within one run and the results differ — the run's "
            "prompt overrides are supposed to be frozen for its whole life, so this means "
            "that invariant is broken and no provenance for this run can be trusted"
        )
    records[purpose] = record
    return text


@contextmanager
def recording_provenance(into: dict[str, dict] | None = None) -> Iterator[dict[str, dict]]:
    """按用途收集此作用域内实际组装的提示词。

    作用域隔离不同运行；into 允许审批恢复后继续添加此前记录。
    只记录实际执行用途，未发生引用修复就不补造修复提示词来源。"""
    records: dict[str, dict] = {} if into is None else into
    token = _provenance.set(records)
    try:
        yield records
    finally:
        _provenance.reset(token)


def _checked(purpose: str) -> str:
    """未知用途立即报错，避免拼写错误让整个图按错误指令继续运行。"""
    if purpose not in PURPOSE_CONSTANTS:
        raise KeyError(
            f"unknown prompt purpose {purpose!r}; known purposes: "
            f"{', '.join(sorted(PURPOSE_CONSTANTS))}"
        )
    return purpose
