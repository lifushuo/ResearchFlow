"""记录构建版本及其对应提交。

构建时由 scripts/stamp_build.py 生成被 Git 忽略的 _build.py。
没有构建戳就返回 unknown，不能拿当前工作区 HEAD 猜测产物的来源。
源码运行时若版本、HEAD 或未提交状态与构建戳矛盾，也返回 unknown。
工作区信息只用于否决旧戳，不用于猜测版本；打包程序或不可读的仓库不做此检查。
本模块放在引擎层，保证桌面包无需服务端配置也能读取版本。"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import structlog

logger = structlog.get_logger()

UNKNOWN = "unknown"
#: 构建戳声明的源码身份；工作区只能核验这些事实。
IDENTITY_FIELDS = ("VERSION", "GIT_SHA", "DIRTY")
#: 从源码目录运行时对应的仓库根目录。
_CHECKOUT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BuildInfo:
    """构建可以披露的五项身份信息。"""

    version: str
    git_sha: str
    #: 构建时是否存在未提交修改。仅有提交 SHA 不能证明本地产物与该提交完全相同。
    dirty: bool
    #: 标准 API 契约的哈希；哈希相同表示接口契约相同。
    contract_version: str
    built_at: str

    def as_dict(self) -> dict:
        return asdict(self)


_UNSTAMPED = BuildInfo(
    version=UNKNOWN,
    git_sha=UNKNOWN,
    # 没有检查过工作区就不能填 False，否则会把未知误报成干净。
    dirty=True,
    contract_version=UNKNOWN,
    built_at=UNKNOWN,
)


def build_info() -> BuildInfo:
    """返回构建身份；没有可靠构建戳时返回 unknown。"""
    try:
        from research_engine import _build  # type: ignore[attr-defined]
    except ImportError:
        return _UNSTAMPED
    claimed = tuple((key, getattr(_build, key, None)) for key in IDENTITY_FIELDS)
    if not getattr(sys, "frozen", False) and _refuted_by_checkout(claimed):
        return _UNSTAMPED
    return BuildInfo(
        version=getattr(_build, "VERSION", UNKNOWN),
        git_sha=getattr(_build, "GIT_SHA", UNKNOWN),
        dirty=bool(getattr(_build, "DIRTY", True)),
        contract_version=getattr(_build, "CONTRACT_VERSION", UNKNOWN),
        built_at=getattr(_build, "BUILT_AT", UNKNOWN),
    )


def checkout_identity(root: Path) -> dict | None:
    """读取 root 的 VERSION、HEAD 和未提交状态；不是可读仓库根目录时返回 None。

    未跟踪文件也计入 DIRTY，因为新增文件同样可能进入构建产物。"""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != root.resolve():
            return None
        return {
            "VERSION": (root / "VERSION").read_text(encoding="utf-8").strip(),
            "GIT_SHA": git("rev-parse", "HEAD"),
            "DIRTY": bool(git("status", "--porcelain")),
        }
    except OSError, subprocess.CalledProcessError:
        return None


def stamp_contradiction(
    claimed: Mapping[str, object], checkout: Mapping[str, object]
) -> str | None:
    """返回构建戳与工作区不符的原因；一致时返回 None。"""
    for key in IDENTITY_FIELDS:
        if claimed.get(key) != checkout[key]:
            return f"stamp says {key}={claimed.get(key)!r}, checkout says {checkout[key]!r}"
    return None


@cache
def _refuted_by_checkout(claimed: tuple[tuple[str, object], ...]) -> bool:
    """按进程和构建戳缓存，避免每次请求 /version 都启动 Git 子进程。"""
    checkout = checkout_identity(_CHECKOUT)
    if checkout is None:
        return False
    reason = stamp_contradiction(dict(claimed), checkout)
    if reason:
        logger.warning("build_stamp_refuted_by_checkout", reason=reason, reporting=UNKNOWN)
    return reason is not None
