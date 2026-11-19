"""查询是否存在新版本，无法确认时明确披露。

构建身份是本地事实，检查更新是可能失败的网络测量，必须分开。结果分最新、可更新、检查失败、本地版本未知四态，不能把网络故障冒报成已是最新。
由后端发请求，避免为 WebView 访问 GitHub 而放宽前端 CSP；仅依赖标准库和 httpx，不引入桌面缺少的服务端配置。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import httpx

# 固定发布接口，不由用户输入拼接，因此不形成任意 URL 请求入口。
RELEASES_URL = (
    "https://api.github.com/repos/adityamhaske/Multi-Agent-Research-Assistant/releases/latest"
)

# 检查由用户点击触发，应快速返回；连不上时直接说明，不让界面等几十秒。
TIMEOUT_SECONDS = 8.0

UpdateState = Literal["up_to_date", "update_available", "check_failed", "unknown_local_version"]

_VERSION = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)")


class UnknownVersion(ValueError):
    """版本字符串无法比较，不等于比新版本更旧。"""


@dataclass(frozen=True)
class UpdateCheck:
    """检查结果以 state 为主，其余字段用于界面说明。"""

    state: UpdateState
    running_version: str
    # 已查到的最新版本；只有检查本身失败才为空。即使本地版本未知，远程查到的事实仍可报告。
    latest_version: str | None = None
    release_url: str | None = None
    # check_failed 的具体原因，供用户决定如何处理。
    detail: str = ""


def _parts(version: str | None) -> tuple[int, int, int]:
    if not isinstance(version, str):
        raise UnknownVersion(f"not a version string: {version!r}")
    found = _VERSION.match(version.strip())
    if not found:
        raise UnknownVersion(f"not a version string: {version!r}")
    return tuple(int(g) for g in found.groups())  # type: ignore[return-value]


def is_newer(*, running: str | None, latest: str | None) -> bool:
    """按各部分数值比较 latest 是否新于 running，不能按字符串比较 2.0.10 与 2.0.9。

    任一版本无法解析就抛 UnknownVersion，不能瞎给新旧结论。"""
    return _parts(latest) > _parts(running)


async def check(running_version: str, transport: httpx.BaseTransport | None = None) -> UpdateCheck:
    """查询 GitHub 最新发布并与当前构建比较。

    transport 可注入，测试可以覆盖正常和网络失败而不访问真实服务。"""
    try:
        async with httpx.AsyncClient(
            transport=transport,
            timeout=TIMEOUT_SECONDS,
            follow_redirects=True,
            headers={"Accept": "application/vnd.github+json"},
        ) as client:
            response = await client.get(RELEASES_URL)
        if response.status_code != 200:
            return UpdateCheck(
                state="check_failed",
                running_version=running_version,
                detail=f"GitHub answered {response.status_code}.",
            )
        tag = response.json().get("tag_name")
        release_url = response.json().get("html_url")
    except Exception as e:  # noqa: BLE001 — 所有此类故障都表示本次检查未完成
        # DNS、超时、TLS 和非 JSON 响应统一归为未完成检查，绝不能进入已是最新分支。
        return UpdateCheck(
            state="check_failed",
            running_version=running_version,
            detail=f"Could not reach GitHub: {type(e).__name__}.",
        )

    if not isinstance(tag, str) or not _VERSION.match(tag.strip()):
        return UpdateCheck(
            state="check_failed",
            running_version=running_version,
            detail="GitHub's response did not name a release version.",
        )

    latest = tag.strip().lstrip("v")
    try:
        newer = is_newer(running=running_version, latest=latest)
    except UnknownVersion:
        # 本地构建版本未记录时，可以披露已查到的远程发布，但不能声称当前落后或最新。
        return UpdateCheck(
            state="unknown_local_version",
            running_version=running_version,
            latest_version=latest,
            release_url=release_url,
            detail="This build did not record its own version, so it cannot be compared.",
        )

    return UpdateCheck(
        state="update_available" if newer else "up_to_date",
        running_version=running_version,
        latest_version=latest,
        release_url=release_url,
    )
