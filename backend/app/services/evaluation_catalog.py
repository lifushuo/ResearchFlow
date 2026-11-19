"""评测证据文件访问：限定根目录，拒绝符号链接。"""

from pathlib import Path


class CatalogError(ValueError):
    """评测索引或产物无效、不安全；不能把服务器私有路径返回客户端。"""


def safe_file(root: Path, relative: str) -> Path:
    """只解析根目录内的普通文件，拒绝路径穿越与符号链接。"""
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise CatalogError("invalid relative artifact path")
    raw = Path(relative)
    if raw.is_absolute() or any(part in (".", "..", "") for part in raw.parts):
        raise CatalogError("invalid relative artifact path")
    base = root.resolve(strict=True)
    child = base
    for part in raw.parts:
        child = child / part
        if child.is_symlink():
            raise CatalogError("artifact symlink is forbidden")
    try:
        resolved = child.resolve(strict=True)
        resolved.relative_to(base)
    except (OSError, ValueError) as exc:
        raise CatalogError("artifact not available inside configured root") from exc
    if not resolved.is_file():
        raise CatalogError("artifact must be a regular file")
    return resolved
