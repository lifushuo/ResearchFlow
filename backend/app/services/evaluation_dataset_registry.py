"""固定白名单、只读的数据集注册表，不是执行器。

不运行代理、不读用户任意路径、不写数据集，也不把样本数量当模型成绩。来源固定于仓库清单，浏览器不能指定路径。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_DATASETS = 30
MAX_PAGE_SIZE = 100
DATASET_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")
CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
COLLECTIONS = {"queries", "items", "cases", "replay_manifest", "declared_cases", "unlinked"}
EXECUTION_MODES = {
    "optional_paid",
    "offline",
    "controlled_free",
    "controlled_fixture",
    "archived_only",
}
ORIGINS = {"upstream", "custom"}
SPLITS = {"benchmark", "regression", "dev_holdout", "fixture"}


class DatasetRegistryError(ValueError):
    """注册表配置错误或数据集来源不可读。"""


def _backend_root() -> Path:
    # 从服务文件位置向上定位 backend 根目录。
    return Path(__file__).resolve().parents[2]


def _safe_source(backend_root: Path, relative: str) -> Path:
    """只允许读取 backend/evals 内的文件，拒绝父目录穿越和符号链接。"""
    if not isinstance(relative, str) or "\\" in relative or "\x00" in relative:
        raise DatasetRegistryError("invalid dataset source")
    path = Path(relative)
    if (
        not relative.startswith("evals/")
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise DatasetRegistryError("dataset source must be within evals")
    base = backend_root.resolve(strict=True)
    current = base
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise DatasetRegistryError("dataset symlink forbidden")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(base / "evals")
    except (OSError, ValueError) as exc:
        raise DatasetRegistryError("dataset source unavailable") from exc
    if not resolved.is_file() or resolved.stat().st_size > MAX_SOURCE_BYTES:
        raise DatasetRegistryError("dataset source is not a supported file")
    return resolved


def _load_json_bytes(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    if len(raw) > MAX_SOURCE_BYTES:
        raise DatasetRegistryError("dataset JSON too large")
    try:
        doc = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, ValueError) as exc:
        raise DatasetRegistryError("invalid dataset JSON") from exc
    if not isinstance(doc, dict):
        raise DatasetRegistryError("dataset JSON must be an object")
    return doc, hashlib.sha256(raw).hexdigest()


def _manifest(backend_root: Path) -> tuple[list[dict[str, Any]], str]:
    manifest = _safe_source(backend_root, "evals/datasets/dataset_registry_v1.json")
    data, digest = _load_json_bytes(manifest)
    if data.get("schema_version") != 1 or not isinstance(data.get("datasets"), list):
        raise DatasetRegistryError("registry schema invalid")
    rows = data["datasets"]
    if not 1 <= len(rows) <= MAX_DATASETS:
        raise DatasetRegistryError("registry dataset count invalid")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise DatasetRegistryError("invalid dataset row")
        if not DATASET_ID.fullmatch(row["id"]) or row["id"] in seen:
            raise DatasetRegistryError("invalid or duplicate dataset ID")
        seen.add(row["id"])
        for field in ("title", "category", "score_family", "description"):
            if not isinstance(row.get(field), str) or not 1 <= len(row[field]) <= 500:
                raise DatasetRegistryError("invalid dataset metadata")
        if (
            row.get("origin") not in ORIGINS
            or row.get("split") not in SPLITS
            or row.get("execution") not in EXECUTION_MODES
            or row.get("collection") not in COLLECTIONS
        ):
            raise DatasetRegistryError("invalid dataset classification")
        if type(row.get("declared_count")) is not int or row["declared_count"] < 0:
            raise DatasetRegistryError("invalid declared case count")
        if not isinstance(row.get("case_ids"), list) or len(row["case_ids"]) > 1000:
            raise DatasetRegistryError("invalid declared case list")
        if any(not isinstance(cid, str) or not CASE_ID.fullmatch(cid) for cid in row["case_ids"]):
            raise DatasetRegistryError("invalid declared case identifier")
        if len(set(row["case_ids"])) != len(row["case_ids"]):
            raise DatasetRegistryError("duplicate declared case identifier")
        if not isinstance(row.get("supporting_files"), list) or len(row["supporting_files"]) > 10:
            raise DatasetRegistryError("invalid supporting file list")
        if row["collection"] == "unlinked":
            if row.get("source") is not None:
                raise DatasetRegistryError("unlinked dataset cannot claim a source")
        elif not isinstance(row.get("source"), str):
            raise DatasetRegistryError("dataset source required")
        if row["collection"] == "declared_cases" and len(row["case_ids"]) != row["declared_count"]:
            raise DatasetRegistryError("declared cases and count do not match")
        for filename in [row.get("source"), *row["supporting_files"]]:
            if filename is not None and (
                not isinstance(filename, str)
                or not filename.startswith("evals/")
                or "\\" in filename
                or "\x00" in filename
                or Path(filename).is_absolute()
                or any(p in ("", ".", "..") for p in Path(filename).parts)
            ):
                raise DatasetRegistryError("invalid registry source path")
    return rows, digest


def _preview(item: dict[str, Any], index: int) -> dict[str, Any]:
    case_id = item.get("case_id", item.get("id"))
    if not isinstance(case_id, str) or not CASE_ID.fullmatch(case_id):
        raise DatasetRegistryError("case ID missing or invalid")
    result = {"case_id": case_id, "index": index + 1}
    for field in ("domain", "kind", "depth", "match_mode"):
        value = item.get(field)
        if isinstance(value, str):
            result[field] = value[:120]
    # 只有问题类数据集可展示问题文本；工具参数、证据正文、提示词和私有参考片段不经此 API 返回。
    if "query" in item and isinstance(item["query"], str):
        result["question_preview"] = item["query"][:160]
    return result


def _source_cases(backend_root: Path, row: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    source = _safe_source(backend_root, row["source"])
    if row["collection"] == "declared_cases":
        # 使用清单明确声明的 ID，不从 Python 文件猜测发现的测试项。
        return (
            [{"case_id": x, "index": i + 1} for i, x in enumerate(row["case_ids"])],
            hashlib.sha256(source.read_bytes()).hexdigest(),
        )
    data, sha = _load_json_bytes(source)
    if row["collection"] == "replay_manifest":
        items = data.get("cases")
        if data.get("schema_version") != 1 or not isinstance(items, list):
            raise DatasetRegistryError("invalid replay manifest")
        cases = []
        for i, name in enumerate(items):
            # 只接受当前目录中指定的 JSON 文件名，不接受远程包含或目录穿越。
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+\.json", name):
                raise DatasetRegistryError("invalid replay case filename")
            child = (Path(row["source"]).parent / name).as_posix()
            case_doc, _ = _load_json_bytes(_safe_source(backend_root, child))
            obj = _preview(case_doc, i)
            fault = case_doc.get("fault")
            if isinstance(fault, dict) and isinstance(fault.get("kind"), str):
                obj["kind"] = fault["kind"][:100]
            cases.append(obj)
    else:
        items = data.get(row["collection"])
        if not isinstance(items, list):
            raise DatasetRegistryError("dataset case array unavailable")
        cases = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise DatasetRegistryError("dataset case must be object")
            cases.append(_preview(item, i))
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise DatasetRegistryError("duplicate case IDs in dataset source")
    if len(cases) > 5000:
        raise DatasetRegistryError("dataset has too many cases")
    return cases, sha


def _describe(
    backend_root: Path, row: dict[str, Any], *, with_cases: bool, offset: int = 0, limit: int = 50
) -> dict[str, Any]:
    metadata = {
        key: row[key]
        for key in (
            "id",
            "title",
            "origin",
            "category",
            "split",
            "collection",
            "source",
            "declared_count",
            "execution",
            "score_family",
            "description",
        )
    }
    metadata.update(
        {
            "source_sha256": None,
            "source_status": "unlinked" if row["collection"] == "unlinked" else "unavailable",
            "verified_count": None,
            "count_basis": "reported_only" if row["collection"] == "unlinked" else "unmeasured",
            "supporting_file_count": len(row["supporting_files"]),
            "supporting_files_valid": None,
            "warnings": [],
        }
    )
    cases: list[dict[str, Any]] = []
    if row["collection"] == "unlinked":
        metadata["warnings"].append(
            "Original calibration samples are not linked; case count not verified from source bytes"
        )
    else:
        try:
            cases, digest = _source_cases(backend_root, row)
            metadata["source_sha256"] = digest
            metadata["verified_count"] = (
                len(cases) if row["collection"] != "declared_cases" else None
            )
            metadata["count_basis"] = (
                "registry_declared" if row["collection"] == "declared_cases" else "source_verified"
            )
            metadata["source_status"] = "available"
            if len(cases) != row["declared_count"]:
                metadata["source_status"] = "count_mismatch"
                metadata["warnings"].append("Source case count differs from registry declaration")
            supports_valid = True
            for name in row["supporting_files"]:
                try:
                    _safe_source(backend_root, name)
                except DatasetRegistryError:
                    supports_valid = False
            metadata["supporting_files_valid"] = supports_valid
            if not supports_valid:
                metadata["source_status"] = "incomplete"
                metadata["warnings"].append("One or more supporting files unavailable")
        except (DatasetRegistryError, OSError) as _exc:
            # 异常变量故意不用，但保留 as 和括号形式。
            # 否则 Python 3.14 格式化器可能改成旧版 3.13 CLI 无法解析的 except A, B 语法。
            metadata["warnings"].append("Dataset source unavailable or invalid")
            metadata["source_status"] = "invalid_or_missing"
    if with_cases:
        metadata["cases"] = cases[offset : offset + limit]
        metadata["offset"] = offset
        metadata["limit"] = limit
        metadata["has_more"] = len(cases) > offset + limit
    return metadata


def list_datasets(backend_root: Path | None = None) -> dict[str, Any]:
    root = backend_root or _backend_root()
    datasets, digest = _manifest(root)
    return {
        "schema_version": 1,
        "registry_sha256": digest,
        "read_only": True,
        "datasets": [_describe(root, row, with_cases=False) for row in datasets],
    }


def get_dataset(
    dataset_id: str, *, offset: int = 0, limit: int = 50, backend_root: Path | None = None
) -> dict[str, Any]:
    if not isinstance(dataset_id, str) or not DATASET_ID.fullmatch(dataset_id):
        raise DatasetRegistryError("invalid dataset ID")
    if (
        type(offset) is not int
        or type(limit) is not int
        or offset < 0
        or not 1 <= limit <= MAX_PAGE_SIZE
    ):
        raise DatasetRegistryError("invalid dataset pagination")
    root = backend_root or _backend_root()
    rows, digest = _manifest(root)
    for row in rows:
        if row["id"] == dataset_id:
            return {
                "schema_version": 1,
                "registry_sha256": digest,
                "read_only": True,
                "dataset": _describe(root, row, with_cases=True, offset=offset, limit=limit),
            }
    raise KeyError("dataset not found")
