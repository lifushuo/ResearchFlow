"""Read-only dataset inventory and safe native integration tests; no API calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.evaluation_dataset_registry import (
    DatasetRegistryError,
    _safe_source,
    get_dataset,
    list_datasets,
)

PACKAGE = Path(__file__).resolve().parents[2]
REGISTRY = PACKAGE / "evals/datasets/dataset_registry_v1.json"


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def backend(tmp_path: Path) -> Path:
    raw = json.loads(REGISTRY.read_text(encoding="utf-8"))
    write(tmp_path / "evals/datasets/dataset_registry_v1.json", raw)
    for entry in raw["datasets"]:
        source = entry["source"]
        if source is None:
            continue
        collection = entry["collection"]
        if collection == "declared_cases":
            path = tmp_path / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# real test source simulated by fixture\n", encoding="utf-8")
        elif collection == "replay_manifest":
            names = [f"replay_{i}.json" for i in range(4)]
            write(tmp_path / source, {"schema_version": 1, "cases": names})
            for i, name in enumerate(names):
                write(
                    tmp_path / Path(source).parent / name,
                    {"case_id": f"replay-{i}", "fault": {"kind": "none"}},
                )
        else:
            count = entry["declared_count"]
            entries = [{"id": f"case-{i + 1}", "query": "private question"} for i in range(count)]
            if collection == "cases":
                entries = [
                    {
                        "case_id": f"case-{i + 1}",
                        "expected_calls": [{"name": "web_search", "args": {"password": "HIDDEN"}}],
                    }
                    for i in range(count)
                ]
            write(tmp_path / source, {collection: entries})
        for supporting in entry["supporting_files"]:
            write(tmp_path / supporting, {"case": "support"})
    return tmp_path


def test_list_finds_all_nine(backend: Path) -> None:
    result = list_datasets(backend)
    assert result["read_only"] is True
    assert len(result["datasets"]) == 9
    assert [d["id"] for d in result["datasets"]].count("report-benchmark") == 1


def test_official_counts_verified_from_bytes(backend: Path) -> None:
    rows = {d["id"]: d for d in list_datasets(backend)["datasets"]}
    for name, count in (
        ("report-benchmark", 10),
        ("corpus-retrieval", 12),
        ("selection-retrieval", 16),
        ("grounding-v1", 425),
        ("tool-contract-v1", 4),
        ("planner-replay-v1", 4),
        ("planner-evolution-reviewed", 30),
    ):
        assert rows[name]["verified_count"] == count
        assert rows[name]["count_basis"] == "source_verified"
        assert rows[name]["source_status"] == "available"
        assert len(rows[name]["source_sha256"]) == 64


def test_grounding_pagination(backend: Path) -> None:
    result = get_dataset("grounding-v1", offset=400, limit=50, backend_root=backend)["dataset"]
    assert result["verified_count"] == 425
    assert len(result["cases"]) == 25
    assert result["cases"][0]["case_id"] == "case-401"
    assert result["has_more"] is False


def test_replay_cases_are_loaded_from_files(backend: Path) -> None:
    row = get_dataset("planner-replay-v1", backend_root=backend)["dataset"]
    assert [c["case_id"] for c in row["cases"]] == [f"replay-{i}" for i in range(4)]
    assert row["cases"][0]["kind"] == "none"


def test_unlinked_judge_count_not_claimed_as_verified(backend: Path) -> None:
    # Pin the unlinked scenario explicitly; the shipped registry now links the original 24 cases.
    path = backend / "evals/datasets/dataset_registry_v1.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    entry = next(row for row in registry["datasets"] if row["id"] == "judge-calibration")
    entry.update(collection="unlinked", source=None, supporting_files=[], case_ids=[])
    write(path, registry)
    row = get_dataset("judge-calibration", backend_root=backend)["dataset"]
    assert row["declared_count"] == 24
    assert row["verified_count"] is None
    assert row["count_basis"] == "reported_only"
    assert row["cases"] == []
    assert row["source_status"] == "unlinked"


def test_declared_executor_count_not_claimed_as_source_verified(backend: Path) -> None:
    # A declared list remains unverified even though the current product has a real manifest.
    path = backend / "evals/datasets/dataset_registry_v1.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    entry = next(row for row in registry["datasets"] if row["id"] == "executor-regression-v1")
    entry.update(
        collection="declared_cases",
        source="evals/agent_eval_ext/declared_executor.py",
        supporting_files=[],
        case_ids=[f"executor-{i}" for i in range(7)],
    )
    (backend / "evals/agent_eval_ext").mkdir(parents=True, exist_ok=True)
    (backend / entry["source"]).write_text("# Test-only declared source\n", encoding="utf-8")
    write(path, registry)
    row = get_dataset("executor-regression-v1", backend_root=backend)["dataset"]
    assert row["verified_count"] is None
    assert row["count_basis"] == "registry_declared"
    assert len(row["cases"]) == 7
    assert row["source_status"] == "available"


def test_does_not_expose_tool_arguments(backend: Path) -> None:
    row = get_dataset("tool-contract-v1", backend_root=backend)["dataset"]
    assert len(row["cases"]) == 4
    assert "HIDDEN" not in json.dumps(row)
    assert "expected_calls" not in json.dumps(row)


def test_listing_does_not_modify_files(backend: Path) -> None:
    path = backend / "evals/queries.json"
    original = path.read_bytes()
    _ = list_datasets(backend)
    assert path.read_bytes() == original


def test_missing_source_is_disclosed(backend: Path) -> None:
    (backend / "evals/queries.json").unlink()
    row = get_dataset("report-benchmark", backend_root=backend)["dataset"]
    assert row["source_status"] == "invalid_or_missing"
    assert row["verified_count"] is None


def test_count_mismatch_is_disclosed(backend: Path) -> None:
    write(backend / "evals/queries.json", {"queries": [{"id": "only-one"}]})
    row = get_dataset("report-benchmark", backend_root=backend)["dataset"]
    assert row["source_status"] == "count_mismatch"
    assert row["verified_count"] == 1


def test_missing_support_file_is_disclosed(backend: Path) -> None:
    (backend / "evals/datasets/grounding/v1/labels.json").unlink()
    row = get_dataset("grounding-v1", backend_root=backend)["dataset"]
    assert row["source_status"] == "incomplete"
    assert row["supporting_files_valid"] is False


@pytest.mark.parametrize(
    "name",
    [
        "../secret.json",
        "/tmp/a.json",
        "evals/../secret.json",
        "evals\\bad.json",
        "not-evals/doc.json",
        "evals/../../tmp.json",
        "evals/\x00broken.json",
    ],
)
def test_path_traversal_refused(backend: Path, name: str) -> None:
    with pytest.raises(DatasetRegistryError):
        _safe_source(backend, name)


def test_symlink_refused(backend: Path) -> None:
    path = backend / "evals/sneaky.json"
    path.symlink_to(backend / "evals/queries.json")
    with pytest.raises(DatasetRegistryError):
        _safe_source(backend, "evals/sneaky.json")


def test_duplicate_dataset_id_refused(backend: Path) -> None:
    path = backend / "evals/datasets/dataset_registry_v1.json"
    raw = json.loads(path.read_text())
    raw["datasets"].append(raw["datasets"][0])
    write(path, raw)
    with pytest.raises(DatasetRegistryError):
        list_datasets(backend)


def test_duplicate_case_id_is_reported_not_exposed(backend: Path) -> None:
    write(backend / "evals/queries.json", {"queries": [{"id": "same"}] * 10})
    row = get_dataset("report-benchmark", backend_root=backend)["dataset"]
    assert row["source_status"] == "invalid_or_missing"
    assert row["cases"] == []


def test_unknown_dataset_rejected(backend: Path) -> None:
    with pytest.raises(KeyError):
        get_dataset("not-here", backend_root=backend)


@pytest.mark.parametrize("offset,limit", [(-1, 20), (0, 0), (0, 101), (True, 10), (0, False)])
def test_invalid_pagination(backend: Path, offset: int, limit: int) -> None:
    with pytest.raises(DatasetRegistryError):
        get_dataset("report-benchmark", offset=offset, limit=limit, backend_root=backend)
