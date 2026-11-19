"""CLI: collect past results or run the EXISTING v0.6 Replay suite then normalize.

No paid API is invoked. Real report/retrieval/grounding artifacts are read-only;
only v0.6 controlled fake Replay is executed in `run` mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .artifacts import check_output_path
from .unified_eval import (
    adapt_grounding,
    adapt_report,
    adapt_retrieval,
    build_unified,
    compare_unified,
    freeze_dataset,
    load_dataset,
    load_json,
    save_new,
)


def _source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--report", help="Existing report-mode result JSON; read-only")
    parser.add_argument("--retrieval", help="Existing retrieval result JSON; read-only")
    parser.add_argument("--grounding", help="Existing grounding benchmark JSON; read-only")


def _preflight_sources(args: argparse.Namespace) -> dict:
    # Validate all inputs BEFORE any replay process is started or artifacts created.
    expected = {}
    for name, adapter in (
        ("report", adapt_report),
        ("retrieval", adapt_retrieval),
        ("grounding", adapt_grounding),
    ):
        source = getattr(args, name)
        if source:
            expected[name] = adapter(source)["source"]["sha256"]
    return expected


def _freeze_sources(args, output, expected):
    frozen = {name: getattr(args, name) for name in ("report", "retrieval", "grounding")}
    for name, digest in expected.items():
        source = Path(frozen[name])
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("source changed after preflight")
        target = output / "sources" / name / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
        frozen[name] = target
    return frozen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ResearchFlow Unified Evaluation v0.7")
    sub = parser.add_subparsers(dest="command", required=True)
    valid = sub.add_parser("validate", help="Validate the versioned replay dataset and its bytes")
    valid.add_argument("--dataset", required=True)
    run = sub.add_parser("run", help="Run fake Replay then attach optional historical metrics")
    run.add_argument("--dataset", required=True)
    run.add_argument("--out-dir", required=True, help="New output directory; must not exist")
    run.add_argument("--timeout", type=int, default=180, help="Per-case process timeout seconds")
    run.add_argument("--revision", default="unspecified", help="Unverified human-supplied label")
    _source_arguments(run)
    collect = sub.add_parser("collect", help="Normalize only existing artifacts; never run Agent")
    collect.add_argument("--replay-suite")
    collect.add_argument("--dataset", help="Optional: verify replay manifest and case hashes")
    collect.add_argument("--out", required=True, help="New unified JSON file; cannot overwrite")
    _source_arguments(collect)
    compare = sub.add_parser(
        "compare", help="Strict replay contract comparison; no false paired A/B"
    )
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--out")
    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            dataset = load_dataset(args.dataset)
            print(
                json.dumps(
                    {
                        "dataset_id": dataset["dataset_id"],
                        "dataset_version": dataset["dataset_version"],
                        "case_count": len(dataset["cases"]),
                        "cases": dataset["cases"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "compare":
            result = compare_unified(load_json(args.baseline), load_json(args.candidate))
            if args.out:
                save_new(args.out, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["comparable"] and result["quality_gate_passed"] else 1
        dataset = load_dataset(args.dataset) if args.dataset else None
        expected_sources = _preflight_sources(args)
        sources = {name: getattr(args, name) for name in ("report", "retrieval", "grounding")}
        replay_path = args.replay_suite if args.command == "collect" else None
        if args.command == "run":
            from .replay_suite import run_suite

            output = Path(args.out_dir)
            # Fail fast BEFORE running a single case; do not pre-create the dir.
            if output.exists():
                raise FileExistsError(f"new out-dir already exists: {output}")
            if type(args.timeout) is not int or not 1 <= args.timeout <= 3600:
                raise ValueError("timeout must be an integer 1..3600")
            if not args.revision.strip() or len(args.revision) > 128:
                raise ValueError("revision must be a nonempty <=128 character string")
            # v0.6 creates the replay directory with exist_ok=False.
            output.mkdir(parents=True, exist_ok=False)
            check_output_path(output / "unified.json")
            manifest = freeze_dataset(dataset, output / "dataset")
            sources = _freeze_sources(args, output, expected_sources)
            replay_path = output / "replay" / "suite.json"
            run_suite(manifest, output / "replay", timeout_s=args.timeout, revision=args.revision)
        normalized = build_unified(
            dataset=dataset,
            replay_suite=replay_path,
            report=sources["report"],
            retrieval=sources["retrieval"],
            grounding=sources["grounding"],
        )
        target = (Path(args.out_dir) / "unified.json") if args.command == "run" else Path(args.out)
        save_new(target, normalized)
        print(
            json.dumps(
                {
                    "output": str(target),
                    "dimensions": {
                        key: val["status"] for key, val in normalized["dimensions"].items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        reliability = normalized["dimensions"]["reliability"]
        if args.command == "run" and reliability["status"] == "measured":
            return 0 if all(c["test_result"] == "pass" for c in reliability["cases"]) else 1
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Evaluation error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
