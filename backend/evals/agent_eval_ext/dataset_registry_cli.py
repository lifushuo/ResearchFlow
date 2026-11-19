"""Offline, read-only inspection of the native Evaluation Dataset Registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.services.evaluation_dataset_registry import (
    DatasetRegistryError,
    get_dataset,
    list_datasets,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect registered evaluation datasets (no API)")
    parser.add_argument("command", choices=("list", "validate", "show"))
    parser.add_argument("--dataset-id", help="Required with show")
    parser.add_argument("--backend-root", type=Path, help="Override backend root for local tests")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)
    try:
        if args.command == "show":
            if not args.dataset_id:
                parser.error("show requires --dataset-id")
            result = get_dataset(
                args.dataset_id,
                offset=args.offset,
                limit=args.limit,
                backend_root=args.backend_root,
            )
        else:
            result = list_datasets(backend_root=args.backend_root)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if args.command == "validate":
            broken = [
                row["id"]
                for row in result["datasets"]
                if row["source_status"] not in ("available", "unlinked")
            ]
            if broken:
                print("Registry sources need review: " + ", ".join(broken), file=sys.stderr)
                return 1
        return 0
    except (DatasetRegistryError, OSError, KeyError, ValueError) as exc:
        print(f"Registry error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
