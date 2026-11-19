"""Output checks precede paid calls; final exclusive creation closes the name race (docs/08 §5)."""

import json
import tempfile
from pathlib import Path


def check_output_path(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=".agent-eval-write-check-", dir=path.parent
    ) as probe:
        probe.write("write-check\n")
        probe.flush()


def save_artifact(result: dict, path: Path) -> None:
    payload = {"extension_schema_version": 1, **result}
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(text)
