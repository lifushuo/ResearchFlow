"""Batch-local Judge labels; both scoring entry points share the bounds (docs/08)."""

import re


def claim_labels(text: str, batch_size: int) -> dict[int, bool]:
    """Keep the legacy last-valid-label-wins policy, without crossing batch boundaries."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    labels = {}
    for match in re.finditer(r"Claim\s+(\d+)\s*:\s*(YES|NO)", text, re.IGNORECASE):
        try:
            index = int(match.group(1)) - 1
        except ValueError:
            continue
        if 0 <= index < batch_size:
            labels[index] = match.group(2).upper() == "YES"
    return labels
