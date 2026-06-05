"""Append-only ledger backed by a JSONL file.

One line == one `LedgerEntry`. Used both as the durable record of produced content
and as the dedup source so the pipeline never re-processes the same Reddit post.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .models import LedgerEntry


def append_entry(ledger_path: Path, entry: LedgerEntry) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")


def iter_entries(ledger_path: Path) -> Iterator[LedgerEntry]:
    if not ledger_path.exists():
        return
    with open(ledger_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield LedgerEntry.from_dict(json.loads(line))


def existing_post_ids(ledger_path: Path) -> set[str]:
    """Reddit post_ids already recorded -- used to skip duplicates."""
    return {
        entry.source.get("post_id")
        for entry in iter_entries(ledger_path)
        if entry.source.get("post_id")
    }
