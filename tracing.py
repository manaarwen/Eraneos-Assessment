"""One trace record per turn: built in app.py, shown in the UI, appended here."""

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

TRACE_PATH = Path(__file__).parent / "traces.jsonl"


def new_record(mode: str, question: str) -> dict:
    """Start a trace record for one turn."""
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "turn_id": uuid4().hex[:8],
        "mode": mode,
        "question": question,
    }


def write_turn(record: dict) -> None:
    """Append one finished turn as a single JSON line."""
    with open(TRACE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
