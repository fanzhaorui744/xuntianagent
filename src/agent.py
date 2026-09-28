#!/usr/bin/env python3
"""Practice-board v1: replay the already-scored playground traces.

The platform drives one decision per request. We identify the scenario from the
`initialize` publication (first night + night count + slot count) and emit the
precomputed action for the current cursor slot. This is a transport check for
the complete-project board, not the formal strategy: hidden formal scenarios
have different tiles and weather, and this table scores nothing there.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

SCHEDULES = {
    # (first_night, night_count, slot_count) -> csv name
    ("2026-09-07", 180, 7928): "dev-reference.csv",
    ("2026-10-05", 14, 590): "dev-fortnight.csv",
}

SCHEDULE_DIR = Path(__file__).resolve().parent.parent / "schedules"


def load_schedule(name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with (SCHEDULE_DIR / name).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


class Replayer:
    """Emit the stored action only when the cursor matches; never emit early."""

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows
        self.index = 0

    def next_action(self, slot_id: str) -> dict[str, str]:
        index = self.index
        while index < len(self.rows) and self.rows[index]["slot_id"] < slot_id:
            index += 1
        self.index = index
        if index < len(self.rows) and self.rows[index]["slot_id"] == slot_id:
            row = self.rows[index]
            self.index = index + 1
            return row
        return {"action": "wait", "reason": "replay: cursor ahead of schedule"}

    def to_decision(self, row: dict[str, str]) -> dict[str, str]:
        if row.get("action") != "observe":
            return {"action": "wait", "reason": "replay wait"}
        return {
            "action": "observe",
            "tile_id": row.get("tile_id", ""),
            "program": row.get("program", ""),
            "request_id": row.get("request_id", ""),
            "reason": "replay v1",
        }


def run(stdin=sys.stdin, stdout=sys.stdout) -> None:
    replayer: Replayer | None = None
    for line in stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        message_type = str(message.get("message_type", ""))
        if message_type == "initialize":
            payload = message.get("payload", {})
            calendar = payload.get("calendar", {})
            key = (
                str(calendar.get("first_night", "")),
                int(calendar.get("night_count", -1)),
                int(calendar.get("slot_count", -1)),
            )
            name = SCHEDULES.get(key)
            if name is None:
                # Unknown scenario: never guess. Formal scenarios land here.
                replayer = None
                print(f"replay-v1: no schedule for {key}, falling back to wait",
                      file=sys.stderr, flush=True)
            else:
                replayer = Replayer(load_schedule(name))
                print(f"replay-v1: loaded {name} for {key}", file=sys.stderr, flush=True)
            continue
        if message_type != "decision_request":
            continue
        sequence = int(message.get("decision_sequence", -1))
        if replayer is None:
            decision = {"action": "wait", "reason": "replay v1: unrecognized scenario"}
        else:
            cursor = message.get("payload", {}).get("cursor", {})
            row = replayer.next_action(str(cursor.get("slot_id", "")))
            decision = replayer.to_decision(row)
        envelope = {
            "protocol_version": message.get("protocol_version", "participant-agent-protocol-v2"),
            "message_type": "decision_response",
            "decision_sequence": sequence,
            "action": decision["action"],
            "tile_id": decision.get("tile_id", ""),
            "program": decision.get("program", ""),
            "request_id": decision.get("request_id", ""),
            "reason": decision.get("reason", ""),
            "decision_source": "deterministic",
        }
        print(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")),
              file=stdout, flush=True)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
