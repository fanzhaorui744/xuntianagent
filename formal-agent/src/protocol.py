"""JSONL protocol layer: accept v1/v2 envelopes, emit matching responses.

One `initialize` arrives first (never answered); every `decision_request` gets
exactly one `decision_response` with the same sequence and protocol version.
Stdlib only.
"""

from __future__ import annotations

import json
import sys
from typing import Any

PROTOCOL_V2 = "participant-agent-protocol-v2"
ACCEPTED = ("participant-agent-protocol-v1", PROTOCOL_V2)


def read_message(line: str) -> tuple[str, str, dict[str, Any]]:
    """Return (message_type, protocol_version, payload)."""
    message = json.loads(line)
    version = str(message.get("protocol_version", ""))
    if version not in ACCEPTED:
        raise ValueError(f"unsupported protocol_version {version!r}")
    payload = message.get("payload")
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    return str(message.get("message_type", "")), version, payload


def sequence_of(message: dict[str, Any], payload: dict[str, Any]) -> int:
    envelope_seq = int(message.get("decision_sequence", -1))
    payload_seq = int(payload.get("decision_sequence", envelope_seq))
    if envelope_seq != payload_seq:
        raise ValueError("decision sequence differs between envelope and payload")
    return envelope_seq


def response(seq: int, version: str, decision: dict[str, Any],
             reports: list[dict[str, str]] | None = None) -> str:
    envelope = {
        "protocol_version": version,
        "message_type": "decision_response",
        "decision_sequence": seq,
        "action": decision["action"],
        "tile_id": decision.get("tile_id", ""),
        "program": decision.get("program", ""),
        "request_id": decision.get("request_id", ""),
        "reason": decision.get("reason", ""),
        "decision_source": decision.get("decision_source", "deterministic"),
    }
    if reports:
        envelope["reports"] = reports
    return json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))


def run_loop(decide, log=None) -> None:
    """Drive stdin/stdout; `decide(payload) -> (decision, reports)`."""
    out = sys.stdout
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message_type, version, payload = read_message(line)
        except ValueError as exc:
            if log:
                log(f"protocol: {exc}")
            continue
        if message_type == "initialize":
            decide.initialize(payload)
            continue
        if message_type != "decision_request":
            continue
        message = json.loads(line)
        seq = sequence_of(message, payload)
        try:
            outcome = decide(payload)
            decision, reports = outcome if isinstance(outcome, tuple) else (outcome, None)
        except Exception as exc:  # never die mid-survey (C5)
            if log:
                log(f"decide error: {type(exc).__name__}: {exc}")
            decision, reports = ({"action": "wait", "reason": "fallback wait"}, None)
        out.write(response(seq, version, decision, reports) + "\n")
        out.flush()
