"""LLM planner (§6.1 night planning, §6.2 weekly adaptation).

Two real touchpoints, both triggered by publications the hot path cannot see:

- §6.1 fires on the first snapshot of each night (`night_start` present):
  window summary + forecast summary + region counters + nights left -> JSON
  {keep_regions: 1-3 regions, notes: one line}. keep_regions only breaks ties
  in the step-4 ranking (x1.01 soft preference) — never a hard filter, so a
  wrong call costs little.
- §6.2 fires when a weekly publication arrives (every 7th night): forecast
  diff + current drops + unfinished REQUIRED -> {drop_tiles: [...]}. Drops go
  through the hot path's existing drop_tiles filter (REQUIRED never dropped).

Schema-validated; on invalid output retry once, then fall back to the
deterministic default (empty keep set, heuristic drops) for that trigger.
Budget: ~1 call/night + ~1/7 nights ≈ 206 calls on a 180-night survey.
"""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PLAN = (
    "You are the night planner for a sky survey. Choose which regions the "
    "telescope should prioritize tonight. Answer with a JSON object containing "
    "keys keep_regions (a list of 1-3 region codes like R00..R07) and notes "
    "(one short sentence)."
)

SYSTEM_WEEK = (
    "You are the weekly planner for a sky survey. A weather forecast revision "
    "just arrived. Decide which tiles should be abandoned (never REQUIRED "
    "tiles). Answer with a JSON object containing keys drop_tiles (a list of "
    "tile ids like T0xxxx, possibly empty) and notes (one short sentence). "
    "Drop sparingly: only tiles whose windows the forecast makes unreachable "
    "or that cannot pay their exposure."
)

MAX_INPUT_CHARS = 4096


def _regions_from(value: Any, valid: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        region = str(item)
        if region in valid and region not in out:
            out.append(region)
    return out[:3]


def _tiles_from(value: Any, droppable: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        tile = str(item)
        if tile in droppable and tile not in out:
            out.append(tile)
    return out


class LlmPlanner:
    def __init__(self, client, scenario, catalog, log=None):
        self.client = client          # LlmClient (may be disabled)
        self.sc = scenario
        self.catalog = catalog
        self.log = log
        self.keep_regions: set[str] = set()
        self.notes: str = ""
        self.drop_tiles: set[str] = set()
        self.last_night_planned = ""  # one §6.1 call per night
        self.last_week_planned = ""   # one §6.2 call per weekly publication
        self.pending_note = ""        # note for THIS decision only (trace), cleared on read
        self.accepted = 0             # model outputs that passed schema
        self.fallbacks = 0            # triggers where the model gave nothing

    # ---------- §6.1 night planning ----------

    def night_payload(self, snapshot: dict[str, Any]) -> str:
        """Night-start prompt input, <= 4 KB: windows, forecasts, counters."""
        night_start = snapshot.get("night_start") or {}
        progress = snapshot.get("progress", {}).get("flexible_completed_by_region", {})
        # Per-region shootable-block counts from tonight's windows.
        by_region: dict[str, int] = {}
        required_soon: list[str] = []
        for row in night_start.get("tile_windows", []):
            region = str(row.get("region_id", ""))
            if region:
                by_region[region] = by_region.get(region, 0) + 1
        # REQUIRED tiles whose windows end within the next two nights.
        for row in night_start.get("tile_windows", []):
            if str(row.get("scheduling_class", "")) != "REQUIRED":
                continue
            required_soon.append(str(row.get("tile_id", "")))
        lines = [f"night={snapshot.get('cursor', {}).get('night_id', '')}",
                 f"nights_total={self.sc.night_count}",
                 f"progress_per_region={json.dumps(progress, separators=(',', ':'))}",
                 f"windows_per_region={json.dumps(by_region, separators=(',', ':'))}",
                 f"required_tonight={','.join(sorted(set(required_soon)))[:200]}"]
        # One-week forecast summary: per condition/event severity x probability.
        weekly = snapshot.get("weekly") or {}
        forecasts = weekly.get("weather_forecast") or []
        if forecasts:
            compact = [
                {"c": str(f.get("condition", "")),
                 "sev": round(float(f.get("severity", 0.0) or 0.0), 2),
                 "p": round(float(f.get("probability", 0.0) or 0.0), 2),
                 "scope": str(f.get("spatial_scope_payload", ""))[:80]}
                for f in forecasts[:20]
            ]
            lines.append("forecast=" + json.dumps(compact, separators=(",", ":")))
        text = "\n".join(lines)
        return text[:MAX_INPUT_CHARS]

    def on_night_start(self, snapshot: dict[str, Any]) -> None:
        """Call once per night when night_start is present."""
        night_id = str(snapshot.get("cursor", {}).get("night_id", ""))
        if not night_id or night_id == self.last_night_planned:
            return
        self.last_night_planned = night_id
        if self.client is None or not self.client.enabled:
            return
        reply = self.client.chat(SYSTEM_PLAN, self.night_payload(snapshot))
        if reply is None:
            self.fallbacks += 1
            self.keep_regions = set()
            self.notes = ""
            return
        regions = _regions_from(reply.get("keep_regions"), set(self.sc.region_ids))
        if not regions:
            # schema miss: retry once, then deterministic fallback
            reply = self.client.chat(SYSTEM_PLAN, self.night_payload(snapshot))
            regions = _regions_from((reply or {}).get("keep_regions"),
                                    set(self.sc.region_ids))
            if not regions:
                self.fallbacks += 1
                self.keep_regions = set()
                self.notes = ""
                return
        self.accepted += 1
        self.keep_regions = set(regions)
        self.notes = str((reply or {}).get("notes", ""))[:120]
        # Append, don't overwrite: the weekly publication can land on the same
        # first snapshot as the night trigger, and both traces matter.
        note = f"night {night_id}: {self.notes}"
        self.pending_note = f"{self.pending_note} + {note}" if self.pending_note else note

    # ---------- §6.2 weekly adaptation ----------

    def week_payload(self, snapshot: dict[str, Any]) -> str:
        weekly = snapshot.get("weekly") or {}
        forecasts = weekly.get("weather_forecast") or []
        compact = [
            {"c": str(f.get("condition", "")),
             "sev": round(float(f.get("severity", 0.0) or 0.0), 2),
             "p": round(float(f.get("probability", 0.0) or 0.0), 2),
             "from": str(f.get("predicted_start_utc", "")),
             "to": str(f.get("predicted_end_utc", "")),
             "scope": str(f.get("spatial_scope_payload", ""))[:100]}
            for f in forecasts[:20]
        ]
        progress = snapshot.get("progress", {}).get("flexible_completed_by_region", {})
        lines = [
            f"issued={weekly.get('issued_at_utc', '')}",
            "forecast=" + json.dumps(compact, separators=(",", ":")),
            f"progress_per_region={json.dumps(progress, separators=(',', ':'))}",
            f"current_drops={','.join(sorted(self.drop_tiles))[:200] or 'none'}",
            f"required_open={','.join(sorted(self.sc.required_ids))[:200]}",
        ]
        return "\n".join(lines)[:MAX_INPUT_CHARS]

    def droppable_tiles(self) -> set[str]:
        """Every non-REQUIRED catalog tile is a legal drop target."""
        return {tile_id for tile_id, row in self.catalog.items()
                if str(row.get("scheduling_class", "")) != "REQUIRED"}

    def on_weekly(self, snapshot: dict[str, Any]) -> None:
        """Call when a weekly publication is present (one call per publication)."""
        weekly = snapshot.get("weekly") or {}
        issued = str(weekly.get("issued_at_utc", ""))
        if not issued or issued == self.last_week_planned:
            return
        self.last_week_planned = issued
        if self.client is None or not self.client.enabled:
            return
        droppable = self.droppable_tiles()
        reply = self.client.chat(SYSTEM_WEEK, self.week_payload(snapshot))
        drops = _tiles_from((reply or {}).get("drop_tiles"), droppable)
        if (reply is None or not _valid_shape(reply, "drop_tiles")):
            reply = self.client.chat(SYSTEM_WEEK, self.week_payload(snapshot))
            drops = _tiles_from((reply or {}).get("drop_tiles"), droppable)
            if not drops and reply is None:
                self.fallbacks += 1
                return  # keep previous drops; heuristic default already applies
        self.accepted += 1
        self.drop_tiles |= set(drops)
        note = str((reply or {}).get("notes", ""))[:120]
        if note:
            self.notes = note
            part = f"weekly {issued[:10]}: {note}"
            self.pending_note = f"{self.pending_note} + {part}" if self.pending_note else part


def _valid_shape(reply: dict | None, key: str) -> bool:
    return isinstance(reply, dict) and isinstance(reply.get(key), list)
