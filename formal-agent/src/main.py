"""Entry point: wire protocol layer to the deterministic hot path + LLM layer.

The LLM planner (V3, spec §6) writes keep_regions / drop_tiles into the hot
path; the deterministic ranking always remains the fallback (C5). Set
MODEL_PROVIDER=deterministic (or leave OPENAI_BASE_URL unset) to run without
the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import protocol
from anomaly import AnomalyWatch
from hotpath import HotPath
from llm import LlmClient
from llm_planner import LlmPlanner
from planner import Planner
from scenario import Scenario, tile_index


class Agent:
    def __init__(self):
        self.hot: HotPath | None = None
        self.planner: Planner | None = None
        self.llm: LlmPlanner | None = None
        self.watch = AnomalyWatch(log=log)

    def initialize(self, payload: dict) -> None:
        scenario = Scenario.from_initialize(payload)
        catalog = tile_index(payload)
        self.hot = HotPath(scenario, catalog)
        self.hot.load_deadlines(catalog)
        self.planner = Planner(scenario, catalog)
        client = LlmClient(log=log)
        client.wallclock_budget = scenario.wallclock  # §11 guard anchor
        self.llm = LlmPlanner(client, scenario, catalog, log=log)
        print(f"formal-agent: nights={scenario.night_count} slots={scenario.slot_count} "
              f"tiles={scenario.tile_count} mechanics={scenario.mechanics} "
              f"coverage={scenario.coverage_weight} wallclock={scenario.wallclock} "
              f"llm={client.enabled}",
              file=sys.stderr, flush=True)

    def __call__(self, payload: dict):
        if self.hot is None:
            return {"action": "wait", "reason": "no initialize"}, None
        reports: list[dict[str, str]] = []
        if self.hot.sc.mechanics:
            # Consume last-finished before deciding so a just-classified suspect
            # can take the one extra visit the tag needs, and the report rides
            # this decision (no slot time).
            reports += self.watch.on_snapshot(payload)
            reports += self.watch.on_fault_status(payload.get("fault_status"))
            self.hot.tag_suspects = self.watch.suspects()
        # LLM triggers ride publications, never slots: §6.1 on the night's
        # first snapshot, §6.2 when a weekly revision lands.
        if self.llm is not None:
            try:
                if payload.get("night_start"):
                    self.llm.on_night_start(payload)
                if payload.get("weekly"):
                    self.llm.on_weekly(payload)
            except Exception as exc:  # model layer must never kill the survey
                if log:
                    log(f"llm planner error: {type(exc).__name__}: {exc}")
            self.hot.keep_regions = set(self.llm.keep_regions)
            self.hot.drop_tiles = set(self.llm.drop_tiles)
            llm_note, self.llm.pending_note = self.llm.pending_note, ""
        decision, extra = self.hot.decide(payload)
        reports += list(extra or [])
        if self.hot.sc.mechanics:
            tile_id = str(decision.get("tile_id", ""))
            if tile_id and decision.get("action") == "observe":
                if str(decision.get("reason", "")).startswith("anomaly suspect"):
                    self.watch.note_extra_visit(tile_id)
                self.watch.record_commit(tile_id, self.hot.last_preview(tile_id),
                                         cold_wave=self.watch.under_cold_wave(payload))
        if llm_note:
            # reason/decision_source carry the LLM trace for review (V3 pass
            # criterion); appended after the suspect check above so the tag
            # machinery sees the original reason prefix. The note stamps only
            # the decision that consumed the trigger, not the whole night.
            decision["reason"] = f"{decision.get('reason', '')}|llm:{llm_note}"[:200]
            decision["decision_source"] = "llm-assisted"
        return decision, reports or None


def log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


if __name__ == "__main__":
    protocol.run_loop(Agent(), log=log)
