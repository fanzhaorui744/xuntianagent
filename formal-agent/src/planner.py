"""Deterministic planner: per-night plan skeleton and drop heuristics.

The LLM layer (not in v1-deterministic) will write keep_regions/drop_tiles.
Everything here is the fallback when the model is absent or fails (C5).
"""

from __future__ import annotations

from typing import Any


class Planner:
    def __init__(self, scenario, catalog: dict[str, dict[str, Any]]):
        self.sc = scenario
        self.catalog = catalog

    def default_drops(self, flexible_by_region: dict[str, int]) -> set[str]:
        """Tiles to abandon when time is scarce: quota already met in region and
        low value per second. REQUIRED is never dropped."""
        drops = set()
        quota = self.sc.quota_per_region or 4
        for tile_id, row in self.catalog.items():
            if str(row.get("scheduling_class", "")) == "REQUIRED":
                continue
            region = str(row.get("region_id", ""))
            if flexible_by_region.get(region, 0) >= quota:
                exptime = float(row.get("nominal_exptime_seconds", 900) or 900)
                value = float(row.get("tile_science_value", 0.0) or 0.0)
                if value / max(exptime, 1.0) < 0.15:  # bottom-rung value density
                    drops.add(tile_id)
        return drops

    def night_context(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """What the LLM planner (§6.1) will consume at night start."""
        return {
            "night": str(snapshot.get("cursor", {}).get("night_id", "")),
            "regions_done": snapshot.get("progress", {}).get("flexible_completed_by_region", {}),
            "weekly_available": snapshot.get("weekly") is not None,
        }
