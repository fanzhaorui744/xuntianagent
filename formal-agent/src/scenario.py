"""Scenario fingerprinting and mode selection from the initialize publication.

Nothing about the scenario is hardcoded: mechanics, coverage weight, quotas and
sizes all come from the scoring contract the platform sends (§2 of the spec).
"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Scenario:
    first_night: str = ""
    night_count: int = 0
    slot_count: int = 0
    tile_count: int = 0
    region_ids: list[str] = field(default_factory=list)
    required_ids: set[str] = field(default_factory=set)
    mechanics: bool = False                 # repeat_observation section present
    coverage_weight: float = 0.0            # coverage_bonus_weight
    quota_per_region: int = 0               # flexible_quota_per_region
    wallclock: float = 3600.0
    faults: bool = False                    # fault_response section present
    dark_threshold: float = 0.65
    bright_threshold: float = 0.40

    @classmethod
    def from_initialize(cls, payload: dict[str, Any]) -> "Scenario":
        calendar = payload.get("calendar", {})
        catalog = payload.get("tile_catalog", {})
        contract = payload.get("scoring_contract", {})
        config = contract.get("score_config", {})
        thresholds = config.get("quality_thresholds", {})
        return cls(
            first_night=str(calendar.get("first_night", "")),
            night_count=int(calendar.get("night_count", 0)),
            slot_count=int(calendar.get("slot_count", 0)),
            tile_count=int(catalog.get("tile_count", 0)),
            region_ids=list(catalog.get("region_ids", [])),
            required_ids=set(catalog.get("required_tile_ids", [])),
            mechanics="repeat_observation" in config,
            coverage_weight=float(config.get("coverage_bonus_weight", 0.0)),
            quota_per_region=int(config.get("flexible_quota_per_region", 0)),
            wallclock=float(payload.get("global_wallclock_seconds", 3600.0)),
            faults="fault_response" in config,
            dark_threshold=float(thresholds.get("dark", 0.65)),
            bright_threshold=float(thresholds.get("bright", 0.40)),
        )


def tile_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """tile_id -> catalog row (region, class, exptime, value)."""
    return {
        str(row["tile_id"]): row
        for row in payload.get("tile_catalog", {}).get("tiles", [])
    }
