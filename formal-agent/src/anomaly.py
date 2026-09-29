"""Anomaly side-channel (§7 of the spec): realized-vs-preview detection and
report emission. Deterministic, no model calls, never changes scheduling
except the one capped extra visit for unreported suspects.

Detection: compare `tile_last_finished.score` (realized, includes the hidden
instrument side) with the preview estimate recorded when the exposure was
committed. Feedback arrives one or more slots after commit and several
exposures may be in flight, so previews are kept as a per-tile FIFO and
paired with feedback in commit order — comparing a newer preview against an
older exposure's realized score is what produced the T00018 false reddening.

Two whole-population effects masquerade as anomalies and are removed before
classification (both measured on finals-preview):

- forecasted cold waves carry a public, legitimate efficiency dip. The weekly
  payload is published only on night-open of every Nth night, so the forecast
  list is cached and reused on later nights; reads committed under one are
  skipped entirely.
- unforecasted suppression (a cold wave whose forecast was issued after the
  only weekly publication) reads as a NIGHT-WIDE ratio trough: every tile on
  the night lands in the same 0.55-0.87 band, while a real tag collapses one
  tile and a real fault collapses one region deeply. Three such reads
  (mean <= 0.75, none above 0.90, none below 0.50) latch the night as
  suppressed and its reads are purged from every tile's history.

Ratios: ~0.90-1.00 jitter, ~1.30-1.65 nova, ~0.70-0.87 reddening,
persistently <= 0.55 -> instrument fault. A night whose reads sit mostly in
the reddening band (>30%, min 6 reads) is a suppression tail, not a tag
outbreak: a true tag collapses one tile, never half the population — its
in-band reads are retracted at the next night start.
"""
from __future__ import annotations

from typing import Any

# Bands from the published factors (nova ×1.5, reddening ×0.8) tightened the
# way the reference detector is: a tag needs a dominant multi-night history,
# one in-band reading is weather noise.
NOVA = (1.30, 1.65)
REDDENING = (0.70, 0.87)
# finals-preview EV0030 co-times with the EV0026 cold wave, so its first-night
# reads collapse to ~0.47-0.53 (0.649 x 0.812 x jitter) — well under the line.
# A solo fault only reaches ~0.58-0.65, which this line deliberately does not
# chase: that range overlaps double-cold-wave dips (0.543-0.60) and reporting
# there misfires (the old 0.68 line swallowed entire cold-wave nights).
FAULT = 0.55
TAG_MIN_READS = 2
TAG_MIN_FRACTION = 0.80
FAULT_WINDOW = 8
FAULT_HITS = 2
# At most this many extra visits are granted to a tile to settle its tag;
# beyond that the odds say stop (each visit costs a slot).
MAX_EXTRA_VISITS = 2
# Night-wide suppression latch: N reads with a common mid dip and no clean
# and no deeply-collapsed tile is weather-level suppression, not an anomaly.
SUPPRESS_MIN_READS = 3
SUPPRESS_MAX_MEAN = 0.75
SUPPRESS_CEILING = 0.90
SUPPRESS_FLOOR = 0.50
# A closed night whose reads are mostly in the reddening band is a
# suppression tail: retract its band evidence before suspects can chase it.
BAND_NIGHT_MAX_FRACTION = 0.30
BAND_NIGHT_MIN_READS = 6


class AnomalyWatch:
    def __init__(self, log=None) -> None:
        self.log = log
        self.pending: dict[str, list[tuple[float, bool]]] = {}  # tile -> FIFO of commit previews
        self.reads: dict[str, list[tuple[str | None, str]]] = {}  # tile -> (band, night)
        self.reported: set[tuple[str, str]] = set()   # (tile, kind)
        self.recent_ratios: list[tuple[str, float]] = []       # (night, ratio)
        self.night_ratios: dict[str, list[float]] = {}         # night -> every read ratio
        self.suppressed_nights: set[str] = set()               # latched suppression
        self._closed_nights: set[str] = set()                  # band-fraction evaluated
        self._forecasts: list[dict[str, Any]] = []             # cached weekly forecasts
        self.fault_pending = False
        self.repair_until: str = ""
        self.misreports = 0
        self.correct_reports = 0
        self.extra_visits: dict[str, int] = {}   # tile -> granted extra visits
        self.trace: list[str] = []               # human-readable read log

    def suspects(self) -> set[str]:
        """Tiles whose latest read is in-band, unreported, and still owed an
        extra visit. The extra read is what lets a tag span two nights; after
        MAX_EXTRA_VISITS granted visits the tag odds say stop."""
        out: set[str] = set()
        for tile_id, reads in self.reads.items():
            if not reads:
                continue
            band, _night = reads[-1]
            if band is None or (tile_id, band) in self.reported:
                continue
            if self.extra_visits.get(tile_id, 0) >= MAX_EXTRA_VISITS:
                continue
            out.add(tile_id)
        return out

    def suspects_active(self, snapshot: dict[str, Any]) -> bool:
        """False on nights whose reads carry no anomaly signal: a forecasted
        cold wave over the cursor, or a latched suppressed night. Chasing tags
        there burns slots on reads that get skipped anyway."""
        night = str(snapshot.get("cursor", {}).get("night_id", ""))
        return not (night in self.suppressed_nights or self.under_cold_wave(snapshot))

    def note_extra_visit(self, tile_id: str) -> None:
        self.extra_visits[tile_id] = self.extra_visits.get(tile_id, 0) + 1

    def record_commit(self, tile_id: str, preview_score: float,
                      cold_wave: bool = False) -> None:
        if preview_score > 0:
            queue = self.pending.setdefault(tile_id, [])
            queue.append((preview_score, cold_wave))
            del queue[:-3]  # safety cap: never pair across more than 3 exposures

    def under_cold_wave(self, snapshot: dict[str, Any]) -> bool:
        """Published forecast says cold_wave over the cursor right now.

        The weekly payload (the only carrier of forecasts) is published on
        night-open of every Nth night, so the forecast list is cached here and
        reused on every later night — finals-preview publishes exactly once,
        on the first night."""
        weekly = snapshot.get("weekly")
        if isinstance(weekly, dict) and weekly.get("weather_forecast") is not None:
            self._forecasts = list(weekly["weather_forecast"])
        now = str((snapshot.get("cursor") or {}).get("timestamp_utc", ""))
        if not now:
            return False
        for forecast in self._forecasts:
            if forecast.get("condition") != "cold_wave":
                continue
            if str(forecast.get("predicted_start_utc", "")) <= now < \
                    str(forecast.get("predicted_end_utc", "")):
                return True
        return False

    def on_snapshot(self, snapshot: dict[str, Any]) -> list[dict[str, str]]:
        """Consume tile_last_finished; return reports to attach to this decision."""
        last = snapshot.get("tile_last_finished")
        if not last:
            return []
        tile_id = str(last.get("tile_id", ""))
        try:
            realized = float(last.get("score", 0.0))
        except (TypeError, ValueError):
            realized = 0.0
        # Pair feedback with the OLDEST unpaired commit of this tile (FIFO):
        # exposures finish in commit order, so does their feedback.
        queue = self.pending.get(tile_id)
        entry = queue.pop(0) if queue else None
        if queue is not None and not queue:
            self.pending.pop(tile_id, None)
        # Cache weekly forecasts whenever one arrives; close out finished
        # nights before classifying so retraction precedes any new suspect.
        weekly = snapshot.get("weekly")
        if isinstance(weekly, dict) and weekly.get("weather_forecast") is not None:
            self._forecasts = list(weekly["weather_forecast"])
        self._close_passed_nights(str(snapshot.get("cursor", {}).get("night_id", "")))
        if entry is None:
            return []
        preview, cold_wave = entry
        if preview <= 0 or realized <= 0:
            return []  # interrupted exposure: no anomaly signal, preview consumed
        ratio = realized / preview
        night = str(snapshot.get("cursor", {}).get("night_id", ""))
        self.night_ratios.setdefault(night, []).append(ratio)
        self._evaluate_suppression(night)
        deep = ratio <= FAULT  # a cold wave bottoms at ~0.54; a fault goes deeper
        if cold_wave and not deep:
            # Forecasted cold wave: mid-band dips are legitimate weather and are
            # skipped, but a collapse to <= FAULT rides THROUGH the wave — the
            # deepest EV0030 evidence (0.476-0.53 on N20261008) sits inside the
            # EV0026 forecast window and would otherwise be swallowed.
            self.trace.append(f"{night} {tile_id} ratio={ratio:.3f} SKIP cold_wave")
            if self.log:
                self.log(f"anomaly: {self.trace[-1]}")
            return []
        if night in self.suppressed_nights and not deep:
            self.trace.append(f"{night} {tile_id} ratio={ratio:.3f} SKIP suppressed night")
            if self.log:
                self.log(f"anomaly: {self.trace[-1]}")
            return []
        reports: list[dict[str, str]] = []
        band = None
        if NOVA[0] <= ratio <= NOVA[1]:
            band = "NOVA"
        elif REDDENING[0] <= ratio <= REDDENING[1]:
            band = "Reddening"
        self.trace.append(f"{night} {tile_id} realized={realized:.2f} "
                          f"preview={preview:.2f} ratio={ratio:.3f} band={band}")
        if self.log:
            self.log(f"anomaly: {self.trace[-1]}")
        reads = self.reads.setdefault(tile_id, [])
        reads.append((band, night))
        if band is not None and (tile_id, band) not in self.reported:
            hits = [n for b, n in reads if b == band]
            if len(reads) >= TAG_MIN_READS and len(hits) / len(reads) >= TAG_MIN_FRACTION \
                    and len(set(hits)) >= 2:
                self.reported.add((tile_id, band))
                reports.append({"kind": band, "tile_id": tile_id})
                if self.log:
                    self.log(f"anomaly: report {band} {tile_id}")
        # fault_status is published only AFTER a correct report, so the report
        # itself has to come from the ratio collapse, not from that status.
        if not self._repair_active(snapshot):
            self.recent_ratios.append((night, ratio))
            del self.recent_ratios[:-FAULT_WINDOW]
            if sum(r <= FAULT for _n, r in self.recent_ratios) >= FAULT_HITS \
                    and not self.fault_pending:
                self.fault_pending = True
                self.recent_ratios.clear()
                reports.append({"kind": "Instrument_Failure"})
                if self.log:
                    self.log("anomaly: report Instrument_Failure")
        return reports

    def on_fault_status(self, status: dict[str, Any] | None) -> list[dict[str, str]]:
        """Track the platform's answer. Never emit from here: status=='fault'
        exists only after a correct report, and status=='normal' is a misreport."""
        if not status:
            return []
        if status.get("status") == "fault":
            self.fault_pending = False
            self.recent_ratios.clear()
            self.repair_until = str(status.get("repair_complete_utc", ""))
            self.correct_reports += 1
        elif status.get("status") == "normal":
            self.fault_pending = False
            self.recent_ratios.clear()
            self.misreports += 1
        return []

    def _evaluate_suppression(self, night: str) -> None:
        """Latch a night as suppressed once its population dip is unambiguous,
        then purge the reads it already contributed (they were classified
        before the pattern was visible)."""
        if night in self.suppressed_nights:
            return
        ratios = self.night_ratios.get(night) or []
        if len(ratios) < SUPPRESS_MIN_READS:
            return
        mean = sum(ratios) / len(ratios)
        if mean <= SUPPRESS_MAX_MEAN and max(ratios) < SUPPRESS_CEILING \
                and min(ratios) > SUPPRESS_FLOOR:
            self.suppressed_nights.add(night)
            self._purge_night(night, band_only=False)
            if self.log:
                self.log(f"anomaly: night {night} suppressed "
                         f"(n={len(ratios)} mean={mean:.3f})")

    def _close_passed_nights(self, cursor_night: str) -> None:
        """Retract band evidence from finished nights that were suppression
        tails (mostly in-band reads), before today's suspects are chosen."""
        for night, ratios in list(self.night_ratios.items()):
            if night == cursor_night or night in self._closed_nights:
                continue
            self._closed_nights.add(night)
            if len(ratios) < BAND_NIGHT_MIN_READS:
                continue
            inband = sum(1 for r in ratios if REDDENING[0] <= r <= REDDENING[1])
            if inband / len(ratios) > BAND_NIGHT_MAX_FRACTION:
                self._purge_night(night, band_only=True)
                if self.log:
                    self.log(f"anomaly: night {night} band reads retracted "
                             f"({inband}/{len(ratios)} in-band)")

    def _purge_night(self, night: str, band_only: bool) -> None:
        for reads in self.reads.values():
            if band_only:
                reads[:] = [e for e in reads if not (e[1] == night and e[0] is not None)]
            else:
                reads[:] = [e for e in reads if e[1] != night]
        if not band_only:
            self.recent_ratios = [(n, r) for n, r in self.recent_ratios if n != night]

    def _repair_active(self, snapshot: dict[str, Any]) -> bool:
        if not self.repair_until:
            return False
        now = str(snapshot.get("cursor", {}).get("timestamp_utc", ""))
        return bool(now) and now < self.repair_until
