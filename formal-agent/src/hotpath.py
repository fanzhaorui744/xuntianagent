"""Slot-level hot path (§4 of the spec): hard filters, requests, REQUIRED
deadlines, value-per-second with marginal Jain coverage, wait rule.

Everything here is O(candidates) scalar work per slot — no ILP, no model calls.
The M1-measured break-even (gain / 0.001 per waiting second) is baked into
REPEAT as the cost of reaching a later start.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

REQUEST_WEIGHTS = {"TOMORROW_NIGHT": 0.9, "ONE_WEEK": 0.5, "TWO_WEEKS": 0.32, "ONE_MONTH": 0.18}


def jain(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    n = len(counts)
    return total * total / (n * sum(c * c for c in counts))


class HotPath:
    """Stateful selector; one instance per run."""

    def __init__(self, scenario, catalog: dict[str, dict[str, Any]]):
        self.sc = scenario
        self.catalog = catalog
        self.labels = sorted(scenario.region_ids)
        self.counts = {r: 0 for r in self.labels}      # completed tiles per region (all classes)
        self.base_estimate = 0.0                       # running base-science estimate for coverage pricing
        self.banked: dict[str, float] = {}             # tile -> banked score+bonus
        self.last_feasible: dict[str, str] = {}        # tile -> last night it was completable
        self.keep_regions: set[str] = set()            # LLM soft preference
        self.drop_tiles: set[str] = set()              # LLM+heuristic abandon list
        # M1 patience knobs (swept on dev-reference: 0.55/0.85 best so far)
        self.quality_floor = 0.55
        self.patience_ratio = 0.85
        # Catch-up waiver: a lagging region's tile may be observed down to this
        # floor. Twilight slots where every candidate sits below quality_floor
        # are otherwise pure waits (run3: 370 in-band waits, avoidable_wait
        # 698.7) — filling them with a lagging-region shot is zero-displacement
        # evenness: positive base, no leading-region completion lost.
        self.catchup_floor = 0.15
        # Catch-up machinery is a big-survey tool: on a 1600-tile stress survey
        # there is always spare budget for lagging-region fill, but on a compact
        # 64-tile official survey every slot is zero-sum — the ranking delta
        # displaced an above-floor completion (finals-preview D000008) and broke
        # the T00054 Reddening chain (report 100 -> 0). Gate both the ranking
        # delta and the below-floor waiver by scenario scale; the REQUIRED
        # window-margin fix stays global (inert on official scenes).
        self.catchup = self.sc.tile_count >= 500
        self._night_slot_counts: dict[str, int] = {}   # cursor visits per night
        self._night_confirmed_midnight_shot: dict[str, bool] = {}  # a short shot finished mid-night
        self._pending_request: dict[str, str] = {}     # tile -> request_id still needing visits
        self._best_q: dict[str, float] = {}            # tile -> best quality seen so far (M1 patience)
        self._last_preview: dict[str, float] = {}      # tile -> preview estimate of the last commit (anomaly watch)
        self._available_until: dict[str, str] = {}     # tile -> YYYYMMDD deadline from catalog
        self._shots: dict[tuple[str, str], int] = {}   # (tile, night) -> attempts this night
        self.tag_suspects: set[str] = set()            # in-band unreported tiles (from AnomalyWatch)

    def last_preview(self, tile_id: str) -> float:
        """Preview score of the tile's most recent committed exposure (§7)."""
        return self._last_preview.get(tile_id, 0.0)

    def _shot_count(self, tile_id: str, night: str) -> int:
        return self._shots.get((tile_id, night), 0)

    def _note_shot(self, tile_id: str, night: str) -> None:
        self._shots[(tile_id, night)] = self._shot_count(tile_id, night) + 1

    def load_deadlines(self, catalog: dict[str, dict[str, Any]]) -> None:
        """Tile availability deadlines come from the catalog, not the nightly
        window: candidate window_end_utc resets every night, so the urgent
        branch would otherwise never fire (finals-preview: 7 REQUIRED missed)."""
        for tile_id, row in catalog.items():
            self._available_until[tile_id] = str(row.get("available_until_utc", ""))[:10].replace("-", "")

    # ---- progress bookkeeping from each snapshot -------------------------
    def observe_progress(self, snapshot: dict[str, Any]) -> None:
        progress = snapshot.get("progress", {})
        done = set(progress.get("completed_tile_ids", []))
        self._last_done = done
        # Evenness counts EVERY completed tile (REQUIRED included) over the
        # whole set — mirrors scoring_core._coverage_evenness exactly.
        self.counts = {r: 0 for r in self.labels}
        est = 0.0
        for tile_id in done:
            row = self.catalog.get(tile_id)
            if row:
                region = str(row.get("region_id", ""))
                if region in self.counts:
                    self.counts[region] += 1
                try:
                    est += 0.75 * float(row.get("tile_science_value", 0.0))
                except (TypeError, ValueError):
                    pass
        if est > 0:
            self.base_estimate = est
        last = snapshot.get("tile_last_finished") or {}
        last_id = str(last.get("tile_id") or "")
        try:
            realized = float(last.get("score") or 0.0)
        except (TypeError, ValueError):
            realized = 0.0
        if last_id and realized > 0:
            # Bank the public baseline of the last commit (efficiency-free),
            # so a later visit is only scheduled when quality actually improves.
            # Fall back to realized if we somehow lost the preview.
            public = self._last_preview.get(last_id, 0.0)
            self.banked[last_id] = max(self.banked.get(last_id, 0.0),
                                       public if public > 0 else realized)

    def hot_completed(self) -> set[str]:
        progress = getattr(self, "_last_done", None)
        return progress or set()

    # ---- scoring ----------------------------------------------------------
    def _quality(self, cand: dict[str, Any]) -> float:
        weather = cand.get("effective_weather", {})
        geo = cand.get("geometry", {})
        try:
            atmospheric = (float(weather["transparency"]) * float(weather["sky_quality"])
                           / (float(weather["seeing_arcsec"]) * float(geo["airmass"])))
        except (KeyError, TypeError, ZeroDivisionError):
            return -1.0
        return atmospheric * float(geo.get("lunar_quality_factor", 1.0))

    def _program(self, quality: float) -> str:
        if quality >= self.sc.dark_threshold:
            return "DARK"
        if quality >= self.sc.bright_threshold:
            return "BRIGHT"
        return "BACKUP"

    def _bonus(self, program: str) -> float:
        return {"DARK": 0.25, "BRIGHT": 0.15, "BACKUP": 0.08}[program]

    def _coverage_delta(self, region: str) -> float:
        """Marginal coverage-bonus value of completing one more tile in `region`.

        The platform's coverage_bonus is coverage_weight * base_science *
        Jain(completed-by-region) over ALL completed tiles (REQUIRED included,
        repeats not double-counted). A full Jain derivative mispriced this on
        the V1 stress scenario: its negative term on leading regions pushed
        marginal repeats into gain<=0 (26 high-value completions lost) while
        the positive term (+63) was still far too small to out-rank the 02Z
        window stampede in lagging regions (R04/R05 windows are 15-60 min at
        450-1350 s exptime). Only the POSITIVE side of the derivative is real
        opportunity — a lagging region's bonus grows superlinearly — so price
        exactly that: a one-tile Jain uplift in `region`, floored at zero.
        Self-limiting (fades as the deficit closes), zero risk to even
        surveys (all deficits ~0 -> all deltas ~0), and never subtracts."""
        if self.sc.coverage_weight <= 0.0:
            return 0.0
        labels = self.labels
        counts = [float(self.counts[r]) for r in labels]
        total = sum(counts)
        if total <= 0:
            return 0.0
        n = len(counts)
        sum_sq = sum(c * c for c in counts)
        j = total * total / (n * sum_sq)
        i = labels.index(region)
        # Jain after +1 tile in `region`, minus Jain now; only the gain counts.
        new_sq = sum_sq + 2.0 * counts[i] + 1.0
        j_after = (total + 1.0) * (total + 1.0) / (n * new_sq)
        one_tile = self.sc.coverage_weight * self.base_estimate * (j_after - j)
        if one_tile <= 0.0:
            return 0.0  # never subtract: the negative side cost 26 completions in V1
        # Catch-up premium, gentler than the first ×4 attempt (run3: J 0.844
        # but base -3138 from displaced leading-region work): scale by the
        # equal-share deficit instead of min-vs-max spread.
        m = min(counts)
        spread = max(counts) - m
        if spread <= 0:
            return one_tile
        share = total / n
        boost = 1.0 + min(3.0, max(0.0, share - counts[i]) / max(share, 1.0) * 3.0)
        return one_tile * boost

    def decide(self, snapshot: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]] | None]:
        self.observe_progress(snapshot)
        now_night = str(snapshot.get("cursor", {}).get("night_id", ""))
        # Scarcity-adaptive patience (§4): short surveys cannot afford a high
        # quality floor — unfinished work per remaining night decides. Nights
        # seen so far approximate the remaining budget.
        nights_seen = len(self._night_slot_counts) or 1
        if self.sc.night_count:
            remaining_fraction = max(0.15, 1.0 - (nights_seen - 1) / max(1, self.sc.night_count))
        else:
            remaining_fraction = 1.0
        unfinished = max(0, self.sc.tile_count - len(self.hot_completed()))
        scarcity = unfinished / max(1.0, self.sc.tile_count * remaining_fraction)
        if self.sc.night_count <= 30:
            # Short surveys: low floor, loose patience — every completion counts.
            self.quality_floor = 0.40
            self.patience_ratio = 0.70
        else:
            # Long surveys: EARLY-style patience pays (measured 12095 @ 0.55/0.85).
            self.quality_floor = 0.55
            self.patience_ratio = 0.85
        offset = int(snapshot.get("cursor", {}).get("slot_offset_seconds", 0))
        self._night_slot_counts[now_night] = self._night_slot_counts.get(now_night, 0) + 1
        raw = [c for c in snapshot.get("candidate_tiles", [])
               if c.get("effective_weather", {}).get("is_observable")]
        now_iso = str(snapshot.get("cursor", {}).get("timestamp_utc", ""))
        candidates = [c for c in raw if self._fits_remaining(c, offset, now_night, now_iso)]
        # REQUIRED kept alive even when the crossing gate filters them: -1000
        # miss outweighs the -100 interruption risk (§4 step 3). But a shot the
        # tile's own window cannot cover is a guaranteed -100 with zero score —
        # only rescue crossing-gate rejects, not window-fits rejects.
        for c in raw:
            tile_id = str(c["tile_id"])
            if tile_id in self.sc.required_ids and c not in candidates \
                    and self._window_can_cover(c, now_iso):
                candidates.append(c)
        # Requested tiles with remaining visits bypass the completed-tile filter:
        # a visit on a completed tile is legal when it carries the request tag.
        for req in snapshot.get("active_requests", []):
            for r in req.get("tile_requirements", []):
                if int(r.get("remaining_visits", 0)) > 0:
                    tile_id = str(r["tile_id"])
                    for c in raw:
                        if str(c["tile_id"]) == tile_id and c not in candidates:
                            candidates.append(c)

        # 1. hard filters
        requested_now = set()
        for req in snapshot.get("active_requests", []):
            for r in req.get("tile_requirements", []):
                if int(r.get("remaining_visits", 0)) > 0:
                    requested_now.add(str(r["tile_id"]))
        pool = []
        for c in candidates:
            tile_id = str(c["tile_id"])
            done = bool(c.get("already_completed"))
            if tile_id in self.drop_tiles and not done:
                continue
            if done and not self.sc.mechanics and tile_id not in requested_now:
                continue
            pool.append(c)
        if not pool:
            return ({"action": "wait", "reason": "no observable candidate"}, None)

        # 2. due requests (expire soonest first; weight only breaks ties)
        due = []
        self._pending_request.clear()
        for req in snapshot.get("active_requests", []):
            remaining_nights = int(req.get("nights_remaining", 99)) if "nights_remaining" in req else None
            for r in req.get("tile_requirements", []):
                if int(r.get("remaining_visits", 0)) <= 0:
                    continue
                tile_id = str(r["tile_id"])
                match = [c for c in pool if str(c["tile_id"]) == tile_id]
                if match:
                    # remember WHICH request still needs this tile (§4 step 2)
                    self._pending_request[tile_id] = str(req["request_id"])
                    due.append((int(req.get("required_tile_count", 1)), tile_id, match[0]))
        if due:
            due.sort(key=lambda item: item[0])
            c = due[0][2]
            # A requested visit banks reward regardless of program band, but the
            # shot quality still sets the tile's banked science: prefer waiting
            # for a DARK-band night when the deadline allows (M1 coefficient:
            # quality delta is worth ~1000 s of wait per 0.01 of q at V~270).
            q_now = self._quality(c)
            if (not c.get("already_completed")) and q_now < self.sc.dark_threshold:
                try:
                    nights_left = self._window_left_nights(c, now_night)
                except ValueError:
                    nights_left = 99.0
                if nights_left > 3 and q_now < 0.85 * self._best_q.get(str(c["tile_id"]), q_now):
                    pass  # fall through to value ranking; request stays pending
                else:
                    return self._observe(c, snapshot, "due request")
            else:
                return self._observe(c, snapshot, "due request")

        # 3. REQUIRED with a closing window. The catalog deadline alone is not
        # urgency: T00041 (finals-preview) stays available until 10-11 but is
        # only observable in a 45-minute window on each of the first three
        # nights — by catalog arithmetic it always looks "6 nights left", so
        # its real chances silently expired. A REQUIRED whose window ends
        # within ~75 minutes is closing TONIGHT: shoot it. A miss is -1000
        # against -100 for one interrupted corner case.
        def _closes_soon(c: dict[str, Any]) -> bool:
            try:
                now = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
                end = datetime.fromisoformat(str(c.get("window_end_utc", "")).replace("Z", "+00:00"))
                return 0.0 <= (end - now).total_seconds() <= 4500.0
            except (TypeError, ValueError):
                return False

        urgent = [c for c in pool
                  if str(c["tile_id"]) in self.sc.required_ids
                  and (self._window_left_nights(c, now_night) <= 3 or _closes_soon(c))
                  and self._quality(c) > 0.05
                  and self._shot_count(str(c["tile_id"]), now_night) < 2]
        if urgent:
            def _fits_window(c: dict[str, Any]) -> bool:
                try:
                    now = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
                    end = datetime.fromisoformat(str(c.get("window_end_utc", "")).replace("Z", "+00:00"))
                    return now + timedelta(seconds=float(c.get("nominal_exptime_seconds", 900))
                                           + (0.0 if str(c.get("scheduling_class")) == "REQUIRED" else 120.0)) <= end
                except (TypeError, ValueError):
                    return True
            c = max(urgent, key=lambda c: (_fits_window(c), self._quality(c)))
            self._note_shot(str(c["tile_id"]), now_night)
            # REQUIRED miss is -1000 vs interruption -100: a REQUIRED that can
            # only finish by crossing slots takes the crossing risk.
            decision, reports = self._observe(c, snapshot, "REQUIRED closing")
            if decision["action"] == "observe":
                exptime = float(c.get("nominal_exptime_seconds", 900))
                if exptime > 900 - offset:
                    self._night_confirmed_midnight_shot[now_night] = True
            return decision, reports

        # 3.5 anomaly suspects (§7): a tag settlement needs in-band reads on two
        # nights, and a suspect whose banked score blocks a positive-gain
        # repeat would otherwise never be revisited. The expected gain here
        # includes the +100 settlement, so this is still a "gain > 0" repeat —
        # bounded to MAX_EXTRA_VISITS per tile by the watch. The visit yields
        # to unfinished work above the quality floor: a +100 tag never
        # justifies a −100 quota shortfall, and the tag stays winnable on
        # later nights (or at night edges when quality dips anyway).
        if self.tag_suspects and not any(
                not c.get("already_completed")
                and self._quality(c) >= self.quality_floor for c in pool):
            for c in pool:
                if str(c["tile_id"]) in self.tag_suspects \
                        and self._fits_remaining(c, offset, now_night, now_iso):
                    decision, reports = self._observe(c, snapshot, "anomaly suspect extra visit")
                    if decision["action"] == "observe":
                        exptime = float(c.get("nominal_exptime_seconds", 900))
                        if exptime > 900 - offset:
                            self._night_confirmed_midnight_shot[now_night] = True
                    return decision, reports

        # 4. value-per-second with marginal coverage (M1 coefficient applied to repeats)
        # Pass 1 ranks only candidates that clear the normal floor; pass 2
        # (catch-up waiver) fires ONLY when pass 1 finds nothing shootable —
        # the waiver exists to fill twilight slots that would otherwise burn a
        # wait, not to out-rank legitimate work. On finals-preview the lagging
        # region's q=0.374 shot was displacing a q=0.436 completion (−222 base,
        # and the displaced slot broke an anomaly report chain, −100).
        best, best_score = None, 0.0
        waiver_best, waiver_score = None, 0.0
        for c in pool:
            tile_id = str(c["tile_id"])
            q = self._quality(c)
            if q < 0:
                continue
            # M1 patience: waiting is cheap (32.5 pts buys 9 h). Skip a tile
            # whose quality is below an absolute floor or far below the best
            # this tile has offered — unless requested (visit banks reward
            # regardless) or a REQUIRED closing tonight (step 3).
            seen = self._best_q.get(tile_id, 0.0)
            if q > seen:
                self._best_q[tile_id] = q
            waived = False
            if tile_id not in requested_now:
                # Catch-up waiver: a tile in a lagging region (its coverage
                # delta is positive) may be shot below the quality floor.
                # Twilight slots otherwise reject every candidate and burn a
                # wait; a lagging-region completion there is free evenness.
                # Long surveys only (see __init__ catchup note).
                waive = self.catchup and self._coverage_delta(str(c.get("region_id", ""))) > 0.0
                if q < (self.catchup_floor if waive else self.quality_floor):
                    continue
                if waive and q < self.quality_floor:
                    waived = True  # below-floor lagging shot: only a fallback
                elif not waive and seen > 0 and q < self.patience_ratio * seen:
                    continue
            program = self._program(q)
            gain = q * float(c.get("tile_science_value", 0.0)) * (1.0 + self._bonus(program))
            if self.sc.mechanics and c.get("already_completed"):
                gain -= self.banked.get(tile_id, 0.0)
                # Repeats need expected gain > 0. One extra visit is allowed for
                # an in-band unreported tile so the tag can span two nights.
                if gain <= 0 and tile_id not in self.tag_suspects:
                    continue
            if self.catchup:
                gain += self._coverage_delta(str(c.get("region_id", "")))
            if self.keep_regions and str(c.get("region_id", "")) in self.keep_regions:
                gain *= 1.01  # soft tiebreak only
            per_second = gain / max(1.0, float(c.get("nominal_exptime_seconds", 900)))
            if waived:
                if per_second > waiver_score:
                    waiver_best, waiver_score = c, per_second
            elif per_second > best_score:
                best, best_score = c, per_second
        if best is None:
            best, best_score = waiver_best, waiver_score
        if best is None:
            return ({"action": "wait", "reason": "no positive-gain candidate"}, None)
        exptime = float(best.get("nominal_exptime_seconds", 900))
        if exptime <= 900 - offset:
            # A shot that finishes inside this slot proves the night has
            # capacity beyond the current cursor position.
            self._night_confirmed_midnight_shot[now_night] = True
        return self._observe(best, snapshot, "value/sec")

    def _preview_score(self, cand: dict[str, Any]) -> float:
        """Public baseline of the exposure: V × quality × (1+bonus).

        No 900/exptime fraction — that made long exposures look like reddening
        (realized/preview fell into 0.65–0.85) and produced a −150 misreport.
        """
        q = self._quality(cand)
        if q < 0:
            return 0.0
        program = self._program(q)
        value = float(cand.get("tile_science_value", 0.0))
        return q * value * (1.0 + self._bonus(program))

    def _observe(self, cand: dict[str, Any], snapshot: dict[str, Any], why: str):
        tile_id = str(cand["tile_id"])
        q = self._quality(cand)
        request_id = self._pending_request.get(tile_id, "")
        self._last_preview[tile_id] = self._preview_score(cand)
        decision = {
            "action": "observe",
            "tile_id": tile_id,
            "program": self._program(q),
            "request_id": request_id,
            "reason": f"{why} q={q:.3f}",
            "decision_source": "deterministic",
        }
        return decision, None

    def _fits_remaining(self, cand: dict[str, Any], offset: int, night_id: str,
                        now_iso: str = "") -> bool:
        """Exposure must fit before the night ends (night-crossing is -100).

        The scorer kills any exposure still running at night end
        (geometry_or_night_interrupted). Without the night calendar in the
        snapshot, use the conservative rule: an exposure longer than the slot
        remainder is only allowed when the night certainly has another full
        slot after this one — approximated via the night's remaining capacity
        carried by the caller (slots_seen/night), else never allowed. Cold
        start: track how many slots each night has produced so far.
        Altitude-edge cases (>=30° now, dipping below mid-shot) also require
        slot remainder >= exptime when snapshot altitude < 40°.
        The candidate's own window_end_utc (same info the official preview
        uses) is checked first: a shot the window cannot cover is killed
        mid-exposure and scores zero.
        """
        try:
            exptime = float(cand.get("nominal_exptime_seconds", 900))
            altitude = float(cand.get("geometry", {}).get("altitude_deg", 90.0))
        except (TypeError, ValueError):
            return True
        if now_iso and str(cand.get("window_end_utc", "")):
            try:
                now = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
                end = datetime.fromisoformat(str(cand["window_end_utc"]).replace("Z", "+00:00"))
                # REQUIRED tiles take the exact boundary: their windows can be
                # as short as the exposure itself (V1 T00601 rises above the
                # altitude floor only in the night's last 15 min), and the
                # cushion otherwise hides a -1000 required_miss behind a
                # -100 interruption risk. The platform completes exposures
                # ending exactly at window_end (baseline proved it twice).
                margin = 0.0 if str(cand.get("scheduling_class")) == "REQUIRED" else 120.0
                if now + timedelta(seconds=exptime + margin) > end:
                    return False
            except (TypeError, ValueError):
                pass
        remaining = 900 - offset
        if exptime <= remaining:
            return True
        # Crossing into the next slot: the snapshot cannot tell whether another
        # slot follows in this night (last-slot shots die at night end, -100).
        # Long exposures (>900 s) cross by construction; with mechanics=false an
        # interrupted tile never completes, so retrying it every slot just burns
        # 100 each time. v1 rule: only start a >900 s exposure at a slot whose
        # remaining time plus one full slot (900) covers it — i.e. offset 0 —
        # and never in what might be the night's last slot. Since we cannot see
        # the calendar, use the night's request pattern: allow crossing at
        # offset 0 only before the (unknown) last slot; the workflow resets
        # offset to 0 at each new slot, so "offset 0" is every slot. Instead:
        # allow crossing only when this night already had a shot FINISH
        # mid-night (tracked), else wait.
        if offset == 0 and self._night_confirmed_midnight_shot.get(night_id):
            return altitude >= 40.0
        return False

    def _window_can_cover(self, cand: dict[str, Any], now_iso: str) -> bool:
        """True when the tile's own window can still cover a full exposure.
        Unknown window/timestamp counts as coverable (old behavior)."""
        if not now_iso or not str(cand.get("window_end_utc", "")):
            return True
        try:
            now = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(cand["window_end_utc"]).replace("Z", "+00:00"))
            return now + timedelta(seconds=float(cand.get("nominal_exptime_seconds", 900))
                                   + (0.0 if str(cand.get("scheduling_class")) == "REQUIRED" else 120.0)) <= end
        except (TypeError, ValueError):
            return True

    def _window_left_nights(self, cand: dict[str, Any], now_night: str) -> float:
        """Nights until the TILE's availability deadline (catalog, not nightly
        window): the candidate's window_end_utc resets every night."""
        try:
            night_num = int(now_night[1:9]) if len(now_night) >= 9 else 0
            tile_id = str(cand["tile_id"])
            deadline = self._available_until.get(tile_id, "")
            end_day = int(deadline) if deadline.isdigit() else 99991231
            return max(0.0, float(end_day - night_num))
        except (ValueError, IndexError):
            return 99.0
