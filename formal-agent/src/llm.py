"""LLM access layer (§6.3): OpenAI-compatible chat via stdlib urllib.

- Endpoint/keys come from the platform-injected OPENAI_BASE_URL / OPENAI_API_KEY
  (relay or encrypted store — both transparent to us).
- Timeout 30 s, one retry, then give up: the deterministic hot path is always
  the fallback (C5). No exception may escape this module.
- Guards (§11): ≤400 calls per scenario, stop calling when wall-clock reserve
  drops under 600 s, and a 4-consecutive-failure breaker for unstable relays.
- MODEL_PROVIDER=deterministic (local dev) or a missing base URL disables the
  network entirely and returns None — callers treat that as "no model".
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

TIMEOUT_SECONDS = 30.0  # grok-4.7-class reasoning models often need 10-25 s
WALLCLOCK_RESERVE = 600.0  # §11: inside this reserve, run deterministic only
MAX_CONSECUTIVE_FAILURES = 4  # breaker: an unstable relay must not burn budget


class LlmClient:
    def __init__(self, log=None):
        self.log = log
        self.base_url = str(os.environ.get("OPENAI_BASE_URL", "")).rstrip("/")
        self.api_key = str(os.environ.get("OPENAI_API_KEY", ""))
        self.model = str(os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
        provider = str(os.environ.get("MODEL_PROVIDER", "")).lower()
        # Deterministic mode or missing endpoint: never touch the network.
        self.enabled = provider != "deterministic" and bool(self.base_url) and bool(self.api_key)
        self.wallclock_budget = 7200.0  # overwritten by initialize()
        self._started = time.monotonic()
        self.calls = 0            # completed chat calls
        self.failures = 0         # calls that produced nothing
        self.disabled_by_budget = False
        self.disabled_by_clock = False
        self.disabled_by_breaker = False
        self._consecutive_failures = 0

    def remaining_seconds(self) -> float:
        """Wall clock left, per spec §11 (budget from initialize)."""
        return self.wallclock_budget - (time.monotonic() - self._started)

    def allow_call(self) -> bool:
        if self.calls >= 400:  # V3 pass criterion: ≤400 calls per scenario
            if not self.disabled_by_budget and self.log:
                self.log(f"llm: call budget (400) reached, model layer off")
            self.disabled_by_budget = True
            return False
        if self.remaining_seconds() < WALLCLOCK_RESERVE:  # §11 guard
            if not self.disabled_by_clock and self.log:
                self.log(f"llm: wall-clock reserve <{WALLCLOCK_RESERVE:.0f}s, "
                         "model layer off")
            self.disabled_by_clock = True
            return False
        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            if not self.disabled_by_breaker and self.log:
                self.log(f"llm: {MAX_CONSECUTIVE_FAILURES} consecutive failures, "
                         "model layer off (deterministic fallback)")
            self.disabled_by_breaker = True  # relay unstable: stop retrying
            return False
        return True

    def _record(self, ok: bool) -> None:
        self._consecutive_failures = 0 if ok else self._consecutive_failures + 1

    def chat(self, system: str, user: str, max_tokens: int = 200) -> dict | None:
        """One chat completion; parsed JSON object or None. Never raises."""
        if not self.enabled:
            return None
        if not self.allow_call():
            return None
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}
        payload = None
        for attempt in (1, 2):  # one retry after a failure
            self.calls += 1
            try:
                request = urllib.request.Request(
                    self.base_url + "/chat/completions", data=body, headers=headers)
                with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except (OSError, ValueError, urllib.error.URLError,
                    urllib.error.HTTPError, KeyError) as exc:
                if self.log:
                    self.log(f"llm: attempt {attempt} failed: {type(exc).__name__}")
                payload = None
        if payload is None:
            self.failures += 1
            self._record(False)
            return None
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            if self.log:
                self.log(f"llm: malformed envelope: {type(exc).__name__}")
            self.failures += 1
            self._record(False)
            return None
        # Strip optional markdown fences; models often wrap JSON in ``` blocks.
        text = str(content).strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            obj = json.loads(text)
        except ValueError as exc:
            if self.log:
                self.log(f"llm: unparsable reply: {type(exc).__name__}")
            self.failures += 1
            self._record(False)
            return None
        self._record(True)
        return obj if isinstance(obj, dict) else None
