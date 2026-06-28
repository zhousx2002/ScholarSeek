from __future__ import annotations

import json
import http.client
import os
import socket
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class ServiceRateLimited(RuntimeError):
    pass


class RequestGate:
    """Process-wide request spacing shared by concurrent API clients."""

    def __init__(self, min_interval: float):
        self.min_interval = max(0.0, min_interval)
        self._lock = threading.Lock()
        self._next_allowed = 0.0
        self._blocked_until = 0.0

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                if self._blocked_until > now:
                    raise ServiceRateLimited(
                        f"rate-limit circuit open for {self._blocked_until - now:.1f}s"
                    )
                delay = self._next_allowed - now
                if delay <= 0:
                    self._next_allowed = now + self.min_interval
                    return
            time.sleep(delay)

    def defer(self, seconds: float) -> None:
        with self._lock:
            self._next_allowed = max(self._next_allowed, time.monotonic() + max(0.0, seconds))

    def block(self, seconds: float) -> None:
        with self._lock:
            self._blocked_until = max(self._blocked_until, time.monotonic() + max(0.0, seconds))


def request_json(
    request: Request,
    *,
    service: str,
    gate: RequestGate,
    timeout: int,
    max_retries: int | None = None,
) -> dict[str, Any]:
    retries = max_retries if max_retries is not None else _env_int("SCHOLARSEEK_HTTP_MAX_RETRIES", 2)
    max_retry_wait = _env_float("SCHOLARSEEK_HTTP_MAX_RETRY_WAIT", 20.0)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            gate.wait()
        except ServiceRateLimited as exc:
            raise ServiceRateLimited(f"{service} temporarily disabled: {exc}") from exc
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            last_error = exc
            if exc.code not in RETRYABLE_STATUS_CODES:
                break
            delay = _retry_delay(exc, attempt)
            if exc.code == 429 and delay > max_retry_wait:
                gate.block(delay)
                raise ServiceRateLimited(
                    f"{service} rate limited; server requested retry after {delay:.1f}s"
                ) from exc
            if attempt >= retries:
                if exc.code == 429:
                    gate.block(max(60.0, delay))
                    raise ServiceRateLimited(
                        f"{service} rate limited after {attempt + 1} attempts"
                    ) from exc
                break
        except (
            URLError,
            TimeoutError,
            socket.timeout,
            ConnectionError,
            http.client.HTTPException,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if attempt >= retries:
                break
            delay = min(30.0, 2.0**attempt)
        gate.defer(delay)
        print(
            f"[warn] {service} request throttled or unavailable; "
            f"retrying in {delay:.1f}s ({attempt + 1}/{retries})",
            flush=True,
        )
    raise RuntimeError(f"{service} request failed after {retries + 1} attempts: {last_error}") from last_error


def _retry_delay(error: HTTPError, attempt: int) -> float:
    retry_after = (error.headers or {}).get("Retry-After")
    parsed = _parse_retry_after(retry_after)
    if parsed is not None:
        return min(120.0, max(1.0, parsed))
    return min(60.0, 2.0 * (2**attempt))


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


OPENALEX_GATE = RequestGate(_env_float("SCHOLARSEEK_OPENALEX_MIN_INTERVAL", 0.35))
SEMANTIC_SCHOLAR_GATE = RequestGate(_env_float("SCHOLARSEEK_S2_MIN_INTERVAL", 0.75))
SEMANTIC_SCHOLAR_PUBLIC_GATE = RequestGate(_env_float("SCHOLARSEEK_S2_PUBLIC_MIN_INTERVAL", 2.5))


def semantic_scholar_gate(api_key: str | None) -> RequestGate:
    return SEMANTIC_SCHOLAR_GATE if api_key else SEMANTIC_SCHOLAR_PUBLIC_GATE
