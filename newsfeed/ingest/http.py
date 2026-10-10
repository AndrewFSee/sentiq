"""Rate-limited, retrying HTTP client shared by adapters."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class RateLimiter:
    """Minimum-interval limiter: at most ``rate`` acquisitions per second, thread-safe.

    Clock and sleep are injectable so tests run instantly and deterministically.
    """

    def __init__(
        self,
        rate: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.interval = 1.0 / rate
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            wait = self._next - now
            if wait > 0:
                self._sleep(wait)
                now += wait
            self._next = now + self.interval


class RetryableHTTPError(Exception):
    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"HTTP {response.status_code} for {response.request.url}")
        self.response = response


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, (RetryableHTTPError, httpx.TransportError))


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    log.warning("retry %d after %r", state.attempt_number, exc)


class HttpClient:
    def __init__(
        self,
        *,
        user_agent: str,
        rate_per_sec: float,
        timeout: float = 30.0,
        max_retries: int = 5,
        transport: httpx.BaseTransport | None = None,
        limiter: RateLimiter | None = None,
        retry_wait_initial: float = 1.0,
        retry_wait_max: float = 60.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._client = httpx.Client(
            headers={
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
                **(headers or {}),
            },
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )
        self.limiter = limiter or RateLimiter(rate_per_sec)
        self.request_count = 0
        self._get = retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential_jitter(
                multiplier=retry_wait_initial, max=retry_wait_max, jitter=retry_wait_initial
            ),
            before_sleep=_log_retry,
            reraise=True,
        )(self._get_once)

    def _get_once(self, url: str) -> httpx.Response:
        self.limiter.acquire()
        self.request_count += 1
        resp = self._client.get(url)
        if resp.status_code in RETRY_STATUS:
            raise RetryableHTTPError(resp)
        if resp.status_code == 403:
            log.error("403 from %s -- SEC blocks undeclared/abusive clients; check User-Agent", url)
        resp.raise_for_status()
        return resp

    def get(self, url: str) -> httpx.Response:
        return self._get(url)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
