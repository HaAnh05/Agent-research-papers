"""Per-host request pacing for arXiv-family hosts.

``arxiv.org/robots.txt`` asks for ``Crawl-delay: 15`` and the paper reader
runs several worker threads.  Every request to a paced host therefore goes
through one shared :class:`HostRateLimiter`:

* the limiter lock is held for the whole request (connect, headers, body), so
  at most one connection per host is open at a time;
* the next request may start only ``min_interval`` seconds after the previous
  request *started*.

The clock and the sleep function are injectable so tests never really sleep.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager, nullcontext
from typing import Callable, ContextManager, Dict, Iterator, Optional

from config import config

ARXIV_HOSTS = frozenset({"arxiv.org", "www.arxiv.org"})
AR5IV_HOSTS = frozenset({"ar5iv.labs.arxiv.org"})

_ARXIV_KEY = "arxiv.org"
_AR5IV_KEY = "ar5iv.labs.arxiv.org"


class HostRateLimiter:
    """Serialise requests to one host and space their starts apart."""

    def __init__(
        self,
        min_interval: Callable[[], float],
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._last_start: Optional[float] = None

    @contextmanager
    def slot(self) -> Iterator[None]:
        """Wait for the host's turn; hold it until the request is finished."""

        with self._lock:
            if self._last_start is not None:
                wait = self._last_start + max(0.0, float(self._min_interval())) - self._clock()
                if wait > 0:
                    self._sleep(wait)
            self._last_start = self._clock()
            yield


_registry_lock = threading.Lock()
_limiters: Dict[str, HostRateLimiter] = {}
_clock: Callable[[], float] = time.monotonic
_sleep: Callable[[float], None] = time.sleep


def configure(
    *,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> None:
    """Drop all limiter state and (optionally) install a clock and sleep.

    Tests call this with a fake clock.  Calling it with no arguments restores
    the real clock.
    """

    global _clock, _sleep
    with _registry_lock:
        _clock = clock or time.monotonic
        _sleep = sleep or time.sleep
        _limiters.clear()


def _interval_for(key: str) -> Callable[[], float]:
    if key == _AR5IV_KEY:
        return lambda: config.AR5IV_MIN_INTERVAL_SEC
    return lambda: config.ARXIV_HOST_MIN_INTERVAL_SEC


def limiter_for_host(host: str) -> Optional[HostRateLimiter]:
    """Return the shared limiter for a paced host, or ``None`` for other hosts."""

    name = (host or "").lower().rstrip(".")
    if name in ARXIV_HOSTS:
        key = _ARXIV_KEY
    elif name in AR5IV_HOSTS:
        key = _AR5IV_KEY
    else:
        return None
    with _registry_lock:
        limiter = _limiters.get(key)
        if limiter is None:
            limiter = HostRateLimiter(_interval_for(key), clock=_clock, sleep=_sleep)
            _limiters[key] = limiter
        return limiter


def host_slot(host: str) -> ContextManager[None]:
    """Context manager that paces one request to ``host`` (no-op for other hosts)."""

    limiter = limiter_for_host(host)
    return limiter.slot() if limiter is not None else nullcontext()
