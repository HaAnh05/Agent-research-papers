"""Shared fixtures: keep every test off the real clock and the network pacing."""

import pytest

from tools import rate_limit


@pytest.fixture(autouse=True)
def _fake_rate_limit_clock():
    """Replace the arXiv host limiters' clock so no test ever really sleeps."""

    state = {"now": 0.0}

    def now() -> float:
        return state["now"]

    def sleep(seconds: float) -> None:
        state["now"] += seconds

    rate_limit.configure(clock=now, sleep=sleep)
    yield
    rate_limit.configure()
