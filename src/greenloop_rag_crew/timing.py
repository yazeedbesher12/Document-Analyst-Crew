"""Safe request and task timing utilities without prompt or content logging."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter

LOGGER = logging.getLogger(__name__)


@dataclass
class RequestTiming:
    """Track named stages for one fresh CrewAI request."""

    _started: dict[str, float] = field(default_factory=dict)
    durations: dict[str, float] = field(default_factory=dict)
    request_started: float = field(default_factory=perf_counter)

    def begin(self, stage: str) -> None:
        self._started[stage] = perf_counter()

    def finish(self, stage: str) -> float:
        started = self._started.pop(stage, None)
        if started is None:
            return self.durations.get(stage, 0.0)
        elapsed = perf_counter() - started
        self.durations[stage] = elapsed
        LOGGER.info("timing event=%s elapsed_seconds=%.3f", stage, elapsed)
        return elapsed

    def finish_request(self) -> float:
        elapsed = perf_counter() - self.request_started
        self.durations["total_request_execution"] = elapsed
        LOGGER.info("timing event=total_request_execution elapsed_seconds=%.3f", elapsed)
        return elapsed

    def snapshot(self) -> dict[str, float]:
        return dict(self.durations)
