"""Per-session inactivity scheduling for bounded autonomous refinement.

The coordinator owns asyncio tasks only; durable truth remains DesignState.
Tasks are cancellable on user activity and are never allowed to exceed the
session's persisted refinement budget.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

log = logging.getLogger(__name__)


class SessionCoordinator:
    def __init__(self, *, inactivity_seconds: float = 30.0) -> None:
        if inactivity_seconds <= 0:
            raise ValueError("inactivity_seconds must be positive")
        self.inactivity_seconds = inactivity_seconds
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stopped = False

    async def stop(self) -> None:
        self._stopped = True
        tasks = list(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def user_activity(self, session_id: str, callback: Callable[[], Awaitable[None]]) -> None:
        """Debounce a self-review after the latest user message."""
        self.cancel(session_id)
        if not self._stopped:
            self._tasks[session_id] = asyncio.create_task(self._wait_then_run(session_id, callback))

    def cancel(self, session_id: str) -> None:
        task = self._tasks.pop(session_id, None)
        if task is not None:
            task.cancel()

    async def _wait_then_run(self, session_id: str, callback: Callable[[], Awaitable[None]]) -> None:
        try:
            await asyncio.sleep(self.inactivity_seconds)
            if not self._stopped:
                await callback()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("session_self_review_failed", extra={"session_id": session_id})
        finally:
            self._tasks.pop(session_id, None)
