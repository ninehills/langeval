import threading
import time
from types import TracebackType
from typing import Optional, Type


class ThreadingRateLimiter:
    def __init__(self, query_per_second: float):
        """Rate limiter for threading

        Args:
            query_per_second (float): query per second
        """
        self._query_per_second = query_per_second
        if query_per_second < 0:
            raise ValueError("query_per_second must be positive")
        if query_per_second == 0:
            # No rate limit
            return
        if query_per_second > 1:
            self._query_per_period = query_per_second
        else:
            self._query_per_period = 1
        self._token_count = self._query_per_period
        self._last_leak_timestamp = time.perf_counter()
        self._sync_lock = threading.Lock()

    def _leak(self) -> None:
        timestamp = time.perf_counter()
        delta = timestamp - self._last_leak_timestamp
        self._last_leak_timestamp = timestamp
        self._token_count = min(
            self._query_per_period,
            self._token_count + delta * self._query_per_second,
        )

    def __enter__(self) -> None:
        if self._query_per_second == 0:
            return
        with self._sync_lock:
            while True:
                self._leak()
                if self._token_count >= 1:
                    self._token_count -= 1
                    return
                time.sleep((1 - self._token_count) / self._query_per_second)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """
        exit
        """
        return
