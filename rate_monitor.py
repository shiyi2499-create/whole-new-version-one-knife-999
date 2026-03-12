"""
Rate Monitor - Sampling rate watchdog
======================================
Monitors actual sensor sampling rate using a simple counter approach.
If rate drops below threshold (default 96Hz) for consecutive checks,
triggers alert and signals the collector to stop.

FIXED: uses counter-based calculation instead of per-sample timestamps
to avoid inflated rates when samples arrive in batches.
"""

import time
import threading
from typing import Callable, Optional


class RateMonitor:
    """
    Monitors sensor sampling rate by counting samples per time interval.

    Call tick(count) whenever you drain N sensor samples.
    The monitor checks the rate every check_interval seconds.
    """

    def __init__(
        self,
        min_rate_hz: int = 96,
        check_interval_sec: float = 1.0,
        on_rate_drop: Optional[Callable[[float], None]] = None,
    ):
        self.min_rate_hz = min_rate_hz
        self.check_interval = check_interval_sec
        self.on_rate_drop = on_rate_drop

        # Simple counter: samples received since last check
        self._counter = 0
        self._counter_lock = threading.Lock()

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Public state
        self.current_rate: float = 0.0
        self.rate_ok: bool = True
        self.rate_history: list[tuple[float, float]] = []  # (wall_time, rate)
        self._consecutive_low: int = 0
        self._started_at: float = 0.0

    def start(self):
        self._running = True
        self.rate_ok = True
        self._consecutive_low = 0
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def tick(self, count: int = 1):
        """Call this whenever you drain count samples from the sensor."""
        with self._counter_lock:
            self._counter += count

    def _check_loop(self):
        """Every check_interval, compute rate = samples_since_last / elapsed."""
        last_check = time.perf_counter()

        while self._running:
            time.sleep(self.check_interval)
            now = time.perf_counter()
            elapsed = now - last_check
            last_check = now

            # Grab and reset counter
            with self._counter_lock:
                count = self._counter
                self._counter = 0

            # Compute rate for this interval
            rate = count / elapsed if elapsed > 0 else 0.0
            self.current_rate = rate
            self.rate_history.append((time.time(), rate))

            # Skip threshold check for the first 4 seconds (warmup)
            uptime = time.time() - self._started_at
            if uptime < 4.0:
                continue

            # Check threshold: require 3 consecutive low readings
            if 0 < rate < self.min_rate_hz:
                self._consecutive_low += 1
                if self._consecutive_low >= 3:
                    self.rate_ok = False
                    if self.on_rate_drop:
                        self.on_rate_drop(rate)
            else:
                self._consecutive_low = 0

    def get_rate_summary(self) -> dict:
        """Return summary statistics of rate history."""
        if not self.rate_history:
            return {"min": 0, "max": 0, "avg": 0, "current": 0}
        # Exclude warmup readings (first 4 seconds) and zero-rate readings
        cutoff = self._started_at + 4.0
        rates = [r for t, r in self.rate_history if r > 0 and t > cutoff]
        if not rates:
            return {"min": 0, "max": 0, "avg": 0, "current": self.current_rate}
        return {
            "min": min(rates),
            "max": max(rates),
            "avg": sum(rates) / len(rates),
            "current": self.current_rate,
        }