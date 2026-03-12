"""
Sensor Reader - Continuous IMU data acquisition
================================================
Wraps the macimu library to continuously read accelerometer + gyroscope data
from the Apple Silicon SPU sensor. Runs in a dedicated thread.

Requires: pip install macimu   (and sudo to run)
"""

import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class SensorSample:
    """One fused sensor reading with both accel and gyro."""
    timestamp_ns: int          # perf_counter_ns
    accel_x: float             # g
    accel_y: float             # g
    accel_z: float             # g
    gyro_x: float              # deg/s
    gyro_y: float              # deg/s
    gyro_z: float              # deg/s


class SensorReader:
    """
    Continuously reads accelerometer + gyroscope from Apple Silicon SPU.
    """

    def __init__(self, buffer_maxlen: int = 500_000):
        self._buffer: deque[SensorSample] = deque(maxlen=buffer_maxlen)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._imu = None
        self._total_samples = 0
        # Cache last known good readings (fallback when a sensor returns None)
        self._last_accel = (0.0, 0.0, 0.0)
        self._last_gyro = (0.0, 0.0, 0.0)

    def start(self):
        from macimu import IMU
        self._imu = IMU()
        self._imu.__enter__()
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._imu:
            self._imu.__exit__(None, None, None)
            self._imu = None

    def _safe_accel(self, sample) -> tuple[float, float, float]:
        """Extract accel xyz; cache and fall back to last good value if None."""
        if sample is not None and hasattr(sample, 'x') and sample.x is not None:
            self._last_accel = (sample.x, sample.y, sample.z)
        return self._last_accel

    def _safe_gyro(self, sample) -> tuple[float, float, float]:
        """Extract gyro xyz; cache and fall back to last good value if None."""
        if sample is not None and hasattr(sample, 'x') and sample.x is not None:
            self._last_gyro = (sample.x, sample.y, sample.z)
        return self._last_gyro

    def _poll_loop(self):
        while self._running:
            try:
                accel_samples = list(self._imu.read_accel())
                gyro_samples = list(self._imu.read_gyro())

                fused = []
                max_len = max(len(accel_samples), len(gyro_samples))

                for i in range(max_len):
                    ts = time.perf_counter_ns()
                    a_raw = accel_samples[i] if i < len(accel_samples) else None
                    g_raw = gyro_samples[i] if i < len(gyro_samples) else None
                    ax, ay, az = self._safe_accel(a_raw)
                    gx, gy, gz = self._safe_gyro(g_raw)
                    fused.append(SensorSample(
                        timestamp_ns=ts,
                        accel_x=ax, accel_y=ay, accel_z=az,
                        gyro_x=gx, gyro_y=gy, gyro_z=gz,
                    ))

                if fused:
                    with self._lock:
                        self._buffer.extend(fused)
                        self._total_samples += len(fused)

            except Exception as e:
                # Silently skip None-related errors; log others
                msg = str(e)
                if "NoneType" not in msg:
                    print(f"[SensorReader] Error: {e}")

            time.sleep(0.002)

    def drain(self) -> list[SensorSample]:
        with self._lock:
            samples = list(self._buffer)
            self._buffer.clear()
        return samples

    def peek_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def total_samples(self) -> int:
        return self._total_samples