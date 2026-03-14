"""
Sensor Reader - Continuous IMU data acquisition
================================================
Uses the direct AppleSPUHIDDevice backend by default and falls back to
macimu when requested or when the direct path is unavailable.

Important semantic choice for this trial:
- The direct backend receives separate accel and gyro callbacks
- We pair one accel callback with one gyro callback before emitting one
  collector-visible SensorSample
- This keeps the output stream close to the legacy ~200 Hz fused shape
  instead of exposing the raw ~400 rows/sec callback stream
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class SensorSample:
    """One fused sensor reading with both accel and gyro."""
    timestamp_ns: int
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float


class SensorReader:
    """
    Continuously reads accelerometer + gyroscope from Apple Silicon SPU.

    Default path:
      - direct SPU backend via AppleSPUHIDDevice (non-root)

    Optional fallback:
      - macimu (legacy root path)
    """

    def __init__(self, buffer_maxlen: int = 500_000, force_macimu: bool = False):
        self._buffer: deque[SensorSample] = deque(maxlen=buffer_maxlen)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._total_samples = 0
        self._force_macimu = force_macimu
        self._backend_name: str = "none"

        self._spu = None
        self._imu = None

        self._last_accel = (0.0, 0.0, 0.0)
        self._last_gyro = (0.0, 0.0, 0.0)

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def start(self):
        if self._running:
            return

        if not self._force_macimu:
            try:
                from spu_backend import SPUBackend

                self._spu = SPUBackend(buffer_maxlen=500_000)
                self._spu.start()
                self._backend_name = "spu_direct"
                self._running = True
                self._thread = threading.Thread(target=self._spu_drain_loop, daemon=True)
                self._thread.start()
                print("[SensorReader] Using direct SPU IOKit backend (non-root, paired ~200Hz stream)")
                return
            except Exception as e:
                print(f"[SensorReader] SPU backend failed: {e}")
                print("[SensorReader] Falling back to macimu...")
                self._spu = None

        from macimu import IMU

        self._imu = IMU()
        self._imu.__enter__()
        self._backend_name = "macimu"
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print("[SensorReader] Using macimu backend (legacy root path)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._spu:
            self._spu.stop()
            self._spu = None
        if self._imu:
            self._imu.__exit__(None, None, None)
            self._imu = None
        self._backend_name = "none"

    def _spu_drain_loop(self):
        """
        Convert separate accel/gyro callbacks into one fused sample per pair.

        The emitted timestamp is the later callback time of the paired accel/gyro
        events, which keeps us in the same perf_counter_ns clock domain as the
        keyboard listener without inventing synthetic timestamps.
        """
        accel_queue = deque()
        gyro_queue = deque()

        while self._running:
            raw = self._spu.drain()
            if not raw:
                time.sleep(0.002)
                continue

            for s in raw:
                if s.sensor == "accel":
                    accel_queue.append(s)
                else:
                    gyro_queue.append(s)

            fused = []
            while accel_queue and gyro_queue:
                a = accel_queue.popleft()
                g = gyro_queue.popleft()
                fused.append(
                    SensorSample(
                        timestamp_ns=max(a.timestamp_ns, g.timestamp_ns),
                        accel_x=a.x,
                        accel_y=a.y,
                        accel_z=a.z,
                        gyro_x=g.x,
                        gyro_y=g.y,
                        gyro_z=g.z,
                    )
                )

            if fused:
                with self._lock:
                    self._buffer.extend(fused)
                    self._total_samples += len(fused)

    def _safe_accel(self, sample) -> tuple[float, float, float]:
        """Extract accel xyz; cache and fall back to last good value if None."""
        if sample is not None and hasattr(sample, "x") and sample.x is not None:
            self._last_accel = (sample.x, sample.y, sample.z)
        return self._last_accel

    def _safe_gyro(self, sample) -> tuple[float, float, float]:
        """Extract gyro xyz; cache and fall back to last good value if None."""
        if sample is not None and hasattr(sample, "x") and sample.x is not None:
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
                    fused.append(
                        SensorSample(
                            timestamp_ns=ts,
                            accel_x=ax,
                            accel_y=ay,
                            accel_z=az,
                            gyro_x=gx,
                            gyro_y=gy,
                            gyro_z=gz,
                        )
                    )

                if fused:
                    with self._lock:
                        self._buffer.extend(fused)
                        self._total_samples += len(fused)

            except Exception as e:
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
