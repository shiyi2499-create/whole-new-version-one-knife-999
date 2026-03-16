"""
Sensor Reader - Continuous IMU data acquisition
================================================
Wraps the SPU direct IOKit backend (non-root) or the macimu library (root
fallback) to continuously read accelerometer + gyroscope data from the
Apple Silicon SPU sensor.

Backend priority:
  1. spu_backend (direct AppleSPUHIDDevice IOKit path) — no root needed
  2. macimu (legacy) — requires root; used only when spu_backend is
     unavailable or explicitly requested via force_macimu=True

Requires: macOS on Apple Silicon with BMI286 IMU
Optional: pip install macimu   (only for legacy fallback)
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

    Uses the direct IOKit SPU backend by default (non-root).
    Falls back to macimu if the SPU backend is unavailable.
    """

    def __init__(self, buffer_maxlen: int = 500_000, force_macimu: bool = False):
        self._buffer: deque[SensorSample] = deque(maxlen=buffer_maxlen)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._total_samples = 0
        self._force_macimu = force_macimu
        self._backend_name: str = "none"

        # SPU direct backend
        self._spu = None

        # Legacy macimu backend
        self._imu = None

        # Cache last known good readings (fallback when a sensor returns None)
        # Used only by the macimu poll path
        self._last_accel = (0.0, 0.0, 0.0)
        self._last_gyro = (0.0, 0.0, 0.0)

    @property
    def backend_name(self) -> str:
        """Return which backend is active: 'spu_direct', 'macimu', or 'none'."""
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
                print("[SensorReader] Using direct SPU IOKit backend (non-root)")
                return
            except Exception as e:
                print(f"[SensorReader] SPU backend failed: {e}")
                print("[SensorReader] Falling back to macimu...")
                self._spu = None

        # Fallback: macimu (requires root)
        try:
            from macimu import IMU
            self._imu = IMU()
            self._imu.__enter__()
            self._backend_name = "macimu"
            self._running = True
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
            print("[SensorReader] Using macimu backend (requires root)")
        except ImportError:
            raise RuntimeError(
                "Neither SPU direct backend nor macimu is available. "
                "On Apple Silicon Mac, the SPU backend should work without root. "
                "Check that this is macOS on Apple Silicon."
            )

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

    # ── SPU direct backend drain loop ────────────────────────

    def _spu_drain_loop(self):
        """
        Drain samples from SPUBackend and fuse accel+gyro into
        SensorSample records matching the existing schema.

        The SPU backend delivers separate accel and gyro callbacks.
        We fuse them by pairing the most recent value from each sensor
        at each callback timestamp — the same "last known good" strategy
        used by the original macimu path.

        Every output sample comes from a real sensor callback.
        No interpolation or synthetic timestamps.
        """
        last_accel = (0.0, 0.0, 0.0)
        last_gyro = (0.0, 0.0, 0.0)

        while self._running:
            raw = self._spu.drain()
            if not raw:
                time.sleep(0.002)
                continue

            fused = []
            for s in raw:
                if s.sensor == "accel":
                    last_accel = (s.x, s.y, s.z)
                else:
                    last_gyro = (s.x, s.y, s.z)

                fused.append(SensorSample(
                    timestamp_ns=s.timestamp_ns,
                    accel_x=last_accel[0],
                    accel_y=last_accel[1],
                    accel_z=last_accel[2],
                    gyro_x=last_gyro[0],
                    gyro_y=last_gyro[1],
                    gyro_z=last_gyro[2],
                ))

            if fused:
                with self._lock:
                    self._buffer.extend(fused)
                    self._total_samples += len(fused)

    # ── Legacy macimu poll loop ──────────────────────────────

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

    # ── Public API (unchanged) ───────────────────────────────

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
