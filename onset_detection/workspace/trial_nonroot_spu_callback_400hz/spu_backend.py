"""
SPU Backend - Direct IOKit HID access to Apple Silicon IMU
==========================================================
Reads accelerometer (usage 3) and gyroscope (usage 9) from
AppleSPUHIDDevice via the direct IOKit service path.

This backend does NOT require root.  It bypasses the IOHIDManager
layer (which is gated by Input Monitoring on Tahoe) and goes
straight to IOServiceMatching("AppleSPUHIDDevice").

Verified on:
  - macOS 26 Tahoe (M4) - non-root, Input Monitoring OFF
  - See /权限问题/README.md for full A/B evidence

Reference C implementation:
  - exp7_phase2_capture.c  (phase-2 compatible CSV emitter)
  - exp2_iokit_imu.c       (minimal PoC)
"""

import ctypes
import ctypes.util
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

# ── Framework loading ────────────────────────────────────────

_cf_path = ctypes.util.find_library("CoreFoundation")
_iokit_path = ctypes.util.find_library("IOKit")

if not _cf_path or not _iokit_path:
    raise ImportError(
        "CoreFoundation or IOKit framework not found. "
        "This backend only works on macOS."
    )

CF = ctypes.cdll.LoadLibrary(_cf_path)
IOKit = ctypes.cdll.LoadLibrary(_iokit_path)

# ── Type aliases ─────────────────────────────────────────────

CFAllocatorRef = ctypes.c_void_p
CFTypeRef = ctypes.c_void_p
CFStringRef = ctypes.c_void_p
CFNumberRef = ctypes.c_void_p
CFDictionaryRef = ctypes.c_void_p
CFMutableDictionaryRef = ctypes.c_void_p
CFRunLoopRef = ctypes.c_void_p
CFRunLoopMode = ctypes.c_void_p
CFIndex = ctypes.c_long
CFStringEncoding = ctypes.c_uint32
CFNumberType = ctypes.c_uint32
CFTypeID = ctypes.c_ulong

IOReturn = ctypes.c_int32
IOOptionBits = ctypes.c_uint32
io_iterator_t = ctypes.c_uint32
io_service_t = ctypes.c_uint32
io_registry_entry_t = ctypes.c_uint32
io_object_t = ctypes.c_uint32
kern_return_t = ctypes.c_int32
IOHIDDeviceRef = ctypes.c_void_p
IOHIDReportType = ctypes.c_int32

IO_OBJECT_NULL = 0
KERN_SUCCESS = 0
kIOReturnSuccess = 0
kIOHIDOptionsTypeNone = 0
kCFStringEncodingUTF8 = 0x08000100
kCFNumberSInt32Type = 3
kCFAllocatorDefault = None

# ── CoreFoundation function signatures ───────────────────────

CF.CFRunLoopGetCurrent.restype = CFRunLoopRef
CF.CFRunLoopGetCurrent.argtypes = []

CF.CFRunLoopRunInMode.restype = ctypes.c_int32
CF.CFRunLoopRunInMode.argtypes = [CFRunLoopMode, ctypes.c_double, ctypes.c_bool]

CF.CFRunLoopStop.restype = None
CF.CFRunLoopStop.argtypes = [CFRunLoopRef]

CF.CFStringCreateWithCString.restype = CFStringRef
CF.CFStringCreateWithCString.argtypes = [CFAllocatorRef, ctypes.c_char_p, CFStringEncoding]

CF.CFNumberCreate.restype = CFNumberRef
CF.CFNumberCreate.argtypes = [CFAllocatorRef, CFNumberType, ctypes.c_void_p]

CF.CFNumberGetValue.restype = ctypes.c_bool
CF.CFNumberGetValue.argtypes = [CFNumberRef, CFNumberType, ctypes.c_void_p]

CF.CFRelease.restype = None
CF.CFRelease.argtypes = [CFTypeRef]

CF.CFGetTypeID.restype = CFTypeID
CF.CFGetTypeID.argtypes = [CFTypeRef]

CF.CFNumberGetTypeID.restype = CFTypeID
CF.CFNumberGetTypeID.argtypes = []

CF.CFDictionaryCreateMutable.restype = CFMutableDictionaryRef
CF.CFDictionaryCreateMutable.argtypes = [
    CFAllocatorRef, CFIndex, ctypes.c_void_p, ctypes.c_void_p,
]

CF.CFDictionarySetValue.restype = None
CF.CFDictionarySetValue.argtypes = [CFMutableDictionaryRef, ctypes.c_void_p, ctypes.c_void_p]

# Get the default key/value callbacks from CF
try:
    _kCFTypeDictionaryKeyCallBacks = ctypes.c_void_p.in_dll(CF, "kCFTypeDictionaryKeyCallBacks")
    _kCFTypeDictionaryValueCallBacks = ctypes.c_void_p.in_dll(CF, "kCFTypeDictionaryValueCallBacks")
except (ValueError, AttributeError):
    _kCFTypeDictionaryKeyCallBacks = None
    _kCFTypeDictionaryValueCallBacks = None

# ── IOKit function signatures ────────────────────────────────

IOKit.IOServiceMatching.restype = CFMutableDictionaryRef
IOKit.IOServiceMatching.argtypes = [ctypes.c_char_p]

IOKit.IOServiceGetMatchingServices.restype = kern_return_t
IOKit.IOServiceGetMatchingServices.argtypes = [
    ctypes.c_uint32, CFDictionaryRef, ctypes.POINTER(io_iterator_t),
]

IOKit.IOIteratorNext.restype = io_object_t
IOKit.IOIteratorNext.argtypes = [io_iterator_t]

IOKit.IOObjectRelease.restype = kern_return_t
IOKit.IOObjectRelease.argtypes = [io_object_t]

IOKit.IORegistryEntryCreateCFProperty.restype = CFTypeRef
IOKit.IORegistryEntryCreateCFProperty.argtypes = [
    io_registry_entry_t, CFStringRef, CFAllocatorRef, IOOptionBits,
]

IOKit.IORegistryEntrySetCFProperty.restype = kern_return_t
IOKit.IORegistryEntrySetCFProperty.argtypes = [
    io_registry_entry_t, CFStringRef, CFTypeRef,
]

IOKit.IOHIDDeviceCreate.restype = IOHIDDeviceRef
IOKit.IOHIDDeviceCreate.argtypes = [CFAllocatorRef, io_service_t]

IOKit.IOHIDDeviceOpen.restype = IOReturn
IOKit.IOHIDDeviceOpen.argtypes = [IOHIDDeviceRef, IOOptionBits]

IOKit.IOHIDDeviceClose.restype = IOReturn
IOKit.IOHIDDeviceClose.argtypes = [IOHIDDeviceRef, IOOptionBits]

# IOHIDReportCallback signature:
#   void (*)(void *context, IOReturn result, void *sender,
#            IOHIDReportType type, uint32_t reportID,
#            uint8_t *report, CFIndex reportLength)
IOHIDReportCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,   # context
    IOReturn,           # result
    ctypes.c_void_p,   # sender
    IOHIDReportType,    # type
    ctypes.c_uint32,    # reportID
    ctypes.POINTER(ctypes.c_uint8),  # report
    CFIndex,            # reportLength
)

IOKit.IOHIDDeviceRegisterInputReportCallback.restype = None
IOKit.IOHIDDeviceRegisterInputReportCallback.argtypes = [
    IOHIDDeviceRef,
    ctypes.POINTER(ctypes.c_uint8),  # report buffer
    CFIndex,                          # report buffer length
    IOHIDReportCallback,              # callback
    ctypes.c_void_p,                  # context
]

IOKit.IOHIDDeviceScheduleWithRunLoop.restype = None
IOKit.IOHIDDeviceScheduleWithRunLoop.argtypes = [
    IOHIDDeviceRef, CFRunLoopRef, CFRunLoopMode,
]

IOKit.IOHIDDeviceUnscheduleFromRunLoop.restype = None
IOKit.IOHIDDeviceUnscheduleFromRunLoop.argtypes = [
    IOHIDDeviceRef, CFRunLoopRef, CFRunLoopMode,
]

# kIOMainPortDefault (was kIOMasterPortDefault)
try:
    _kIOMainPortDefault = ctypes.c_uint32.in_dll(IOKit, "kIOMasterPortDefault").value
except (ValueError, AttributeError):
    _kIOMainPortDefault = 0

# kCFRunLoopDefaultMode
try:
    _kCFRunLoopDefaultMode = CFRunLoopMode.in_dll(CF, "kCFRunLoopDefaultMode")
except (ValueError, AttributeError):
    _kCFRunLoopDefaultMode = CF.CFStringCreateWithCString(
        kCFAllocatorDefault, b"kCFRunLoopDefaultMode", kCFStringEncodingUTF8
    )

# ── Helper: create CFString ─────────────────────────────────

def _cfstr(s: str) -> CFStringRef:
    return CF.CFStringCreateWithCString(
        kCFAllocatorDefault, s.encode("utf-8"), kCFStringEncodingUTF8
    )

def _cfint32(value: int) -> CFNumberRef:
    val = ctypes.c_int32(value)
    return CF.CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, ctypes.byref(val))

def _get_registry_int(service: io_service_t, key: str) -> int:
    """Read an int32 IORegistry property. Returns -1 on failure."""
    cf_key = _cfstr(key)
    ref = IOKit.IORegistryEntryCreateCFProperty(service, cf_key, kCFAllocatorDefault, 0)
    CF.CFRelease(cf_key)
    if not ref:
        return -1
    type_id = CF.CFGetTypeID(ref)
    num_type_id = CF.CFNumberGetTypeID()
    if type_id == num_type_id:
        val = ctypes.c_int32(0)
        ok = CF.CFNumberGetValue(ref, kCFNumberSInt32Type, ctypes.byref(val))
        CF.CFRelease(ref)
        return val.value if ok else -1
    CF.CFRelease(ref)
    return -1


# ── SPU device discovery and wake ────────────────────────────

def _wake_spu_drivers() -> None:
    """
    Set SensorPropertyReportingState / PowerState / ReportInterval
    on all AppleSPUHIDDriver instances to wake sensors.
    """
    match = IOKit.IOServiceMatching(b"AppleSPUHIDDriver")
    if not match:
        return
    it = io_iterator_t()
    kr = IOKit.IOServiceGetMatchingServices(_kIOMainPortDefault, match, ctypes.byref(it))
    if kr != KERN_SUCCESS:
        return

    while True:
        svc = IOKit.IOIteratorNext(it)
        if svc == IO_OBJECT_NULL:
            break
        enabled_num = _cfint32(1)
        interval_num = _cfint32(5000)
        key_report = _cfstr("SensorPropertyReportingState")
        key_power = _cfstr("SensorPropertyPowerState")
        key_interval = _cfstr("ReportInterval")
        if enabled_num:
            IOKit.IORegistryEntrySetCFProperty(svc, key_report, enabled_num)
            IOKit.IORegistryEntrySetCFProperty(svc, key_power, enabled_num)
            CF.CFRelease(enabled_num)
        if interval_num:
            IOKit.IORegistryEntrySetCFProperty(svc, key_interval, interval_num)
            CF.CFRelease(interval_num)
        CF.CFRelease(key_report)
        CF.CFRelease(key_power)
        CF.CFRelease(key_interval)
        IOKit.IOObjectRelease(svc)
    IOKit.IOObjectRelease(it)


def _find_spu_service(usage: int) -> io_service_t:
    """
    Find AppleSPUHIDDevice with PrimaryUsagePage=0xFF00 and PrimaryUsage=usage.
    Returns io_service_t or IO_OBJECT_NULL.
    """
    match = IOKit.IOServiceMatching(b"AppleSPUHIDDevice")
    if not match:
        return IO_OBJECT_NULL
    it = io_iterator_t()
    kr = IOKit.IOServiceGetMatchingServices(_kIOMainPortDefault, match, ctypes.byref(it))
    if kr != KERN_SUCCESS:
        return IO_OBJECT_NULL

    found = IO_OBJECT_NULL
    while True:
        svc = IOKit.IOIteratorNext(it)
        if svc == IO_OBJECT_NULL:
            break
        up = _get_registry_int(svc, "PrimaryUsagePage")
        u = _get_registry_int(svc, "PrimaryUsage")
        if up == 0xFF00 and u == usage:
            found = svc
            break
        IOKit.IOObjectRelease(svc)

    IOKit.IOObjectRelease(it)
    return found


# ── Sample dataclass ─────────────────────────────────────────

@dataclass
class SPUSample:
    """One IMU sample from an SPU sensor callback."""
    timestamp_ns: int
    x: float
    y: float
    z: float
    sensor: str   # "accel" or "gyro"


# ── SPU Backend ──────────────────────────────────────────────

class SPUBackend:
    """
    Non-root direct IOKit backend for Apple SPU IMU.

    Opens accel (usage 3) and gyro (usage 9) via the direct
    AppleSPUHIDDevice service path and collects samples via
    input report callbacks on a CFRunLoop thread.

    Thread-safe: samples are pushed into a deque from the
    RunLoop thread and drained by the consumer.
    """

    SCALE = 65536.0
    REPORT_MIN_LEN = 18
    REPORT_BUF_SIZE = 4096

    def __init__(self, buffer_maxlen: int = 500_000):
        self._buffer: deque[SPUSample] = deque(maxlen=buffer_maxlen)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._run_loop: Optional[CFRunLoopRef] = None
        self._run_loop_ready = threading.Event()

        # Sensor state
        self._accel_service = IO_OBJECT_NULL
        self._gyro_service = IO_OBJECT_NULL
        self._accel_device: Optional[IOHIDDeviceRef] = None
        self._gyro_device: Optional[IOHIDDeviceRef] = None

        # Report buffers (must stay alive while callbacks are active)
        self._accel_report_buf = (ctypes.c_uint8 * self.REPORT_BUF_SIZE)()
        self._gyro_report_buf = (ctypes.c_uint8 * self.REPORT_BUF_SIZE)()

        # Keep strong references to prevent GC of callback trampolines
        self._accel_cb: Optional[IOHIDReportCallback] = None
        self._gyro_cb: Optional[IOHIDReportCallback] = None

        self._total_accel_callbacks = 0
        self._total_gyro_callbacks = 0

    # ── Callback factories ──────────────────────────────────

    def _make_callback(self, sensor_name: str) -> IOHIDReportCallback:
        """Create a ctypes callback for the given sensor."""

        @IOHIDReportCallback
        def _cb(context, result, sender, report_type, report_id, report, length):
            if not self._running or length < self.REPORT_MIN_LEN or not report:
                return
            ts = time.perf_counter_ns()

            # Parse BMI286 report: header(6) + X(4) + Y(4) + Z(4)
            raw = bytes(report[i] for i in range(min(length, 18)))
            x_raw, y_raw, z_raw = struct.unpack_from("<iii", raw, 6)

            x = x_raw / self.SCALE
            y = y_raw / self.SCALE
            z = z_raw / self.SCALE

            sample = SPUSample(
                timestamp_ns=ts, x=x, y=y, z=z, sensor=sensor_name,
            )
            with self._lock:
                self._buffer.append(sample)

            if sensor_name == "accel":
                self._total_accel_callbacks += 1
            else:
                self._total_gyro_callbacks += 1

        return _cb

    # ── Lifecycle ────────────────────────────────────────────

    def start(self) -> None:
        """Start the backend: wake SPU, open devices, begin RunLoop thread."""
        if self._running:
            return

        _wake_spu_drivers()

        # Find services
        self._accel_service = _find_spu_service(3)
        self._gyro_service = _find_spu_service(9)

        if self._accel_service == IO_OBJECT_NULL and self._gyro_service == IO_OBJECT_NULL:
            raise RuntimeError(
                "No AppleSPUHIDDevice found for accel (usage 3) or gyro (usage 9). "
                "Is this an Apple Silicon Mac with BMI286 IMU?"
            )

        # Create HID device refs
        if self._accel_service != IO_OBJECT_NULL:
            self._accel_device = IOKit.IOHIDDeviceCreate(kCFAllocatorDefault, self._accel_service)
        if self._gyro_service != IO_OBJECT_NULL:
            self._gyro_device = IOKit.IOHIDDeviceCreate(kCFAllocatorDefault, self._gyro_service)

        # Open devices (non-exclusive)
        opened = []
        for name, dev in [("accel", self._accel_device), ("gyro", self._gyro_device)]:
            if dev:
                ret = IOKit.IOHIDDeviceOpen(dev, kIOHIDOptionsTypeNone)
                if ret != kIOReturnSuccess:
                    raise RuntimeError(
                        f"IOHIDDeviceOpen({name}, None) failed: 0x{ret & 0xFFFFFFFF:08x}"
                    )
                opened.append(name)

        if not opened:
            raise RuntimeError("Failed to open any SPU sensor device.")

        # Build callbacks (prevent GC)
        self._accel_cb = self._make_callback("accel")
        self._gyro_cb = self._make_callback("gyro")

        self._running = True
        self._run_loop_ready.clear()
        self._thread = threading.Thread(target=self._runloop_thread, daemon=True)
        self._thread.start()
        # Wait for the RunLoop to be set up before returning
        self._run_loop_ready.wait(timeout=5.0)

    def stop(self) -> None:
        """Stop the backend: tear down RunLoop, close devices, release services."""
        if not self._running:
            return
        self._running = False

        if self._run_loop:
            CF.CFRunLoopStop(self._run_loop)

        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

        # Close and release devices
        for dev in [self._accel_device, self._gyro_device]:
            if dev:
                IOKit.IOHIDDeviceClose(dev, kIOHIDOptionsTypeNone)
                CF.CFRelease(dev)
        self._accel_device = None
        self._gyro_device = None

        # Release services
        for svc in [self._accel_service, self._gyro_service]:
            if svc != IO_OBJECT_NULL:
                IOKit.IOObjectRelease(svc)
        self._accel_service = IO_OBJECT_NULL
        self._gyro_service = IO_OBJECT_NULL

        self._run_loop = None

    def drain(self) -> list[SPUSample]:
        """Drain all buffered samples. Thread-safe."""
        with self._lock:
            samples = list(self._buffer)
            self._buffer.clear()
        return samples

    @property
    def total_callbacks(self) -> int:
        return self._total_accel_callbacks + self._total_gyro_callbacks

    # ── RunLoop thread ──────────────────────────────────────

    def _runloop_thread(self) -> None:
        """
        Dedicated thread that runs a CFRunLoop to receive HID callbacks.
        Must register callbacks on this thread's RunLoop.
        """
        self._run_loop = CF.CFRunLoopGetCurrent()

        # Register callbacks and schedule on this RunLoop
        if self._accel_device:
            IOKit.IOHIDDeviceRegisterInputReportCallback(
                self._accel_device,
                self._accel_report_buf,
                self.REPORT_BUF_SIZE,
                self._accel_cb,
                None,
            )
            IOKit.IOHIDDeviceScheduleWithRunLoop(
                self._accel_device, self._run_loop, _kCFRunLoopDefaultMode,
            )

        if self._gyro_device:
            IOKit.IOHIDDeviceRegisterInputReportCallback(
                self._gyro_device,
                self._gyro_report_buf,
                self.REPORT_BUF_SIZE,
                self._gyro_cb,
                None,
            )
            IOKit.IOHIDDeviceScheduleWithRunLoop(
                self._gyro_device, self._run_loop, _kCFRunLoopDefaultMode,
            )

        self._run_loop_ready.set()

        # Run the loop in short increments so we can check _running
        while self._running:
            CF.CFRunLoopRunInMode(_kCFRunLoopDefaultMode, 0.05, False)

        # Unschedule
        if self._accel_device and self._run_loop:
            IOKit.IOHIDDeviceUnscheduleFromRunLoop(
                self._accel_device, self._run_loop, _kCFRunLoopDefaultMode,
            )
        if self._gyro_device and self._run_loop:
            IOKit.IOHIDDeviceUnscheduleFromRunLoop(
                self._gyro_device, self._run_loop, _kCFRunLoopDefaultMode,
            )
