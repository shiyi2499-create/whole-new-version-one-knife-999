"""
Keyboard Listener - Captures key press/release events
======================================================
Uses pynput to monitor keyboard events and record them with
high-precision timestamps synchronized with the sensor stream.

Requires: pip install pynput
"""

import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class KeyEvent:
    """One keyboard event."""
    timestamp_ns: int       # perf_counter_ns (same clock as sensor)
    key: str                # normalized key name: 'a', 'b', '0', 'space', etc.
    event_type: str         # 'press' or 'release'


class KeyboardListener:
    """
    Listens for keyboard events in background.
    
    Usage:
        listener = KeyboardListener()
        listener.start()
        ...
        events = listener.drain()
        ...
        listener.stop()
    """

    def __init__(self, buffer_maxlen: int = 100_000):
        self._buffer: deque[KeyEvent] = deque(maxlen=buffer_maxlen)
        self._lock = threading.Lock()
        self._listener = None
        self._total_events = 0

    def start(self):
        """Start listening for keyboard events."""
        from pynput import keyboard
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def stop(self):
        """Stop the keyboard listener."""
        if self._listener:
            self._listener.stop()
            self._listener.join(timeout=3.0)
            self._listener = None

    @staticmethod
    def _normalize_key(key) -> str:
        """Convert pynput key to a clean string label."""
        # pynput gives either keyboard.Key (special) or keyboard.KeyCode (char)
        try:
            # Normal character key
            if hasattr(key, 'char') and key.char is not None:
                return key.char.lower()
        except AttributeError:
            pass

        # Special keys → readable names
        name = str(key).replace("Key.", "")
        # Map common special keys
        special_map = {
            "space": "space",
            "enter": "enter",
            "backspace": "backspace",
            "tab": "tab",
            "shift": "shift",
            "shift_l": "shift",
            "shift_r": "shift",
            "ctrl": "ctrl",
            "ctrl_l": "ctrl",
            "ctrl_r": "ctrl",
            "alt": "alt",
            "alt_l": "alt",
            "alt_r": "alt",
            "cmd": "cmd",
            "cmd_l": "cmd",
            "cmd_r": "cmd",
            "caps_lock": "capslock",
            "esc": "esc",
            "delete": "delete",
        }
        return special_map.get(name, name)

    def _on_press(self, key):
        ts = time.perf_counter_ns()
        k = self._normalize_key(key)
        event = KeyEvent(timestamp_ns=ts, key=k, event_type="press")
        with self._lock:
            self._buffer.append(event)
            self._total_events += 1

    def _on_release(self, key):
        ts = time.perf_counter_ns()
        k = self._normalize_key(key)
        event = KeyEvent(timestamp_ns=ts, key=k, event_type="release")
        with self._lock:
            self._buffer.append(event)
            self._total_events += 1

    def drain(self) -> list[KeyEvent]:
        """Return all buffered events and clear."""
        with self._lock:
            events = list(self._buffer)
            self._buffer.clear()
        return events

    @property
    def total_events(self) -> int:
        return self._total_events
