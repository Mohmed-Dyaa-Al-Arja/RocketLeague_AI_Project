"""runtime/refresh_loop.py — Guaranteed-rescheduling Tkinter poll loop.

The bare ``root.after(500, callback)`` pattern breaks if ``callback``
raises an exception, because it never re-registers itself.  RefreshLoop
uses try/finally so the next tick is always scheduled regardless of errors.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional


class RefreshLoop:
    """Drive a periodic Tkinter callback that cannot die from exceptions.

    Usage::

        loop = RefreshLoop(my_callback, interval_ms=2000)
        loop.start(root)          # pass any tk.Widget (root / Toplevel)
        ...
        loop.stop()
    """

    def __init__(self, callback: Callable[[], None], interval_ms: int = 2000) -> None:
        self._callback    = callback
        self._interval_ms = interval_ms
        self._running     = False
        self._widget: Optional[object] = None   # tk.Widget
        self._after_id: Optional[str]  = None
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, widget: object) -> None:
        """Register and fire the first tick.  Idempotent."""
        with self._lock:
            if self._running:
                return
            self._widget  = widget
            self._running = True
        self._schedule()

    def stop(self) -> None:
        """Cancel the next pending tick.  Safe to call multiple times."""
        with self._lock:
            self._running = False
            aid = self._after_id
            self._after_id = None
        if aid and self._widget:
            try:
                self._widget.after_cancel(aid)  # type: ignore[attr-defined]
            except Exception:
                pass

    def set_interval(self, ms: int) -> None:
        """Change the polling interval (takes effect on the next tick)."""
        self._interval_ms = ms

    # ── Internal ──────────────────────────────────────────────────────────────

    def _schedule(self) -> None:
        with self._lock:
            if not self._running or self._widget is None:
                return
            try:
                self._after_id = self._widget.after(  # type: ignore[attr-defined]
                    self._interval_ms, self._tick
                )
            except Exception:
                pass

    def _tick(self) -> None:
        try:
            self._callback()
        finally:
            self._schedule()
