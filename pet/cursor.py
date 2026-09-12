"""Picks the best available way to track the global cursor, preferring
evdev (works everywhere) and falling back to Qt's QCursor (works on
X11/XWayland, not on native Wayland)."""
from PySide6.QtGui import QCursor

from .cursor_tracker import EvdevCursorTracker


class QtCursorProvider:
    def position(self) -> tuple[float, float]:
        p = QCursor.pos()
        return float(p.x()), float(p.y())


def get_cursor_provider(screen_size: tuple[int, int]):
    try:
        tracker = EvdevCursorTracker(screen_size)
        print("cursor tracking: evdev (works under native Wayland)")
        return tracker
    except Exception as e:
        print(f"evdev cursor tracking unavailable ({e})")
        print("falling back to Qt's QCursor — needs QT_QPA_PLATFORM=xcb on Wayland")
        return QtCursorProvider()
