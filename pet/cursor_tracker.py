"""
Cross-compositor cursor tracking via raw evdev input.

Native Wayland deliberately blocks any client from querying the global
pointer position — that's intentional (X11 allows it via
`XQueryPointer`; Wayland's security model doesn't have an equivalent
for surfaces that aren't receiving input). Reading straight from
/dev/input/event* sidesteps the compositor entirely, so this works the
same under Wayland, X11, or headless.

Requires read access to /dev/input/event* — on most distros that means
being in the `input` group:
    sudo usermod -aG input $USER
then log out and back in (group membership doesn't apply to already-
running sessions).
"""
import threading

try:
    import evdev
    from evdev import ecodes
except ImportError:
    evdev = None


class EvdevCursorTracker:
    """Accumulates relative mouse motion into a virtual absolute position,
    clamped to screen bounds. Starts at screen center since relative
    deltas carry no absolute reference point."""

    def __init__(self, screen_size: tuple[int, int]):
        if evdev is None:
            raise RuntimeError("python-evdev not installed")

        self.width, self.height = screen_size
        self.x = self.width / 2
        self.y = self.height / 2
        self._lock = threading.Lock()

        self._devices = self._find_pointer_devices()
        if not self._devices:
            raise RuntimeError(
                "no readable pointer devices under /dev/input — "
                "check you're in the `input` group (sudo usermod -aG input $USER, then re-login)"
            )

        self._threads = [
            threading.Thread(target=self._read_loop, args=(dev,), daemon=True) for dev in self._devices
        ]
        for t in self._threads:
            t.start()

    def _find_pointer_devices(self):
        devices = []
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
            except (PermissionError, OSError):
                continue
            rel_caps = dev.capabilities().get(ecodes.EV_REL, [])
            if ecodes.REL_X in rel_caps and ecodes.REL_Y in rel_caps:
                devices.append(dev)
        return devices

    def _read_loop(self, dev):
        try:
            for event in dev.read_loop():
                if event.type != ecodes.EV_REL:
                    continue
                with self._lock:
                    if event.code == ecodes.REL_X:
                        self.x = max(0.0, min(self.width, self.x + event.value))
                    elif event.code == ecodes.REL_Y:
                        self.y = max(0.0, min(self.height, self.y + event.value))
        except OSError:
            pass  # device unplugged mid-read etc.

    def position(self) -> tuple[float, float]:
        with self._lock:
            return self.x, self.y
