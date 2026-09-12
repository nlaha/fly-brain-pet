"""Transparent, click-through, always-on-top fruit-fly overlay."""
import math
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget, QApplication


class FlyOverlay(QWidget):
    def __init__(self, screen_size: tuple[int, int], step_callback):
        super().__init__()
        self.step_callback = step_callback
        self.pos = (screen_size[0] / 2, screen_size[1] / 2)
        self.heading = 0.0
        self.escape = False
        self.wing_drive = 0.0
        self.wing_phase = 0.0

        # Frame/update timing, similar to a game engine profiler.
        self.frame_count = 0
        self.last_frame_time = time.perf_counter()
        self.update_ms = 0.0
        self.frame_ms = 0.0
        self._timing_samples = []

        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setGeometry(0, 0, *screen_size)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

    def tick(self):
        # Measure the actual update callback separately from the Qt timer
        # interval.  update_ms is the time spent advancing the brain + body
        # simulation for one rendered frame.
        now = time.perf_counter()
        self.frame_ms = (now - self.last_frame_time) * 1000.0
        self.last_frame_time = now

        update_start = time.perf_counter()
        result = self.step_callback()
        self.update_ms = (time.perf_counter() - update_start) * 1000.0

        self.frame_count += 1
        self._timing_samples.append((self.frame_ms, self.update_ms))
        if len(self._timing_samples) > 60:
            self._timing_samples.pop(0)

        # Print a rolling 60-frame timing summary.  This is intentionally
        # independent of the neural debug output in main.py.
        if self.frame_count % 60 == 0 and self._timing_samples:
            avg_frame = sum(x[0] for x in self._timing_samples) / len(self._timing_samples)
            avg_update = sum(x[1] for x in self._timing_samples) / len(self._timing_samples)
            fps = 1000.0 / avg_frame if avg_frame > 0 else 0.0
            print(
                f"frame={self.frame_count:6d}  "
                f"update={avg_update:7.2f} ms  "
                f"frame={avg_frame:7.2f} ms  "
                f"fps={fps:6.1f}"
            )
        if len(result) == 4:
            self.pos, self.heading, self.escape, self.wing_drive = result
        else:
            self.pos, self.heading = result
            self.escape = False
            self.wing_drive = 0.0

        # Wings beat slowly during ordinary motion and rapidly when the
        # connectome's escape-wing population is active.
        self.wing_phase += 0.20 + 1.4 * float(self.wing_drive) + (0.9 if self.escape else 0.0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(*self.pos)
        painter.rotate(self.heading)

        # The fly faces +X.  This makes the body orientation line up naturally
        # with the physics heading.
        body = QColor(43, 29, 24, 245)
        thorax = QColor(53, 35, 28, 250)
        abdomen = QColor(67, 40, 28, 248)
        abdomen_dark = QColor(35, 25, 22, 235)
        eye = QColor(160, 28, 38, 255)
        eye_hi = QColor(255, 105, 115, 230)
        leg = QColor(27, 22, 20, 235)
        wing = QColor(205, 225, 236, 78)
        wing_edge = QColor(160, 185, 205, 120)
        wing_vein = QColor(130, 155, 175, 85)

        # Legs: six articulated, fine strokes.  They are kept behind the body.
        painter.setPen(QPen(leg, 1.1))
        painter.setBrush(Qt.NoBrush)

        for y in (-2.2, 2.2):
            # front
            painter.drawLine(-1, y, -7, y * 3.0)
            painter.drawLine(-7, y * 3.0, -11, y * 3.3)
            # middle
            painter.drawLine(-4, y, -8, y * 4.0)
            painter.drawLine(-8, y * 4.0, -12, y * 3.7)
            # rear
            painter.drawLine(-8, y, -3, y * 4.3)
            painter.drawLine(-3, y * 4.3, 2, y * 4.0)

        # Wings.  The sinusoidal phase changes their angle slightly so they
        # visibly beat instead of looking like two static ovals.
        flap = math.sin(self.wing_phase)
        painter.save()
        painter.rotate(-10 + 7 * flap)
        painter.setPen(QPen(wing_edge, 0.8))
        painter.setBrush(wing)
        painter.drawEllipse(-5, -15, 22, 10)
        painter.setPen(QPen(wing_vein, 0.65))
        painter.drawLine(0, -10, 14, -13)
        painter.drawLine(0, -9, 14, -7)
        painter.restore()

        painter.save()
        painter.rotate(10 - 7 * flap)
        painter.setPen(QPen(wing_edge, 0.8))
        painter.setBrush(wing)
        painter.drawEllipse(-5, 5, 22, 10)
        painter.setPen(QPen(wing_vein, 0.65))
        painter.drawLine(0, 10, 14, 7)
        painter.drawLine(0, 11, 14, 13)
        painter.restore()

        # Abdomen: tapered, segmented, with the characteristic dark bands.
        painter.setPen(Qt.NoPen)
        painter.setBrush(abdomen)
        painter.drawEllipse(-15, -4.2, 18, 8.4)

        painter.setBrush(abdomen_dark)
        painter.drawRect(-10, -3.5, 2, 7)
        painter.drawRect(-5, -3.7, 2, 7.4)
        painter.drawRect(0, -3.2, 2, 6.4)

        # Thorax.
        painter.setBrush(thorax)
        painter.drawEllipse(-5, -5.0, 10, 10)

        # Head at the +X/front of the body.
        painter.setBrush(body)
        painter.drawEllipse(3, -4.5, 8, 9)

        # Large red compound eyes, split slightly by the head silhouette.
        painter.setBrush(eye)
        painter.drawEllipse(7, -4.3, 5, 4.6)
        painter.drawEllipse(7, -0.3, 5, 4.6)

        painter.setBrush(eye_hi)
        painter.drawEllipse(9, -3.0, 1.1, 1.1)
        painter.drawEllipse(9, 1.0, 1.1, 1.1)

        # Antennae.
        painter.setPen(QPen(leg, 0.8))
        painter.drawLine(9, -2.7, 14, -6.5)
        painter.drawLine(14, -6.5, 17, -7.0)
        painter.drawLine(9, 2.7, 14, 6.5)
        painter.drawLine(14, 6.5, 17, 7.0)

        # Tiny mouthparts.
        painter.drawLine(11, -0.5, 14, 0)
        painter.drawLine(14, 0, 11, 0.5)

        painter.end()


def run_overlay(step_callback, app=None):
    app = app or QApplication.instance() or QApplication([])
    screen = app.primaryScreen().size()
    overlay = FlyOverlay((screen.width(), screen.height()), step_callback)
    overlay.show()
    app.exec()
