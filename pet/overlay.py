"""Transparent, click-through, always-on-top fruit-fly overlay."""
import math
import time

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget, QApplication


class FlyOverlay(QWidget):
    def __init__(self, screen_size, step_callback, debug=False, num_flies=1):
        super().__init__()
        self.step_callback = step_callback
        self.screen_size = screen_size
        self.num_flies = num_flies
        self.views = []
        self.debug_mode = debug
        self.frame_count = 0
        self.last_frame_time = time.perf_counter()
        self._timing_samples = []

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setGeometry(0, 0, *screen_size)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def tick(self):
        now = time.perf_counter()
        frame_ms = (now - self.last_frame_time) * 1000.0
        self.last_frame_time = now

        start = time.perf_counter()
        self.views = self.step_callback()
        update_ms = (time.perf_counter() - start) * 1000.0

        self.frame_count += 1
        self._timing_samples.append((frame_ms, update_ms))
        if len(self._timing_samples) > 60:
            self._timing_samples.pop(0)
        if self.frame_count % 60 == 0:
            avg_frame = sum(x[0] for x in self._timing_samples) / len(self._timing_samples)
            avg_update = sum(x[1] for x in self._timing_samples) / len(self._timing_samples)
            fps = 1000.0 / avg_frame if avg_frame > 0 else 0.0
            print(f"frame={self.frame_count:6d} update={avg_update:7.2f} ms frame={avg_frame:7.2f} ms fps={fps:6.1f} flies={len(self.views)}")
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        for view in self.views:
            self._draw_fly(painter, view)

        if self.debug_mode:
            self._draw_world_debug(painter)
        painter.end()

    def _draw_fly(self, painter, view):
        x, y = view["pos"]
        painter.save()
        painter.translate(x, y)
        painter.rotate(view["heading"])

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

        painter.setPen(QPen(leg, 1.1))
        painter.setBrush(Qt.NoBrush)
        for yy in (-2.2, 2.2):
            painter.drawLine(-1, yy, -7, yy * 3.0)
            painter.drawLine(-7, yy * 3.0, -11, yy * 3.3)
            painter.drawLine(-4, yy, -8, yy * 4.0)
            painter.drawLine(-8, yy * 4.0, -12, yy * 3.7)
            painter.drawLine(-8, yy, -3, yy * 4.3)
            painter.drawLine(-3, yy * 4.3, 2, yy * 4.0)

        flap = math.sin(time.perf_counter() * (12.0 if view["escape"] else 4.0))
        for sign in (-1, 1):
            painter.save()
            painter.rotate(sign * (10 - 7 * flap))
            painter.setPen(QPen(wing_edge, 0.8))
            painter.setBrush(wing)
            y0 = -15 if sign < 0 else 5
            painter.drawEllipse(-5, y0, 22, 10)
            painter.setPen(QPen(wing_vein, 0.65))
            painter.drawLine(0, y0 + 5, 14, y0 + (2 if sign < 0 else 8))
            painter.restore()

        painter.setPen(Qt.NoPen)
        painter.setBrush(abdomen)
        painter.drawEllipse(-15, -4.2, 18, 8.4)
        painter.setBrush(abdomen_dark)
        painter.drawRect(-10, -3.5, 2, 7)
        painter.drawRect(-5, -3.7, 2, 7.4)
        painter.drawRect(0, -3.2, 2, 6.4)
        painter.setBrush(thorax)
        painter.drawEllipse(-5, -5.0, 10, 10)
        painter.setBrush(body)
        painter.drawEllipse(3, -4.5, 8, 9)
        painter.setBrush(eye)
        painter.drawEllipse(7, -4.3, 5, 4.6)
        painter.drawEllipse(7, -0.3, 5, 4.6)
        painter.setBrush(eye_hi)
        painter.drawEllipse(9, -3.0, 1.1, 1.1)
        painter.drawEllipse(9, 1.0, 1.1, 1.1)
        painter.setPen(QPen(leg, 0.8))
        painter.drawLine(9, -2.7, 14, -6.5)
        painter.drawLine(14, -6.5, 17, -7.0)
        painter.drawLine(9, 2.7, 14, 6.5)
        painter.drawLine(14, 6.5, 17, 7.0)
        painter.restore()

    def _draw_world_debug(self, painter):
        # Debug visualization is tied to the actual virtual retina, not a
        # generic spinning cone. The field is a broad compound-eye FOV and the
        # small cyan marks show where each eye sits on the rendered fly.
        for view in self.views:
            cx, cy = view["pos"]
            heading = math.radians(view["heading"])
            fov = math.radians(300.0)
            radius = 125.0
            start = heading - fov / 2.0
            painter.setPen(QPen(QColor(80, 180, 255, 95), 1.0, Qt.DashLine))
            painter.setBrush(QColor(80, 170, 255, 10))
            points = [QPointF(cx, cy)]
            for i in range(49):
                a = start + fov * i / 48
                points.append(QPointF(cx + math.cos(a) * radius, cy + math.sin(a) * radius))
            painter.drawPolygon(points)

            # Eye locations in the fly's local coordinates, rotated with it.
            ex = math.cos(heading) * 9.5
            ey = math.sin(heading) * 9.5
            perp_x, perp_y = -math.sin(heading), math.cos(heading)
            for side in (-1, 1):
                eye_x = cx + ex + perp_x * side * 2.8
                eye_y = cy + ey + perp_y * side * 2.8
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(70, 240, 255, 220))
                painter.drawEllipse(QPointF(eye_x, eye_y), 2.2, 2.2)

            # Draw projected objects as what the retina currently sees.
            for obj in view.get("debug_data", {}).get("visual_objects", []):
                if not obj.get("visible"):
                    continue
                angle = heading + obj["angle"]
                d = min(radius, max(18.0, obj["distance"] * 0.35))
                tx = cx + math.cos(angle) * d
                ty = cy + math.sin(angle) * d
                intensity = min(1.0, max(0.0, obj.get("expansion_rate", 0.0) / 0.30))
                alpha = int(55 + 170 * intensity)
                painter.setPen(QPen(QColor(120, 220, 255, alpha), 1.0))
                painter.setBrush(QColor(120, 220, 255, int(20 + 90 * intensity)))
                size = max(3.0, min(16.0, obj.get("apparent_radius", 0.0) * 45.0))
                painter.drawEllipse(QPointF(tx, ty), size, size)

        # Bottom-right debug panel: actual retina + named neural populations.
        panel_w = 610
        panel_h = min(self.height() - 30, 165 + 70 * len(self.views))
        px = max(10, self.width() - panel_w - 20)
        py = max(10, self.height() - panel_h - 20)
        painter.setPen(QPen(QColor(210, 220, 230, 190), 1))
        painter.setBrush(QColor(10, 15, 20, 220))
        painter.drawRoundedRect(px, py, panel_w, panel_h, 8, 8)
        painter.setPen(QColor(235, 240, 245, 230))
        painter.drawText(px + 12, py + 20, "FLY BRAIN DEBUG  •  VIRTUAL COMPOUND-EYE RETINA")

        y = py + 40
        for view in self.views:
            data = view.get("debug_data", {})
            sensory = data.get("sensory", {})
            signal = data.get("signal", {})
            painter.setPen(QColor(235, 240, 245, 230))
            painter.drawText(
                px + 12, y,
                f"FLY {view['id']}  speed={data.get('speed', 0):.2f}  "
                f"loom={sensory.get('looming', 0):.3f}  "
                f"expansion={sensory.get('approach_speed', 0):.3f} rad/s"
            )

            # Panoramic retina strip. This is the actual sensory image fed to
            # the visual interface, not a decorative FOV indicator.
            retina = data.get("vision")
            rx, ry, rw, rh = px + 12, y + 8, 210, 28
            painter.setPen(QPen(QColor(100, 120, 130, 180), 1))
            painter.setBrush(QColor(25, 30, 35, 220))
            painter.drawRect(rx, ry, rw, rh)
            if retina is not None:
                h = len(retina)
                w = len(retina[0]) if h else 0
                if w and h:
                    for ix in range(w):
                        value = float(max(retina[:, ix]))
                        if value <= 0:
                            continue
                        a = int(30 + 220 * min(1.0, value))
                        painter.setPen(QColor(100, 210, 255, a))
                        xx = rx + int(ix * rw / w)
                        painter.drawLine(xx, ry + 2, xx, ry + rh - 2)
            painter.setPen(QColor(150, 170, 180, 210))
            painter.drawText(rx + 4, ry + 19, "RETINA  ← rear   front   rear →")

            y += 44
            for role, label, color in [
                ("looming", "LOOM", QColor(100, 180, 255, 230)),
                ("escape", "ESCAPE", QColor(255, 100, 100, 240)),
                ("forward", "WALK", QColor(120, 230, 150, 230)),
                ("steering", "STEER", QColor(230, 200, 100, 230)),
                ("escape_wing", "WINGS", QColor(210, 150, 255, 230)),
            ]:
                values = list(data.get(role, ()))[:36]
                painter.setPen(QColor(180, 190, 200, 210))
                painter.drawText(px + 240, y + 4, label)
                peak = max(values, default=0.0)
                for i, value in enumerate(values):
                    intensity = max(0.08, min(1.0, float(value) / max(peak, 1e-6)))
                    c = QColor(color.red(), color.green(), color.blue(), int(35 + 220 * intensity))
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(c)
                    painter.drawEllipse(px + 300 + i * 8, y - 3, 6, 6)
                y += 14
            y += 7


def run_overlay(step_callback, app=None, debug=False, num_flies=1):
    app = app or QApplication.instance() or QApplication([])
    screen = app.primaryScreen().size()
    overlay = FlyOverlay((screen.width(), screen.height()), step_callback, debug=debug, num_flies=num_flies)
    overlay.show()
    app.exec()
