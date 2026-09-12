"""Transparent, click-through, always-on-top fruit-fly overlay."""
import math
import time

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QImage
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
        self._brain_panel_cache = None
        self._brain_panel_generation = -1

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setGeometry(0, 0, *screen_size)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(0)  # run continuously; simulation update time determines the frame rate

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
        # Debug is deliberately separated from the simulation: it renders a
        # compact 3-D projection of the *actual soma coordinates* from the
        # loaded Male CNS annotation. Bright points are the currently most
        # active neurons; the faint cloud is the anatomical reference.
        for view in self.views:
            cx, cy = view["pos"]
            heading = math.radians(view["heading"])
            fov = math.radians(300.0)
            radius = 105.0
            start = heading - fov / 2.0
            painter.setPen(QPen(QColor(80, 180, 255, 45), 1.0, Qt.DashLine))
            painter.setBrush(QColor(80, 170, 255, 4))
            points = [QPointF(cx, cy)]
            for i in range(37):
                a = start + fov * i / 36
                points.append(QPointF(cx + math.cos(a) * radius, cy + math.sin(a) * radius))
            painter.drawPolygon(points)

            perp_x, perp_y = -math.sin(heading), math.cos(heading)
            for side in (-1, 1):
                eye_x = cx + math.cos(heading) * 8.5 + perp_x * side * 2.7
                eye_y = cy + math.sin(heading) * 8.5 + perp_y * side * 2.7
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(70, 240, 255, 180))
                painter.drawEllipse(QPointF(eye_x, eye_y), 1.8, 1.8)
            painter.setPen(QPen(QColor(240, 240, 240, 120), 1.0))
            painter.drawLine(QPointF(cx, cy), QPointF(cx + math.cos(heading) * 16,
                                                        cy + math.sin(heading) * 16))

            objects = view.get("debug_data", {}).get("visual_objects", [])
            visible = [o for o in objects if o.get("visible")]
            visible.sort(key=lambda o: (o.get("expansion_rate", 0.0),
                                        -o.get("distance", 1e9)), reverse=True)
            for obj in visible[:3]:
                angle = heading + obj["angle"]
                d = min(radius * 0.9, max(14.0, obj["distance"] * 0.18))
                tx = cx + math.cos(angle) * d
                ty = cy + math.sin(angle) * d
                intensity = min(1.0, max(0.0, obj.get("expansion_rate", 0.0) / 0.08))
                alpha = int(55 + 150 * intensity)
                painter.setPen(QPen(QColor(120, 220, 255, alpha), 1.0))
                painter.setBrush(Qt.NoBrush)
                size = max(2.0, min(10.0, obj.get("apparent_radius", 0.0) * 55.0))
                painter.drawEllipse(QPointF(tx, ty), size, size)

        # The connectome panel is expensive to paint (thousands of translucent
        # point operations). Brain activity is sampled at ~8 Hz in main.py, so
        # cache the finished panel image and simply blit it between samples.
        generation = -1
        if self.views:
            generation = self.views[0].get("debug_data", {}).get("brain_generation", -1)
        if generation != self._brain_panel_generation or self._brain_panel_cache is None:
            self._rebuild_brain_panel()
            self._brain_panel_generation = generation
        if self._brain_panel_cache is not None:
            px = self._brain_panel_cache[1]
            py = self._brain_panel_cache[2]
            painter.drawImage(QPointF(px, py), self._brain_panel_cache[0])

    def _rebuild_brain_panel(self):
        cols = 3 if len(self.views) <= 9 else 2
        card_w = 300
        card_h = 154
        shown = self.views[:15]
        rows = max(1, math.ceil(len(shown) / cols))
        panel_w = cols * card_w + 18
        panel_h = rows * card_h + 42
        px = max(10, self.width() - panel_w - 18)
        py = max(10, self.height() - panel_h - 18)

        image = QImage(panel_w, panel_h, QImage.Format_ARGB32_Premultiplied)
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(210, 220, 230, 180), 1))
        painter.setBrush(QColor(7, 11, 16, 225))
        painter.drawRoundedRect(0, 0, panel_w, panel_h, 9, 9)
        painter.setPen(QColor(235, 240, 245, 235))
        painter.drawText(12, 21,
                         "CONNECTOME ACTIVITY  •  3-D SOMA PROJECTION  •  bright = active")

        for i, view in enumerate(shown):
            col = i % cols
            row = i // cols
            x0 = 6 + col * card_w
            y0 = 30 + row * card_h
            painter.setPen(QPen(QColor(90, 105, 120, 130), 1))
            painter.setBrush(QColor(12, 18, 25, 215))
            painter.drawRoundedRect(x0, y0, card_w - 8, card_h - 8, 6, 6)

            data = view.get("debug_data", {})
            painter.setPen(QColor(230, 235, 240, 230))
            painter.drawText(x0 + 8, y0 + 16,
                             f"FLY {view['id']}  v={data.get('speed', 0):.1f}  "
                             f"loom={data.get('sensory', {}).get('looming', 0):.2f}")

            cloud = data.get("brain_cloud")
            bx, by, bw, bh = x0 + 7, y0 + 23, 178, 116
            painter.setPen(QPen(QColor(70, 85, 100, 120), 1))
            painter.setBrush(QColor(4, 7, 11, 230))
            painter.drawRoundedRect(bx, by, bw, bh, 5, 5)
            if cloud is not None:
                ids, xyz, vals = cloud
                if len(xyz):
                    sx = (xyz[:, 0] - 0.5) * 1.20
                    sy = (xyz[:, 1] - 0.5) * 0.70
                    sz = (xyz[:, 2] - 0.5) * 0.90
                    iso_x = sx - sy
                    iso_y = (sx + sy) * 0.25 - sz
                    order = sorted(range(len(vals)), key=lambda j: float(xyz[j, 2]))
                    for j in order:
                        xx = bx + bw * 0.50 + iso_x[j] * bw * 0.39
                        yy = by + bh * 0.57 + iso_y[j] * bh * 0.55
                        if xx < bx + 2 or xx > bx + bw - 2 or yy < by + 2 or yy > by + bh - 2:
                            continue
                        activity = min(1.0, float(vals[j]) / 3.0)
                        alpha = int(18 + 220 * activity)
                        # Keep neurons close to subpixel/1px scale so dense
                        # anatomy remains legible instead of becoming blobs.
                        radius = 0.45 + 0.9 * activity
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(QColor(85, 205, 255, alpha))
                        painter.drawEllipse(QPointF(xx, yy), radius, radius)

            labels = [("ESC", "escape"), ("WALK", "forward"),
                      ("TURN", "flight_steering"), ("LOOM", "looming")]
            tx = x0 + 190
            ty = y0 + 37
            for label, role in labels:
                values = data.get(role, ())
                peak = max((float(v) for v in values), default=0.0)
                painter.setPen(QColor(185, 195, 205, 210))
                painter.drawText(tx, ty, label)
                painter.setPen(QPen(QColor(95, 210, 255, 180), 3))
                painter.drawLine(tx + 32, ty - 3, tx + 32 + min(55, int(peak * 6)), ty - 3)
                painter.setPen(QColor(155, 165, 175, 180))
                painter.drawText(tx + 32, ty + 13, f"{peak:.2f}")
                ty += 18

        if len(self.views) > 15:
            painter.setPen(QColor(170, 180, 190, 190))
            painter.drawText(12, panel_h - 8, f"showing 15 of {len(self.views)} brains")
        painter.end()
        self._brain_panel_cache = (image, px, py)



def run_overlay(step_callback, app=None, debug=False, num_flies=1):
    app = app or QApplication.instance() or QApplication([])
    screen = app.primaryScreen().size()
    overlay = FlyOverlay((screen.width(), screen.height()), step_callback, debug=debug, num_flies=num_flies)
    overlay.show()
    app.exec()
