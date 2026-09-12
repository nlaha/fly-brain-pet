"""Transparent, click-through, always-on-top window that draws the fly."""
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor
from PySide6.QtWidgets import QWidget, QApplication


class FlyOverlay(QWidget):
    def __init__(self, screen_size: tuple[int, int], step_callback):
        super().__init__()
        self.step_callback = step_callback  # called each frame, returns (x, y, heading)
        self.pos = (screen_size[0] / 2, screen_size[1] / 2)
        self.heading = 0.0

        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # click-through
        self.setGeometry(0, 0, *screen_size)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)  # ~60fps

    def tick(self):
        self.pos, self.heading = self.step_callback()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.translate(*self.pos)
        painter.rotate(self.heading)

        # -------------------------
        # Colors
        # -------------------------
        body = QColor(45, 30, 25, 245)
        abdomen = QColor(65, 40, 30, 245)
        abdomen_dark = QColor(35, 25, 22, 245)

        eye = QColor(150, 25, 35, 255)
        eye_highlight = QColor(230, 90, 100, 220)

        leg_color = QColor(30, 25, 23, 230)

        wing = QColor(210, 225, 235, 80)
        wing_outline = QColor(170, 190, 205, 110)

        # -------------------------
        # Legs
        # -------------------------
        painter.setPen(leg_color)
        painter.setBrush(Qt.NoBrush)

        # Rear legs
        painter.drawLine(2, -2, 9, -8)
        painter.drawLine(9, -8, 13, -7)

        painter.drawLine(2, 2, 9, 8)
        painter.drawLine(9, 8, 13, 7)

        # Middle legs
        painter.drawLine(-1, -2, 5, -9)
        painter.drawLine(5, -9, 8, -10)

        painter.drawLine(-1, 2, 5, 9)
        painter.drawLine(5, 9, 8, 10)

        # Front legs
        painter.drawLine(-5, -2, -8, -7)
        painter.drawLine(-8, -7, -11, -6)

        painter.drawLine(-5, 2, -8, 7)
        painter.drawLine(-8, 7, -11, 6)

        # -------------------------
        # Wings
        # -------------------------
        painter.setPen(wing_outline)
        painter.setBrush(wing)

        # Upper wing
        painter.drawEllipse(-5, -14, 18, 11)

        # Lower wing
        painter.drawEllipse(-5, 3, 18, 11)

        # Wing veins
        painter.setPen(QColor(150, 170, 185, 90))
        painter.drawLine(1, -9, 10, -11)
        painter.drawLine(1, -8, 9, -5)

        painter.drawLine(1, 9, 10, 11)
        painter.drawLine(1, 8, 9, 5)

        # -------------------------
        # Abdomen
        # -------------------------
        painter.setPen(Qt.NoPen)
        painter.setBrush(abdomen)

        # Slightly tapered abdomen
        painter.drawEllipse(-2, -3, 15, 6)

        # Abdomen stripes
        painter.setBrush(abdomen_dark)
        painter.drawRect(7, -2, 2, 4)
        painter.drawRect(11, -2, 2, 4)

        # -------------------------
        # Thorax
        # -------------------------
        painter.setBrush(body)
        painter.drawEllipse(-7, -4, 9, 8)

        # -------------------------
        # Head
        # -------------------------
        painter.drawEllipse(-11, -4, 7, 8)

        # -------------------------
        # Eyes
        # -------------------------
        painter.setBrush(eye)

        painter.drawEllipse(-12, -4, 4, 4)
        painter.drawEllipse(-12, 0, 4, 4)

        # Tiny eye highlights
        painter.setBrush(eye_highlight)
        painter.drawEllipse(-11, -3, 1, 1)
        painter.drawEllipse(-11, 1, 1, 1)

        # -------------------------
        # Antennae
        # -------------------------
        painter.setPen(QColor(25, 20, 18, 230))
        painter.drawLine(-9, -3, -14, -7)
        painter.drawLine(-14, -7, -16, -8)

        painter.drawLine(-9, 3, -14, 7)
        painter.drawLine(-14, 7, -16, 8)

        painter.end()

def run_overlay(step_callback, app=None):
    app = app or QApplication.instance() or QApplication([])
    screen = app.primaryScreen().size()
    overlay = FlyOverlay((screen.width(), screen.height()), step_callback)
    overlay.show()
    app.exec()
