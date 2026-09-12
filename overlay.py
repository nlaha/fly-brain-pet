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
        painter.setBrush(QColor(20, 20, 20, 230))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(-6, -3, 12, 6)   # body, placeholder sprite
        painter.drawEllipse(-9, -4, 4, 3)    # left wing hint
        painter.drawEllipse(-9, 1, 4, 3)     # right wing hint
        painter.end()


def run_overlay(step_callback):
    app = QApplication.instance() or QApplication([])
    screen = app.primaryScreen().size()
    overlay = FlyOverlay((screen.width(), screen.height()), step_callback)
    overlay.show()
    app.exec()
