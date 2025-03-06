from PySide6.QtCore import QRectF
from PySide6.QtGui import QPen
from PySide6.QtWidgets import QGraphicsItem

#
# class CustomRectItem(QGraphicsItem):
#     def __init__(self, x, y, width, height):
#         super().__init__()
#         self.rect = QRectF(x, y, width, height)
#         self.pen = QPen()
#
#     def boundingRect(self):
#         return self.rect
#
#     def paint(self, painter, option, widget):
#         painter.setPen(self.pen)
#         painter.drawRect(self.rect)