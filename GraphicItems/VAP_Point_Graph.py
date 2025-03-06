from PySide6.QtCore import Qt, QPoint, QRectF, QRect, QPointF
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QGraphicsItem



class VAP_Point_Graph(QGraphicsItem):

    pointMarkerPen = QPen(Qt.SolidLine)
    pointCenterPen = QPen(Qt.SolidLine)
    def __init__(self, pixmapItem, vap_point, color=Qt.green, pad_width=10, pad_height=10, penThickness=1):
        super().__init__()

        self.setAcceptHoverEvents(True)
        self.selected = False
        self.vap_point = vap_point

        self.imagePoint = QPoint(vap_point.x, vap_point.y)
        self.centerPoint = pixmapItem.mapToScene(QPointF(vap_point.x, vap_point.y))

        self.pad_width = pad_width
        self.pad_height = pad_height
        self.color_default = color
        self.color_current = self.color_default
        self.color_selected = Qt.magenta
        self.penThickness = penThickness
        self.showMarker = True
        self.showCenter = True

        self.setToolTip("x:"+str(vap_point.x)+", y:"+str(vap_point.y))
        self.update()

    def updateIt(self):
        self.update(self.boundingRect())

    def boundingRect(self):

        return QRectF(0.5 + self.centerPoint.x() - self.pad_width / 2, 0.5 + self.centerPoint.y() - self.pad_height / 2, self.pad_width, self.pad_height)

    def paint(self, painter, option, widget):
        #pen = QPen(Qt.SolidLine)
        #pen.setWidth(self.penThickness)
        #pen.setColor(self.color)
        #painter.setPen(pen)
        if self.showMarker:

            VAP_Point_Graph.pointMarkerPen.setWidth(self.penThickness)
            VAP_Point_Graph.pointMarkerPen.setColor(self.color_current)
            painter.setPen(VAP_Point_Graph.pointMarkerPen)
            painter.drawEllipse(self.boundingRect())
        if self.showCenter:
            #pen = QPen(Qt.SolidLine)
            VAP_Point_Graph.pointCenterPen.setWidth(self.penThickness)
            VAP_Point_Graph.pointCenterPen.setColor(self.color_current)
            VAP_Point_Graph.pointCenterPen.setWidth(0)
            painter.setPen(VAP_Point_Graph.pointCenterPen)
            #painter.setPen(pen)
            painter.setBrush(self.color_current)
            painter.drawRect(self.centerPoint.x(), self.centerPoint.y(), 1, 1)

    def mousePressEvent(self, event):
        self.selected = not self.selected  # Toggle selection
        self.color_current = self.color_selected if self.selected else self.color_default
        self.update()  # Update the item to trigger repaint

    def hoverEnterEvent(self, event):
        self.setCursor(Qt.PointingHandCursor)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
    # def itemChange(self, change, value):
    #     if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
    #         self.color_current = self.color_selected if value else self.color_default
    #         self.update()  # Update the item to trigger repaint
    #     return super().itemChange(change, value)

    # def sceneEvent(self, event: QEvent):
    #     if event.type() == QEvent.GraphicsItemChange:
    #         if event.change() == QEvent.ItemSelectedChange:
    #             self.color_current = self.color_selected if self.isSelected() else self.color_default
    #             self.update(self.boundingRect())  # Update the item to trigger repaint
    #     return super().sceneEvent(event)