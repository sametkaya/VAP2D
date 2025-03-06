from PySide6.QtCore import Qt, QPoint, QRectF, QRect, QPointF, QEvent
from PySide6.QtGui import QBrush, QColor, QPen, QPainterPath, QFont
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPathItem


class VAP_Vein_Graph(QGraphicsItem):
    pointMarkerPen = QPen(Qt.SolidLine)
    lenghtDotDigit =3
    def __init__(self, pixmapItem, vap_vein, color=Qt.blue, penThickness=0):
        super().__init__()

        self.setAcceptHoverEvents(True)
        self.selected = False

        self.vap_vein = vap_vein
        #self.tip_points= []
        #self.branch_points = []
        #if vap_vein.idn==4:
        #    print(vap_vein.idn)
        self.imagePathCenterPoint = QPointF(self.vap_vein.mid_point.x, self.vap_vein.mid_point.y)
        self.scene_mapped_point_list = [pixmapItem.mapToScene(QPointF(p.x, p.y)) for p in self.vap_vein.vap_point_list]
        self.scenePathCenterPoint = pixmapItem.mapToScene(self.imagePathCenterPoint)

        self.showId = True
        self.showLenght = True
        self.showInfo = self.showId or self.showLenght
        self.lenghtDotDigit = round(self.vap_vein.length, VAP_Vein_Graph.lenghtDotDigit)

        self.color_default = color
        self.color_current = self.color_default
        self.color_selected = Qt.magenta

        self.penThickness = penThickness
        self.font = QFont()
        self.font.setPointSize(self.penThickness+3)
        #self.setZValue(1)

        self.update()

    def updateIt(self):
        self.update(self.boundingRect())

    def boundingRect(self):
        pf = QPointF(0.5 + self.scene_mapped_point_list[0].x(), 0.5 + self.scene_mapped_point_list[0].y())
        rect = QRectF(pf.x(), pf.y(), 1, 1)
        for point in self.scene_mapped_point_list[1:]:
            pf = QPointF(0.5 + point.x(), 0.5 + point.y())
            rect = rect.united(QRectF(pf.x(), pf.y(), 1, 1))
        return rect

    def paint(self, painter, option, widget):

        pen = QPen(Qt.SolidLine)
        pen.setWidth(self.penThickness)
        pen.setColor(self.color_current)
        painter.setPen(pen)
        painter.setBrush(self.color_current)

        #painter.drawPath(self.path)
        for point in self.scene_mapped_point_list:
            painter.drawRect(point.x()+0.5, point.y()+0.5, 1, 1)
        #if self.showLenght and self.idn == 0:
        if self.showInfo:
            painter.setFont(self.font)
            pen.setColor(Qt.yellow)

            painter.setPen(pen)

            text = ""
            if self.showId:
                text = str(self.vap_vein.idn)
            if self.showLenght:
                if self.showId:
                    text = text + "->"+str(self.lenghtDotDigit)
                else:
                    text = str(self.lenghtDotDigit)
            painter.drawText(self.scenePathCenterPoint, text)
            pen.setWidth(0)
            painter.setPen(pen)
            painter.setBrush(self.color_current)
            painter.drawRect(self.scenePathCenterPoint.x(), self.scenePathCenterPoint.y(), 1, 1)

    def mousePressEvent(self, event):
        self.selected = not self.selected  # Toggle selection
        self.color_current = self.color_selected if self.selected else self.color_default
        self.setZValue(float(self.selected))
        #print(self.vap_vein.idn)


        #self.update()  # Update the item to trigger repaint


    def hoverEnterEvent(self, event):
        self.setCursor(Qt.PointingHandCursor)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)

