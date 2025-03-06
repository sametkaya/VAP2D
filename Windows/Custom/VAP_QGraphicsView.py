from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import *

from Windows.Custom.VAP_Scene import VAP_Scene


class VAP_QGraphicsView(QtWidgets.QGraphicsView):
    photoClicked = QtCore.Signal(QtCore.QPoint)

    def __init__(self, parent):
        super(VAP_QGraphicsView, self).__init__(parent)
        self.setAutoFillBackground(True)
        pal = QPalette()
        pal.setColor(QPalette.Window, "#3a3a3a")
        self.setPalette(pal)

        self.imagePath = None
        self.zoom = 0
        self.isEmpty = True
        self.scenes = []
        self.scenes.append(VAP_Scene(self))
        self.scene = self.scenes[0]

        self.setScene(self.scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor("#3a3a3a")))

        # El aracı için değişkenler
        self.isPanning = False
        self.lastPanPoint = None

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self.scene.vap_image.image_qimage.rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.scene.HasImage():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self.zoom = 0

    def setImageBackground(self, qimage=None):
        self.scene.vap_image.image_pixmap = qimage
        pixmap = QPixmap.fromImage(self.image)
        self.zoom = 0
        if pixmap and not pixmap.isNull():
            self.isEmpty = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self.scene.vap_image.image_pixmap.setPixmap(pixmap)
        else:
            self.isEmpty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self.scene.vap_image.image_pixmap.setPixmap(QtGui.QPixmap())

        self.fitInView()

    def wheelEvent(self, event):
        if self.scene.vap_image.HasImage():
            zoom_in_factor = 1.25
            zoom_out_factor = 0.8
            zoom_max = 10
            zoom_min = -10

            if event.angleDelta().y() > 0:
                if self.zoom < zoom_max:
                    factor = zoom_in_factor
                    self.zoom += 1
                else:
                    return
            else:
                if self.zoom > zoom_min:
                    factor = zoom_out_factor
                    self.zoom -= 1
                else:
                    return

            self.scale(factor, factor)

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self.scene.vap_image.image_pixmap.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.isPanning = True
            self.lastPanPoint = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
        super(VAP_QGraphicsView, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.isPanning and self.lastPanPoint is not None:
            delta = event.pos() - self.lastPanPoint
            self.lastPanPoint = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        super(VAP_QGraphicsView, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.isPanning = False
            self.setCursor(QtCore.Qt.ArrowCursor)
        super(VAP_QGraphicsView, self).mouseReleaseEvent(event)
