from PySide6 import QtWidgets
from PySide6.QtCore import QFileInfo
from PySide6.QtGui import QImage, QPixmap


class VAP_Image():
    def __init__(self):
        self.image_raw_name = ""
        self.image_raw_path = ""
        self.image_qimage = None
        self.image_pixmap = QtWidgets.QGraphicsPixmapItem()

        #image
        self.image_raw = None
        self.image_gray = None
        self.image_byte8 = None
        self.image_segmented_byte8 = None
        self.image_skeletonized_byte8 = None

        self.is_empty = True

        #status
        self.status = 0

        #info
        self.vascularAreaFraction = 0
        self.whitePixelCount = 0
        self.blackPixelCount = 0
        self.vap_veins = []
        self.vap_vein_graphs = []
        self.total_vein_length = 0
        self.vaf=0
        self.average_vein_length =0
        self.average_vein_length = 0
        self.branchPoints = []
        self.tipPoints = []

    def SetImage(self, image_raw_path, image_raw=None, image_gray=None):
        self.image_raw_name = QFileInfo(image_raw_path).fileName()
        self.image_raw_path = image_raw_path
        self.image_qimage = QImage(image_raw_path)
        pixmap = QPixmap.fromImage(self.image_qimage)
        self.image_pixmap.setPixmap(pixmap)

        self.image_raw = image_raw
        self.image_gray = image_gray
        self.image_byte8 = image_gray
        self.is_empty = False

    def SetProcessImage(self, image_byte8):
        self.image_byte8 = image_byte8
        self.image_qimage = QImage(image_byte8.tobytes(), image_byte8.shape[1], image_byte8.shape[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(self.image_qimage)
        self.image_pixmap.setPixmap(pixmap)

    def HasImage(self):
        return not self.is_empty

    def AddVein(self, vap_vein_graph):
        self.vap_veins.append(vap_vein_graph.vap_vein)
        self.vap_vein_graphs.append(vap_vein_graph)
        self.total_vein_length = self.total_vein_length + vap_vein_graph.vap_vein.length

    def AddBranchPoint(self, vpoint):
        self.branchPoints.append(vpoint)

    def AddTipPoint(self, vpoint):
        self.tipPoints.append(vpoint)

    def ResetImageObcjects(self):
        self.branchPoints.clear()
        self.tipPoints.clear()
        self.vap_veins.clear()
        self.total_vein_length=0
        self.vap_vein_graphs.clear()