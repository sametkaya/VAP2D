from PySide6 import QtWidgets
from PySide6.QtGui import QImage, QPixmap, Qt

from GraphicItems.VAP_Image import VAP_Image
from GraphicItems.VAP_Point_Graph import VAP_Point_Graph
from GraphicItems.VAP_Vein_Graph import VAP_Vein_Graph


class VAP_Scene(QtWidgets.QGraphicsScene):


    def __init__(self, parent):
        super(VAP_Scene, self).__init__(parent)
        self.vap_image = VAP_Image()
        self.addItem(self.vap_image.image_pixmap)
    def SetImage(self, image_raw_path, image_raw=None, image_byte8=None):
        self.vap_image.SetImage(image_raw_path, image_raw, image_byte8)
        self.is_empty = False
    def SetProcessImage(self, image_byte8):
        self.vap_image.SetProcessImage(image_byte8)

    def HasImage(self):
        return not self.vap_image.HasImage()
    def Reset(self):
        for item in self.vap_image.vap_vein_graphs:
            self.removeItem(item)
        for item in self.vap_image.branchPoints:
            self.removeItem(item)
        for item in self.vap_image.tipPoints:
            self.removeItem(item)
        self.vap_image.ResetImageObcjects()

    def AddVeins(self, veins):
        for vap_vein in veins:
            vap_vein_graph = VAP_Vein_Graph(self.vap_image.image_pixmap, vap_vein)
            self.addItem(vap_vein_graph)
            self.vap_image.AddVein(vap_vein_graph)
        self.vap_image.average_vein_length = self.vap_image.total_vein_length / (len(self.vap_image.vap_veins) + 1)

    def AddBranchPoints(self, vap_point_branch_list):
        for vap_point in vap_point_branch_list:
            vap_point_graph = VAP_Point_Graph(self.vap_image.image_pixmap, vap_point, color=Qt.red)
            self.addItem(vap_point_graph)
            self.vap_image.AddBranchPoint(vap_point_graph)
    def AddTipPoints(self, vap_point_tip_list):
        for vap_point in vap_point_tip_list:
            vap_point_graph = VAP_Point_Graph(self.vap_image.image_pixmap, vap_point, color=Qt.green)
            self.addItem(vap_point_graph)
            self.vap_image.AddTipPoint(vap_point_graph)
    def Update_Branch_Points(self, isVisible=True, showCenter=True, showMarker=True):
        for vpoint in self.vap_image.branchPoints:
            vpoint.setVisible(isVisible)
            vpoint.showMarker = showMarker
            vpoint.showCenter = showCenter
            # vpoint.updateIt()
        #self.update()
    def Update_Tip_Points(self, isVisible=True, showCenter=True, showMarker=True):
        for vpoint in self.vap_image.tipPoints:
            vpoint.setVisible(isVisible)
            vpoint.showMarker = showMarker
            vpoint.showCenter = showCenter
            # vpoint.updateIt()
        self.update()
    def Update_Branch_Paths(self, isVisible=True, showId=True, showLenght=True):
        for vap_vein_graph in self.vap_image.vap_vein_graphs:
            vap_vein_graph.setVisible(isVisible)
            vap_vein_graph.showId = showId
            vap_vein_graph.showLenght = showLenght
            vap_vein_graph.showInfo = vap_vein_graph.showId or vap_vein_graph.showLenght
        self.update()




