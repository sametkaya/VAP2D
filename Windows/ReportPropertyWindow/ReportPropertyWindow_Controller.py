# from PySide6.QtWidgets import QMainWindow
#
# from System.ImageOperation import ImageOperation
# from Windows.MainWindow import MainWindow_Controller
# from Windows.ReportPropertyWindow.ReportPropertyWindow_UI import Ui_ReportPropertyWindow
#
#
#
#
# class SecondWindow_Controller():
#     def __init__(self, secondWindow: QMainWindow, ui: Ui_ReportPropertyWindow, mainWndw: MainWindow_Controller):
#         self.secondWindow = secondWindow
#         self.ui = ui
#         self.mainWndw=mainWndw
#         self.gv_image=self.mainWndw.ui.gv_image.scene.vap_image
#         return
#
#     def pbtn_create_pdf_clicked(self):
#         self.secondWindow.hide()
#
#         branchPointsCount = self.ui.checkBox_sw_bpCount.isChecked()
#         tipPointsCount = self.ui.checkBox_sw_tpCount.isChecked()
#         veinCount = self.ui.checkBox_sw_vCount.isChecked()
#         veinStartEndPoints = self.ui.checkBox_sw_veinSEP.isChecked()
#         totalVeinLength = self.ui.checkBox_sw_tvLength.isChecked()
#         averageVeinLength = self.ui.checkBox_sw_avLength.isChecked()
#         eachVeinLength = self.ui.checkBox_sw_evLength.isChecked()
#         veinStartEndPointsType = self.ui.checkBox_sw_vSEPType.isChecked()
#
#         informationDict={}
#         informationDict["vaf(%)"] = True
#         informationDict["id"] = True
#         informationDict["branch points count"]=branchPointsCount
#         informationDict["tip point count"] = tipPointsCount
#         informationDict["vein count"] = veinCount
#         informationDict["total vein length"] = totalVeinLength
#         informationDict["average vein length"] = averageVeinLength
#         informationDict["p1.x, p1.y"] = veinStartEndPoints
#         informationDict["p2.x, p2.y"] = veinStartEndPoints
#         informationDict["length"] = eachVeinLength
#         informationDict["p1_type"] = veinStartEndPointsType
#         informationDict["p2_type"] = veinStartEndPointsType
#
#
#         csv_file_path, pdf_file_path = ImageOperation.SaveInfos(self.gv_image,informationDict)
#
#         self.mainWndw.ui.wgts_sceneContent.setCurrentWidget(self.mainWndw.ui.page_report)
#         self.mainWndw.load_pdf(pdf_file_path)
#         return
#
#
#
