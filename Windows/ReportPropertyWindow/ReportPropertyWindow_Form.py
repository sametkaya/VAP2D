# from PySide6.QtGui import QIcon
# from PySide6.QtWidgets import QMainWindow
#
# from Windows.ReportPropertyWindow.ReportPropertyWindow_Controller import SecondWindow_Controller
# from Windows.ReportPropertyWindow.ReportPropertyWindow_UI import Ui_ReportPropertyWindow
#
#
# class SecondWindow_Form(QMainWindow):
#     def __init__(self,mainWndw):
#         super(SecondWindow_Form, self).__init__()
#         self.ui = Ui_ReportPropertyWindow()
#         self.mainWndw=mainWndw
#         self.gv_image=self.mainWndw.ui.gv_image.scene.vap_image
#         self.ui.setupUi(self)
#         self.cntrlr = SecondWindow_Controller(self,self.ui,self.mainWndw)
#         self.initilizeComponent()
#         appIcon = QIcon("Resources/app_icon.ico")
#         self.setWindowIcon(appIcon)
#         return
#
#     def initilizeComponent(self):
#         # region buttons click
#         self.ui.pbtn_create_pdf.clicked.connect(self.cntrlr.pbtn_create_pdf_clicked)
#
