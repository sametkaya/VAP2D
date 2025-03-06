from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow, QGraphicsScene

from Windows.Custom.VAP_QGraphicsView import VAP_QGraphicsView
from Windows.MainWindow.MainWindow_Controller import MainWindow_Controller
from Windows.MainWindow.MainWindow_UI import Ui_MainWindow


class MainWindow_Form(QMainWindow):
    def __init__(self):
        super(MainWindow_Form, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.cntlr = MainWindow_Controller(self,self.ui)
        self.initilizeComponent()


        self.ui.radioBtn_imgPrcssng.setChecked(True)

        appIcon = QIcon("Resources/app_icon.ico")
        self.setWindowIcon(appIcon)
        return

    def initilizeComponent(self):
        # region image
        self.ui.gv_image = VAP_QGraphicsView(self.ui.frm_image_processing_content)
        self.ui.lyt_image_processing_content.addWidget(self.ui.gv_image)
        self.ui.wgts_sceneContent.setCurrentWidget(self.ui.page_image_processing)


        #endregion


        # region buttons click
        self.ui.pbtn_menu_loadImage.clicked.connect(self.cntlr.pbtn_menu_loadImage_clicked)
        self.ui.pbtn_menu_denoise.clicked.connect(self.cntlr.pbtn_menu_denoise_clicked)
        self.ui.pbtn_menu_segment.clicked.connect(self.cntlr.pbtn_menu_segment_clicked)
        self.ui.pbtn_menu_skeletonize.clicked.connect(self.cntlr.pbtn_menu_skeletonize_clicked)
        self.ui.pbtn_menu_analyse.clicked.connect(self.cntlr.pbtn_menu_analyse_clicked)
        self.ui.pbtn_menu_report.clicked.connect(self.cntlr.pbtn_menu_report_clicked)
        self.ui.pbtn_menu_close.clicked.connect(self.cntlr.pbtn_menu_close_clicked)

        self.ui.chbx_analyse_showBranchPoints.clicked.connect(self.cntlr.chbx_analyse_showBranchPoints_clicked)
        self.ui.chbx_analyse_showBranchPointMarker.clicked.connect(self.cntlr.chbx_analyse_showBranchPoints_clicked)
        self.ui.chbx_analyse_showBranchPointCenter.clicked.connect(self.cntlr.chbx_analyse_showBranchPoints_clicked)

        self.ui.chbx_analyse_showTipPoints.clicked.connect(self.cntlr.chbx_analyse_showTipPoints_clicked)
        self.ui.chbx_analyse_showTipPointMarker.clicked.connect(self.cntlr.chbx_analyse_showTipPoints_clicked)
        self.ui.chbx_analyse_showTipPointCenter.clicked.connect(self.cntlr.chbx_analyse_showTipPoints_clicked)

        #self.ui.chbx_analyse_showTipPoints.clicked.connect(self.cntlr.chbx_analyse_point_properties_clicked)

        self.ui.chbx_analyse_showBranchPaths.clicked.connect(self.cntlr.chbx_analyse_showBranchPaths_clicked)
        self.ui.chbx_analyse_showBranchPathLenght.clicked.connect(self.cntlr.chbx_analyse_showBranchPaths_clicked)
        self.ui.chbx_analyse_showBranchPathId.clicked.connect(self.cntlr.chbx_analyse_showBranchPaths_clicked)

        #self.ui.radioBtn_imgPrcssng.toggled.connect(self.cntlr.start_menu_radio_btn_clicked)
        self.ui.radioBtn_imgPrcssng.clicked.connect(self.cntlr.radioBtn_imgPrcssng_clicked)
        
        self.ui.scene = QGraphicsScene(self)
        self.ui.pdf_view.setScene(self.ui.scene)

        self.ui.pbtn_next_page.clicked.connect(self.cntlr.next_page)
        self.ui.pbtn_prev_page.clicked.connect(self.cntlr.previous_page)

        # endregion
        return



