# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindow_UI.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QFrame, QGraphicsView,
    QGroupBox, QHBoxLayout, QMainWindow, QPushButton,
    QRadioButton, QSizePolicy, QSpacerItem, QStackedWidget,
    QVBoxLayout, QWidget)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1000, 850)
        MainWindow.setMinimumSize(QSize(1000, 850))
        MainWindow.setMaximumSize(QSize(1000, 850))
        MainWindow.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        MainWindow.setAnimated(False)
        self.wgt_main = QWidget(MainWindow)
        self.wgt_main.setObjectName(u"wgt_main")
        self.wgt_main.setStyleSheet(u"#frm_main{\n"
"background-color : #282828;\n"
"}")
        self.verticalLayout_2 = QVBoxLayout(self.wgt_main)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.frm_main = QFrame(self.wgt_main)
        self.frm_main.setObjectName(u"frm_main")
        self.frm_main.setStyleSheet(u"")
        self.lyt_main = QVBoxLayout(self.frm_main)
        self.lyt_main.setObjectName(u"lyt_main")
        self.lyt_main.setContentsMargins(15, 15, 15, 15)
        self.frm_middle_content = QFrame(self.frm_main)
        self.frm_middle_content.setObjectName(u"frm_middle_content")
        self.frm_middle_content.setStyleSheet(u"#frm_left_menu{\n"
"background-color:#5d5d5d;\n"
"border: 1px solid white;\n"
"border-radius:10px;\n"
"}\n"
"\n"
"#frm_right_scene{\n"
"background-color:#5d5d5d;\n"
"border: 1px solid white;\n"
"border-radius:10px;\n"
"}\n"
"QPushButton {\n"
"    background-color: qradialgradient(\n"
"        cx: 0.5, cy: 0.5, radius: 0.5,\n"
"        fx: 0.5, fy: 0.3,\n"
"        stop: 0 #FFA500, /* Orta turuncu */\n"
"        stop: 0.7 #FF8C00, /* Daha koyu turuncu kenar */\n"
"        stop: 1 #CC8400 /* En d\u0131\u015f koyu turuncu */\n"
"    );\n"
"    border: 1px solid white; /* Gri kenar */\n"
"    border-radius: 15px; /* Yuvarlat\u0131lm\u0131\u015f k\u00f6\u015feler */\n"
"    box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.4); /* G\u00f6lge efekti */\n"
"    color: black; /* Yaz\u0131 siyah */\n"
"    font-weight: bold; /* Yaz\u0131 kal\u0131n */\n"
"    padding: 10px;\n"
"    text-align: center;\n"
"    transition: all 0.3s ease; /* Yumu\u015fak ge\u00e7i\u015f efekti */\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    backgroun"
                        "d-color: qradialgradient(\n"
"        cx: 0.5, cy: 0.5, radius: 0.5,\n"
"        fx: 0.5, fy: 0.3,\n"
"        stop: 0 #FFD700, /* Daha parlak turuncu */\n"
"        stop: 0.7 #FFA500, /* Orta turuncu */\n"
"        stop: 1 #FF8C00 /* Koyu turuncu */\n"
"    );\n"
"    transform: scale(1.05); /* Hafif b\u00fcy\u00fcme efekti */\n"
"    box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.5); /* Daha belirgin g\u00f6lge */\n"
"    color: black; /* Yaz\u0131 siyah */\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: qradialgradient(\n"
"        cx: 0.5, cy: 0.5, radius: 0.5,\n"
"        fx: 0.5, fy: 0.3,\n"
"        stop: 0 #CC8400, /* Bas\u0131ld\u0131\u011f\u0131nda koyula\u015f\u0131r */\n"
"        stop: 0.7 #A65E00,\n"
"        stop: 1 #7A4600\n"
"    );\n"
"    box-shadow: inset 0px 4px 6px rgba(0, 0, 0, 0.6); /* \u0130\u00e7 g\u00f6lge efekti */\n"
"    transform: scale(0.95); /* Hafif k\u00fc\u00e7\u00fclme efekti */\n"
"    color: black; /* Yaz\u0131 siyah */\n"
"}\n"
"\n"
"QPushButton:checked {\n"
" "
                        "   background-color: qradialgradient(\n"
"        cx: 0.5, cy: 0.5, radius: 0.5,\n"
"        fx: 0.5, fy: 0.3,\n"
"        stop: 0 #32CD32, /* Ye\u015fil merkez */\n"
"        stop: 0.7 #228B22, /* Koyu ye\u015fil kenar */\n"
"        stop: 1 #006400 /* En koyu ye\u015fil */\n"
"    );\n"
"    border: 1px solid white; /* K\u0131rm\u0131z\u0131 kenar */\n"
"    box-shadow: 0px 8px 15px white; /* K\u0131rm\u0131z\u0131 g\u00f6lge */\n"
"    color: black; /* Yaz\u0131 siyah */\n"
"}\n"
"\n"
"/* Disabled Durumu */\n"
"QPushButton:disabled {\n"
"    background-color: qradialgradient(\n"
"        cx: 0.5, cy: 0.5, radius: 0.5,\n"
"        fx: 0.5, fy: 0.3,\n"
"        stop: 0 #A0A0A0, /* Soluk gri merkez */\n"
"        stop: 0.7 #808080, /* Gri tonlar */\n"
"        stop: 1 #606060 /* Daha koyu gri */\n"
"    );\n"
"    border: 1px solid white; /* Gri kenar */\n"
"    box-shadow: none; /* G\u00f6lge kald\u0131r\u0131l\u0131r */\n"
"    color: black; /* Yaz\u0131 siyah */\n"
"}\n"
"")
        self.lyt_middle_content = QHBoxLayout(self.frm_middle_content)
        self.lyt_middle_content.setSpacing(15)
        self.lyt_middle_content.setObjectName(u"lyt_middle_content")
        self.frm_left_menu = QFrame(self.frm_middle_content)
        self.frm_left_menu.setObjectName(u"frm_left_menu")
        self.frm_left_menu.setMinimumSize(QSize(140, 0))
        self.frm_left_menu.setMaximumSize(QSize(140, 16777215))
        self.frm_left_menu.setStyleSheet(u"#frm_left_menu{\n"
"background-color: #3A3A3A;\n"
"border-radius:10px;\n"
"}\n"
"")
        self.lyt_left_menu = QVBoxLayout(self.frm_left_menu)
        self.lyt_left_menu.setObjectName(u"lyt_left_menu")
        self.lyt_left_menu.setContentsMargins(6, 6, 6, 6)
        self.frm_left_menu_buttons = QFrame(self.frm_left_menu)
        self.frm_left_menu_buttons.setObjectName(u"frm_left_menu_buttons")
        self.frm_left_menu_buttons.setFrameShape(QFrame.Shape.StyledPanel)
        self.frm_left_menu_buttons.setFrameShadow(QFrame.Shadow.Raised)
        self.lyt_left_menu_buttons = QVBoxLayout(self.frm_left_menu_buttons)
        self.lyt_left_menu_buttons.setObjectName(u"lyt_left_menu_buttons")
        self.lyt_left_menu_buttons.setContentsMargins(0, 0, 0, 0)
        self.pbtn_menu_loadImage = QPushButton(self.frm_left_menu_buttons)
        self.pbtn_menu_loadImage.setObjectName(u"pbtn_menu_loadImage")
        self.pbtn_menu_loadImage.setMinimumSize(QSize(60, 30))
        self.pbtn_menu_loadImage.setStyleSheet(u"")

        self.lyt_left_menu_buttons.addWidget(self.pbtn_menu_loadImage)

        self.frm_left_menu_stack_style = QFrame(self.frm_left_menu_buttons)
        self.frm_left_menu_stack_style.setObjectName(u"frm_left_menu_stack_style")
        self.frm_left_menu_stack_style.setMaximumSize(QSize(16777215, 150))
        self.frm_left_menu_stack_style.setStyleSheet(u"#frm_left_menu_stack_style{\n"
"background-color: #5d5d5d;\n"
"border-radius:10px;\n"
"}")
        self.frm_left_menu_stack_style.setFrameShape(QFrame.Shape.Box)
        self.frm_left_menu_stack_style.setFrameShadow(QFrame.Shadow.Plain)
        self.verticalLayout_3 = QVBoxLayout(self.frm_left_menu_stack_style)
        self.verticalLayout_3.setSpacing(5)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(5, 5, 5, 5)
        self.frm_image_processing_btns = QFrame(self.frm_left_menu_stack_style)
        self.frm_image_processing_btns.setObjectName(u"frm_image_processing_btns")
        self.verticalLayout_5 = QVBoxLayout(self.frm_image_processing_btns)
        self.verticalLayout_5.setSpacing(6)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 5, 5, 1)
        self.pbtn_menu_denoise = QPushButton(self.frm_image_processing_btns)
        self.pbtn_menu_denoise.setObjectName(u"pbtn_menu_denoise")
        self.pbtn_menu_denoise.setEnabled(False)
        self.pbtn_menu_denoise.setMinimumSize(QSize(60, 30))
        self.pbtn_menu_denoise.setMaximumSize(QSize(16777215, 16777215))

        self.verticalLayout_5.addWidget(self.pbtn_menu_denoise)

        self.pbtn_menu_segment = QPushButton(self.frm_image_processing_btns)
        self.pbtn_menu_segment.setObjectName(u"pbtn_menu_segment")
        self.pbtn_menu_segment.setEnabled(False)
        self.pbtn_menu_segment.setMinimumSize(QSize(60, 30))

        self.verticalLayout_5.addWidget(self.pbtn_menu_segment)

        self.pbtn_menu_skeletonize = QPushButton(self.frm_image_processing_btns)
        self.pbtn_menu_skeletonize.setObjectName(u"pbtn_menu_skeletonize")
        self.pbtn_menu_skeletonize.setEnabled(False)
        self.pbtn_menu_skeletonize.setMinimumSize(QSize(60, 30))

        self.verticalLayout_5.addWidget(self.pbtn_menu_skeletonize)


        self.verticalLayout_3.addWidget(self.frm_image_processing_btns)


        self.lyt_left_menu_buttons.addWidget(self.frm_left_menu_stack_style)

        self.frm_left_menu_analyse = QFrame(self.frm_left_menu_buttons)
        self.frm_left_menu_analyse.setObjectName(u"frm_left_menu_analyse")
        self.frm_left_menu_analyse.setEnabled(False)
        self.frm_left_menu_analyse.setStyleSheet(u"#frm_left_menu_analyse{\n"
"background-color: #5d5d5d;\n"
"border-radius:10px;\n"
"}")
        self.frm_left_menu_analyse.setFrameShape(QFrame.Shape.StyledPanel)
        self.frm_left_menu_analyse.setFrameShadow(QFrame.Shadow.Raised)
        self.lyt_left_menu_analyse = QVBoxLayout(self.frm_left_menu_analyse)
        self.lyt_left_menu_analyse.setObjectName(u"lyt_left_menu_analyse")
        self.lyt_left_menu_analyse.setContentsMargins(5, 5, 5, 10)
        self.gbx_left_menu_analyse_method = QGroupBox(self.frm_left_menu_analyse)
        self.gbx_left_menu_analyse_method.setObjectName(u"gbx_left_menu_analyse_method")
        self.gbx_left_menu_analyse_method.setEnabled(False)
        self.gbx_left_menu_analyse_method.setMaximumSize(QSize(16777215, 80))
        self.verticalLayout = QVBoxLayout(self.gbx_left_menu_analyse_method)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(5, 3, 1, 5)
        self.radioBtn_imgPrcssng = QRadioButton(self.gbx_left_menu_analyse_method)
        self.radioBtn_imgPrcssng.setObjectName(u"radioBtn_imgPrcssng")
        self.radioBtn_imgPrcssng.setChecked(True)

        self.verticalLayout.addWidget(self.radioBtn_imgPrcssng)

        self.radioBtn_deepLearning = QRadioButton(self.gbx_left_menu_analyse_method)
        self.radioBtn_deepLearning.setObjectName(u"radioBtn_deepLearning")

        self.verticalLayout.addWidget(self.radioBtn_deepLearning)


        self.lyt_left_menu_analyse.addWidget(self.gbx_left_menu_analyse_method)

        self.pbtn_menu_analyse = QPushButton(self.frm_left_menu_analyse)
        self.pbtn_menu_analyse.setObjectName(u"pbtn_menu_analyse")
        self.pbtn_menu_analyse.setEnabled(False)
        self.pbtn_menu_analyse.setMinimumSize(QSize(60, 30))

        self.lyt_left_menu_analyse.addWidget(self.pbtn_menu_analyse)

        self.grpbx_analyse_branchPoint = QGroupBox(self.frm_left_menu_analyse)
        self.grpbx_analyse_branchPoint.setObjectName(u"grpbx_analyse_branchPoint")
        self.lyt_analyse_branchPoint = QVBoxLayout(self.grpbx_analyse_branchPoint)
        self.lyt_analyse_branchPoint.setObjectName(u"lyt_analyse_branchPoint")
        self.lyt_analyse_branchPoint.setContentsMargins(5, 6, 1, 6)
        self.chbx_analyse_showBranchPoints = QCheckBox(self.grpbx_analyse_branchPoint)
        self.chbx_analyse_showBranchPoints.setObjectName(u"chbx_analyse_showBranchPoints")
        self.chbx_analyse_showBranchPoints.setChecked(True)

        self.lyt_analyse_branchPoint.addWidget(self.chbx_analyse_showBranchPoints)

        self.chbx_analyse_showBranchPointMarker = QCheckBox(self.grpbx_analyse_branchPoint)
        self.chbx_analyse_showBranchPointMarker.setObjectName(u"chbx_analyse_showBranchPointMarker")
        self.chbx_analyse_showBranchPointMarker.setChecked(True)

        self.lyt_analyse_branchPoint.addWidget(self.chbx_analyse_showBranchPointMarker)

        self.chbx_analyse_showBranchPointCenter = QCheckBox(self.grpbx_analyse_branchPoint)
        self.chbx_analyse_showBranchPointCenter.setObjectName(u"chbx_analyse_showBranchPointCenter")
        self.chbx_analyse_showBranchPointCenter.setChecked(True)

        self.lyt_analyse_branchPoint.addWidget(self.chbx_analyse_showBranchPointCenter)


        self.lyt_left_menu_analyse.addWidget(self.grpbx_analyse_branchPoint)

        self.grpbx_analyse_tipPoint = QGroupBox(self.frm_left_menu_analyse)
        self.grpbx_analyse_tipPoint.setObjectName(u"grpbx_analyse_tipPoint")
        self.lyt_analyse_tipPoint = QVBoxLayout(self.grpbx_analyse_tipPoint)
        self.lyt_analyse_tipPoint.setObjectName(u"lyt_analyse_tipPoint")
        self.lyt_analyse_tipPoint.setContentsMargins(5, 6, 1, 6)
        self.chbx_analyse_showTipPoints = QCheckBox(self.grpbx_analyse_tipPoint)
        self.chbx_analyse_showTipPoints.setObjectName(u"chbx_analyse_showTipPoints")
        self.chbx_analyse_showTipPoints.setChecked(True)

        self.lyt_analyse_tipPoint.addWidget(self.chbx_analyse_showTipPoints)

        self.chbx_analyse_showTipPointMarker = QCheckBox(self.grpbx_analyse_tipPoint)
        self.chbx_analyse_showTipPointMarker.setObjectName(u"chbx_analyse_showTipPointMarker")
        self.chbx_analyse_showTipPointMarker.setChecked(True)

        self.lyt_analyse_tipPoint.addWidget(self.chbx_analyse_showTipPointMarker)

        self.chbx_analyse_showTipPointCenter = QCheckBox(self.grpbx_analyse_tipPoint)
        self.chbx_analyse_showTipPointCenter.setObjectName(u"chbx_analyse_showTipPointCenter")
        self.chbx_analyse_showTipPointCenter.setChecked(True)

        self.lyt_analyse_tipPoint.addWidget(self.chbx_analyse_showTipPointCenter)


        self.lyt_left_menu_analyse.addWidget(self.grpbx_analyse_tipPoint)

        self.grpbx_analyse_branchPath = QGroupBox(self.frm_left_menu_analyse)
        self.grpbx_analyse_branchPath.setObjectName(u"grpbx_analyse_branchPath")
        self.lyt_analyse_branchPath = QVBoxLayout(self.grpbx_analyse_branchPath)
        self.lyt_analyse_branchPath.setObjectName(u"lyt_analyse_branchPath")
        self.lyt_analyse_branchPath.setContentsMargins(5, 6, 1, 6)
        self.chbx_analyse_showBranchPaths = QCheckBox(self.grpbx_analyse_branchPath)
        self.chbx_analyse_showBranchPaths.setObjectName(u"chbx_analyse_showBranchPaths")
        self.chbx_analyse_showBranchPaths.setChecked(True)

        self.lyt_analyse_branchPath.addWidget(self.chbx_analyse_showBranchPaths)

        self.chbx_analyse_showBranchPathId = QCheckBox(self.grpbx_analyse_branchPath)
        self.chbx_analyse_showBranchPathId.setObjectName(u"chbx_analyse_showBranchPathId")
        self.chbx_analyse_showBranchPathId.setChecked(True)

        self.lyt_analyse_branchPath.addWidget(self.chbx_analyse_showBranchPathId)

        self.chbx_analyse_showBranchPathLenght = QCheckBox(self.grpbx_analyse_branchPath)
        self.chbx_analyse_showBranchPathLenght.setObjectName(u"chbx_analyse_showBranchPathLenght")
        self.chbx_analyse_showBranchPathLenght.setChecked(True)

        self.lyt_analyse_branchPath.addWidget(self.chbx_analyse_showBranchPathLenght)


        self.lyt_left_menu_analyse.addWidget(self.grpbx_analyse_branchPath)


        self.lyt_left_menu_buttons.addWidget(self.frm_left_menu_analyse)

        self.pbtn_menu_report = QPushButton(self.frm_left_menu_buttons)
        self.pbtn_menu_report.setObjectName(u"pbtn_menu_report")
        self.pbtn_menu_report.setEnabled(False)
        self.pbtn_menu_report.setMinimumSize(QSize(60, 30))
        self.pbtn_menu_report.setCheckable(True)

        self.lyt_left_menu_buttons.addWidget(self.pbtn_menu_report)

        self.pbtn_menu_close = QPushButton(self.frm_left_menu_buttons)
        self.pbtn_menu_close.setObjectName(u"pbtn_menu_close")
        self.pbtn_menu_close.setMinimumSize(QSize(60, 30))

        self.lyt_left_menu_buttons.addWidget(self.pbtn_menu_close)


        self.lyt_left_menu.addWidget(self.frm_left_menu_buttons)


        self.lyt_middle_content.addWidget(self.frm_left_menu)

        self.frm_right_scene = QFrame(self.frm_middle_content)
        self.frm_right_scene.setObjectName(u"frm_right_scene")
        self.frm_right_scene.setStyleSheet(u"#frm_right_scene QFrame{\n"
"background-color: #5d5d5d;\n"
"\n"
"}\n"
"#page_image_processing{\n"
"background-color: #3A3A3A;\n"
"\n"
"}\n"
"\n"
"#page_report{\n"
"background-color: #3A3A3A;\n"
"\n"
"}\n"
"\n"
"")
        self.lyt_right_scene = QVBoxLayout(self.frm_right_scene)
        self.lyt_right_scene.setSpacing(0)
        self.lyt_right_scene.setObjectName(u"lyt_right_scene")
        self.lyt_right_scene.setContentsMargins(9, 9, 9, 9)
        self.frm_right_scene_content = QFrame(self.frm_right_scene)
        self.frm_right_scene_content.setObjectName(u"frm_right_scene_content")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frm_right_scene_content.sizePolicy().hasHeightForWidth())
        self.frm_right_scene_content.setSizePolicy(sizePolicy)
        self.frm_right_scene_content.setStyleSheet(u"")
        self.frm_right_scene_content.setFrameShape(QFrame.Shape.StyledPanel)
        self.frm_right_scene_content.setFrameShadow(QFrame.Shadow.Raised)
        self.lyt_right_scene_content = QVBoxLayout(self.frm_right_scene_content)
        self.lyt_right_scene_content.setSpacing(0)
        self.lyt_right_scene_content.setObjectName(u"lyt_right_scene_content")
        self.lyt_right_scene_content.setContentsMargins(0, 0, 0, 0)
        self.wgts_sceneContent = QStackedWidget(self.frm_right_scene_content)
        self.wgts_sceneContent.setObjectName(u"wgts_sceneContent")
        sizePolicy.setHeightForWidth(self.wgts_sceneContent.sizePolicy().hasHeightForWidth())
        self.wgts_sceneContent.setSizePolicy(sizePolicy)
        self.wgts_sceneContent.setStyleSheet(u"")
        self.page_report = QWidget()
        self.page_report.setObjectName(u"page_report")
        sizePolicy.setHeightForWidth(self.page_report.sizePolicy().hasHeightForWidth())
        self.page_report.setSizePolicy(sizePolicy)
        self.verticalLayout_4 = QVBoxLayout(self.page_report)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(5, 5, 5, 5)
        self.page_report_lwdgt = QWidget(self.page_report)
        self.page_report_lwdgt.setObjectName(u"page_report_lwdgt")
        sizePolicy.setHeightForWidth(self.page_report_lwdgt.sizePolicy().hasHeightForWidth())
        self.page_report_lwdgt.setSizePolicy(sizePolicy)
        self.horizontalLayout_2 = QHBoxLayout(self.page_report_lwdgt)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.report_widget = QWidget(self.page_report_lwdgt)
        self.report_widget.setObjectName(u"report_widget")
        self.verticalLayout_6 = QVBoxLayout(self.report_widget)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 9, 0)
        self.pdf_view = QGraphicsView(self.report_widget)
        self.pdf_view.setObjectName(u"pdf_view")

        self.verticalLayout_6.addWidget(self.pdf_view)


        self.horizontalLayout_2.addWidget(self.report_widget)

        self.page_report_pageControl = QFrame(self.page_report_lwdgt)
        self.page_report_pageControl.setObjectName(u"page_report_pageControl")
        self.page_report_pageControl.setMaximumSize(QSize(120, 16777215))
        self.page_report_pageControl.setFrameShape(QFrame.Shape.StyledPanel)
        self.page_report_pageControl.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_7 = QVBoxLayout(self.page_report_pageControl)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_7.addItem(self.verticalSpacer)

        self.pbtn_prev_page = QPushButton(self.page_report_pageControl)
        self.pbtn_prev_page.setObjectName(u"pbtn_prev_page")
        self.pbtn_prev_page.setMinimumSize(QSize(100, 0))

        self.verticalLayout_7.addWidget(self.pbtn_prev_page)

        self.pbtn_next_page = QPushButton(self.page_report_pageControl)
        self.pbtn_next_page.setObjectName(u"pbtn_next_page")
        self.pbtn_next_page.setMinimumSize(QSize(100, 0))

        self.verticalLayout_7.addWidget(self.pbtn_next_page)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_7.addItem(self.verticalSpacer_2)


        self.horizontalLayout_2.addWidget(self.page_report_pageControl)


        self.verticalLayout_4.addWidget(self.page_report_lwdgt)

        self.wgts_sceneContent.addWidget(self.page_report)
        self.page_image_processing = QWidget()
        self.page_image_processing.setObjectName(u"page_image_processing")
        sizePolicy.setHeightForWidth(self.page_image_processing.sizePolicy().hasHeightForWidth())
        self.page_image_processing.setSizePolicy(sizePolicy)
        self.lyt_image_processing = QVBoxLayout(self.page_image_processing)
        self.lyt_image_processing.setObjectName(u"lyt_image_processing")
        self.lyt_image_processing.setContentsMargins(5, 5, 5, 5)
        self.frm_image_processing_content = QFrame(self.page_image_processing)
        self.frm_image_processing_content.setObjectName(u"frm_image_processing_content")
        self.frm_image_processing_content.setStyleSheet(u"#frm_image_processing_content\n"
"{\n"
"\n"
"background-color: #5d5d5d;\n"
"}\n"
"\n"
"")
        self.frm_image_processing_content.setFrameShape(QFrame.Shape.StyledPanel)
        self.frm_image_processing_content.setFrameShadow(QFrame.Shadow.Raised)
        self.lyt_image_processing_content = QVBoxLayout(self.frm_image_processing_content)
        self.lyt_image_processing_content.setObjectName(u"lyt_image_processing_content")
        self.lyt_image_processing_content.setContentsMargins(0, 0, 0, 0)

        self.lyt_image_processing.addWidget(self.frm_image_processing_content)

        self.wgts_sceneContent.addWidget(self.page_image_processing)

        self.lyt_right_scene_content.addWidget(self.wgts_sceneContent)


        self.lyt_right_scene.addWidget(self.frm_right_scene_content)


        self.lyt_middle_content.addWidget(self.frm_right_scene)


        self.lyt_main.addWidget(self.frm_middle_content)


        self.verticalLayout_2.addWidget(self.frm_main)

        MainWindow.setCentralWidget(self.wgt_main)

        self.retranslateUi(MainWindow)

        self.wgts_sceneContent.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"VAP2D: Vessel Analiysis Program", None))
        self.pbtn_menu_loadImage.setText(QCoreApplication.translate("MainWindow", u"Load Image", None))
        self.pbtn_menu_denoise.setText(QCoreApplication.translate("MainWindow", u"Denoise", None))
        self.pbtn_menu_segment.setText(QCoreApplication.translate("MainWindow", u"Segment", None))
        self.pbtn_menu_skeletonize.setText(QCoreApplication.translate("MainWindow", u"Skeletonize", None))
        self.gbx_left_menu_analyse_method.setTitle(QCoreApplication.translate("MainWindow", u"Method", None))
        self.radioBtn_imgPrcssng.setText(QCoreApplication.translate("MainWindow", u"Img Processing", None))
        self.radioBtn_deepLearning.setText(QCoreApplication.translate("MainWindow", u"Deep Learning", None))
        self.pbtn_menu_analyse.setText(QCoreApplication.translate("MainWindow", u"Analyse", None))
        self.grpbx_analyse_branchPoint.setTitle(QCoreApplication.translate("MainWindow", u"Branch Points", None))
        self.chbx_analyse_showBranchPoints.setText(QCoreApplication.translate("MainWindow", u"Show Points", None))
        self.chbx_analyse_showBranchPointMarker.setText(QCoreApplication.translate("MainWindow", u"Show Marker", None))
        self.chbx_analyse_showBranchPointCenter.setText(QCoreApplication.translate("MainWindow", u"Show Center", None))
        self.grpbx_analyse_tipPoint.setTitle(QCoreApplication.translate("MainWindow", u"Tip Points", None))
        self.chbx_analyse_showTipPoints.setText(QCoreApplication.translate("MainWindow", u"Show Points", None))
        self.chbx_analyse_showTipPointMarker.setText(QCoreApplication.translate("MainWindow", u"Show Marker", None))
        self.chbx_analyse_showTipPointCenter.setText(QCoreApplication.translate("MainWindow", u"Show Center", None))
        self.grpbx_analyse_branchPath.setTitle(QCoreApplication.translate("MainWindow", u"Branch Paths", None))
        self.chbx_analyse_showBranchPaths.setText(QCoreApplication.translate("MainWindow", u"Show Paths", None))
        self.chbx_analyse_showBranchPathId.setText(QCoreApplication.translate("MainWindow", u"Show Id", None))
        self.chbx_analyse_showBranchPathLenght.setText(QCoreApplication.translate("MainWindow", u"Show Lenght", None))
        self.pbtn_menu_report.setText(QCoreApplication.translate("MainWindow", u"Report", None))
        self.pbtn_menu_close.setText(QCoreApplication.translate("MainWindow", u"Close", None))
        self.pbtn_prev_page.setText(QCoreApplication.translate("MainWindow", u"< Prev", None))
        self.pbtn_next_page.setText(QCoreApplication.translate("MainWindow", u"Next >", None))
    # retranslateUi

