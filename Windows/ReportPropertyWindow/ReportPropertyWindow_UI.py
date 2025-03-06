# # -*- coding: utf-8 -*-
#
# ################################################################################
# ## Form generated from reading UI file 'ReportPropertyWindow_UI.ui'
# ##
# ## Created by: Qt User Interface Compiler version 6.7.2
# ##
# ## WARNING! All changes made in this file will be lost when recompiling UI file!
# ################################################################################
#
# from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
#     QMetaObject, QObject, QPoint, QRect,
#     QSize, QTime, QUrl, Qt)
# from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
#     QFont, QFontDatabase, QGradient, QIcon,
#     QImage, QKeySequence, QLinearGradient, QPainter,
#     QPalette, QPixmap, QRadialGradient, QTransform)
# from PySide6.QtWidgets import (QApplication, QCheckBox, QFrame, QHBoxLayout,
#     QMainWindow, QPushButton, QSizePolicy, QVBoxLayout,
#     QWidget)
#
# class Ui_ReportPropertyWindow(object):
#     def setupUi(self, ReportPropertyWindow):
#         if not ReportPropertyWindow.objectName():
#             ReportPropertyWindow.setObjectName(u"SecondWindow")
#         ReportPropertyWindow.resize(424, 201)
#         self.wdgt_sw = QWidget(ReportPropertyWindow)
#         self.wdgt_sw.setObjectName(u"wdgt_sw")
#         self.wdgt_sw.setStyleSheet(u"#wdgt_sw{\n"
# "background-color : #282828;\n"
# "}")
#         self.verticalLayout_2 = QVBoxLayout(self.wdgt_sw)
#         self.verticalLayout_2.setObjectName(u"verticalLayout_2")
#         self.frm_sw = QFrame(self.wdgt_sw)
#         self.frm_sw.setObjectName(u"frm_sw")
#         self.frm_sw.setStyleSheet(u"#frm_sw{\n"
# "background-color:#5d5d5d;\n"
# "border: 1px solid white;\n"
# "border-radius:10px;\n"
# "}")
#         self.verticalLayout = QVBoxLayout(self.frm_sw)
#         self.verticalLayout.setObjectName(u"verticalLayout")
#         self.frm_choices_sw = QFrame(self.frm_sw)
#         self.frm_choices_sw.setObjectName(u"frm_choices_sw")
#         self.horizontalLayout = QHBoxLayout(self.frm_choices_sw)
#         self.horizontalLayout.setObjectName(u"horizontalLayout")
#         self.frm_menu2_sw = QFrame(self.frm_choices_sw)
#         self.frm_menu2_sw.setObjectName(u"frm_menu2_sw")
#         self.verticalLayout_4 = QVBoxLayout(self.frm_menu2_sw)
#         self.verticalLayout_4.setObjectName(u"verticalLayout_4")
#         self.checkBox_sw_bpCount = QCheckBox(self.frm_menu2_sw)
#         self.checkBox_sw_bpCount.setObjectName(u"checkBox_sw_bpCount")
#
#         self.verticalLayout_4.addWidget(self.checkBox_sw_bpCount)
#
#         self.checkBox_sw_tpCount = QCheckBox(self.frm_menu2_sw)
#         self.checkBox_sw_tpCount.setObjectName(u"checkBox_sw_tpCount")
#
#         self.verticalLayout_4.addWidget(self.checkBox_sw_tpCount)
#
#         self.checkBox_sw_vCount = QCheckBox(self.frm_menu2_sw)
#         self.checkBox_sw_vCount.setObjectName(u"checkBox_sw_vCount")
#
#         self.verticalLayout_4.addWidget(self.checkBox_sw_vCount)
#
#         self.checkBox_sw_veinSEP = QCheckBox(self.frm_menu2_sw)
#         self.checkBox_sw_veinSEP.setObjectName(u"checkBox_sw_veinSEP")
#
#         self.verticalLayout_4.addWidget(self.checkBox_sw_veinSEP)
#
#
#         self.horizontalLayout.addWidget(self.frm_menu2_sw)
#
#         self.frm_menu1_sw = QFrame(self.frm_choices_sw)
#         self.frm_menu1_sw.setObjectName(u"frm_menu1_sw")
#         self.verticalLayout_3 = QVBoxLayout(self.frm_menu1_sw)
#         self.verticalLayout_3.setObjectName(u"verticalLayout_3")
#         self.checkBox_sw_tvLength = QCheckBox(self.frm_menu1_sw)
#         self.checkBox_sw_tvLength.setObjectName(u"checkBox_sw_tvLength")
#
#         self.verticalLayout_3.addWidget(self.checkBox_sw_tvLength)
#
#         self.checkBox_sw_avLength = QCheckBox(self.frm_menu1_sw)
#         self.checkBox_sw_avLength.setObjectName(u"checkBox_sw_avLength")
#
#         self.verticalLayout_3.addWidget(self.checkBox_sw_avLength)
#
#         self.checkBox_sw_evLength = QCheckBox(self.frm_menu1_sw)
#         self.checkBox_sw_evLength.setObjectName(u"checkBox_sw_evLength")
#
#         self.verticalLayout_3.addWidget(self.checkBox_sw_evLength)
#
#         self.checkBox_sw_vSEPType = QCheckBox(self.frm_menu1_sw)
#         self.checkBox_sw_vSEPType.setObjectName(u"checkBox_sw_vSEPType")
#
#         self.verticalLayout_3.addWidget(self.checkBox_sw_vSEPType)
#
#
#         self.horizontalLayout.addWidget(self.frm_menu1_sw)
#
#
#         self.verticalLayout.addWidget(self.frm_choices_sw)
#
#         self.frm_pbtn_sw = QFrame(self.frm_sw)
#         self.frm_pbtn_sw.setObjectName(u"frm_pbtn_sw")
#         self.frm_pbtn_sw.setFrameShape(QFrame.Shape.StyledPanel)
#         self.frm_pbtn_sw.setFrameShadow(QFrame.Shadow.Raised)
#         self.pbtn_create_pdf = QPushButton(self.frm_pbtn_sw)
#         self.pbtn_create_pdf.setObjectName(u"pbtn_create_pdf")
#         self.pbtn_create_pdf.setGeometry(QRect(300, 0, 75, 24))
#         self.pbtn_create_pdf.setStyleSheet(u"QPushButton{\n"
# " background-color: #FE6E00;\n"
# " border-style: solid;\n"
# " border-width:1px;\n"
# " border-radius:9px;\n"
# " border-color: white;\n"
# "}\n"
# "\n"
# "QPushButton:hover{\n"
# "background-color: #FF8303;\n"
# "}")
#
#         self.verticalLayout.addWidget(self.frm_pbtn_sw)
#
#
#         self.verticalLayout_2.addWidget(self.frm_sw)
#
#         ReportPropertyWindow.setCentralWidget(self.wdgt_sw)
#
#         self.retranslateUi(ReportPropertyWindow)
#
#         QMetaObject.connectSlotsByName(ReportPropertyWindow)
#     # setupUi
#
#     def retranslateUi(self, SecondWindow):
#         SecondWindow.setWindowTitle(QCoreApplication.translate("SecondWindow", u"Vessel Analiysis Program | Report", None))
#         self.checkBox_sw_bpCount.setText(QCoreApplication.translate("SecondWindow", u"branch points count", None))
#         self.checkBox_sw_tpCount.setText(QCoreApplication.translate("SecondWindow", u"tip point count", None))
#         self.checkBox_sw_vCount.setText(QCoreApplication.translate("SecondWindow", u"vein count", None))
#         self.checkBox_sw_veinSEP.setText(QCoreApplication.translate("SecondWindow", u"vein start/end points", None))
#         self.checkBox_sw_tvLength.setText(QCoreApplication.translate("SecondWindow", u"total vein length", None))
#         self.checkBox_sw_avLength.setText(QCoreApplication.translate("SecondWindow", u"average vein length", None))
#         self.checkBox_sw_evLength.setText(QCoreApplication.translate("SecondWindow", u"length of each vein", None))
#         self.checkBox_sw_vSEPType.setText(QCoreApplication.translate("SecondWindow", u"vein start/end point types", None))
#         self.pbtn_create_pdf.setText(QCoreApplication.translate("SecondWindow", u"CREATE PDF", None))
#     # retranslateUi
#
