# import json
# import os
#
# from PySide6 import QtWidgets
# from PySide6.QtCore import QDir, QFileInfo
# from PySide6.QtWidgets import QFileDialog, QWidget
# import csv
#
#
#
# class FileSystem(object):
#     projectParentFolderPath = None
#     projectFolderGeneralName = "BAT_Project"
#     projectConfigFileName = "BAT_Project_Config.json"
#     projectFolderPath = None
#
#     projectConfigFilePath = None
#     projectFolderImagesPath = None
#
#     @staticmethod
#     def CreateNewProject(parentWidget: QWidget):
#
#         dlg = QFileDialog(parentWidget)
#         dlg.setFileMode(QFileDialog.FileMode.Directory)
#         home = os.getcwd()
#         dlg.setDirectory(home)
#         dlg.setWindowTitle("Create Project")
#         # dlg.set(QFileDialog.ShowDirsOnly)
#
#         if dlg.exec_():
#
#             path = dlg.selectedFiles()[0]
#             FileSystem.projectParentFolderPath = QDir.toNativeSeparators(path)
#             path = QDir.toNativeSeparators(os.path.join(FileSystem.projectParentFolderPath, FileSystem.projectFolderGeneralName))
#             # print(path)
#             num = 0
#             extend = ""
#             while os.path.exists(path + extend):
#                 extend = "_" + str(num)
#                 num = num + 1
#             path = path + extend
#             FileSystem.projectFolderPath = path
#             # print(path)
#             os.mkdir(path)
#             FileSystem.projectFolderImagesPath = QDir.toNativeSeparators(os.path.join(path, "Images"))
#             os.mkdir(FileSystem.projectFolderImagesPath)
#             FileSystem.projectConfigFilePath = QDir.toNativeSeparators(
#                 os.path.join(path, FileSystem.projectConfigFileName))
#             open(FileSystem.projectConfigFilePath, "w")
#
#     @staticmethod
#     def SaveProject(vimage):
#
#         #jsonDict={}
#         #jsonDict[""]
#         #batImageDict = [bi.toDict() for bi in Data.batImageList]
#         #data = {'Bat Images': batImageDict}
#         #json_string = json.dumps(data)
#         os.makedirs("folder_path")
#
#         os.path.join(folder_path, "my_file.csv")
#         csv_file = open(vimage.image_raw_name.csv, 'w', newline='')
#         csv_writer= csv.writer(csv_file)
#         field = ["vaf(%)", "branch_point_count", "tip_point_count","vein_count","total_vein_lenght",]
#         csv_writer.writerow(field)
#
#         row = [vimage.vascularAreaFraction, len(vimage.branchPoints), len(vimage.tipPoints), len(vimage.vap_veins), vimage.total_vein_length]
#         csv_writer.writerow(row)
#
#         csv_file.close()
#         return
#
#     @staticmethod
#     def OpenProject(parentWidget: QWidget):
#         dlg = QFileDialog(parentWidget)
#         dlg.setFileMode(QFileDialog.ExistingFile)
#         home = os.getcwd()
#         dlg.setDirectory(home)
#         dlg.setWindowTitle("Open Project")
#         dlg.setNameFilter("BAT Project File (BAT_Project_Config.json)")
#
#         if dlg.exec_():
#             batConfigFile = dlg.selectedFiles()[0]
#             FileSystem.projectParentFolderPath = QDir.toNativeSeparators(QFileInfo(batConfigFile).dir().path())
#             FileSystem.projectFolderImagesPath = QDir.toNativeSeparators(os.path.join(FileSystem.projectParentFolderPath, "Images"))
#
#         return
