import fitz
from PySide6.QtGui import QImage, Qt, QPixmap
from PySide6.QtWidgets import QMessageBox, QMainWindow

from DeepLearning import DeepLearningObjectDetection
from System import ImageProcessing
from System.ImageOperation import ImageOperation
from Windows.MainWindow.MainWindow_UI import Ui_MainWindow
#from Windows.ReportPropertyWindow.ReportPropertyWindow_Form import SecondWindow_Form



#rom System.ImageOperation import ImageOperation
class MainWindow_Controller():
    def __init__(self, mainWindow:QMainWindow ,ui:Ui_MainWindow):
        self.mainWindow=mainWindow
        self.ui = ui
        self.pdf_document=None
        self.page_index=0
        self.MySecondWindow=None
        return

    def radioBtn_imgPrcssng_clicked(self):
        #self.ui.stack_wdgt_left_menu_btns.setCurrentIndex(0)
        return

    def radioBtn_deepLearning_clicked(self):
        #self.ui.stack_wdgt_left_menu_btns.setCurrentIndex(1)
        return
        
    def pbtn_menu_loadImage_clicked(self):
        imagePath= ImageOperation.LoadImages(self.ui.wgt_main)
        if not imagePath == None:
            image_raw, image_gray = ImageProcessing.GetImageFormats(imagePath)
            self.ui.gv_image.scene.SetImage(imagePath, image_raw, image_gray)

            self.ui.gv_image.scene.Update_Branch_Points(False, False, False)
            self.ui.gv_image.scene.Update_Tip_Points(False, False, False)
            self.ui.gv_image.scene.Update_Branch_Paths(False, False, False)

            self.ui.pbtn_menu_denoise.setEnabled(True)
            self.ui.pbtn_menu_skeletonize.setEnabled(True)
            self.ui.pbtn_menu_segment.setEnabled(True)
            self.ui.pbtn_menu_analyse.setEnabled(True)
            self.ui.gbx_left_menu_analyse_method.setEnabled(True)
            self.ui.frm_left_menu_analyse.setEnabled(True)

        return

    def pbtn_menu_denoise_clicked(self):

        if (self.ui.gv_image.scene.vap_image.HasImage()):
            image_byte8 = ImageProcessing.DenoiseImage(self.ui.gv_image.scene.vap_image.image_byte8)
            image_byte8= image_byte8.reshape(image_byte8.shape[0], image_byte8.shape[1])
            self.ui.gv_image.scene.vap_image.SetProcessImage(image_byte8)

            # Save the processed image
            savename = self.ui.gv_image.scene.vap_image.image_raw_name.replace(".jpeg", "_denoiseed.png");
            save_path = self.ui.gv_image.scene.vap_image.image_raw_path.replace(
                self.ui.gv_image.scene.vap_image.image_raw_name, savename);
            import cv2
            cv2.imwrite(save_path, image_byte8)
            #image_byte= self.ui.gv_image.scene.image_byte8
            #qimage = QImage(image_byte.tobytes(), image_byte.shape[1], image_byte.shape[0], QImage.Format_Grayscale8)
            #self.ui.gv_image.setImageBackground(qimage)

        return

    def pbtn_menu_segment_clicked(self):
        if (self.ui.gv_image.scene.vap_image.HasImage()):
            savename = self.ui.gv_image.scene.vap_image.image_raw_name.replace(".jpeg", "_segmented.png");
            save_path = self.ui.gv_image.scene.vap_image.image_raw_path.replace(
                self.ui.gv_image.scene.vap_image.image_raw_name, savename);
            image_byte8 = ImageProcessing.Segment(self.ui.gv_image.scene.vap_image.image_byte8)

            #savename = self.ui.gv_image.scene.vap_image.image_raw_name.replace(".jpeg", "_segmented.png");
            #save_path = self.ui.gv_image.scene.vap_image.image_raw_path.replace(
            #    self.ui.gv_image.scene.vap_image.image_raw_name, savename);
            #import cv2
            #cv2.imwrite(save_path, image_byte8)

            #self.ui.gv_image.scene.qimage_byte = ImageProcessing.BinaryClosing(self.qimage_byte)
            image_byte8 = ImageProcessing.RemoveSmallObject(image_byte8)
            self.ui.gv_image.scene.vap_image.SetProcessImage(image_byte8)
            self.ui.gv_image.scene.vap_image.image_segmented_byte8 = image_byte8

            #image_byte = self.ui.gv_image.scene.qimage_byte
            #qimage = QImage(image_byte.tobytes(), image_byte.shape[1], image_byte.shape[0],  QImage.Format_Grayscale8)
            #self.ui.gv_image.setImageBackground(qimage)

        return

    def pbtn_menu_skeletonize_clicked(self):
        if (self.ui.gv_image.scene.vap_image.HasImage()):
            image_byte8 = ImageProcessing.Skeletonize(self.ui.gv_image.scene.vap_image.image_byte8)
            self.ui.gv_image.scene.vap_image.SetProcessImage(image_byte8)
            self.ui.gv_image.scene.vap_image.image_skeletonized_byte8 = image_byte8
            #image_byte = self.qimage_byte
            #qimage = QImage(image_byte.tobytes(), image_byte.shape[1], image_byte.shape[0],  QImage.Format_Grayscale8)
            #self.ui.gv_image.setImageBackground(qimage)
        return

    def pbtn_menu_analyse_clicked(self):
        self.ui.gv_image.scene.Reset()
        if (self.ui.gv_image.scene.vap_image.HasImage()):
            if (self.ui.radioBtn_imgPrcssng.isChecked()):
                # vap_vein_list= ImageProcessing.find_veins(self.ui.gv_image.scene.vap_image.image_byte8)
                # self.ui.gv_image.scene.AddVeins(vap_vein_list)
                #
                # vap_point_branch_list = ImageProcessing.find_branch_pts(self.ui.gv_image.scene.vap_image.image_byte8)
                # self.ui.gv_image.scene.AddBranchPoints(vap_point_branch_list)
                #
                # vap_point_tip_list = ImageProcessing.find_tips(self.ui.gv_image.scene.vap_image.image_byte8)
                # self.ui.gv_image.scene.AddTipPoints(vap_point_tip_list)
                vascularAreaFraction, whitePixelCount, blackPixelCount = ImageProcessing.GetVascularAreaFractionValues(
                    self.ui.gv_image.scene.vap_image.image_byte8)
                self.ui.gv_image.scene.vap_image.vascularAreaFraction = vascularAreaFraction
                self.ui.gv_image.scene.vap_image.whitePixelCount = whitePixelCount
                self.ui.gv_image.scene.vap_image.blackPixelCount = blackPixelCount

                vap_vein_list,vap_point_branch_list, vap_point_tip_list= ImageProcessing.Analyze(self.ui.gv_image.scene.vap_image.image_byte8)
                self.ui.gv_image.scene.AddVeins(vap_vein_list)
                self.ui.gv_image.scene.AddBranchPoints(vap_point_branch_list)
                self.ui.gv_image.scene.AddTipPoints(vap_point_tip_list)

                self.ui.pbtn_menu_report.setEnabled(True)
            elif (self.ui.radioBtn_deepLearning.isChecked()):
                predicted_results=DeepLearningObjectDetection.predict_objects_from_image(self.ui.gv_image.scene.vap_image.image_byte8)
                vap_point_branch_list,vap_point_tip_list = DeepLearningObjectDetection.find_branch_and_tip_points(self.ui.gv_image.scene.vap_image.image_byte8, predicted_results)
                self.ui.gv_image.scene.AddBranchPoints(vap_point_branch_list)
                self.ui.gv_image.scene.AddTipPoints(vap_point_tip_list)

                # Print results
                for i, (x_min, y_min, x_max, y_max, label) in enumerate(predicted_results):
                    print(f"Object {i + 1}: X_min={x_min}, Y_min={y_min}, X_max={x_max}, Y_max={y_max}, Class={label}")
        return

    def chbx_analyse_showBranchPoints_clicked(self):
        isVisible = self.ui.chbx_analyse_showBranchPoints.isChecked()
        showMarker = self.ui.chbx_analyse_showBranchPointMarker.isChecked()
        showCenter = self.ui.chbx_analyse_showBranchPointCenter.isChecked()


        self.ui.gv_image.scene.Update_Branch_Points(isVisible,showCenter,showMarker)


    def chbx_analyse_showTipPoints_clicked(self):
        isVisible = self.ui.chbx_analyse_showTipPoints.isChecked()
        showMarker = self.ui.chbx_analyse_showTipPointMarker.isChecked()
        showCenter = self.ui.chbx_analyse_showTipPointCenter.isChecked()

        self.ui.gv_image.scene.Update_Tip_Points(isVisible,showCenter,showMarker)


    def chbx_analyse_showBranchPaths_clicked(self):
        isVisible = self.ui.chbx_analyse_showBranchPaths.isChecked()
        showId = self.ui.chbx_analyse_showBranchPathId.isChecked()
        showLenght = self.ui.chbx_analyse_showBranchPathLenght.isChecked()

        self.ui.gv_image.scene.Update_Branch_Paths(isVisible, showId, showLenght)



    def pbtn_menu_report_clicked(self):
        if self.ui.pbtn_menu_report.isChecked():
           # if self.MySecondWindow==None:
            #    self.MySecondWindow = SecondWindow_Form(self)
            #self.MySecondWindow.show()

            informationDict = {}
            informationDict["vaf(%)"] = True
            informationDict["id"] = True
            informationDict["branch points count"] = True
            informationDict["tip point count"] = True
            informationDict["vein count"] = True
            informationDict["total vein length"] = True
            informationDict["average vein length"] = True
            informationDict["p1.x, p1.y"] = True
            informationDict["p2.x, p2.y"] = True
            informationDict["length"] = True
            informationDict["p1_type"] = True
            informationDict["p2_type"] = True

            csv_file_path, pdf_file_path = ImageOperation.SaveInfos(self.ui.gv_image.scene.vap_image, informationDict)
            if csv_file_path!=None:
                self.ui.wgts_sceneContent.setCurrentWidget(self.ui.page_report)
                self.load_pdf(pdf_file_path)
                self.ui.pbtn_menu_report.setText("View")
            else:
                self.ui.pbtn_menu_report.setChecked(False)

        else:
            #self.MySecondWindow.setVisible(False)
            self.ui.wgts_sceneContent.setCurrentWidget(self.ui.page_image_processing)
            self.ui.pbtn_menu_report.setText("Report")
        return

    def pbtn_menu_close_clicked(self):
        confirm = QMessageBox.question(self.mainWindow, "Confirmation", "Are you sure you want to close the application?",
                                       QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.mainWindow.close()
        return

    def load_pdf(self,file_name):
        if file_name:
            self.pdf_document = fitz.open(file_name)
            self.show_page()

    def show_page(self):
        if not self.pdf_document:
            return

        # Load the page and create the pixmap with a high DPI value for better quality
        page = self.pdf_document.load_page(self.page_index)
        dpi = 400  # Increase DPI for even higher quality
        zoom = dpi / 72  # Scale factor for DPI (72 is the default PDF DPI)
        matrix = fitz.Matrix(zoom, zoom)  # Create a transformation matrix for scaling
        image = page.get_pixmap(matrix=matrix, alpha=False)  # Render the page as a pixmap (no alpha for RGB)

        # Convert the pixmap to a QImage
        qt_image = QImage(image.samples, image.width, image.height, image.stride, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Calculate the scale factor to fit the page to the view while maintaining quality
        view_width = self.ui.pdf_view.width()
        view_height = self.ui.pdf_view.height()
        scale_factor = min(view_width / pixmap.width(), view_height / pixmap.height())

        # Scale the pixmap to fit the view while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            pixmap.width() * scale_factor,
            pixmap.height() * scale_factor,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation  # Use smooth transformation for better scaling quality
        )

        # Calculate the center position to display the page in the view
        center_x = (view_width - scaled_pixmap.width()) / 2
        center_y = (view_height - scaled_pixmap.height()) / 2

        # Clear the scene and add the high-quality pixmap
        self.ui.scene.clear()
        pixmap_item = self.ui.scene.addPixmap(scaled_pixmap)

        # Set the scene rectangle to match the view dimensions
        self.ui.pdf_view.setSceneRect(0, 0, view_width, view_height)

        # Center the pixmap in the scene
        pixmap_item.setPos(center_x, center_y)

        # Ensure the view updates properly
        self.ui.pdf_view.update()
    def previous_page(self):
        if self.page_index > 0:
            self.page_index -= 1
            self.show_page()

    def next_page(self):
        if self.pdf_document and self.page_index < len(self.pdf_document) - 1:
            self.page_index += 1
            self.show_page()
