# This is a sample Python script.
from PySide6 import QtWidgets
from PySide6.QtGui import QIcon

from Windows.MainWindow.MainWindow_Form import MainWindow_Form



# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    appIcon = QIcon("Resources/app_icon.ico")
    app.setWindowIcon(appIcon)
    MyMainWindow = MainWindow_Form()
    MyMainWindow.show()
    sys.exit(app.exec())


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
